"""Hybrid retrieval: BM25 + vector candidate lists fused by Reciprocal Rank Fusion,
re-weighted by recency, deduped to distinct sessions, and optionally cross-encoder reranked.

Two tiers:
  - 'hot'  : keyword + vector, NO rerank (prompt hot path; latency-critical)
  - 'full' : + cross-encoder rerank of the top candidates (dashboard / `mem query --rerank`)
"""
from __future__ import annotations

import math
import re
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone

from .config import Config
from .lifecycle import fact_is_active, fact_is_recallable
from .providers.embeddings import EmbeddingProvider, get_embedding_provider
from .providers.reranker import Reranker, get_reranker
from .store.base import Chunk, Fact, Store


@dataclass
class Result:
    chunk: Chunk
    score: float
    fused: float
    reranked: bool = False


class Retriever:
    def __init__(self, cfg: Config, store: Store,
                 embedder: EmbeddingProvider | None = None, reranker: Reranker | None = None,
                 chunk_rank_multiplier: Callable[[Chunk], float] | None = None,
                 fact_rank_multiplier: Callable[[Fact], float] | None = None):
        self.cfg = cfg
        self.store = store
        self.embedder = embedder or get_embedding_provider(cfg)
        self.reranker = reranker or get_reranker(cfg)
        # Offline experiments may inject bounded multipliers. Production construction never does;
        # learned feedback therefore cannot silently alter the prompt path.
        self.chunk_rank_multiplier = chunk_rank_multiplier
        self.fact_rank_multiplier = fact_rank_multiplier

    # ---- fusion + weighting ----
    def _rrf(self, lists: list[list]) -> dict[int, float]:
        k = self.cfg.recall.rrf_k
        scores: dict[int, float] = defaultdict(float)
        for lst in lists:
            for cand in lst:
                scores[cand.chunk_id] += 1.0 / (k + cand.rank)
        return scores

    def _recency(self, ts: datetime | None) -> float:
        if not ts:
            return 0.7
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        age = max(0.0, (datetime.now(timezone.utc) - ts).total_seconds() / 86400.0)
        return 0.5 ** (age / max(1.0, self.cfg.recall.recency_half_life_days))

    # ---- main search ----
    def embed_query(self, query: str) -> list[float] | None:
        """Embed once; callers pass the result to both search() and search_facts()
        so the hot path embeds the query a single time."""
        if not query or not query.strip() or not self.embedder.available():
            return None
        try:
            return self.embedder.embed_query(query)
        except Exception:
            return None

    def search(self, query: str, *, tier: str = "full", exclude_session: str | None = None,
               k: int | None = None, do_rerank: bool | None = None,
               qvec: list[float] | None = None, use_bm25: bool = True,
               use_vector: bool = True) -> list[Result]:
        if not query or not query.strip():
            return []
        rc = self.cfg.recall
        k = k or rc.top_k
        bm25 = (self.store.search_bm25(query, rc.bm25_k, exclude_session=exclude_session)
                if use_bm25 else [])

        if qvec is None and use_vector:
            qvec = self.embed_query(query)
        vec = (self.store.search_vector(qvec, rc.vector_k, exclude_session=exclude_session)
               if use_vector and qvec else [])

        fused = self._rrf([bm25, vec])
        if not fused:
            return []

        chunks = {c.id: c for c in self.store.get_chunks(list(fused.keys()))}

        # Anti-memory: deliberately suppressed content (stale or wrong advice) never resurfaces.
        try:
            anti = self.store.list_anti_memory()
        except Exception:
            anti = []
        if anti:
            anti_ids = {a.get("chunk_id") for a in anti if a.get("chunk_id")}
            anti_keys = [a["key"].lower() for a in anti if a.get("key") and len(a["key"]) >= 6]
            chunks = {cid: ch for cid, ch in chunks.items()
                      if cid not in anti_ids
                      and not any(k in ch.content.lower() for k in anti_keys)}

        # recency-weighted fused score
        scored: list[tuple[int, float, float]] = []  # (id, weighted, fused_raw)
        for cid, f in fused.items():
            ch = chunks.get(cid)
            if ch is None:
                continue
            w = f * (0.7 + 0.3 * self._recency(ch.ts))
            if self.chunk_rank_multiplier is not None:
                try:
                    w *= max(0.5, min(1.5, float(self.chunk_rank_multiplier(ch))))
                except Exception:
                    pass
            scored.append((cid, w, f))
        scored.sort(key=lambda x: x[1], reverse=True)

        # dedupe to distinct sessions (keep best per session; None session = always distinct)
        seen: set[str] = set()
        deduped: list[tuple[int, float, float]] = []
        for cid, w, f in scored:
            sess = chunks[cid].session_id
            if sess:
                if sess in seen:
                    continue
                seen.add(sess)
            deduped.append((cid, w, f))

        # optional rerank (full tier only)
        want_rerank = (do_rerank if do_rerank is not None
                       else (tier == "full" or self.cfg.reranker.hot_path))
        results: list[Result]
        if want_rerank and self.reranker.available():
            cand = deduped[: self.cfg.reranker.candidates]
            items = [(cid, self._rerank_text(chunks[cid])) for cid, _w, _f in cand]
            ranked = self.reranker.rerank(query, items, top_n=k)
            results = [Result(chunk=chunks[cid], score=s, fused=fused[cid], reranked=True)
                       for cid, s in ranked]
        else:
            results = [Result(chunk=chunks[cid], score=w, fused=f) for cid, w, f in deduped[:k]]
        return results

    @staticmethod
    def _rerank_text(ch: Chunk) -> str:
        blurb = (ch.context_blurb + " ") if ch.context_blurb else ""
        return (blurb + ch.content)[:2000]

    # ---- curated facts ----
    def search_facts(self, query: str, k: int | None = None,
                     qvec: list[float] | None = None, *, automatic: bool = False,
                     project: str | None = None) -> list[Fact]:
        k = k or self.cfg.recall.facts_k
        if qvec is None:
            qvec = self.embed_query(query)
        # Fetch beyond the final cap so expired/superseded notes cannot crowd out current ones.
        candidates = self.store.search_facts(query, max(k * 6, 24), qvec=qvec)
        # Explicit author metadata can refine rank without model inference. Defaults are nearly
        # neutral, so legacy notes preserve their established ordering.
        weighted_candidates = list(enumerate(candidates, 1))
        def authored_score(item):
            rank, fact = item
            meta = fact.meta or {}
            try:
                importance = max(0.0, min(1.0, float(meta.get("importance", 0.7))))
                confidence = max(0.0, min(1.0, float(meta.get("confidence", 1.0))))
            except (TypeError, ValueError):
                importance, confidence = 0.7, 1.0
            # RRF-scale rank gaps are intentionally small enough for strong explicit metadata to
            # move an adjacent item, while the bounded factors cannot catapult distant noise.
            return (1.0 / (self.cfg.recall.rrf_k + rank)) * (0.9 + 0.1 * importance) * (
                0.95 + 0.05 * confidence)
        weighted_candidates.sort(key=authored_score, reverse=True)
        candidates = [fact for _rank, fact in weighted_candidates]
        if self.fact_rank_multiplier is not None:
            ranked = list(enumerate(candidates, 1))
            def learned_score(item):
                rank, fact = item
                try:
                    multiplier = max(0.5, min(1.5, float(self.fact_rank_multiplier(fact))))
                except Exception:
                    multiplier = 1.0
                return (1.0 / (self.cfg.recall.rrf_k + rank)) * multiplier
            ranked.sort(key=learned_score, reverse=True)
            candidates = [fact for _rank, fact in ranked]
        out: list[Fact] = []
        seen: set[str] = set()
        for fact in candidates:
            if automatic:
                eligible = fact_is_recallable(fact, project=project)
            else:
                eligible = fact_is_active(fact)
            if not eligible:
                continue
            key = " ".join(re.findall(r"[a-z0-9]+", (fact.name or fact.title).lower()))
            if key and key in seen:
                continue
            if key:
                seen.add(key)
            out.append(fact)
            if len(out) >= k:
                break
        return out
