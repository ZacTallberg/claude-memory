"""Hybrid retrieval: BM25 + vector candidate lists fused by Reciprocal Rank Fusion,
re-weighted by recency, deduped to distinct sessions, and optionally cross-encoder reranked.

Two tiers:
  - 'hot'  : keyword + vector, NO rerank (prompt hot path; latency-critical)
  - 'full' : + cross-encoder rerank of the top candidates (dashboard / `mem query --rerank`)
"""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone

from .config import Config
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
                 embedder: EmbeddingProvider | None = None, reranker: Reranker | None = None):
        self.cfg = cfg
        self.store = store
        self.embedder = embedder or get_embedding_provider(cfg)
        self.reranker = reranker or get_reranker(cfg)

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
               qvec: list[float] | None = None) -> list[Result]:
        if not query or not query.strip():
            return []
        rc = self.cfg.recall
        k = k or rc.top_k
        bm25 = self.store.search_bm25(query, rc.bm25_k, exclude_session=exclude_session)

        if qvec is None:
            qvec = self.embed_query(query)
        vec = self.store.search_vector(qvec, rc.vector_k, exclude_session=exclude_session) if qvec else []

        fused = self._rrf([bm25, vec])
        if not fused:
            return []

        chunks = {c.id: c for c in self.store.get_chunks(list(fused.keys()))}

        # recency-weighted fused score
        scored: list[tuple[int, float, float]] = []  # (id, weighted, fused_raw)
        for cid, f in fused.items():
            ch = chunks.get(cid)
            if ch is None:
                continue
            w = f * (0.7 + 0.3 * self._recency(ch.ts))
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
                     qvec: list[float] | None = None) -> list[Fact]:
        k = k or self.cfg.recall.facts_k
        if qvec is None:
            qvec = self.embed_query(query)
        return self.store.search_facts(query, k, qvec=qvec)[:k]
