from __future__ import annotations

import time
import uuid
from dataclasses import replace

from .clock import utc_iso
from .embeddings import EmbeddingProvider
from .ids import content_hash
from .inference import InferenceScheduler, InferenceShed, InferenceTimeout, WorkKind
from .models import RecallEvidence, RecallQuery, RecallResult
from .store import MemoryStore, SearchHit
from .vector_index import VectorIndexCache


class RecallEngine:
    """Bounded fast-path retrieval with truthful empty/fallback outcomes."""

    def __init__(
        self,
        store: MemoryStore,
        *,
        lexical_candidates: int = 40,
        abstention_min_score: float = 0.30,
        vector_min_similarity: float = 0.72,
        embedder: EmbeddingProvider | None = None,
        scheduler: InferenceScheduler | None = None,
        query_timeout_seconds: float = 2.0,
    ) -> None:
        self.store = store
        self.lexical_candidates = lexical_candidates
        self.abstention_min_score = abstention_min_score
        self.vector_min_similarity = vector_min_similarity
        self.embedder = embedder
        self.scheduler = scheduler
        self.query_timeout_seconds = query_timeout_seconds
        self.vector_cache = VectorIndexCache()

    def recall(self, request: RecallQuery) -> RecallResult:
        request_id = f"recall_{uuid.uuid4().hex}"
        requested_at = utc_iso()
        started = time.perf_counter()
        generation = self.store.active_generation_id()
        hard = request.scope.hard_filter

        lexical_started = time.perf_counter()
        hits = self.store.lexical_search(
            request.query,
            limit=max(self.lexical_candidates, request.limit),
            current_project_id=request.current_project_id,
            hard_project_ids=request.scope.project_ids if hard else (),
            hard_providers=request.scope.providers if hard else (),
            hard_session_ids=request.scope.session_ids if hard else (),
            exclude_session_ids=request.scope.exclude_session_ids,
            hard_roles=tuple(role.value for role in request.scope.roles) if hard else (),
            as_of=utc_iso(request.as_of) if request.as_of else None,
        )
        lexical_ms = (time.perf_counter() - lexical_started) * 1000

        vector_hits: list[SearchHit] = []
        vector_ran = False
        vector_failure: str | None = None
        vector_ms = 0.0
        manifest = self.store.embedding_manifest(generation) if generation else None
        if manifest and self.embedder:
            if manifest != self.embedder.manifest:
                vector_failure = "active embedding manifest does not match the loaded model"
            elif not self.scheduler:
                vector_failure = "inference scheduler is not configured"
            else:
                vector_started = time.perf_counter()
                try:
                    query_vector = self.scheduler.submit(
                        WorkKind.QUERY_EMBEDDING,
                        lambda: self.embedder.embed_query(request.query),
                        timeout=self.query_timeout_seconds,
                    )
                    index = self.vector_cache.get(self.store, generation)
                    vector_hits = index.search(
                        query_vector,
                        limit=max(self.lexical_candidates, request.limit),
                        hard_project_ids=request.scope.project_ids if hard else (),
                        hard_providers=request.scope.providers if hard else (),
                        hard_session_ids=request.scope.session_ids if hard else (),
                        exclude_session_ids=request.scope.exclude_session_ids,
                        hard_roles=(
                            tuple(role.value for role in request.scope.roles) if hard else ()
                        ),
                        as_of=utc_iso(request.as_of) if request.as_of else None,
                    )
                    vector_hits = [
                        hit for hit in vector_hits if hit.vector_score >= self.vector_min_similarity
                    ]
                    vector_ran = True
                except (InferenceShed, InferenceTimeout, ValueError, RuntimeError) as error:
                    vector_failure = type(error).__name__
                finally:
                    vector_ms = (time.perf_counter() - vector_started) * 1000

        fused = self._fuse(hits, vector_hits) if vector_ran else hits

        # Soft facets increase relevance without blocking cross-project or cross-provider
        # memory. Explicit hard scope was already applied in SQL.
        rescored = [self._apply_soft_scope(hit, request) for hit in fused]
        rescored.sort(key=lambda hit: (-hit.score, hit.document_id))
        accepted = [hit for hit in rescored if hit.score >= self.abstention_min_score]
        selected = self._fit_budget(accepted, request.limit, request.max_chars)

        elapsed_ms = (time.perf_counter() - started) * 1000
        if not selected:
            mode = "empty"
            reason = "no evidence crossed the calibrated fast-path threshold"
        elif vector_ran:
            mode = "hybrid"
            reason = None
        else:
            mode = "keyword_only"
            reason = vector_failure or "dense retrieval generation is not active"

        evidence = tuple(self._evidence(hit) for hit in selected)
        self.store.record_retrieval(
            request_id=request_id,
            query_sha256=content_hash(request.query),
            requested_at=requested_at,
            delivered_at=utc_iso(),
            mode=mode,
            generation_id=generation,
            current_project_id=request.current_project_id,
            stage_latency={
                "lexical_ms": round(lexical_ms, 3),
                "vector_ms": round(vector_ms, 3),
                "total_ms": round(elapsed_ms, 3),
            },
            result_ids=[item.document_id for item in selected],
            fallback_reason=reason if mode == "keyword_only" else None,
        )
        return RecallResult(
            request_id=request_id,
            mode=mode,
            evidence=evidence,
            elapsed_ms=elapsed_ms,
            generation_id=generation,
            abstained=not bool(selected),
            reason=reason,
        )

    @staticmethod
    def _fuse(lexical: list[SearchHit], vector: list[SearchHit]) -> list[SearchHit]:
        weights = ((lexical, 1.2), (vector, 1.0))
        maximum = sum(weight / 61.0 for _, weight in weights)
        by_id: dict[str, SearchHit] = {}
        scores: dict[str, float] = {}
        for lane, weight in weights:
            for rank, hit in enumerate(lane, start=1):
                scores[hit.document_id] = scores.get(hit.document_id, 0.0) + weight / (60 + rank)
                current = by_id.get(hit.document_id)
                if current is None:
                    by_id[hit.document_id] = hit
                else:
                    by_id[hit.document_id] = replace(
                        current,
                        lexical_score=max(current.lexical_score, hit.lexical_score),
                        exact_score=max(current.exact_score, hit.exact_score),
                        vector_score=max(current.vector_score, hit.vector_score),
                    )
        return [
            replace(hit, fusion_score=scores[document] / maximum) for document, hit in by_id.items()
        ]

    @staticmethod
    def _apply_soft_scope(hit: SearchHit, request: RecallQuery) -> SearchHit:
        if request.scope.hard_filter:
            return hit
        project = 0.0
        if request.scope.project_ids and hit.project_id in request.scope.project_ids:
            project += 0.10
        if request.current_provider and hit.provider == request.current_provider:
            project += 0.04
        if request.scope.providers and hit.provider in request.scope.providers:
            project += 0.04
        if request.scope.session_ids and hit.session_id in request.scope.session_ids:
            project += 0.05
        if request.scope.roles and hit.role in {role.value for role in request.scope.roles}:
            project += 0.03
        if not project:
            return hit
        return SearchHit(**{**hit.__dict__, "project_boost": hit.project_boost + project})

    @staticmethod
    def _fit_budget(hits: list[SearchHit], limit: int, max_chars: int) -> list[SearchHit]:
        chosen: list[SearchHit] = []
        used = 0
        for hit in hits:
            cost = len(hit.title) + len(hit.body) + 160
            if chosen and used + cost > max_chars:
                continue
            if not chosen and cost > max_chars:
                shortened = hit.body[: max(64, max_chars - len(hit.title) - 180)].rstrip()
                hit = SearchHit(**{**hit.__dict__, "body": shortened + "…"})
                cost = len(hit.title) + len(hit.body) + 160
            chosen.append(hit)
            used += cost
            if len(chosen) >= limit:
                break
        return chosen

    @staticmethod
    def _evidence(hit: SearchHit) -> RecallEvidence:
        reasons = ["lexical-match"]
        if hit.exact_score:
            reasons.append("exact-term-support")
        if hit.project_boost:
            reasons.append("context-boost")
        return RecallEvidence(
            document_id=hit.document_id,
            memory_type=hit.memory_type,
            ref_id=hit.ref_id,
            title=hit.title,
            text=hit.body,
            provider=hit.provider,
            project_id=hit.project_id,
            session_id=hit.session_id,
            role=hit.role,
            authority=hit.authority,
            occurred_at=hit.occurred_at,
            score=hit.score,
            reasons=tuple(reasons),
        )
