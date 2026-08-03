"""Local embedding provider via fastembed (ONNX Runtime, CPU, no torch)."""
from __future__ import annotations

import threading
from functools import lru_cache

import numpy as np

from ..config import Config
from ..log import get_logger
from .embeddings import EmbeddingProvider

log = get_logger(__name__)


class FastEmbedProvider(EmbeddingProvider):
    def __init__(self, cfg: Config):
        from fastembed import TextEmbedding  # lazy

        self.cfg = cfg
        self.model_name = cfg.embeddings.model
        self._model = TextEmbedding(model_name=self.model_name)
        # Probe native dimension once.
        probe = np.asarray(next(iter(self._model.embed(["probe"]))), dtype=np.float32)
        self._native_dim = int(probe.shape[0])
        want = int(cfg.embeddings.dim)
        if want > self._native_dim:
            raise ValueError(
                f"configured embeddings.dim={want} exceeds model native dim {self._native_dim} "
                f"for {self.model_name!r}; lower dim or pick a larger model")
        self.dim = want
        self.name = f"fastembed:{self.model_name}:{self.dim}"
        self._qpref = cfg.embeddings.query_prefix
        self._dpref = cfg.embeddings.doc_prefix
        # ONNX Runtime already parallelizes one inference across CPU cores. The lane therefore
        # stays serialized, but query waiters have strict priority over document indexing. A
        # document list is split into small batches so live indexing can never monopolize the
        # model for an entire 64-chunk batch while prompt hooks time out around it.
        self._lane = threading.Condition()
        self._lane_busy = False
        self._waiting_queries = 0
        self._document_microbatch_size = cfg.embeddings.document_microbatch_size
        log.info("FastEmbedProvider ready model=%s native_dim=%d use_dim=%d",
                 self.model_name, self._native_dim, self.dim)

    def _post(self, vec) -> list[float]:
        v = np.asarray(vec, dtype=np.float32)
        if self.dim < v.shape[0]:
            v = v[: self.dim]          # Matryoshka truncation
        n = float(np.linalg.norm(v))
        if n > 0:
            v = v / n                  # L2-normalize for cosine
        return v.astype(np.float32).tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        if self._dpref:
            texts = [self._dpref + t for t in texts]
        out: list[list[float]] = []
        size = self._document_microbatch_size
        for start in range(0, len(texts), size):
            batch = texts[start:start + size]
            self._acquire_lane(query=False)
            try:
                out.extend(self._post(v) for v in self._model.embed(batch))
            finally:
                self._release_lane()
        return out

    def embed_query(self, text: str) -> list[float]:
        t = (self._qpref + text) if self._qpref else text
        return list(self._embed_query_cached(t))

    @lru_cache(maxsize=256)
    def _embed_query_cached(self, text: str) -> tuple[float, ...]:
        """Bounded hot-query cache; immutable values are safe to share across callers."""
        self._acquire_lane(query=True)
        try:
            v = next(iter(self._model.embed([text])))
        finally:
            self._release_lane()
        return tuple(self._post(v))

    def _acquire_lane(self, *, query: bool) -> None:
        """Acquire one model-inference lane; queued queries jump ahead of documents."""
        with self._lane:
            if query:
                self._waiting_queries += 1
            try:
                while self._lane_busy or (not query and self._waiting_queries):
                    self._lane.wait()
                self._lane_busy = True
            finally:
                if query:
                    self._waiting_queries -= 1

    def _release_lane(self) -> None:
        with self._lane:
            self._lane_busy = False
            self._lane.notify_all()
