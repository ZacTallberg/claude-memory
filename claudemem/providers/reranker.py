"""Reranker interface + factory. Used on the full/on-demand path (dashboard, `mem query
--rerank`); NOT on the prompt hot path by default (CPU latency). Degrades to a no-op
pass-through if disabled or unavailable.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from ..config import Config
from ..log import get_logger

log = get_logger(__name__)


class Reranker(ABC):
    name: str

    @abstractmethod
    def rerank(self, query: str, items: list[tuple[int, str]], top_n: int) -> list[tuple[int, float]]:
        """Return [(chunk_id, score), ...] sorted desc, length <= top_n."""

    def available(self) -> bool:
        return True


class NoopReranker(Reranker):
    name = "noop"

    def rerank(self, query: str, items: list[tuple[int, str]], top_n: int) -> list[tuple[int, float]]:
        # Preserve incoming order; assign descending pseudo-scores.
        return [(cid, float(len(items) - i)) for i, (cid, _t) in enumerate(items[:top_n])]

    def available(self) -> bool:
        return False


_cache: dict[str, Reranker] = {}


def get_reranker(cfg: Config) -> Reranker:
    if not cfg.reranker.enabled or cfg.reranker.provider == "none":
        return NoopReranker()
    key = f"{cfg.reranker.provider}:{cfg.reranker.model}"
    if key in _cache:
        return _cache[key]
    try:
        if cfg.reranker.provider == "local":
            from .local_reranker import FastEmbedReranker
            inst: Reranker = FastEmbedReranker(cfg)
        elif cfg.reranker.provider == "cohere":
            from .cloud_rerank import CohereReranker
            inst = CohereReranker(cfg)
        else:
            inst = NoopReranker()
    except Exception as e:
        log.warning("reranker %s unavailable: %s", cfg.reranker.provider, e)
        inst = NoopReranker()
    _cache[key] = inst
    return inst
