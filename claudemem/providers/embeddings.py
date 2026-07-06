"""Embedding provider interface + factory.

The factory NEVER raises: if the configured backend can't load (missing dep, no model,
no API key) it returns a provider whose .available() is False, and callers degrade to
keyword-only retrieval. Concrete impls live in sibling modules and are imported lazily.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from ..config import Config
from ..log import get_logger

log = get_logger(__name__)


class EmbeddingProvider(ABC):
    name: str
    dim: int

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    @abstractmethod
    def embed_query(self, text: str) -> list[float]: ...

    def available(self) -> bool:
        return True


class NullEmbeddingProvider(EmbeddingProvider):
    """Sentinel used when no embedder is usable. Retrieval falls back to keyword-only."""
    name = "null"

    def __init__(self, dim: int, reason: str = ""):
        self.dim = dim
        self.reason = reason

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return []

    def embed_query(self, text: str) -> list[float]:
        return []

    def available(self) -> bool:
        return False


_cache: dict[str, EmbeddingProvider] = {}


def get_embedding_provider(cfg: Config) -> EmbeddingProvider:
    """Singleton per provider+model. Lazy import so the package loads without heavy deps."""
    key = f"{cfg.embeddings.provider}:{cfg.embeddings.model}:{cfg.embeddings.dim}"
    if key in _cache:
        return _cache[key]

    provider = cfg.embeddings.provider
    inst: EmbeddingProvider
    try:
        if provider == "local":
            from .local_fastembed import FastEmbedProvider
            inst = FastEmbedProvider(cfg)
        elif provider == "ollama":
            from .ollama_embed import OllamaEmbedProvider
            inst = OllamaEmbedProvider(cfg)
        elif provider in ("voyage", "cohere"):
            from .cloud_embed import CloudEmbedProvider
            inst = CloudEmbedProvider(cfg)
        else:
            inst = NullEmbeddingProvider(cfg.embeddings.dim, f"unknown provider {provider!r}")
    except Exception as e:  # missing dep / model / key
        log.warning("embedding provider %s unavailable: %s", provider, e)
        inst = NullEmbeddingProvider(cfg.embeddings.dim, str(e))

    _cache[key] = inst
    return inst
