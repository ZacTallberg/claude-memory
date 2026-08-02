from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import numpy as np

from .models import EmbeddingManifest


class EmbeddingProvider(Protocol):
    manifest: EmbeddingManifest

    def embed_query(self, text: str) -> list[float]: ...

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...


class FastEmbedProvider:
    """FastEmbed wrapper that enforces the declared model contract exactly."""

    def __init__(self, manifest: EmbeddingManifest) -> None:
        from fastembed import TextEmbedding

        self.manifest = manifest
        self._model = TextEmbedding(model_name=manifest.model)
        probe = np.asarray(next(iter(self._model.embed(["dimension probe"]))), dtype=np.float32)
        native = int(probe.shape[0])
        if native != manifest.native_dimension:
            raise ValueError(
                f"model native dimension differs from manifest: runtime={native} "
                f"manifest={manifest.native_dimension}"
            )

    def _finish(self, value) -> list[float]:
        vector = np.asarray(value, dtype=np.float32)
        if self.manifest.dimension != self.manifest.native_dimension:
            if not self.manifest.matryoshka:
                raise ValueError("dimension truncation is forbidden for this model")
            vector = vector[: self.manifest.dimension]
        if int(vector.shape[0]) != self.manifest.dimension:
            raise ValueError("embedding output dimension differs from manifest")
        if self.manifest.normalized:
            norm = float(np.linalg.norm(vector))
            if norm <= 0:
                raise ValueError("embedding provider returned a zero vector")
            vector = vector / norm
        return vector.astype(np.float32).tolist()

    def embed_query(self, text: str) -> list[float]:
        prepared = f"{self.manifest.query_prefix}{text}"
        return self._finish(next(iter(self._model.embed([prepared]))))

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        if len(texts) > 8:
            raise ValueError("background embedding batches are capped at eight documents")
        prepared = [f"{self.manifest.document_prefix}{text}" for text in texts]
        return [self._finish(vector) for vector in self._model.embed(prepared)]


class DeterministicEmbeddingProvider:
    """Small test provider; never used for production generations."""

    def __init__(self, manifest: EmbeddingManifest, vocabulary: tuple[str, ...]) -> None:
        if manifest.dimension != len(vocabulary):
            raise ValueError("test vocabulary length must equal manifest dimension")
        self.manifest = manifest
        self.vocabulary = tuple(term.casefold() for term in vocabulary)

    def _embed(self, text: str) -> list[float]:
        lowered = text.casefold()
        vector = np.asarray([lowered.count(term) for term in self.vocabulary], dtype=np.float32)
        norm = float(np.linalg.norm(vector))
        if norm:
            vector = vector / norm
        elif len(vector):
            vector[-1] = 1.0
        return vector.tolist()

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]
