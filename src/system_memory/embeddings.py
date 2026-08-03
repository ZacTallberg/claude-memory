from __future__ import annotations

import hashlib
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

import numpy as np

from .models import EmbeddingManifest


class EmbeddingProvider(Protocol):
    manifest: EmbeddingManifest

    def embed_query(self, text: str) -> list[float]: ...

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...


class FastEmbedProvider:
    """FastEmbed wrapper that enforces the declared model contract exactly."""

    def __init__(
        self,
        manifest: EmbeddingManifest,
        *,
        cache_dir: Path | None = None,
        threads: int | None = None,
        local_files_only: bool = True,
    ) -> None:
        from fastembed import TextEmbedding

        self.manifest = manifest
        if manifest.provider != "fastembed":
            raise ValueError("FastEmbedProvider requires a fastembed manifest")
        if not manifest.artifact_sha256:
            raise ValueError("fastembed generations require an exact artifact fingerprint")
        self._model = TextEmbedding(
            model_name=manifest.model,
            cache_dir=str(cache_dir) if cache_dir else None,
            threads=threads,
            local_files_only=local_files_only,
        )
        revision, artifact = self._model_identity(self._model)
        if revision != manifest.revision:
            raise ValueError(
                f"model revision differs from manifest: runtime={revision} "
                f"manifest={manifest.revision}"
            )
        if artifact != manifest.artifact_sha256:
            raise ValueError("model artifact fingerprint differs from manifest")
        probe = np.asarray(next(iter(self._model.embed(["dimension probe"]))), dtype=np.float32)
        native = int(probe.shape[0])
        if native != manifest.native_dimension:
            raise ValueError(
                f"model native dimension differs from manifest: runtime={native} "
                f"manifest={manifest.native_dimension}"
            )

    @classmethod
    def discover(
        cls,
        model: str,
        *,
        cache_dir: Path | None = None,
        threads: int | None = None,
        normalized: bool = True,
        query_prefix: str = "",
        document_prefix: str = "",
    ) -> FastEmbedProvider:
        """Download/load one model and bind a manifest to its exact cached artifact."""
        from fastembed import TextEmbedding

        loaded = TextEmbedding(
            model_name=model,
            cache_dir=str(cache_dir) if cache_dir else None,
            threads=threads,
            local_files_only=False,
        )
        probe = np.asarray(next(iter(loaded.embed(["dimension probe"]))), dtype=np.float32)
        native = int(probe.shape[0])
        revision, artifact = cls._model_identity(loaded)
        manifest = EmbeddingManifest(
            provider="fastembed",
            model=model,
            revision=revision,
            artifact_sha256=artifact,
            dimension=native,
            native_dimension=native,
            normalized=normalized,
            query_prefix=query_prefix,
            document_prefix=document_prefix,
            matryoshka=False,
        )
        provider = cls.__new__(cls)
        provider.manifest = manifest
        provider._model = loaded
        return provider

    @staticmethod
    def _model_identity(model) -> tuple[str, str]:
        inner = getattr(model, "model", None)
        directory_value = getattr(inner, "_model_dir", None)
        if not directory_value:
            raise ValueError("fastembed did not expose the resolved model snapshot")
        directory = Path(directory_value).resolve()
        if not directory.is_dir():
            raise ValueError("resolved fastembed model snapshot does not exist")
        digest = hashlib.sha256(b"system-memory:fastembed-artifact:v1\n")
        files = sorted(path for path in directory.rglob("*") if path.is_file())
        if not files:
            raise ValueError("resolved fastembed model snapshot is empty")
        for path in files:
            relative = path.relative_to(directory).as_posix()
            digest.update(relative.encode("utf-8"))
            digest.update(b"\0")
            digest.update(str(path.stat().st_size).encode("ascii"))
            digest.update(b"\0")
            with path.open("rb") as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(block)
            digest.update(b"\n")
        return directory.name, digest.hexdigest()

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
