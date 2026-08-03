from __future__ import annotations

import threading
from dataclasses import replace
from typing import Any

import numpy as np

from .store import MemoryStore, SearchHit


class VectorIndex:
    def __init__(
        self,
        *,
        generation_id: str,
        documents: tuple[SearchHit, ...],
        matrix: np.ndarray,
    ) -> None:
        self.generation_id = generation_id
        self.documents = documents
        self.matrix = matrix

    @classmethod
    def load(cls, store: MemoryStore, generation_id: str) -> VectorIndex:
        manifest = store.embedding_manifest(generation_id)
        if not manifest:
            raise ValueError("generation has no embedding manifest")
        records = store.vector_records(generation_id)
        return cls._from_records(generation_id, manifest.dimension, records)

    @classmethod
    def load_live(cls, store: MemoryStore, generation_id: str) -> VectorIndex | None:
        manifest = store.embedding_manifest(generation_id)
        if not manifest:
            raise ValueError("generation has no embedding manifest")
        records = store.live_vector_records(generation_id)
        if not records:
            return None
        return cls._from_records(generation_id, manifest.dimension, records)

    @classmethod
    def _from_records(
        cls, generation_id: str, dimension: int, records: list[dict[str, Any]]
    ) -> VectorIndex:
        if not records:
            raise ValueError("generation has no vectors")
        matrix = np.empty((len(records), dimension), dtype=np.float32)
        documents: list[SearchHit] = []
        for index, row in enumerate(records):
            if int(row["dimension"]) != dimension:
                raise ValueError("stored vector dimension differs from generation manifest")
            vector = np.frombuffer(row["vector"], dtype="<f4")
            if int(vector.shape[0]) != dimension:
                raise ValueError("stored vector payload has the wrong length")
            matrix[index] = vector
            documents.append(
                SearchHit(
                    document_id=row["id"],
                    memory_type=row["memory_type"],
                    ref_id=row["ref_id"],
                    provider=row["provider"],
                    project_id=row["project_id"],
                    task_id=row["task_id"],
                    session_id=row["session_id"],
                    role=row["role"],
                    authority=row["authority"],
                    occurred_at=row["occurred_at"],
                    title=row["title"],
                    body=row["body"],
                    content_sha256=row["content_sha256"],
                    lexical_score=0.0,
                    exact_score=0.0,
                    project_boost=0.0,
                )
            )
        return cls(generation_id=generation_id, documents=tuple(documents), matrix=matrix)

    def search(
        self,
        query_vector: list[float],
        *,
        limit: int,
        hard_project_ids: tuple[str, ...] = (),
        hard_providers: tuple[str, ...] = (),
        hard_session_ids: tuple[str, ...] = (),
        exclude_session_ids: tuple[str, ...] = (),
        hard_roles: tuple[str, ...] = (),
        as_of: str | None = None,
    ) -> list[SearchHit]:
        query = np.asarray(query_vector, dtype=np.float32)
        if query.ndim != 1 or query.shape[0] != self.matrix.shape[1]:
            raise ValueError("query vector dimension differs from active generation")
        norm = float(np.linalg.norm(query))
        if norm <= 0:
            return []
        query = query / norm
        scores = self.matrix @ query
        eligible = np.ones(len(self.documents), dtype=bool)
        if hard_project_ids:
            eligible &= np.asarray(
                [doc.project_id in hard_project_ids for doc in self.documents], dtype=bool
            )
        if hard_providers:
            eligible &= np.asarray(
                [doc.provider in hard_providers for doc in self.documents], dtype=bool
            )
        if hard_session_ids:
            eligible &= np.asarray(
                [doc.session_id in hard_session_ids for doc in self.documents], dtype=bool
            )
        if exclude_session_ids:
            eligible &= np.asarray(
                [doc.session_id not in exclude_session_ids for doc in self.documents],
                dtype=bool,
            )
        if hard_roles:
            eligible &= np.asarray([doc.role in hard_roles for doc in self.documents], dtype=bool)
        if as_of:
            eligible &= np.asarray(
                [doc.occurred_at is None or doc.occurred_at <= as_of for doc in self.documents],
                dtype=bool,
            )
        indexes = np.flatnonzero(eligible)
        if not len(indexes):
            return []
        ordered = indexes[np.argsort(-scores[indexes], kind="stable")[:limit]]
        return [
            replace(self.documents[int(index)], vector_score=float(scores[int(index)]))
            for index in ordered
        ]


class VectorIndexCache:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._index: VectorIndex | None = None
        self._live_index: VectorIndex | None = None
        self._live_key: tuple[str, int, int] | None = None

    def get(self, store: MemoryStore, generation_id: str) -> VectorIndex:
        current = self._index
        if current and current.generation_id == generation_id:
            return current
        with self._lock:
            current = self._index
            if current and current.generation_id == generation_id:
                return current
            loaded = VectorIndex.load(store, generation_id)
            self._index = loaded
            return loaded

    def get_live(self, store: MemoryStore, generation_id: str) -> VectorIndex | None:
        count, newest = store.live_vector_version(generation_id)
        key = (generation_id, count, newest)
        if self._live_key == key:
            return self._live_index
        with self._lock:
            if self._live_key == key:
                return self._live_index
            loaded = VectorIndex.load_live(store, generation_id) if count else None
            self._live_index = loaded
            self._live_key = key
            return loaded

    def search(
        self,
        store: MemoryStore,
        generation_id: str,
        query_vector: list[float],
        *,
        limit: int,
        **filters,
    ) -> list[SearchHit]:
        hits = self.get(store, generation_id).search(query_vector, limit=limit, **filters)
        live = self.get_live(store, generation_id)
        if live:
            hits.extend(live.search(query_vector, limit=limit, **filters))
        hits.sort(key=lambda item: (-item.vector_score, item.document_id))
        return hits[:limit]
