"""The storage DAO contract. Both backends (ParadeDB primary, SQLite fallback) implement
this exact interface. IMPORTANT: fusion (RRF) lives in the retriever, NOT here — the store
only returns ranked candidate lists, keeping both backends simple and portable.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Iterable, Sequence


@dataclass
class Candidate:
    """One hit from a single retriever (bm25 OR vector), with its 1-based rank in that list."""
    chunk_id: int
    rank: int
    score: float


@dataclass
class Chunk:
    id: int
    source_id: int
    kind: str                 # 'text' | 'thinking' | 'tool'
    role: str                 # 'user' | 'assistant'
    session_id: str | None
    project: str
    cwd: str | None
    ts: datetime | None
    content: str
    context_blurb: str | None = None
    ordinal: int = 0
    meta: dict = field(default_factory=dict)


@dataclass
class Fact:
    id: int
    path: str
    project: str
    name: str
    title: str
    description: str
    type: str                 # user | feedback | project | reference
    tags: list[str]
    origin_session_id: str | None
    body: str
    mtime: float | None = None
    meta: dict = field(default_factory=dict)


class Store(ABC):
    name: str

    # --- lifecycle ---
    @abstractmethod
    def connect(self) -> None: ...
    @abstractmethod
    def migrate(self) -> None: ...
    @abstractmethod
    def health(self) -> dict:
        """{'backend','ok','vector','bm25', plus counts}. Never raises."""
    @abstractmethod
    def close(self) -> None: ...

    # --- sources & indexing ---
    @abstractmethod
    def get_source(self, path: str) -> dict | None:
        """{'id','bytes_indexed','mtime',...} or None."""
    @abstractmethod
    def upsert_source(self, *, path: str, kind: str, project: str,
                      session_id: str | None, bytes_indexed: int,
                      mtime: float, meta: dict | None = None) -> int: ...
    @abstractmethod
    def set_bytes_indexed(self, source_id: int, n: int) -> None: ...
    @abstractmethod
    def add_chunks(self, source_id: int, chunks: list[dict]) -> list[int]:
        """Each chunk dict: kind, role, session_id, project, cwd, ts(datetime|None),
        content, context_blurb, ordinal, token_est, meta. Returns new chunk ids."""
    @abstractmethod
    def set_embeddings(self, rows: list[tuple[int, list[float]]]) -> None: ...
    @abstractmethod
    def chunks_missing_embeddings(self, limit: int) -> list[Chunk]: ...
    @abstractmethod
    def delete_source(self, path: str) -> None: ...

    # --- retrieval (return candidate lists; NO fusion here) ---
    @abstractmethod
    def search_bm25(self, query: str, k: int, *, exclude_session: str | None = None,
                    kinds: Sequence[str] | None = None) -> list[Candidate]: ...
    @abstractmethod
    def search_vector(self, qvec: list[float], k: int, *, exclude_session: str | None = None,
                      kinds: Sequence[str] | None = None) -> list[Candidate]: ...
    @abstractmethod
    def get_chunks(self, ids: Iterable[int]) -> list[Chunk]:
        """Fetch full chunks by id (order need not be preserved; caller reorders)."""

    # --- curated facts ---
    @abstractmethod
    def upsert_fact(self, *, path: str, project: str, name: str, title: str, description: str,
                    type: str, tags: list[str], origin_session_id: str | None, body: str,
                    embedding: list[float] | None, mtime: float, meta: dict | None = None) -> int: ...
    @abstractmethod
    def list_facts(self, *, project: str | None = None, type: str | None = None) -> list[Fact]: ...
    @abstractmethod
    def search_facts(self, query: str, k: int, *, qvec: list[float] | None = None) -> list[Fact]: ...
    @abstractmethod
    def get_fact(self, fact_id: int) -> Fact | None: ...
    @abstractmethod
    def delete_fact(self, path: str) -> None: ...
    @abstractmethod
    def facts_titles_map(self) -> dict[str, list[Fact]]:
        """project -> [Fact,...] for the Unify session-start map."""

    # --- graph (derived from [[wikilinks]] in notes) ---
    @abstractmethod
    def replace_graph(self, nodes: list[dict], edges: list[dict]) -> None: ...
    @abstractmethod
    def graph(self) -> dict:
        """{'nodes':[{id,label,type,group}], 'edges':[{source,target,kind}]}"""

    # --- promotion candidates ---
    @abstractmethod
    def add_promotion_candidate(self, *, title: str, body: str, type: str,
                                support: dict, score: float) -> int: ...
    @abstractmethod
    def list_promotions(self, status: str | None = None) -> list[dict]: ...
    @abstractmethod
    def update_promotion(self, pid: int, status: str) -> None: ...

    # --- anti-memory ("don't repeat this") ---
    @abstractmethod
    def add_anti_memory(self, *, key: str, reason: str, chunk_id: int | None = None) -> int: ...
    @abstractmethod
    def list_anti_memory(self) -> list[dict]: ...

    # --- observability ---
    @abstractmethod
    def log_injection(self, *, hook: str, session_id: str | None, prompt_excerpt: str,
                      n_recalled: int, n_facts: int, chars: int, latency_ms: int,
                      details: dict | None = None) -> None: ...
    @abstractmethod
    def recent_injections(self, limit: int = 50) -> list[dict]: ...
    @abstractmethod
    def record_metric(self, metric: str, value: float, *, run_id: str | None = None,
                      details: dict | None = None) -> None: ...
    @abstractmethod
    def metric_series(self, metric: str) -> list[dict]: ...

    # --- misc ---
    @abstractmethod
    def counts(self) -> dict:
        """{'sources','chunks','chunks_embedded','facts','injections',...}"""
    @abstractmethod
    def kv_get(self, key: str) -> dict | None: ...
    @abstractmethod
    def kv_set(self, key: str, value: dict) -> None: ...
