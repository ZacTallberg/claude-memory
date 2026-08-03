from __future__ import annotations

import threading

from .embeddings import EmbeddingProvider
from .inference import InferenceScheduler, InferenceShed, InferenceTimeout, WorkKind
from .store import MemoryStore


def embed_generation(
    store: MemoryStore,
    generation_id: str,
    provider: EmbeddingProvider,
    scheduler: InferenceScheduler,
    *,
    timeout_per_batch: float = 120.0,
) -> int:
    manifest = store.embedding_manifest(generation_id)
    if not manifest:
        raise ValueError("generation has no embedding manifest")
    if provider.manifest != manifest:
        raise ValueError("embedding provider does not match the generation manifest")
    total = 0
    while True:
        batch = store.pending_embedding_batch(generation_id, limit=8)
        if not batch:
            return total
        identifiers = [document for document, _ in batch]
        texts = [text for _, text in batch]
        vectors = scheduler.submit(
            WorkKind.DOCUMENT_EMBEDDING,
            lambda batch=texts: provider.embed_documents(batch),
            timeout=timeout_per_batch,
        )
        if len(vectors) != len(identifiers):
            raise ValueError("embedding provider returned the wrong number of vectors")
        total += store.put_embeddings(generation_id, dict(zip(identifiers, vectors, strict=True)))


class LiveEmbeddingWorker:
    """Persistently backfill the active generation's live semantic overlay.

    Work enters the same priority scheduler as interactive queries, in batches no larger
    than eight, so freshness cannot monopolize the model.
    """

    def __init__(
        self,
        store: MemoryStore,
        provider: EmbeddingProvider,
        scheduler: InferenceScheduler,
        *,
        poll_seconds: float = 1.0,
        timeout_per_batch: float = 600.0,
    ) -> None:
        self.store = store
        self.provider = provider
        self.scheduler = scheduler
        self.poll_seconds = max(0.05, poll_seconds)
        self.timeout_per_batch = timeout_per_batch
        self._stop = threading.Event()
        self._wake = threading.Event()
        self._thread: threading.Thread | None = None
        self._state_lock = threading.Lock()
        self._embedded = 0
        self._last_error: str | None = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run,
            name="system-memory-live-embeddings",
            daemon=True,
        )
        self._thread.start()

    def wake(self) -> None:
        self._wake.set()

    def close(self, *, wait: bool = True) -> None:
        self._stop.set()
        self._wake.set()
        if wait and self._thread:
            self._thread.join(timeout=30)

    def status(self) -> dict[str, object]:
        with self._state_lock:
            return {
                "running": bool(self._thread and self._thread.is_alive()),
                "embedded": self._embedded,
                "last_error": self._last_error,
            }

    def _run(self) -> None:
        while not self._stop.is_set():
            generation = self.store.active_generation_id()
            if (
                not generation
                or self.store.embedding_manifest(generation) != self.provider.manifest
            ):
                self._wait()
                continue
            batch = self.store.pending_live_embedding_batch(generation, limit=8)
            if not batch:
                self._wait()
                continue
            identifiers = [document for document, _ in batch]
            texts = [body for _, body in batch]
            try:
                vectors = self.scheduler.submit(
                    WorkKind.DOCUMENT_EMBEDDING,
                    lambda values=texts: self.provider.embed_documents(values),
                    timeout=self.timeout_per_batch,
                )
                if len(vectors) != len(identifiers):
                    raise ValueError("embedding provider returned the wrong number of vectors")
                inserted = self.store.put_live_embeddings(
                    generation, dict(zip(identifiers, vectors, strict=True))
                )
                with self._state_lock:
                    self._embedded += inserted
                    self._last_error = None
            except (InferenceShed, InferenceTimeout, ValueError, RuntimeError) as error:
                code = type(error).__name__
                self.store.defer_live_embeddings(
                    generation,
                    identifiers,
                    error_code=code,
                    delay_seconds=5.0,
                )
                with self._state_lock:
                    self._last_error = code
                self._wait()

    def _wait(self) -> None:
        self._wake.wait(self.poll_seconds)
        self._wake.clear()
