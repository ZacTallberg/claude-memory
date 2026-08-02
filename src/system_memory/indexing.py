from __future__ import annotations

from .embeddings import EmbeddingProvider
from .inference import InferenceScheduler, WorkKind
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
