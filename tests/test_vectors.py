from __future__ import annotations

import pytest
from pydantic import ValidationError

from system_memory.embeddings import DeterministicEmbeddingProvider
from system_memory.indexing import embed_generation
from system_memory.inference import InferenceScheduler
from system_memory.models import EmbeddingManifest, RecallQuery
from system_memory.recall import RecallEngine

from .conftest import make_event


def manifest() -> EmbeddingManifest:
    return EmbeddingManifest(
        provider="deterministic-test",
        model="test-vocabulary",
        revision="fixture-1",
        dimension=3,
        native_dimension=3,
        normalized=True,
    )


def build_hybrid(store):
    coral = store.ingest(
        make_event(
            event_key="coral",
            project_id="ethics",
            content="Coral marks the testimony principle in this design.",
        )
    )
    teal = store.ingest(
        make_event(
            event_key="teal",
            session_id="session-teal",
            project_id="game",
            content="Teal marks the procedural memory layer in the game.",
        )
    )
    provider = DeterministicEmbeddingProvider(manifest(), ("coral", "teal", "other"))
    generation = store.create_search_generation(
        corpus_sha256="1" * 64,
        chunker_version="event-v1",
        embedding_manifest=provider.manifest.model_dump(mode="json"),
    )
    store.index_event(coral.event_id, generation)
    store.index_event(teal.event_id, generation)
    return generation, provider


def test_embedding_generation_cannot_activate_partially(store):
    generation, provider = build_hybrid(store)
    status = store.embedding_status(generation)
    assert status == {"documents": 2, "vectors": 0, "pending": 2}
    with pytest.raises(ValueError, match="incomplete"):
        store.activate_generation(generation)

    scheduler = InferenceScheduler(capacity=4)
    assert embed_generation(store, generation, provider, scheduler) == 2
    assert store.embedding_status(generation) == {"documents": 2, "vectors": 2, "pending": 0}
    store.activate_generation(generation)
    scheduler.close()


def test_vector_dimension_is_exact_and_never_blindly_truncated(store):
    generation, _ = build_hybrid(store)
    document = store.pending_embedding_batch(generation, limit=1)[0][0]
    with pytest.raises(ValueError, match="dimension mismatch"):
        store.put_embeddings(generation, {document: [1.0, 0.0]})
    with pytest.raises(ValidationError, match="Matryoshka"):
        EmbeddingManifest(
            provider="unsafe",
            model="ordinary-model",
            revision="1",
            dimension=2,
            native_dimension=3,
            matryoshka=False,
        )


def test_hybrid_recall_uses_global_vector_generation_and_truthful_mode(store):
    generation, provider = build_hybrid(store)
    scheduler = InferenceScheduler(capacity=4)
    embed_generation(store, generation, provider, scheduler)
    store.activate_generation(generation)
    engine = RecallEngine(
        store,
        embedder=provider,
        scheduler=scheduler,
        vector_min_similarity=0.70,
    )
    result = engine.recall(RecallQuery(query="coral testimony", current_project_id="game"))
    assert result.mode == "hybrid"
    assert result.evidence
    assert result.evidence[0].project_id == "ethics"
    assert "exact-term-support" in result.evidence[0].reasons
    scheduler.close()


def test_model_manifest_mismatch_degrades_explicitly_to_keyword(store):
    generation, provider = build_hybrid(store)
    scheduler = InferenceScheduler(capacity=4)
    embed_generation(store, generation, provider, scheduler)
    store.activate_generation(generation)
    wrong_manifest = manifest().model_copy(update={"revision": "different"})
    wrong = DeterministicEmbeddingProvider(wrong_manifest, ("coral", "teal", "other"))
    result = RecallEngine(store, embedder=wrong, scheduler=scheduler).recall(
        RecallQuery(query="coral testimony")
    )
    assert result.mode == "keyword_only"
    assert "manifest" in (result.reason or "")
    scheduler.close()
