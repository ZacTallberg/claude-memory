from __future__ import annotations

from system_memory.models import RecallQuery, RecallScope
from system_memory.recall import RecallEngine

from .conftest import make_event


def prepare(store):
    first = store.ingest(
        make_event(
            event_key="first",
            project_id="ethics",
            content="The testimony principle begins by taking declared experience seriously.",
        )
    )
    second = store.ingest(
        make_event(
            event_key="second",
            session_id="session-2",
            project_id="game",
            content="The testimony principle also shapes how the game records player reports.",
        )
    )
    generation = store.create_search_generation(corpus_sha256="d" * 64, chunker_version="event-v1")
    store.index_event(first.event_id, generation)
    store.index_event(second.event_id, generation)
    store.activate_generation(generation)
    return generation


def test_recall_is_global_by_default_and_truthfully_keyword_only(store):
    generation = prepare(store)
    result = RecallEngine(store).recall(
        RecallQuery(query="testimony principle", current_project_id="ethics")
    )
    assert result.mode == "keyword_only"
    assert result.generation_id == generation
    assert not result.abstained
    assert {item.project_id for item in result.evidence} == {"ethics", "game"}
    assert result.evidence[0].project_id == "ethics"
    with store.database.read() as connection:
        receipt = connection.execute(
            "SELECT mode,result_ids_json FROM retrieval_receipts WHERE request_id=?",
            (result.request_id,),
        ).fetchone()
    assert receipt["mode"] == "keyword_only"
    assert result.evidence[0].document_id in receipt["result_ids_json"]


def test_recall_hard_scope_is_explicit(store):
    prepare(store)
    result = RecallEngine(store).recall(
        RecallQuery(
            query="testimony principle",
            scope=RecallScope(project_ids=("game",), hard_filter=True),
        )
    )
    assert result.evidence
    assert {item.project_id for item in result.evidence} == {"game"}


def test_recall_abstains_instead_of_filling_slots(store):
    prepare(store)
    result = RecallEngine(store).recall(RecallQuery(query="zzqv flarnoblat kestrelonium wugafrax"))
    assert result.mode == "empty"
    assert result.abstained
    assert result.evidence == ()


def test_context_budget_is_enforced(store):
    event = store.ingest(make_event(content="phenomenology " + "careful prose " * 1_000))
    generation = store.create_search_generation(corpus_sha256="e" * 64, chunker_version="event-v1")
    store.index_event(event.event_id, generation)
    store.activate_generation(generation)
    result = RecallEngine(store).recall(
        RecallQuery(query="phenomenology careful prose", max_chars=512)
    )
    assert result.evidence
    assert len(result.evidence[0].text) < 512
