from __future__ import annotations

from .conftest import make_event


def test_global_search_uses_project_as_boost_not_boundary(store):
    event_a = store.ingest(
        make_event(
            event_key="project-a-message",
            project_id="project-a",
            content="The luminous tassel interaction should unfold gradually.",
        )
    )
    event_b = store.ingest(
        make_event(
            event_key="project-b-message",
            project_id="project-b",
            session_id="session-b",
            content="The luminous tassel design also appears in the archive viewer.",
        )
    )
    generation = store.create_search_generation(
        corpus_sha256="a" * 64,
        chunker_version="event-v1",
        lexical_config={"tokenizer": "unicode61"},
    )
    store.index_event(event_a.event_id, generation)
    store.index_event(event_b.event_id, generation)
    store.activate_generation(generation)

    results = store.lexical_search("luminous tassel", current_project_id="project-b", limit=6)
    assert {result.project_id for result in results} == {"project-a", "project-b"}
    assert results[0].project_id == "project-b"


def test_explicit_hard_scope_is_available_but_not_default(store):
    first = store.ingest(
        make_event(project_id="project-a", content="A shared ontology term: noema.")
    )
    second = store.ingest(
        make_event(
            event_key="b",
            project_id="project-b",
            session_id="session-b",
            content="Another project also discusses the noema carefully.",
        )
    )
    generation = store.create_search_generation(corpus_sha256="b" * 64, chunker_version="event-v1")
    store.index_event(first.event_id, generation)
    store.index_event(second.event_id, generation)
    store.activate_generation(generation)
    results = store.lexical_search("noema", hard_project_ids=("project-a",))
    assert results
    assert {result.project_id for result in results} == {"project-a"}


def test_lexical_path_abstains_on_nonsense_and_stopwords(store):
    event = store.ingest(
        make_event(content="A meaningful remembered statement about phenomenology.")
    )
    generation = store.create_search_generation(corpus_sha256="c" * 64, chunker_version="event-v1")
    store.index_event(event.event_id, generation)
    store.activate_generation(generation)
    assert store.lexical_search("zzqv flarnoblat kestrelonium wugafrax") == []
    assert store.lexical_search("the and what you have") == []


def test_search_generations_can_independently_index_the_same_event(store):
    event = store.ingest(make_event(content="A stable event can appear in replacement indexes."))
    first = store.create_search_generation(corpus_sha256="1" * 64, chunker_version="event-v1")
    first_document = store.index_event(event.event_id, first)
    store.activate_generation(first)

    second = store.create_search_generation(corpus_sha256="2" * 64, chunker_version="event-v1")
    second_document = store.index_event(event.event_id, second)

    assert first_document != second_document
    with store.database.read() as connection:
        counts = {
            row["generation_id"]: row["amount"]
            for row in connection.execute(
                """SELECT generation_id,COUNT(*) AS amount
                     FROM search_documents GROUP BY generation_id"""
            )
        }
    assert counts == {first: 1, second: 1}


def test_bulk_event_index_is_resumable_and_corpus_hash_tracks_changes(store):
    first = store.ingest(make_event(content="First bulk-indexed event."))
    initial_hash = store.event_corpus_sha256()
    second = store.ingest(
        make_event(
            event_key="second-bulk-event",
            session_id="session-2",
            content="Second bulk-indexed event.",
        )
    )
    updated_hash = store.event_corpus_sha256()
    assert initial_hash != updated_hash

    generation = store.create_search_generation(
        corpus_sha256=updated_hash, chunker_version="event-v1"
    )
    assert store.index_events([first.event_id], generation) == [
        store.index_event(first.event_id, generation)
    ]
    assert store.index_all_events(generation, batch_size=1) == 1
    assert store.index_all_events(generation, batch_size=1) == 0
    with store.database.read() as connection:
        refs = {
            row["ref_id"]
            for row in connection.execute(
                "SELECT ref_id FROM search_documents WHERE generation_id=?", (generation,)
            )
        }
    assert refs == {first.event_id, second.event_id}


def test_activation_fails_closed_if_corpus_changes_during_build(store):
    store.ingest(make_event(content="The event present when index construction begins."))
    corpus_hash = store.event_corpus_sha256()
    generation = store.create_search_generation(
        corpus_sha256=corpus_hash, chunker_version="event-v1"
    )
    store.index_all_events(generation)
    store.ingest(
        make_event(
            event_key="late-event",
            session_id="late-session",
            content="An event arriving after the generation snapshot.",
        )
    )

    try:
        store.activate_generation(generation, expected_event_corpus_sha256=corpus_hash)
    except ValueError as error:
        assert "corpus changed" in str(error)
    else:
        raise AssertionError("changed corpus was incorrectly activated")

    with store.database.read() as connection:
        status = connection.execute(
            "SELECT status FROM search_generations WHERE id=?", (generation,)
        ).fetchone()["status"]
    assert status == "failed"
