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
