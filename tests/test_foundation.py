from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from system_memory.ids import content_hash, stable_id
from system_memory.normalize import normalize_authored_text
from system_memory.store import IdentityConflict

from .conftest import make_event


def test_stable_ids_are_deterministic_and_type_scoped():
    left = stable_id("evt", "codex", {"b": 2, "a": 1})
    right = stable_id("evt", "codex", {"a": 1, "b": 2})
    assert left == right
    assert left.startswith("evt_")
    assert left != stable_id("src", "codex", {"a": 1, "b": 2})


def test_normalization_preserves_structure_and_removes_injected_blocks():
    raw = (
        "First paragraph.\r\n\r\n"
        "    indented code\r\n"
        "<in-app-browser-context>ambient state</in-app-browser-context>\r\n"
        "Final paragraph."
    )
    result = normalize_authored_text(raw)
    assert "ambient state" not in result.text
    assert "First paragraph.\n\n    indented code" in result.text
    assert result.text.endswith("Final paragraph.")


def test_ingest_is_idempotent_and_archive_precedes_canonical_commit(store):
    event = make_event()
    first = store.ingest(event)
    second = store.ingest(event)

    assert first.inserted is True
    assert second.inserted is False
    assert first.event_id == second.event_id
    assert store.counts()["memory_events"] == 1

    with store.database.read() as connection:
        row = connection.execute(
            "SELECT content,content_sha256,metadata FROM memory_events WHERE id=?",
            (first.event_id,),
        ).fetchone()
        source = connection.execute("SELECT cursor FROM sources").fetchone()
    metadata = json.loads(row["metadata"])
    assert source["cursor"] == 100
    assert row["content_sha256"] == content_hash(row["content"])
    assert store.archive.verify_event(
        path=metadata["archive_ref"],
        event_id=first.event_id,
        payload_hash=row["content_sha256"],
    )


def test_same_provider_identity_with_different_content_is_rejected(store):
    store.ingest(make_event(content="First canonical content."))
    with pytest.raises(IdentityConflict):
        store.ingest(make_event(content="Conflicting content under the same provider identity."))
    assert store.counts()["memory_events"] == 1


def test_concurrent_duplicate_delivery_commits_once(store):
    event = make_event(event_key="duplicate-concurrent")
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda _: store.ingest(event), range(16)))
    assert sum(result.inserted for result in results) == 1
    assert {result.event_id for result in results} == {results[0].event_id}
    assert store.counts()["memory_events"] == 1


def test_secrets_are_redacted_in_database_and_archive(store):
    synthetic = "sk-proj-" + "A1b2" * 8
    result = store.ingest(make_event(content=f"An accidental token appeared: {synthetic}"))
    with store.database.read() as connection:
        row = connection.execute(
            "SELECT content,metadata FROM memory_events WHERE id=?", (result.event_id,)
        ).fetchone()
        tombstones = connection.execute("SELECT COUNT(*) FROM secret_tombstones").fetchone()[0]
    archive_ref = json.loads(row["metadata"])["archive_ref"]
    archive_text = store.archive.resolve_reference(archive_ref).read_text(encoding="utf-8")
    assert synthetic not in row["content"]
    assert synthetic not in archive_text
    assert "[REDACTED_SECRET:openai-token]" in row["content"]
    assert tombstones == 1


def test_secrets_are_also_redacted_from_provenance_metadata(store):
    synthetic = "github_pat_" + "Z9y8" * 8
    event = make_event().model_copy(
        update={
            "source_locator": f"https://example.invalid/?access_token={synthetic}",
            "metadata": {"diagnostic": f"password: {synthetic}"},
        }
    )
    result = store.ingest(event)
    with store.database.read() as connection:
        source = connection.execute(
            "SELECT locator FROM sources WHERE id=?", (result.source_id,)
        ).fetchone()[0]
        metadata = connection.execute(
            "SELECT metadata FROM memory_events WHERE id=?", (result.event_id,)
        ).fetchone()[0]
    assert synthetic not in source
    assert synthetic not in metadata
    assert result.redaction_count >= 2


def test_database_reports_full_durability_and_integrity(store):
    health = store.database.health()
    assert health["ok"] is True
    assert health["schema_version"] == 4
    with store.database.read() as connection:
        assert connection.execute("PRAGMA foreign_keys").fetchone()[0] == 1
        assert connection.execute("PRAGMA synchronous").fetchone()[0] == 2


def test_archive_references_are_portable_and_cannot_escape(store):
    result = store.ingest(make_event(event_key="portable-archive"))
    with store.database.read() as connection:
        row = connection.execute(
            "SELECT archive_ref,metadata FROM memory_events WHERE id=?", (result.event_id,)
        ).fetchone()
    assert row["archive_ref"] and not Path(row["archive_ref"]).is_absolute()
    assert json.loads(row["metadata"])["archive_ref"] == row["archive_ref"]
    assert store.archive.verify_event(row["archive_ref"], result.event_id, result.content_sha256)
    assert not store.archive.verify_event("../outside.json", result.event_id, result.content_sha256)


def test_legacy_absolute_archive_path_is_verified_and_backfilled(store):
    result = store.ingest(make_event(event_key="legacy-absolute-archive"))
    with store.database.read() as connection:
        row = connection.execute(
            "SELECT archive_ref,metadata FROM memory_events WHERE id=?", (result.event_id,)
        ).fetchone()
    absolute = str(store.archive.resolve_reference(row["archive_ref"]))
    metadata = json.loads(row["metadata"])
    metadata.pop("archive_ref")
    metadata["archive_path"] = absolute
    with store.database.write() as connection:
        connection.execute(
            "UPDATE memory_events SET archive_ref=NULL,metadata=? WHERE id=?",
            (json.dumps(metadata), result.event_id),
        )

    assert store.backfill_archive_references() == 1
    with store.database.read() as connection:
        repaired = connection.execute(
            "SELECT archive_ref,metadata FROM memory_events WHERE id=?", (result.event_id,)
        ).fetchone()
    assert repaired["archive_ref"] == row["archive_ref"]
    assert "archive_path" not in json.loads(repaired["metadata"])
