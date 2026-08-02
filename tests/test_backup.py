from __future__ import annotations

import json
import sqlite3

import pytest

from system_memory.archive import CanonicalArchive
from system_memory.backup import BackupError, BackupManager

from .conftest import make_event


def test_backup_is_complete_verified_and_restorable(store, tmp_path):
    first = store.ingest(make_event(content="First canonical backup event."))
    second = store.ingest(
        make_event(
            event_key="backup-second",
            session_id="backup-session-2",
            content="Second canonical backup event.",
        )
    )
    snapshot = BackupManager(store).create(tmp_path / "backups")
    manifest = BackupManager.verify_snapshot(snapshot)

    assert manifest.event_count == 2
    assert manifest.archive_file_count == 2
    assert manifest.database_integrity == "ok"
    assert manifest.secret_findings == {}

    restored = BackupManager.restore(snapshot, tmp_path / "restored")
    database = restored / "data" / "system-memory.db"
    connection = sqlite3.connect(f"file:{database.as_posix()}?mode=ro&immutable=1", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        rows = connection.execute(
            "SELECT id,content_sha256,archive_ref FROM memory_events ORDER BY id"
        ).fetchall()
    finally:
        connection.close()
    archive = CanonicalArchive(restored / "data" / "archive")
    assert {row["id"] for row in rows} == {first.event_id, second.event_id}
    assert all(
        archive.verify_event(row["archive_ref"], row["id"], row["content_sha256"]) for row in rows
    )
    receipt = json.loads((restored / "restore-receipt.json").read_text(encoding="utf-8"))
    assert receipt["snapshot_id"] == manifest.snapshot_id


def test_backup_verification_fails_closed_on_payload_tampering(store, tmp_path):
    store.ingest(make_event(content="Tamper-detection event."))
    snapshot = BackupManager(store).create(tmp_path / "backups")
    archive_file = next((snapshot / "data" / "archive").rglob("*.json"))
    archive_file.write_text("{}\n", encoding="utf-8")

    with pytest.raises(BackupError, match="hash mismatch"):
        BackupManager.verify_snapshot(snapshot)


def test_restore_never_overwrites_an_existing_target(store, tmp_path):
    store.ingest(make_event(content="Isolated restore target event."))
    snapshot = BackupManager(store).create(tmp_path / "backups")
    target = tmp_path / "existing-target"
    target.mkdir()

    with pytest.raises(BackupError, match="already exists"):
        BackupManager.restore(snapshot, target)
