from __future__ import annotations

import json
import sqlite3

from system_memory.legacy_v1 import LegacyV1Importer


def legacy_database(tmp_path):
    available = tmp_path / "available.jsonl"
    available.write_text("{}\n", encoding="utf-8")
    database = tmp_path / "legacy.db"
    connection = sqlite3.connect(database)
    connection.executescript(
        """
        CREATE TABLE sources(
            id INTEGER PRIMARY KEY,path TEXT,kind TEXT,project TEXT,session_id TEXT,
            bytes_indexed INTEGER,mtime REAL,last_indexed TEXT,meta TEXT
        );
        CREATE TABLE chunks(
            id INTEGER PRIMARY KEY,source_id INTEGER,ordinal INTEGER,kind TEXT,role TEXT,
            session_id TEXT,project TEXT,cwd TEXT,ts TEXT,content TEXT,context_blurb TEXT,
            search_text TEXT,token_est INTEGER,meta TEXT
        );
        CREATE TABLE facts(
            id INTEGER PRIMARY KEY,path TEXT,project TEXT,name TEXT,title TEXT,description TEXT,
            type TEXT,tags TEXT,origin_session_id TEXT,body TEXT,search_text TEXT,mtime REAL,
            meta TEXT
        );
        """
    )
    connection.execute(
        "INSERT INTO sources VALUES (1,?,'transcript','code','session-a',1,0,'',?)",
        (str(available), json.dumps({"provider": "claude"})),
    )
    connection.execute(
        "INSERT INTO sources VALUES (2,?,'transcript','code','session-b',1,0,'',?)",
        (str(tmp_path / "missing.jsonl"), json.dumps({})),
    )
    rows = [
        (
            1,
            1,
            0,
            "text",
            "user",
            "session-a",
            "code",
            "C:/code/a",
            "2026-08-01T10:00:00+00:00",
            "Available source memory.",
            "",
            "",
            4,
            "{}",
        ),
        (
            2,
            2,
            0,
            "text",
            "assistant",
            "session-b",
            "code",
            "C:/code/b",
            "2026-08-01T11:00:00+00:00",
            "Recovered missing-source memory.",
            "password: synthetic-secret",
            "",
            4,
            "{}",
        ),
        (
            3,
            2,
            0,
            "text",
            "assistant",
            "session-b",
            "code",
            "C:/code/b",
            "2026-08-01T11:00:00+00:00",
            "Recovered missing-source memory.",
            "password: synthetic-secret",
            "",
            4,
            "{}",
        ),
    ]
    connection.executemany("INSERT INTO chunks VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)", rows)
    connection.execute(
        "INSERT INTO facts VALUES (1,'note.md','code','note','Legacy note','Description',"
        "'fact','[]','session-a','Synthesized body','',1,'{}')"
    )
    connection.commit()
    connection.close()
    return database


def test_legacy_import_is_loss_explicit_deduplicated_and_resumable(store, tmp_path):
    database = legacy_database(tmp_path)
    importer = LegacyV1Importer(store)
    missing = importer.import_database(database, only_missing_sources=True)
    assert missing.chunk_rows_seen == 2
    assert missing.chunk_events_inserted == 1
    assert missing.exact_duplicates_skipped == 1
    assert missing.facts_seen == 0
    assert missing.source_files_missing == 1
    assert missing.source_files_present == 1

    complete = importer.import_database(database)
    assert complete.chunk_events_inserted == 1
    assert complete.chunk_events_existing == 1
    assert complete.fact_events_inserted == 1
    assert complete.exact_duplicates_skipped == 1
    assert store.counts()["memory_events"] == 3

    with store.database.read() as connection:
        rows = connection.execute(
            "SELECT provider,project_id,authority,loss_flags,metadata FROM memory_events "
            "ORDER BY provider"
        ).fetchall()
    assert all(row["project_id"] is None for row in rows)
    assert any(row["provider"] == "legacy-unknown" for row in rows)
    assert any("source_missing" in row["loss_flags"] for row in rows)
    assert any(row["authority"] == "imported_unknown" for row in rows)
    assert all("synthetic-secret" not in row["metadata"] for row in rows)


def test_legacy_import_report_never_contains_memory_bodies(store, tmp_path):
    database = legacy_database(tmp_path)
    report = LegacyV1Importer(store).import_database(database)
    serialized = report.model_dump_json()
    assert "Available source memory" not in serialized
    assert "Recovered missing-source memory" not in serialized
    assert len(report.source_database_sha256) == 64


def test_legacy_import_can_import_facts_without_lossy_chunks(store, tmp_path):
    database = legacy_database(tmp_path)

    report = LegacyV1Importer(store).import_database(database, only_facts=True)

    assert report.only_facts is True
    assert report.chunk_rows_seen == 0
    assert report.facts_seen == 1
    assert report.fact_events_inserted == 1
    assert report.source_files_present == 0
    assert report.source_files_missing == 0
    with store.database.read() as connection:
        rows = connection.execute(
            """SELECT e.provider,s.kind AS source_kind
                 FROM memory_events e JOIN sources s ON s.id=e.source_id
                 ORDER BY e.provider"""
        ).fetchall()
    assert [(row["provider"], row["source_kind"]) for row in rows] == [
        ("legacy-curated", "legacy-v1-fact")
    ]
