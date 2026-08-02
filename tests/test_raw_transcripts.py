from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from system_memory.raw_transcripts import RawTranscriptImporter


def _write_jsonl(path: Path, records: list[dict], *, incomplete: dict | None = None) -> None:
    rendered = "".join(json.dumps(record) + "\n" for record in records)
    if incomplete is not None:
        rendered += json.dumps(incomplete)
    path.write_text(rendered, encoding="utf-8")


def _inventory(tmp_path: Path, sources: list[tuple[Path, str]]) -> Path:
    database = tmp_path / "legacy-inventory.db"
    connection = sqlite3.connect(database)
    connection.execute("CREATE TABLE sources(id INTEGER PRIMARY KEY,path TEXT,meta TEXT)")
    connection.executemany(
        "INSERT INTO sources(path,meta) VALUES (?,?)",
        [(str(path), json.dumps({"provider": provider})) for path, provider in sources],
    )
    connection.commit()
    connection.close()
    return database


def _rows(store):
    with store.database.read() as connection:
        return connection.execute(
            """SELECT provider,provider_event_id,agent_id,session_id,parent_session_id,
                      role,authority,content,metadata
                 FROM memory_events ORDER BY provider,provider_event_id"""
        ).fetchall()


def test_claude_reparse_keeps_authored_text_and_sidechains_only(store, tmp_path):
    transcript = tmp_path / "claude.jsonl"
    _write_jsonl(
        transcript,
        [
            {
                "type": "user",
                "uuid": "user-1",
                "sessionId": "claude-session",
                "timestamp": "2026-08-01T10:00:00Z",
                "message": {"role": "user", "content": "The user's own declaration."},
            },
            {
                "type": "assistant",
                "uuid": "assistant-1",
                "sessionId": "claude-session",
                "timestamp": "2026-08-01T10:00:01Z",
                "message": {
                    "role": "assistant",
                    "content": [
                        {"type": "thinking", "thinking": "Hidden reasoning must not persist."},
                        {"type": "text", "text": "Visible assistant synthesis."},
                        {"type": "tool_use", "name": "Read", "input": {"path": "secret"}},
                    ],
                },
            },
            {
                "type": "user",
                "uuid": "tool-result",
                "sessionId": "claude-session",
                "message": {
                    "role": "user",
                    "content": [{"type": "tool_result", "content": "Tool output."}],
                },
            },
            {
                "type": "user",
                "uuid": "notification",
                "sessionId": "claude-session",
                "message": {
                    "role": "user",
                    "content": "<task-notification>Not authored.</task-notification>",
                },
            },
            {
                "type": "assistant",
                "uuid": "sidechain-1",
                "sessionId": "claude-session",
                "isSidechain": True,
                "message": {"role": "assistant", "content": "Worker-authored synthesis."},
            },
            {
                "type": "user",
                "uuid": "mixed-notice",
                "sessionId": "claude-session",
                "message": {
                    "role": "user",
                    "content": (
                        "<system-reminder>Injected notice.</system-reminder>\n"
                        "Authored text after the notice."
                    ),
                },
            },
        ],
        incomplete={
            "type": "user",
            "uuid": "partial",
            "sessionId": "claude-session",
            "message": {"role": "user", "content": "Incomplete final record."},
        },
    )
    database = _inventory(tmp_path, [(transcript, "claude")])

    report = RawTranscriptImporter(store).import_sources(database)
    rows = _rows(store)
    bodies = "\n".join(row["content"] for row in rows)

    assert report.files_imported == 1
    assert report.events_inserted == 4
    assert report.reasoning_records_skipped == 1
    assert report.tool_records_skipped == 2
    assert report.sidechain_events == 1
    assert "The user's own declaration." in bodies
    assert "Visible assistant synthesis." in bodies
    assert "Worker-authored synthesis." in bodies
    assert "Authored text after the notice." in bodies
    assert "Hidden reasoning" not in bodies
    assert "Tool output" not in bodies
    assert "Injected notice" not in bodies
    assert "Incomplete final record" not in bodies
    sidechain = next(row for row in rows if row["content"] == "Worker-authored synthesis.")
    assert sidechain["agent_id"] == "claude-worker:claude-session"


def test_codex_reparse_preserves_worker_lineage_and_delegation_authority(store, tmp_path):
    transcript = tmp_path / "codex.jsonl"
    _write_jsonl(
        transcript,
        [
            {
                "type": "session_meta",
                "payload": {
                    "id": "worker-thread",
                    "session_id": "parent-main",
                    "parent_thread_id": "parent-main",
                    "agent_path": "/root/reviewer",
                    "thread_source": "subagent",
                    "cwd": "C:/code/game",
                    "agent_nickname": "Sagan",
                    "source": {"subagent": {"thread_spawn": {"agent_nickname": "Sagan"}}},
                },
            },
            {
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": "developer",
                    "content": [{"type": "input_text", "text": "Developer policy."}],
                },
            },
            {"type": "response_item", "payload": {"type": "reasoning", "summary": []}},
            {
                "type": "response_item",
                "payload": {"type": "function_call", "name": "shell_command"},
            },
            {
                "type": "response_item",
                "timestamp": "2026-08-01T11:00:00Z",
                "payload": {
                    "type": "message",
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": (
                                "<codex_delegation><input>Review system-wide memory."
                                "</input></codex_delegation>"
                            ),
                        }
                    ],
                },
            },
            {
                "type": "response_item",
                "timestamp": "2026-08-01T11:00:01Z",
                "payload": {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "Worker review result."}],
                },
            },
            {
                "type": "response_item",
                "timestamp": "2026-08-01T11:00:02Z",
                "payload": {
                    "type": "message",
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": (
                                '<in-app-browser-context source="ambient-ui-state">'
                                "Injected browser state.</in-app-browser-context>\n"
                                "The authored tail remains."
                            ),
                        }
                    ],
                },
            },
        ],
    )
    database = _inventory(tmp_path, [(transcript, "codex")])

    importer = RawTranscriptImporter(store)
    first = importer.import_sources(database)
    second = importer.import_sources(database)
    rows = _rows(store)
    bodies = "\n".join(row["content"] for row in rows)

    assert first.events_inserted == 3
    assert second.events_inserted == 0
    assert second.events_existing == 3
    assert first.reasoning_records_skipped == 1
    assert first.tool_records_skipped == 1
    assert "Developer policy" not in bodies
    assert "Injected browser state" not in bodies
    assert "The authored tail remains." in bodies
    delegated = next(row for row in rows if row["content"] == "Review system-wide memory.")
    assert delegated["session_id"] == "worker-thread"
    assert delegated["parent_session_id"] == "parent-main"
    assert delegated["agent_id"] == "codex:/root/reviewer"
    assert delegated["role"] == "system_observation"
    assert delegated["authority"] == "explicit_decision"
    assert json.loads(delegated["metadata"])["agent_nickname"] == "Sagan"


def test_raw_import_report_never_contains_memory_bodies(store, tmp_path):
    transcript = tmp_path / "claude-private.jsonl"
    private_body = "A private sentence that must never enter an import report."
    _write_jsonl(
        transcript,
        [
            {
                "type": "user",
                "uuid": "private",
                "sessionId": "private-session",
                "message": {"role": "user", "content": private_body},
            }
        ],
    )
    database = _inventory(tmp_path, [(transcript, "claude")])

    report = RawTranscriptImporter(store).import_sources(database)

    assert private_body not in report.model_dump_json()
