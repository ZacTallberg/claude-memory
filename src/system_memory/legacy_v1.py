from __future__ import annotations

import hashlib
import json
import sqlite3
from collections import Counter
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from .ids import content_hash, stable_id
from .models import Authority, EventKind, IngestEvent, Role
from .store import MemoryStore


class LegacyImportReport(BaseModel):
    source_database: str
    source_database_sha256: str
    only_missing_sources: bool
    chunk_rows_seen: int = 0
    chunk_events_inserted: int = 0
    chunk_events_existing: int = 0
    exact_duplicates_skipped: int = 0
    facts_seen: int = 0
    fact_events_inserted: int = 0
    fact_events_existing: int = 0
    source_files_present: int = 0
    source_files_missing: int = 0
    redactions: int = 0
    provider_counts: dict[str, int] = Field(default_factory=dict)
    loss_flag_counts: dict[str, int] = Field(default_factory=dict)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(value: str | None) -> dict[str, Any]:
    try:
        loaded = json.loads(value or "{}")
    except json.JSONDecodeError:
        return {}
    return loaded if isinstance(loaded, dict) else {}


class LegacyV1Importer:
    """One-way, loss-explicit importer for the stabilized v1 SQLite corpus."""

    def __init__(self, store: MemoryStore) -> None:
        self.store = store

    def import_database(
        self,
        database_path: Path,
        *,
        only_missing_sources: bool = False,
        include_facts: bool = True,
        progress_every: int = 1_000,
        on_progress=None,
    ) -> LegacyImportReport:
        database = database_path.resolve()
        if not database.is_file() or database.stat().st_size <= 0:
            raise FileNotFoundError(database)
        report = LegacyImportReport(
            source_database=str(database),
            source_database_sha256=_sha256(database),
            only_missing_sources=only_missing_sources,
        )
        providers: Counter[str] = Counter()
        losses: Counter[str] = Counter()
        imported_chunk_ids, imported_fact_ids = self.store.legacy_v1_row_ids(
            report.source_database_sha256
        )
        connection = sqlite3.connect(f"file:{database.as_posix()}?mode=ro&immutable=1", uri=True)
        connection.row_factory = sqlite3.Row
        try:
            sources: dict[int, dict[str, Any]] = {}
            for row in connection.execute("SELECT * FROM sources ORDER BY id"):
                metadata = _json(row["meta"])
                path = Path(row["path"]) if row["path"] else None
                present = bool(path and path.is_file())
                if present:
                    report.source_files_present += 1
                else:
                    report.source_files_missing += 1
                sources[int(row["id"])] = {
                    "path": str(path) if path else None,
                    "present": present,
                    "provider": str(metadata.get("provider") or "legacy-unknown"),
                    "session_id": row["session_id"],
                    "project_hint": row["project"],
                    "meta": metadata,
                }

            seen: set[tuple[Any, ...]] = set()
            pending_chunks: list[IngestEvent] = []

            def flush_chunks() -> None:
                if not pending_chunks:
                    return
                for result in self.store.ingest_batch(pending_chunks):
                    report.redactions += result.redaction_count
                    if result.inserted:
                        report.chunk_events_inserted += 1
                    else:
                        report.chunk_events_existing += 1
                pending_chunks.clear()

            query = """SELECT c.* FROM chunks c ORDER BY c.source_id,c.ts,c.id"""
            for row in connection.execute(query):
                source = sources[int(row["source_id"])]
                if only_missing_sources and source["present"]:
                    continue
                report.chunk_rows_seen += 1
                body_hash = content_hash(row["content"] or "")
                duplicate_key = (
                    row["source_id"],
                    row["session_id"],
                    row["role"],
                    row["ts"],
                    body_hash,
                )
                if duplicate_key in seen:
                    report.exact_duplicates_skipped += 1
                    continue
                seen.add(duplicate_key)
                if int(row["id"]) in imported_chunk_ids:
                    report.chunk_events_existing += 1
                    continue

                provider = source["provider"]
                providers[provider] += 1
                loss_flags = [
                    "legacy_chunked",
                    "legacy_ordinal_unreliable",
                    "agent_lineage_missing",
                ]
                if source["present"]:
                    loss_flags.append("raw_source_available")
                else:
                    loss_flags.extend(("source_missing", "legacy_recovered"))
                losses.update(loss_flags)
                role = Role.USER if row["role"] == "user" else Role.ASSISTANT
                authority = (
                    Authority.USER_AUTHORED if role == Role.USER else Authority.ASSISTANT_SYNTHESIS
                )
                identity = stable_id(
                    "legacychunk",
                    source["path"] or f"source-{row['source_id']}",
                    row["session_id"],
                    row["role"],
                    row["ts"],
                    body_hash,
                )
                occurred = datetime.fromisoformat(str(row["ts"]).replace("Z", "+00:00"))
                incoming = IngestEvent(
                    provider=provider,
                    source_kind="legacy-v1-chunk",
                    source_locator=(
                        f"legacy-v1://{report.source_database_sha256}/source/{row['source_id']}"
                    ),
                    provider_event_id=identity,
                    agent_id=f"legacy:{provider}:unknown-agent",
                    session_id=row["session_id"],
                    project_id=None,
                    role=role,
                    authority=authority,
                    kind=EventKind.LEGACY_RECOVERED,
                    occurred_at=occurred,
                    content=row["content"] or "",
                    visibility="private",
                    trust="legacy",
                    loss_flags=tuple(loss_flags),
                    metadata={
                        "legacy_source_path": source["path"],
                        "legacy_source_present": source["present"],
                        "legacy_project_hint": row["project"],
                        "legacy_cwd_hint": row["cwd"],
                        "legacy_chunk_id": row["id"],
                        "legacy_ordinal": row["ordinal"],
                        "legacy_kind": row["kind"],
                        "legacy_context_blurb": row["context_blurb"],
                    },
                )
                pending_chunks.append(incoming)
                if len(pending_chunks) >= 250:
                    flush_chunks()
                if on_progress and progress_every and report.chunk_rows_seen % progress_every == 0:
                    on_progress(report)

            flush_chunks()

            if include_facts and not only_missing_sources:
                self._import_facts(
                    connection, report, providers, losses, imported_fact_ids=imported_fact_ids
                )
        finally:
            connection.close()
        report.provider_counts = dict(sorted(providers.items()))
        report.loss_flag_counts = dict(sorted(losses.items()))
        return report

    def _import_facts(
        self,
        connection: sqlite3.Connection,
        report: LegacyImportReport,
        providers: Counter[str],
        losses: Counter[str],
        *,
        imported_fact_ids: set[int],
    ) -> None:
        pending: list[IngestEvent] = []
        for row in connection.execute("SELECT * FROM facts ORDER BY id"):
            report.facts_seen += 1
            if int(row["id"]) in imported_fact_ids:
                report.fact_events_existing += 1
                continue
            rendering = "\n\n".join(
                item.strip()
                for item in (row["title"] or "", row["description"] or "", row["body"] or "")
                if item and item.strip()
            )
            if not rendering:
                continue
            provider = "legacy-curated"
            providers[provider] += 1
            loss_flags = ("legacy_mutable_fact", "assistant_synthesis", "claim_status_unknown")
            losses.update(loss_flags)
            occurred = datetime(1970, 1, 1, tzinfo=UTC) + timedelta(
                seconds=max(0.0, float(row["mtime"] or 0))
            )
            identity = stable_id(
                "legacyfact",
                row["path"] or row["name"] or row["id"],
                content_hash(rendering),
            )
            incoming = IngestEvent(
                provider=provider,
                source_kind="legacy-v1-fact",
                source_locator=(f"legacy-v1://{report.source_database_sha256}/fact/{identity}"),
                provider_event_id=identity,
                agent_id="legacy:curation:unknown-agent",
                session_id=row["origin_session_id"] or "legacy-curated-notes",
                project_id=None,
                role=Role.ASSISTANT,
                authority=Authority.ASSISTANT_SYNTHESIS,
                kind=EventKind.LEGACY_RECOVERED,
                occurred_at=occurred,
                content=rendering,
                visibility="private",
                trust="legacy",
                loss_flags=loss_flags,
                metadata={
                    "legacy_fact_id": row["id"],
                    "legacy_path": row["path"],
                    "legacy_project_hint": row["project"],
                    "legacy_type": row["type"],
                    "legacy_tags": row["tags"],
                },
            )
            pending.append(incoming)
        for result in self.store.ingest_batch(pending):
            report.redactions += result.redaction_count
            if result.inserted:
                report.fact_events_inserted += 1
            else:
                report.fact_events_existing += 1
