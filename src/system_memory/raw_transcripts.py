from __future__ import annotations

import json
import re
import sqlite3
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from .clock import utc_iso
from .ids import content_hash
from .models import Authority, EventKind, IngestEvent, Role
from .normalize import normalize_authored_text, strip_injected_blocks
from .store import IdentityConflict, MemoryStore

_CODEX_WRAPPER = re.compile(
    r"<(?P<outer>codex_delegation|codex_internal_context)\b[^>]*>.*?"
    r"</(?P=outer)\s*>",
    re.DOTALL | re.IGNORECASE,
)
_CODEX_INNER = {
    "codex_delegation": re.compile(r"<input\b[^>]*>(.*?)</input>", re.DOTALL | re.IGNORECASE),
    "codex_internal_context": re.compile(
        r"<objective\b[^>]*>(.*?)</objective>", re.DOTALL | re.IGNORECASE
    ),
}
_NOTIFICATION_BLOCK = re.compile(
    r"<(?P<tag>task-notification|system-reminder)\b[^>]*>.*?"
    r"</(?P=tag)\s*>",
    re.DOTALL | re.IGNORECASE,
)


class RawImportReport(BaseModel):
    source_database: str
    files_seen: int = 0
    files_imported: int = 0
    files_failed: int = 0
    records_seen: int = 0
    events_inserted: int = 0
    events_existing: int = 0
    redactions: int = 0
    reasoning_records_skipped: int = 0
    tool_records_skipped: int = 0
    non_authored_records_skipped: int = 0
    exact_provider_duplicates_skipped: int = 0
    provider_identity_revisions: int = 0
    sidechain_events: int = 0
    compact_summaries_classified: int = 0
    provider_counts: dict[str, int] = Field(default_factory=dict)
    failure_types: dict[str, int] = Field(default_factory=dict)


def _timestamp(value: Any, *, fallback: datetime) -> datetime:
    if value:
        try:
            parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
            return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)
        except ValueError:
            pass
    return fallback


def _text_blocks(content: Any, accepted: frozenset[str]) -> list[tuple[int, str]]:
    if isinstance(content, str):
        return [(0, content)]
    if not isinstance(content, list):
        return []
    return [
        (index, str(block.get("text") or ""))
        for index, block in enumerate(content)
        if isinstance(block, dict) and block.get("type") in accepted and block.get("text")
    ]


def _unwrap_codex(text: str) -> tuple[str, bool]:
    delegated = False

    def replace(match: re.Match[str]) -> str:
        nonlocal delegated
        outer = match.group("outer").casefold()
        inner = _CODEX_INNER[outer].search(match.group(0))
        if inner:
            delegated = True
            return inner.group(1)
        return ""

    return _CODEX_WRAPPER.sub(replace, text), delegated


def _clean_provider_text(text: str) -> str:
    """Remove provider control notices while preserving any authored text around them."""
    previous = None
    while text != previous:
        previous = text
        text = _NOTIFICATION_BLOCK.sub("\n", text)
    return text.strip()


def _memory_eligible(text: str) -> bool:
    """Avoid making one injection-only record reject an otherwise valid atomic batch."""
    return bool(strip_injected_blocks(text).strip())


class RawTranscriptImporter:
    """Reparse available provider files without relying on v1's lossy chunks."""

    def __init__(self, store: MemoryStore) -> None:
        self.store = store

    def import_sources(self, legacy_database: Path) -> RawImportReport:
        database = legacy_database.resolve()
        if not database.is_file():
            raise FileNotFoundError(database)
        report = RawImportReport(source_database=str(database))
        providers: Counter[str] = Counter()
        failures: Counter[str] = Counter()
        connection = sqlite3.connect(f"file:{database.as_posix()}?mode=ro&immutable=1", uri=True)
        try:
            rows = connection.execute("SELECT path,meta FROM sources ORDER BY id").fetchall()
        finally:
            connection.close()
        for path_value, meta_value in rows:
            try:
                metadata = json.loads(meta_value or "{}")
            except json.JSONDecodeError:
                metadata = {}
            provider = metadata.get("provider")
            path = Path(path_value) if path_value else None
            if provider not in ("claude", "codex") or not path or not path.is_file():
                continue
            report.files_seen += 1
            try:
                if provider == "claude":
                    events = self._parse_claude(path, report)
                else:
                    events = self._parse_codex(path, report)
                events = self._deduplicate_provider_events(events, report)
                pending = self._exclude_existing(events, report)
                for start in range(0, len(pending), 250):
                    for result in self.store.ingest_batch(pending[start : start + 250]):
                        report.redactions += result.redaction_count
                        if result.inserted:
                            report.events_inserted += 1
                        else:
                            report.events_existing += 1
                report.files_imported += 1
                providers[provider] += len(events)
            except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
                report.files_failed += 1
                failures[type(error).__name__] += 1
        report.provider_counts = dict(sorted(providers.items()))
        report.failure_types = dict(sorted(failures.items()))
        return report

    @staticmethod
    def _deduplicate_provider_events(
        events: list[IngestEvent], report: RawImportReport
    ) -> list[IngestEvent]:
        """Collapse repeated transcript history and version genuine provider-ID reuse."""
        seen: dict[tuple[str, str, str, str], set[str]] = {}
        unique: list[IngestEvent] = []
        for event in events:
            identity = (
                event.provider_event_id or "",
                event.role.value,
                event.kind.value,
                utc_iso(event.occurred_at),
            )
            body_hash = content_hash(normalize_authored_text(event.content).text)
            bodies = seen.setdefault(identity, set())
            if body_hash in bodies:
                report.exact_provider_duplicates_skipped += 1
                continue
            if bodies:
                original = event.provider_event_id or "provider-event-missing"
                metadata = dict(event.metadata)
                metadata.update(
                    {
                        "provider_identity_reused": True,
                        "original_provider_event_id": event.provider_event_id,
                    }
                )
                event = event.model_copy(
                    update={
                        "provider_event_id": f"{original}:revision:{body_hash[:16]}",
                        "loss_flags": (*event.loss_flags, "provider_identity_reused"),
                        "metadata": metadata,
                    }
                )
                report.provider_identity_revisions += 1
            bodies.add(body_hash)
            unique.append(event)
        return unique

    def _exclude_existing(
        self, events: list[IngestEvent], report: RawImportReport
    ) -> list[IngestEvent]:
        if not events:
            return []
        first = events[0]
        existing = self.store.source_event_hashes(
            source_kind=first.source_kind,
            provider=first.provider,
            source_locator=first.source_locator,
        )
        pending: list[IngestEvent] = []
        for event in events:
            body_hash = content_hash(normalize_authored_text(event.content).text)
            known_hash = existing.get(event.provider_event_id or "")
            if known_hash == body_hash:
                report.events_existing += 1
                continue
            if known_hash is not None:
                raise IdentityConflict(
                    "existing provider event has different canonical content: "
                    f"{event.provider_event_id}"
                )
            pending.append(event)
        return pending

    def _parse_claude(self, path: Path, report: RawImportReport) -> list[IngestEvent]:
        events: list[IngestEvent] = []
        fallback = datetime.fromtimestamp(path.stat().st_mtime, tz=UTC)
        with path.open("rb") as stream:
            for line_number, raw in enumerate(stream, start=1):
                if not raw.endswith(b"\n"):
                    break
                try:
                    record = json.loads(raw.decode("utf-8-sig", "replace"))
                except json.JSONDecodeError:
                    continue
                report.records_seen += 1
                if record.get("type") not in ("user", "assistant") or record.get("isMeta"):
                    report.non_authored_records_skipped += 1
                    continue
                message = record.get("message")
                if not isinstance(message, dict):
                    report.non_authored_records_skipped += 1
                    continue
                role_value = message.get("role") or record.get("type")
                if role_value not in ("user", "assistant"):
                    report.non_authored_records_skipped += 1
                    continue
                content = message.get("content")
                if isinstance(content, list):
                    report.reasoning_records_skipped += sum(
                        1
                        for block in content
                        if isinstance(block, dict) and block.get("type") == "thinking"
                    )
                    report.tool_records_skipped += sum(
                        1
                        for block in content
                        if isinstance(block, dict)
                        and block.get("type") in ("tool_use", "tool_result")
                    )
                blocks = _text_blocks(content, frozenset({"text"}))
                sidechain = bool(record.get("isSidechain"))
                compact_summary = bool(record.get("isCompactSummary"))
                session_id = str(record.get("sessionId") or path.stem)
                agent_id = f"claude-worker:{session_id}" if sidechain else "claude-main"
                for block_index, text in blocks:
                    text = _clean_provider_text(text)
                    if not _memory_eligible(text):
                        report.non_authored_records_skipped += 1
                        continue
                    event_uuid = str(record.get("uuid") or f"line-{line_number}")
                    events.append(
                        IngestEvent(
                            provider="claude",
                            source_kind="claude-transcript-v2",
                            source_locator=str(path.resolve()),
                            provider_event_id=f"{event_uuid}:text:{block_index}",
                            agent_id=agent_id,
                            session_id=session_id,
                            project_id=None,
                            role=(
                                Role.SYSTEM_OBSERVATION
                                if compact_summary
                                else (Role.USER if role_value == "user" else Role.ASSISTANT)
                            ),
                            authority=(
                                Authority.ASSISTANT_SYNTHESIS
                                if compact_summary or role_value == "assistant"
                                else Authority.USER_AUTHORED
                            ),
                            kind=EventKind.CHECKPOINT if compact_summary else EventKind.MESSAGE,
                            occurred_at=_timestamp(record.get("timestamp"), fallback=fallback),
                            content=text,
                            visibility="private",
                            trust="derived" if compact_summary else "authored",
                            loss_flags=("provider_compaction_summary",) if compact_summary else (),
                            metadata={
                                "cwd": record.get("cwd"),
                                "git_branch": record.get("gitBranch"),
                                "parent_message_uuid": record.get("parentUuid"),
                                "is_sidechain": sidechain,
                                "is_compact_summary": compact_summary,
                                "source_line": line_number,
                            },
                        )
                    )
                    if sidechain:
                        report.sidechain_events += 1
                    if compact_summary:
                        report.compact_summaries_classified += 1
        return events

    def _parse_codex(self, path: Path, report: RawImportReport) -> list[IngestEvent]:
        events: list[IngestEvent] = []
        fallback = datetime.fromtimestamp(path.stat().st_mtime, tz=UTC)
        meta: dict[str, Any] = {}
        with path.open("rb") as stream:
            for line_number, raw in enumerate(stream, start=1):
                if not raw.endswith(b"\n"):
                    break
                try:
                    record = json.loads(raw.decode("utf-8-sig", "replace"))
                except json.JSONDecodeError:
                    continue
                report.records_seen += 1
                if record.get("type") == "session_meta" and isinstance(record.get("payload"), dict):
                    meta = record["payload"]
                    continue
                if record.get("type") != "response_item":
                    payload = record.get("payload") or {}
                    if payload.get("type") == "agent_reasoning":
                        report.reasoning_records_skipped += 1
                    else:
                        report.non_authored_records_skipped += 1
                    continue
                payload = record.get("payload")
                if not isinstance(payload, dict):
                    report.non_authored_records_skipped += 1
                    continue
                payload_type = payload.get("type")
                if payload_type == "reasoning":
                    report.reasoning_records_skipped += 1
                    continue
                if payload_type in ("function_call", "function_call_output"):
                    report.tool_records_skipped += 1
                    continue
                if payload_type != "message" or payload.get("role") not in ("user", "assistant"):
                    report.non_authored_records_skipped += 1
                    continue
                role_value = payload["role"]
                blocks = _text_blocks(
                    payload.get("content"), frozenset({"input_text", "output_text", "text"})
                )
                thread_id = str(meta.get("id") or meta.get("session_id") or path.stem)
                parent_thread = meta.get("parent_thread_id")
                agent_path = meta.get("agent_path")
                is_worker = bool(
                    agent_path or parent_thread or meta.get("thread_source") == "subagent"
                )
                agent_id = (
                    f"codex:{agent_path}"
                    if agent_path
                    else (f"codex-worker:{thread_id}" if is_worker else "codex-main")
                )
                for block_index, text in blocks:
                    text, delegated = _unwrap_codex(text)
                    text = _clean_provider_text(text)
                    if not _memory_eligible(text):
                        report.non_authored_records_skipped += 1
                        continue
                    if not text.strip():
                        continue
                    event_key = str(payload.get("id") or f"line-{line_number}")
                    if role_value == "user" and delegated:
                        role = Role.SYSTEM_OBSERVATION
                        authority = Authority.EXPLICIT_DECISION
                    else:
                        role = Role.USER if role_value == "user" else Role.ASSISTANT
                        authority = (
                            Authority.USER_AUTHORED
                            if role_value == "user"
                            else Authority.ASSISTANT_SYNTHESIS
                        )
                    events.append(
                        IngestEvent(
                            provider="codex",
                            source_kind="codex-rollout-v2",
                            source_locator=str(path.resolve()),
                            provider_event_id=f"{event_key}:text:{block_index}",
                            agent_id=agent_id,
                            session_id=thread_id,
                            parent_session_id=str(parent_thread) if parent_thread else None,
                            project_id=None,
                            role=role,
                            authority=authority,
                            kind=EventKind.MESSAGE,
                            occurred_at=_timestamp(record.get("timestamp"), fallback=fallback),
                            content=text,
                            visibility="private",
                            trust="authored" if not delegated else "observed",
                            metadata={
                                "cwd": meta.get("cwd"),
                                "agent_path": agent_path,
                                "agent_nickname": (
                                    (
                                        ((meta.get("source") or {}).get("subagent") or {}).get(
                                            "thread_spawn"
                                        )
                                        or {}
                                    ).get("agent_nickname")
                                    if isinstance(meta.get("source"), dict)
                                    else None
                                ),
                                "thread_source": meta.get("thread_source"),
                                "source_line": line_number,
                            },
                        )
                    )
                    if is_worker:
                        report.sidechain_events += 1
        return events
