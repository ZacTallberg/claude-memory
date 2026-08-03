from __future__ import annotations

from datetime import UTC, datetime

import pytest

from system_memory.archive import CanonicalArchive
from system_memory.database import Database
from system_memory.models import Authority, EventKind, IngestEvent, Role
from system_memory.store import MemoryStore


@pytest.fixture
def store(tmp_path):
    memory = MemoryStore(
        Database(tmp_path / "system-memory.db"),
        CanonicalArchive(tmp_path / "archive"),
    )
    assert memory.initialize() == 6
    return memory


def make_event(
    *,
    event_key: str = "message-1",
    content: str = "The user prefers systems that explain their reasoning clearly.",
    project_id: str | None = "project-a",
    session_id: str = "session-1",
    offset: int = 100,
) -> IngestEvent:
    return IngestEvent(
        provider="codex",
        source_kind="hook-stream",
        source_locator=f"codex://sessions/{session_id}",
        provider_event_id=event_key,
        agent_id="codex-main",
        session_id=session_id,
        project_id=project_id,
        role=Role.USER,
        authority=Authority.USER_DECLARATION,
        kind=EventKind.MESSAGE,
        occurred_at=datetime(2026, 8, 2, 12, 0, tzinfo=UTC),
        content=content,
        source_offset_start=max(0, offset - 100),
        source_offset_end=offset,
    )
