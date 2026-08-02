from __future__ import annotations

import pytest

from system_memory.mcp_server import build_server


class FakeClient:
    def recall(self, *args, **kwargs):
        return {"mode": "empty", "evidence": [], "abstained": True}

    def health(self):
        return {"ok": True}

    def record_checkpoint(self, **kwargs):
        return {"inserted": True, "event_id": "event-1"}


@pytest.mark.asyncio
async def test_mcp_tools_declare_truthful_safety_annotations():
    tools = {tool.name: tool for tool in await build_server(FakeClient()).list_tools()}

    assert set(tools) == {
        "memory_recall",
        "memory_health",
        "memory_record_checkpoint",
    }
    assert tools["memory_recall"].annotations.readOnlyHint is True
    assert tools["memory_health"].annotations.readOnlyHint is True
    checkpoint = tools["memory_record_checkpoint"].annotations
    assert checkpoint.readOnlyHint is False
    assert checkpoint.destructiveHint is False
    assert checkpoint.idempotentHint is True
