from __future__ import annotations

import json

import httpx
import pytest

from system_memory.api_client import MemoryApiClient


def test_adapter_recall_is_global_soft_scoped_and_excludes_current_session(tmp_path):
    token = tmp_path / "codex.token"
    token.write_text("synthetic-token", encoding="utf-8")
    observed = {}

    def handler(request: httpx.Request) -> httpx.Response:
        observed["authorization"] = request.headers.get("Authorization")
        observed["payload"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "request_id": "recall-1",
                "mode": "empty",
                "evidence": [],
                "elapsed_ms": 1.0,
                "generation_id": "generation-1",
                "abstained": True,
            },
        )

    client = MemoryApiClient(
        base_url="http://127.0.0.1:7788",
        token_path=token,
        agent_id="codex-main",
        provider="codex",
        transport=httpx.MockTransport(handler),
    )
    result = client.recall(
        "What does the system remember?",
        current_project_id="game",
        current_session_id="current-session",
    )

    assert result["abstained"] is True
    assert observed["authorization"] == "Bearer synthetic-token"
    assert observed["payload"]["scope"] == {
        "exclude_session_ids": ["current-session"],
        "hard_filter": False,
    }
    assert observed["payload"]["current_project_id"] == "game"
    assert observed["payload"]["current_provider"] == "codex"


def test_checkpoint_authority_cannot_impersonate_the_user(tmp_path):
    token = tmp_path / "claude.token"
    token.write_text("synthetic-token", encoding="utf-8")
    observed = {}

    def handler(request: httpx.Request) -> httpx.Response:
        observed.update(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "event_id": "event-1",
                "source_id": "source-1",
                "episode_id": "episode-1",
                "inserted": True,
                "redaction_count": 0,
                "content_sha256": "a" * 64,
            },
        )

    client = MemoryApiClient(
        base_url="http://localhost:7788",
        token_path=token,
        agent_id="claude-main",
        provider="claude",
        transport=httpx.MockTransport(handler),
    )
    client.record_checkpoint(
        content="A bounded assistant checkpoint.",
        session_id="session-1",
        provider_event_id="checkpoint-1",
    )

    assert observed["agent_id"] == "claude-main"
    assert observed["role"] == "assistant"
    assert observed["authority"] == "assistant_synthesis"
    assert observed["kind"] == "checkpoint"


def test_provider_adapter_refuses_non_loopback_service(tmp_path):
    token = tmp_path / "token"
    token.write_text("synthetic", encoding="utf-8")
    with pytest.raises(ValueError, match="loopback"):
        MemoryApiClient(
            base_url="https://memory.example.com",
            token_path=token,
            agent_id="codex-main",
            provider="codex",
        )
