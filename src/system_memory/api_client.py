from __future__ import annotations

import hashlib
import json
import os
from contextlib import suppress
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

import httpx

from .clock import utc_iso


class MemoryApiError(RuntimeError):
    pass


class MemoryApiClient:
    """Bounded authenticated client used by thin provider adapters."""

    def __init__(
        self,
        *,
        base_url: str,
        token_path: Path,
        agent_id: str,
        provider: str,
        timeout_seconds: float = 5.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        parsed = urlparse(base_url)
        if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
            raise ValueError("system memory provider adapters require a loopback HTTP endpoint")
        self.base_url = base_url.rstrip("/")
        self.token_path = token_path.resolve()
        self.agent_id = agent_id.strip()
        self.provider = provider.strip().lower()
        if not self.agent_id or not self.provider:
            raise ValueError("agent_id and provider are required")
        self.client = httpx.Client(timeout=timeout_seconds, transport=transport)

    @classmethod
    def from_environment(cls) -> MemoryApiClient:
        token_path = os.environ.get("SYSTEM_MEMORY_TOKEN_PATH")
        if not token_path:
            raise MemoryApiError("SYSTEM_MEMORY_TOKEN_PATH is required")
        return cls(
            base_url=os.environ.get("SYSTEM_MEMORY_URL", "http://127.0.0.1:7788"),
            token_path=Path(token_path),
            agent_id=os.environ.get("SYSTEM_MEMORY_AGENT_ID", "provider-main"),
            provider=os.environ.get("SYSTEM_MEMORY_PROVIDER", "unknown"),
            timeout_seconds=float(os.environ.get("SYSTEM_MEMORY_TOOL_TIMEOUT_SECONDS", "5")),
        )

    def close(self) -> None:
        self.client.close()

    def health(self) -> dict:
        return self._request("GET", "/healthz")

    def recall(
        self,
        query: str,
        *,
        current_project_id: str | None = None,
        current_session_id: str | None = None,
        as_of: datetime | None = None,
        limit: int = 6,
        max_chars: int = 8_000,
    ) -> dict:
        scope = {
            "exclude_session_ids": [current_session_id] if current_session_id else [],
            "hard_filter": False,
        }
        return self._request(
            "POST",
            "/v1/recall",
            json={
                "query": query,
                "current_project_id": current_project_id,
                "current_provider": self.provider,
                "as_of": utc_iso(as_of) if as_of else None,
                "scope": scope,
                "limit": limit,
                "max_chars": max_chars,
            },
        )

    def record_checkpoint(
        self,
        *,
        content: str,
        session_id: str,
        provider_event_id: str,
        project_id: str | None = None,
        task_id: str | None = None,
        parent_session_id: str | None = None,
        occurred_at: datetime | None = None,
    ) -> dict:
        return self._request(
            "POST",
            "/v1/events",
            json={
                "provider": self.provider,
                "source_kind": "provider-mcp-checkpoint",
                "source_locator": f"mcp://{self.provider}/{self.agent_id}/{session_id}",
                "provider_event_id": provider_event_id,
                "agent_id": self.agent_id,
                "session_id": session_id,
                "parent_session_id": parent_session_id,
                "project_id": project_id,
                "task_id": task_id,
                "role": "assistant",
                "authority": "assistant_synthesis",
                "kind": "checkpoint",
                "occurred_at": utc_iso(occurred_at),
                "content": content,
                "visibility": "private",
                "trust": "authored",
                "metadata": {"adapter": "mcp", "explicit_checkpoint": True},
            },
        )

    def record_user_prompt(
        self,
        *,
        content: str,
        session_id: str,
        provider_event_id: str,
        source_locator: str,
        project_id: str | None = None,
        occurred_at: datetime | None = None,
        cwd: str | None = None,
    ) -> dict:
        return self._request(
            "POST",
            "/v1/events",
            json={
                "provider": self.provider,
                "source_kind": "provider-user-prompt-hook",
                "source_locator": source_locator,
                "provider_event_id": provider_event_id,
                "agent_id": self.agent_id,
                "session_id": session_id,
                "project_id": project_id,
                "worktree": cwd,
                "role": "user",
                "authority": "user_authored",
                "kind": "message",
                "occurred_at": utc_iso(occurred_at),
                "content": content,
                "visibility": "private",
                "trust": "authored",
                "metadata": {"adapter": "user-prompt-hook"},
            },
        )

    @staticmethod
    def prompt_event_id(
        *, session_id: str, prompt: str, turn_id: str | None, transcript_path: str | None
    ) -> str:
        if turn_id:
            return f"turn:{turn_id}:user-prompt"
        source_size: int | None = None
        if transcript_path:
            with suppress(OSError):
                source_size = Path(transcript_path).stat().st_size
        digest = hashlib.sha256(
            json.dumps(
                [session_id, transcript_path, source_size, prompt],
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        return f"prompt:{digest}"

    def _request(self, method: str, path: str, **kwargs) -> dict:
        token = self._token()
        try:
            response = self.client.request(
                method,
                f"{self.base_url}{path}",
                headers={"Authorization": f"Bearer {token}"},
                **kwargs,
            )
            response.raise_for_status()
            payload = response.json()
        except httpx.TimeoutException as error:
            raise MemoryApiError("system memory request timed out") from error
        except httpx.HTTPStatusError as error:
            raise MemoryApiError(
                f"system memory rejected the request with HTTP {error.response.status_code}"
            ) from error
        except (httpx.HTTPError, ValueError) as error:
            raise MemoryApiError("system memory is unavailable or returned invalid JSON") from error
        if not isinstance(payload, dict):
            raise MemoryApiError("system memory returned a non-object response")
        return payload

    def _token(self) -> str:
        try:
            token = self.token_path.read_text(encoding="utf-8").strip()
        except OSError as error:
            raise MemoryApiError("system memory credential file is unavailable") from error
        if not token or len(token) > 4_096:
            raise MemoryApiError("system memory credential file is invalid")
        return token
