from __future__ import annotations

from datetime import datetime

from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations

from .api_client import MemoryApiClient


def build_server(client: MemoryApiClient | None = None) -> FastMCP:
    api = client or MemoryApiClient.from_environment()
    server = FastMCP(
        "system-memory",
        instructions=(
            "System-wide memory shared across providers and projects. Recall is global by "
            "default; project identity is a relevance facet, not a silo. Current task state "
            "belongs to the relevant project hub, never to memory."
        ),
    )

    @server.tool(
        name="memory_recall",
        description=(
            "Recall relevant historical evidence across sessions and projects. Pass the current "
            "session ID to suppress self-echo. Empty means calibrated abstention, not failure."
        ),
        annotations=ToolAnnotations(
            title="Recall system memory",
            readOnlyHint=True,
            destructiveHint=False,
            idempotentHint=True,
            openWorldHint=False,
        ),
        structured_output=True,
    )
    def memory_recall(
        query: str,
        current_project_id: str | None = None,
        current_session_id: str | None = None,
        as_of: datetime | None = None,
        limit: int = 6,
        max_chars: int = 8_000,
    ) -> dict[str, object]:
        return api.recall(
            query,
            current_project_id=current_project_id,
            current_session_id=current_session_id,
            as_of=as_of,
            limit=limit,
            max_chars=max_chars,
        )

    @server.tool(
        name="memory_health",
        description="Report authenticated memory health, active generation, and canonical counts.",
        annotations=ToolAnnotations(
            title="Inspect memory health",
            readOnlyHint=True,
            destructiveHint=False,
            idempotentHint=True,
            openWorldHint=False,
        ),
        structured_output=True,
    )
    def memory_health() -> dict[str, object]:
        return api.health()

    @server.tool(
        name="memory_record_checkpoint",
        description=(
            "Record one explicit, finite assistant checkpoint with a stable provider event ID. "
            "This cannot create user-authored declarations or alter a project hub task."
        ),
        annotations=ToolAnnotations(
            title="Record memory checkpoint",
            readOnlyHint=False,
            destructiveHint=False,
            idempotentHint=True,
            openWorldHint=False,
        ),
        structured_output=True,
    )
    def memory_record_checkpoint(
        content: str,
        session_id: str,
        provider_event_id: str,
        project_id: str | None = None,
        task_id: str | None = None,
        parent_session_id: str | None = None,
        occurred_at: datetime | None = None,
    ) -> dict[str, object]:
        return api.record_checkpoint(
            content=content,
            session_id=session_id,
            provider_event_id=provider_event_id,
            project_id=project_id,
            task_id=task_id,
            parent_session_id=parent_session_id,
            occurred_at=occurred_at,
        )

    return server


def main() -> None:
    build_server().run(transport="stdio")


if __name__ == "__main__":
    main()
