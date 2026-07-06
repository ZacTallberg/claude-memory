"""Optional MCP server for claude-memory.

Exposes the core memory operations (hybrid search, fact browse/search, curated-note
authoring, and the recall envelope) as MCP tools so Claude Code and any MCP client can
do model-driven memory ops. Runnable as `python -m claudemem.mcp.server` over stdio.
"""
