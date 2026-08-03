# Codex integration

The engine is shared: Claude Code and Codex use the same SQLite corpus, hybrid retriever,
curated notes, warm server, and dashboard. Claude's native lifecycle wiring remains intact;
Codex receives equivalent hooks plus on-demand MCP tools.

## Install

```powershell
cd C:\code\claude-memory
.\mem.cmd install-codex-hooks

$codex = Get-ChildItem "$env:LOCALAPPDATA\OpenAI\Codex\bin" -Recurse -Filter codex.exe |
  Sort-Object LastWriteTime -Descending | Select-Object -First 1 -ExpandProperty FullName
& $codex mcp add claude-memory `
  --env PYTHONPATH=C:/code/claude-memory `
  --env PYTHONUTF8=1 `
  -- C:/code/claude-memory/.venv/Scripts/python.exe -m claudemem.mcp.server
```

Restart the Codex app after registration. Open `/hooks` once and trust the four user-level
shared-memory hook definitions; Codex intentionally requires review whenever hook contents change.
Use `/mcp` to confirm the `claude-memory` tools are available.

## What each connection does

- `UserPromptSubmit` automatically injects bounded hybrid recall.
- `SessionStart` injects the cross-project curated-note title map.
- `PreCompact` and `SessionEnd` detach an incremental index pass.
- The indexer normalizes both `~/.claude/projects/**/*.jsonl` and
  `~/.codex/{sessions,archived_sessions}/**/*.jsonl` into the shared corpus. The Codex parser
  deliberately retains only user and assistant message text; tools, reasoning, system/developer
  context, and unknown rollout records are ignored.
- MCP provides explicit progressive search, fact browsing, exact recall envelopes, typed note
  authoring/supersession, contradiction inspection (`memory_conflicts`), and attributable
  usefulness feedback at any point in a task.

The hooks are user-level and machine-wide (`[scope].activation = "installed_clients"`), so generated
projectless tasks under `Documents\Codex` receive the same recall as repositories under `C:\code`.
The directory is retrieval context, not an activation boundary. Every accepted automatic recall/
unify response posts a delivery receipt; `recall-fallback` remains an explicit degraded result.

After installation or upgrades, run `mem integrations` for connector coverage and
`mem delivery-check --load --cwd <any-project-or-projectless-directory>` for the focused live SLO.

The Codex rollout format is documented as unstable. Its adapter is therefore narrow, fail-safe,
and regression-tested. An unknown future record shape is skipped instead of poisoning or blocking
memory delivery. Claude and Codex are built-ins, not architectural boundaries: additional local
agent formats can register the background-only adapter protocol described in `docs/ADAPTERS.md`.
