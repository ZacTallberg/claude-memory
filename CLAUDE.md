# claude-memory — operating rules

Shared local agent-memory layer for Claude Code and Codex on Windows: hybrid recall (`hooks/recall.py` on UserPromptSubmit) + cross-folder memory map (`hooks/unify.py` on SessionStart), provider-normalized transcript indexing, an MCP interface, a `mem` CLI, and a FastAPI warm server/dashboard. Files are the source of truth; the database is a rebuildable derived index. Full picture: `docs/README.md`, `docs/ARCHITECTURE.md`, and `docs/CODEX.md`; configuration: `config.toml` (defaults < file < `CLAUDEMEM_*` env vars).

Hard rules:
- Hooks must stay FAIL-SAFE: hard timeout, exit 0 on any error. A slow or broken memory layer must never block or crash a user prompt or session start.
- Delivery health must never be silent: keep the TRUNCATED marker on capped memory maps and the health beacon (`~/.claude/memory-health.json`, written by unify, rendered by the statusline). Never remove or bypass them.
- Run `mem selftest` (all checks must pass) before changing anything under `hooks/` — it is the
  test suite; `tests/` is empty.
- The warm server is the only process that may hold heavy embedding/reranking models. Per-client
  MCP processes must remain thin proxies so a worker fleet cannot multiply model memory.
- Canonical curated notes live in the client-neutral `~/.agent-memory/notes` tree. Historical
  Claude memory folders are compatibility inputs. Do not make the harness hub- or client-specific.
- Lifecycle hooks must never fetch or execute remote code. The dashboard/API is loopback-only.
