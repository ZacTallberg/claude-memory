# Shared agent-memory operating rules

This repository is the local memory engine shared by Claude Code and Codex. Read
`docs/README.md`, `docs/ARCHITECTURE.md`, and `docs/CODEX.md` before changing lifecycle,
retrieval, or indexing behavior.

- Hooks are fail-safe: bounded work, exit zero on failure, and never block a prompt.
- Hybrid/vector delivery is the healthy path. `recall-fallback` is visible keyword-only
  degradation and must never be counted as successful vector recall.
- Keep heavy embedding and reranking models only in the singleton supervised warm server.
  Every per-client MCP process stays lightweight.
- Files are source truth; SQLite is a rebuildable derived index. Preserve path confinement,
  partial-line handling, injected-context stripping, and live-session exclusion.
- Run `mem selftest` before and after changes. All checks must pass. The `tests/` directory is
  intentionally empty; the self-test is the regression suite.
