# Shared agent-memory operating rules

This repository is the local memory engine shared by Claude Code and Codex. Read
`docs/README.md`, `docs/ARCHITECTURE.md`, and `docs/CODEX.md` before changing lifecycle,
retrieval, or indexing behavior.

- Hooks are fail-safe: bounded work, exit zero on failure, and never block a prompt.
- Hybrid/vector delivery is the healthy path. `recall-fallback` is visible keyword-only
  degradation and must never be counted as successful vector recall.
- Client delivery receipts—not server computation completion—are the success boundary. Keep the
  installed-client activation machine-wide unless the operator explicitly selects confinement.
- Keep heavy embedding and reranking models only in the singleton supervised warm server.
  Every per-client MCP process stays lightweight.
- Prompt-query inference has priority over document indexing; indexing must remain interruptible
  between small embedding microbatches.
- Files are source truth; SQLite is a rebuildable derived index. The canonical note root is
  `~/.agent-memory/notes`; Claude project memory dirs are compatibility inputs, never the identity
  of the system. Preserve lifecycle filtering, path confinement, partial-line handling,
  injected-context stripping, and live-session exclusion.
- Never download or execute code from lifecycle hooks. The HTTP service stays loopback-only.
- Deep verification stays scheduled/manual and off the prompt path; observable usefulness feedback
  complements delivery receipts.
- Run `mem selftest` before and after changes. All checks must pass. The `tests/` directory is
  intentionally empty; the self-test is the regression suite.
