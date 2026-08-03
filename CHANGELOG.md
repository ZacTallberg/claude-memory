# Changelog

## 0.2.0 — 2026-08-02

- Added a canonical, client-neutral `~/.agent-memory/notes` store while preserving legacy Claude
  note compatibility.
- Added temporal lifecycle, provenance, confidence, visibility, and file-preserving supersession.
- Added progressive-disclosure MCP retrieval and request-linked usefulness feedback.
- Removed the remote startup-code carrier and made hook installation clean up old registrations.
- Enforced loopback-only HTTP access, trusted hosts, and safe browser origins.
- Added atomic note writes, lifecycle-aware promotion and graph edges, and hardened backup handling.
- Added concurrent query microbatching and startup retrieval prewarming to keep four-worker loaded
  delivery below the 3-second hybrid SLO, including immediately after restart.
- Added a verified dependency lock, a manual/weekly offline quality workflow, and a release gate.
- Expanded the regression suite from 42 to 48 checks.
