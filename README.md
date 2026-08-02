# System Memory

System Memory is the machine-wide memory plane shared by Claude, Codex, and future
agent providers. It remembers across sessions and projects while keeping every
project hub independent and authoritative for its own live work.

This repository is the side-by-side v2 rebuild. The current `claude-memory` service
remains the recovery source until the release gates in `docs/RELEASE-GATES.md` pass.

## Boundaries

- Memory is global by default. Project and task identity are ranking facets, never
  mandatory silos.
- A project hub owns current tasks, leases, priorities, progress, and worker state.
  `C:\code\game` is the canonical hub carry-forward; other project hubs upgrade to
  its protocol and runtime without merging their stores.
- Git and artifact stores own code and deliverables.
- Provider sessions own only temporary working state.
- The memory service stores durable episodes, claims, procedures, core memory, and
  provenance. It may consume hub history but never invent or overwrite hub truth.

## Status

The architecture contract is frozen. The implementation is being built beside the
live v1 service so corpus migration and cutover can be proven without destructive
in-place changes.

See:

- `docs/ARCHITECTURE.md`
- `docs/RELEASE-GATES.md`
- `docs/MIGRATION.md`
