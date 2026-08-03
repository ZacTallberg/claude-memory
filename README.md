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

The current v2 branch now includes:

- append-preserving, typed ingestion with deterministic identity, provenance,
  redaction, tombstones, and a content-addressed canonical archive;
- transactional SQLite migrations, immutable lexical/vector generations, atomic
  activation, and verified backup/restore;
- exact FastEmbed model revision and artifact fingerprints, with model loading kept
  local-only after an explicit build;
- hybrid lexical/vector recall with calibrated abstention, explicit delivered modes,
  project/session facets, and a query-priority inference scheduler;
- a durable live-vector overlay that keeps newly ingested memories semantically
  searchable without rebuilding the active generation;
- bounded loopback API, scoped credentials, MCP and lifecycle adapters for Claude
  and Codex, supervised-process identity, and health/readiness diagnostics; and
- raw transcript and legacy-v1 recovery adapters that preserve loss and authority
  uncertainty instead of fabricating trusted provenance.

This is still a side-by-side candidate, not an automatic replacement for v1. The
sealed quality, failure-injection, resource, and migration gates remain the cutover
authority. The full local regression suite is:

```powershell
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m ruff check src tests
```

See:

- `docs/ARCHITECTURE.md`
- `docs/RELEASE-GATES.md`
- `docs/MIGRATION.md`
