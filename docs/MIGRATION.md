# Side-by-side migration and cutover

The migration is roll-forward and corpus-preserving. V1 remains live and read-only as
a recovery source until V2 has passed its sealed gates and a restore rehearsal.

## Stage 1: rescue baseline

- Sanitize the live v1 database and curated notes without displaying matched values.
- Rotate any potentially exposed external credentials.
- Create and verify a clean snapshot.
- Commit the stabilized v1 code and pin its incompatible dependency boundary.
- Record corpus counts, missing sources, duplicate groups, model identity, and schema.

## Stage 2: canonical export

- Export every recoverable authored event using deterministic IDs.
- Reparse available Claude and Codex sources with the v2 normalizer.
- Export database-only rows as `legacy_recovered` with explicit provenance-loss flags.
- Preserve provider, session, role, timestamps, source offsets, project/worktree hints,
  and hashes where evidence supports them.
- Do not invent missing project, task, role, or source identity.
- Produce a signed import manifest with source counts, event counts, omissions,
  redactions, duplicates, and hashes.

## Stage 3: shadow build

- Import canonical events into an isolated v2 database and content archive.
- Build lexical and embedding generations without changing v1.
- Generate claim/procedure proposals; accept only the safe deterministic subset.
- Compile core memory from reviewed current claims.
- Run development evaluation, failure injection, performance soak, and restore.

## Stage 4: provider canary

- Install V2 MCP under a distinct name and hooks in observe-only mode.
- Compare v1 and v2 recall receipts without injecting both into prompts.
- Canary one provider/session at a time.
- Confirm global cross-project recall and exact project attribution.
- Confirm each project hub remains authoritative and independent while conforming to
  the canonical `C:\code\game` hub adapter protocol.

## Stage 5: atomic cutover

- Pause ingestion briefly and flush both providers.
- Import the final delta idempotently.
- Activate the proven index generation and launcher pointer.
- Switch hooks and MCP clients to V2.
- Keep V1 stopped but intact for the bounded rollback window.
- A rollback changes only the launcher/config pointer; it never reverse-mutates V2
  data into the v1 schema.

## Stage 6: retirement

- After the rollback window and a second restore rehearsal, archive the sanitized v1
  snapshot off-device.
- Upgrade older project hubs to the canonical `game` hub runtime without combining
  their stores.
- Remove v1 runtime hooks, obsolete supervisors, polling mailboxes, and unused
  Postgres/ParadeDB configuration.
- Retain migration manifests and audit receipts, not a noisy user-facing task archive.
