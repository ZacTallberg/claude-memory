# System Memory v2 architecture contract

Status: accepted for implementation, 2026-08-02.

This contract reconciles five independent principal reviews: memory architecture,
retrieval science, reliability/platform, evaluation science, and agent/hub
integration. It replaces the shape of v1 instead of decorating it.

## 1. The non-negotiable boundary

System Memory is one machine-wide service. It spans providers, sessions, agents,
worktrees, and projects. Project identity is retained as provenance and can boost a
query, but global recall is the default.

Hubs remain project-specific. Each project hub owns its own current task graph,
priority order, leases, checkpoints, worker presence, and decisions. The common hub
code will be extracted into a reusable engine, but neither its runtime store nor its
user experience becomes global.

```text
Claude sessions ----\
Codex sessions ------+--> System Memory (one global service)
Other providers -----/          ^       ^
                                |       |
Project A hub ------------------+       +---------------- Project B hub
  own store and UI                                  own store and UI
```

Truth ownership is explicit:

| Truth | Owner |
|---|---|
| Current project work | That project's hub |
| Code and artifacts | Git and artifact store |
| Durable personal/agent knowledge | System Memory |
| Historical hub events | Originating hub; memory holds a cited observation |
| Temporary reasoning state | Provider session only |
| UI state | Projection only |

Memory never changes a task status, infers that a `todo` is active, or maintains a
shadow task board. A context pack asks the relevant hub for current state at build
time, then combines it with globally retrieved memory.

## 2. What is being replaced

V1 is a useful rescue source but not the target architecture. It is primarily a flat
retrieval index over transcript chunks plus mutable Markdown notes. It lacks reliable
abstention, stable message identity, atomic cursor/chunk commits, temporal claim
history, procedural memory, explicit authority, proper worker lineage, and an
evaluation set capable of certifying any of those properties.

The rebuild preserves only the proven ideas:

- local-first, private storage;
- SQLite WAL and FTS for this machine's scale;
- local embeddings and reranking;
- warm model processes;
- hybrid lexical/semantic retrieval;
- lifecycle hooks and MCP access;
- provenance, anti-memory, redaction, and audit records;
- graceful keyword-only degradation;
- verified snapshots and a kill switch.

The rebuild retires:

- provider-specific product identity;
- transcript polling as the primary ingestion path;
- mutable fact files as the sole semantic-memory representation;
- giant title maps injected at session start;
- blind embedding-dimension truncation;
- one model lock shared by interactive and background work;
- database cursors committed separately from indexed content;
- irrelevant results returned merely because six slots exist;
- Postgres/ParadeDB code paths that are not used by the memory workload;
- task or worker truth copied into memory;
- unversioned dependencies and model artifacts.

## 3. Memory model

The canonical layer is typed and append-preserving.

### 3.1 Sources and authored events

`source` records locate an original provider transcript, hook stream, imported legacy
database, project hub, or artifact. `memory_event` records normalized, sanitized,
user-visible events with deterministic IDs.

Required identity and provenance:

- provider and provider event ID;
- agent ID, session ID, and parent/worker lineage;
- optional project, task, hub instance, worktree, and commit;
- event kind, role, authority, visibility, and trust;
- occurred-at and ingested-at times;
- canonical content hash, source locator, and loss flags;
- schema and normalizer versions.

Only user-visible/authored turns and selected structured tool outcomes are eligible.
System/developer prompts, hidden reasoning, ambient browser state, injected recall,
transport wrappers, and secret material are excluded or redacted before persistence.

Transcript readers are repair adapters. Hooks and explicit provider events are the
primary feed. Unknown transcript records fail closed.

### 3.2 Episodes

An episode groups causally related events from a session, task, or bounded activity.
It is immutable after closure except for an explicit repair revision. Episode
summaries are derived search aids, not evidence. Retrieval can expand from a matched
event to neighboring events without flattening paragraph or code structure.

### 3.3 Semantic claims

A claim is a versioned assertion with:

- subject, predicate, typed value, and professional natural-language rendering;
- authority and confidence;
- `valid_from`, `valid_to`, and transaction time;
- state: proposed, accepted, disputed, superseded, retracted, or forgotten;
- one or more exact evidence spans;
- relation to predecessor and competing claims.

Consolidation proposes one of `ADD`, `MERGE`, `SUPERSEDE`, `RETRACT`, `DISPUTE`, or
`NOOP`. It never silently overwrites. Direct user statements outrank assistant
synthesis; tool outcomes establish what happened, not what the user believes.

### 3.4 Procedures

Procedures store versioned runbooks, environmental gotchas, and successful operating
patterns. Each revision records evidence, preconditions, steps, expected outcome,
last verification, and failure history. A procedure is not promoted merely because
an assistant suggested it once.

### 3.5 Core memory

Core memory is a deliberately small, always-available compilation of high-value
identity, values, preferences, standing constraints, and operating context. Every
entry is backed by current claims and has a token cost. Compilation has a hard budget,
stable ordering, and a receipt; it is not a dump of note titles.

### 3.6 Forgetting and correction

Deletion is a first-class lifecycle operation. Tombstones identify content and secret
fingerprints that must not be re-imported. Forgetting removes canonical payloads,
derived text, FTS rows, vectors, context caches, backups according to policy, and then
proves non-retrieval. Ordinary correction uses temporal supersession so historical
questions remain answerable without presenting obsolete state as current.

## 4. Storage and durability

SQLite remains the primary local database because this is a single-machine memory
service with one serialized writer and many readers. It is configured with WAL,
foreign keys, a bounded busy timeout, and `synchronous=FULL` unless benchmarks prove a
material problem.

Canonical payloads are also exported into a content-addressed, sanitized archive so
the search index is rebuildable and legacy database-only memories cease to have a
single point of loss. Imported v1 rows whose sources vanished are labeled
`legacy_recovered`; fabricated provenance is forbidden.

Schema evolution uses ordered migrations and `PRAGMA user_version`. Search indexes
use immutable generations. A generation receipt records:

- corpus snapshot hash;
- normalizer and chunker versions;
- embedding repository, exact revision/checksum, dimension, normalization, and
  query/document prefixes;
- lexical configuration;
- reranker identity;
- code and dependency-lock identity.

New generations build in shadow tables and activate atomically. Source event,
canonical payload, search document, FTS row, and source cursor commit in one
transaction. Full rebuild never deletes the active generation first.

## 5. Retrieval

Retrieval is evidence selection, not six-result filling.

### 5.1 Fast path

The fast path runs independent candidate lanes:

1. exact identifiers, quoted phrases, paths, and fielded terms;
2. BM25 over structure-preserving search documents;
3. dense retrieval against one declared embedding generation;
4. current semantic claims with temporal filtering;
5. procedures and core memory when intent warrants them.

Lanes keep authority and memory type visible. Candidates are fused with measured
weights, deduplicated by stable evidence identity, aggregated into episodes where
useful, reranked in a bounded pool, diversified, and expanded to cited neighbors.

Current project, task, and provider are boosts—not global filters—unless the caller
explicitly requests a scope. Recency is query-aware and never substitutes for
temporal validity.

### 5.2 Abstention

Every request ends in exactly one delivered mode: `hybrid`, `keyword_only`, `empty`,
`timeout`, `shed`, or `error`. Results below a calibrated relevance threshold produce
`empty`; fallback is never described as hybrid. Work completing after the caller has
fallen back is recorded as `computed_late`, not as delivered recall.

Confidence is calibrated on the local task bed using rank features, score margins,
lexical support, authority, temporal compatibility, and reranker output. A fixed raw
vector or reciprocal-rank score is not treated as universal confidence.

### 5.3 Deep path

`investigate` is a separately budgeted tool for cross-session synthesis, temporal
reconstruction, and ambiguous premise checking. It may reformulate queries and
iterate, but returns exact sources and distinguishes observed evidence from derived
synthesis. Session hooks never invoke the deep path automatically.

### 5.4 Inference scheduling

One priority scheduler owns local model access:

1. interactive query embeddings;
2. interactive reranking;
3. context compilation;
4. background document embeddings.

Background work uses micro-batches and yields whenever an interactive request waits.
The backfill queue is durable. Model choice is made by a local bakeoff, not a generic
leaderboard; BGE-small remains the control until a challenger wins the sealed gates.

## 6. Context compilation

A provider asks for a compact, typed context pack with a token budget. The compiler:

1. identifies the current provider/session/project/task;
2. asks that project's hub adapter for live task truth when available;
3. retrieves relevant global memory;
4. includes current claims, decisions, procedures, and exact source locators;
5. emits a short-lived projection with a board cursor and generation receipt.

The pack is data, not instruction. Retrieved content is clearly delimited and cannot
override system, developer, user, or project instructions.

## 7. Provider and hub adapters

The core exposes one versioned local API and MCP surface. Claude, Codex, transcript
repair, and each project hub are adapters.

- Claude receives standard MCP tools and lifecycle hooks. Its optional MCP Channel
  adapter can push typed hub events into an active main session, but remains behind a
  feature flag while Channels are a research preview.
- Codex receives standard MCP tools and supported lifecycle hooks. A bounded wait tool
  handles project-hub attention until a native push surface is available.
- Each project hub publishes immutable typed events from its own store. Memory records
  them as historical observations and uses a live adapter for current truth.

MCP tools are annotated accurately. Reads are read-only; proposals and compare-and-
swap revisions are explicit writes. Experimental MCP Tasks may wrap long evaluation
or deep-retrieval jobs but never replace a project's task model.

## 8. Security and privacy

- Loopback HTTP only by default.
- Per-install random API credential; per-agent identities and scoped credentials for
  hub adapters.
- Exact Host and Origin policy, CSRF protection for browser mutations, request-size
  limits, constant-time credential checks, and bounded rate/admission controls.
- MCP remains stdio where practical.
- Deterministic secret redaction at every ingress and egress, plus fingerprint
  tombstones and quarantine receipts without plaintext.
- Data ACL limited to the current user, SYSTEM, and Administrators.
- No arbitrary board-authored shell commands. Project hubs use allowlisted verifier
  IDs with argv, path, timeout, and assurance policy declared in project config.
- Raw source files are treated as private external evidence and are never served by a
  generic browser endpoint.

## 9. Reliability and operations

Supervisor V2 owns an exact child PID and cryptographic instance nonce. `/livez` is
store-free; `/readyz` reports usable modes; `/healthz` provides bounded diagnostics.
An unknown process on the configured port is a visible conflict and is never killed.

Backups contain the canonical database, archive, sanitized configuration, schema and
index receipts, dependency lock, source inventory, and checksums. Local generations
are supplemented by encrypted off-device replication. Restore always occurs into an
isolated location, passes integrity and task-bed checks, then activates atomically.

OpenTelemetry correlation follows a request through hook, context compilation,
retrieval, hub lookup, MCP response, and late work. Logs and metrics are bounded and
never contain prompt bodies or secrets by default.

## 10. Reusable project-hub engine

The hub work is a separate package and release track. It removes copied engine code
without merging project experiences or databases. Each project pins an engine version
and supplies project-specific schema extensions, navigation, styling, verifier
registry, deployment profile, and memory adapter configuration.

The engine guarantees transactional events/outbox, literal status semantics, atomic
claim/lease/fencing, idempotency, authenticated actor identity, cursor-based live
streams, and responsive projections. Project hubs remain free to present entirely
different products.

