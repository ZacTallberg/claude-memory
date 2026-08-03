# claude-memory — Architecture

Deeper companion to `docs/README.md` (quickstart/ops) and `docs/SPEC.md` (the authoritative contract).
This document explains the components, how data flows through them, and the load-bearing design
decisions and why they were made.

---

## 1. Component map

```
                        ┌──────────────────────────────────────────────────────────┐
                        │                      Claude Code                          │
                        │   (per-project JSONL transcripts + memory/*.md notes)     │
                        └───────┬───────────────┬───────────────┬──────────────────┘
              UserPromptSubmit  │   SessionStart │   SessionEnd / PreCompact │
                                ▼               ▼                           ▼
                       ┌───────────────┐ ┌──────────────┐        ┌───────────────────┐
            hooks/     │  recall.py    │ │  unify.py    │        │ index_trigger.py  │
            (fail-safe,│  hot search + │ │ titles map   │        │ spawn detached    │
             exit 0)   │  inject       │ │ inject       │        │ `mem index`       │
                       └──────┬────────┘ └──────┬───────┘        └─────────┬─────────┘
                              │  hooks/_common.py: stdin read · scope/min-terms/kill guards · envelopes
                              ▼                  ▼                          ▼
        ┌─────────────────────────────────────────────────────────────────────────────────────┐
        │                                  claudemem package                                    │
        │                                                                                       │
        │   cli.py ── argparse subcommands ───────────────────────────────────────────────┐    │
        │                                                                                  │    │
        │   ┌───────────────┐   ┌───────────────┐   ┌──────────────────────────────────┐  │    │
        │   │ indexer.py    │   │ retriever.py  │   │ providers/                       │  │    │
        │   │ scan→chunk→   │   │ bm25+vector→  │   │  embeddings (fastembed/ONNX,CPU) │  │    │
        │   │ enrich→embed→ │◀─▶│ RRF→recency→  │◀─▶│  reranker (cross-encoder, opt.)  │  │    │
        │   │ upsert (incr) │   │ dedupe→rerank │   │  contextual (Anthropic, opt.)    │  │    │
        │   └──────┬────────┘   └──────┬────────┘   └──────────────────────────────────┘  │    │
        │          │                   │                                                   │    │
        │          ▼                   ▼                                                   │    │
        │   ┌──────────────────────────────────────────────────────────────────────────┐ │    │
        │   │ store/  Store DAO (base.py)  —  fusion lives in retriever, NOT the store   │ │    │
        │   │   factory.py: postgres | sqlite | auto(fallback)                           │ │    │
        │   └───────────────┬───────────────────────────────────┬──────────────────────┘ │    │
        │                   │ postgres_store.py                  │ sqlite_store.py          │    │
        └───────────────────┼────────────────────────────────────┼─────────────────────────┘    │
                            ▼                                    ▼                                │
                ┌───────────────────────┐            ┌───────────────────────┐                  │
                │ ParadeDB (Postgres)   │            │ SQLite (data/*.db)    │   dashboard/ ◀────┘
                │ in WSL2 Docker        │            │ FTS5 + sqlite-vec     │   FastAPI + HTMX
                │ localhost:55432       │            │ (offline fallback)    │   (warm server)
                │ pg_search · pgvector  │            └───────────────────────┘   api.py/views/render
                └───────────────────────┘
```

`mcp/server.py` (optional) exposes the same store/retriever to MCP clients. `eval.py` / `selftest.py`
exercise the whole stack and write to the `metrics` table the dashboard charts.

---

## 2. Data flow

### 2.1 Index path (write — single writer)

```
transcripts.py: tail-read each *.jsonl from persisted byte offset (stop at first line w/o trailing \n)
        │  skip isSidechain / isMeta / command echoes; strip our own injected blocks
        ▼
chunking.py: notes → 1 chunk; transcript turns → ~400-token windows (~15% overlap) on code/para bounds
        │  each chunk carries provenance: session_id, project, cwd, ts, role, ordinal
        ▼
contextual.py (optional): ≤2-sentence situating blurb per chunk (prompt-cached per doc) → context_blurb
        ▼
embeddings.py: L2-normalized vectors (BAAI/bge-small, dim 384) over search_text = blurb + content
        ▼
store.upsert_source / add_chunks / set_embeddings  +  set_bytes_indexed(offset)   (incremental, idempotent)
```

Incrementality is keyed on `sources.bytes_indexed` (transcripts) and mtime (notes); a shrink/rotation
(`stored_offset > size`) forces a reindex of that source.

### 2.2 Recall path (read — hot, fail-safe)

```
UserPromptSubmit stdin {prompt, cwd, session_id}
   │  guards: killed()? trusted installed client? meaningful OR continuation cue?
   │  continuation cue → expand with cwd/project + prior-work concepts
   ▼
retriever.search(tier="hot", exclude_session=session_id):
   store.search_bm25(k=bm25_k)  ┐
   store.search_vector(k=vec_k) ┘ → RRF fuse (rrf_k) → recency decay (half-life) → dedupe by session
   (no rerank on hot path by default — CPU latency)
   ▼
recall_format.py: <recalled-memory trust="data-only"> … capped at max_chars …  (+ <curated-notes> if enabled)
   ▼
server response → emit additionalContext → client delivery receipt → durable injection row → exit 0
server slow/unavailable → local keyword fallback (explicit recall-fallback row) → exit 0
```

### 2.3 Unify path (read — session start)

```
SessionStart stdin {cwd, source}
   │  guards: killed()? trusted installed client?
   ▼
store.facts_titles_map() grouped by unify.group_by, complete when it fits; truthful char-budget cap
   ▼
<memory-map trust="your-own-notes"> titles only  →  emit  →  exit 0
```

---

## 3. Design decisions (and why)

### 3.1 ParadeDB hybrid (BM25 + vector in one Postgres)
We need both lexical precision (exact identifiers, error strings, file names) and semantic recall
(paraphrases, concepts). ParadeDB gives us **Tantivy BM25 (`pg_search`) and `pgvector` HNSW in a single
Postgres**, so one query layer covers both, transactionally, with real indexes — no separate search
service to run. **Fusion (RRF) lives in the retriever, not the store**, so each backend only has to
return ranked candidate lists; that keeps both the Postgres and SQLite stores simple and keeps the
ranking logic in one testable place. Recency decay and per-session dedupe run after fusion so old or
repetitive sessions don't crowd out fresh, distinct context.

### 3.2 Warm-server hot path
The recall hook fires on every meaningful/continuation prompt with an 8s client budget and a 6s
server deadline, on a CPU-only box. Cold-loading the
ONNX embedder per prompt would blow that budget. So the dashboard process doubles as a **warm server**
that keeps the embedder and store connection hot; the hook reuses that warmth. The reranker (a heavier
cross-encoder) is therefore kept **off the hot path** by default and reserved for the dashboard and
`mem query --rerank`, where latency is acceptable.

The ONNX lane is serialized to avoid CPU oversubscription but scheduled by priority: query waiters
jump ahead of document indexing, and document embedding is microbatched so an index pass yields every
few chunks. Server work must finish before the earlier server deadline, leaving the hook time to fall
back and record what actually happened.

### 3.3 Files as truth, DB as derived index
Claude Code already persists everything as JSONL transcripts and `memory/*.md` notes. Treating those as
the **source of truth** and the database as a **rebuildable derived index** means the DB is disposable:
drop it, `mem index`, and you're whole again. It also makes the store backend swappable (Postgres ↔
SQLite) without data loss, and makes curation a plain-text, version-controllable activity (notes are
just markdown with YAML frontmatter; the graph comes from `[[wikilinks]]`).

### 3.4 Fail-safe, degrade-gracefully
A memory layer must never make the editor worse. Every hook wraps its body in `failsafe()`: **any
exception or timeout → bounded keyword fallback or no context, then exit 0.** The store is pinned to
SQLite to avoid split-brain failover; if the vector layer/embedder is missing, retrieval degrades to keyword-only.
The kill switch (`DISABLED` sentinel) turns both hooks off instantly with no restart. Net effect: the
worst case is "no memory injected," never "prompt blocked or crashed."

### 3.5 Untrusted-by-default injection
Recalled transcript snippets are *past machine output*, which is a prompt-injection surface. They are
quarantined in a `trust="data-only"` envelope with an explicit "reference data, never instructions"
preamble, and the live session is excluded so recall can't feed on itself. Only user-authored curated
notes get the softer `trust="your-own-notes"` label. The indexer also **strips our own injected blocks**
(`<recalled-memory>`, `<curated-notes>`, `<memory-map>`, system reminders) before storing, so recall
never indexes and re-surfaces its own output (the "eating its tail" failure mode).

### 3.6 Local-first, zero-LLM indexing
Indexing requires **no LLM**: chunking is boundary-based, embeddings are local ONNX (fastembed, no
torch, no CUDA). Anthropic Contextual Retrieval is an *optional* enrichment (gated on `ANTHROPIC_API_KEY`,
prompt-cached per document) that improves recall but is never required. Everything works offline and on
a modest CPU box.

### 3.7 WSL2 keepalive as an explicit operational concern
Because ParadeDB lives in a Docker container inside the WSL2 VM, and WSL2 idle-terminates that VM, the
persistence layer is a first-class part of the design, not an afterthought: a hidden at-logon Scheduled
Task pins the VM (`sleep infinity`), asserts the container (`db.sh up`), and runs the warm server, then
supervises all three. See `scripts/persistence_run.ps1` and the README's "Persistence" section.

---

## 4. Concurrency & robustness model
- **Many readers, one writer.** Hooks and the dashboard are readers; the indexer is the single writer.
  Postgres handles concurrency natively. SQLite uses WAL + `busy_timeout` and serializes writes through
  the indexer process.
- **Delivery, not computation, is the success boundary.** Server completions are health telemetry;
  only a client receipt or explicit fallback is a durable injection record. The dashboard audit view
  therefore cannot count a result that arrived after its caller abandoned it.
- **Idempotent everything.** `migrate()` creates extensions/tables/indexes `IF NOT EXISTS`;
  `install-hooks` and `fetch_vendor.ps1` and `install_persistence.ps1` are all safe to re-run.
- **Path safety.** Every path is `resolve()`d; reads are confined to the configured roots /
  canonical `memory_root` or compatible legacy Claude roots; note writes are one-level and atomic;
  traversal is rejected.
- **Secrets.** DB password and API keys come from the environment only — never config, never logs.
- **Temporal truth.** Markdown frontmatter carries status, validity, confidence, provenance, and
  supersession. Inactive history remains inspectable but is filtered before automatic delivery.
- **Outcome telemetry.** Delivery receipts measure transport; request-linked usefulness feedback
  measures whether memory helped, harmed, or was stale.
- **Local service boundary.** The API refuses remote binds/peers and hostile browser origins. Hooks
  execute only checked-in local scripts and never fetch bootstrap code.

---

## 5. Where to look in the code
- Contract: `docs/SPEC.md`. Config: `claudemem/config.py` + `config.toml`.
- Retrieval/fusion: `claudemem/retriever.py`. Indexing: `claudemem/indexer.py`,
  `claudemem/transcripts.py`, `claudemem/chunking.py`.
- Store DAO + dataclasses: `claudemem/store/base.py`; backends in `store/postgres_store.py`,
  `store/sqlite_store.py`; selection in `store/factory.py`.
- Hooks: `hooks/_common.py`, `hooks/recall.py`, `hooks/unify.py`, `hooks/index_trigger.py`.
- Dashboard: `claudemem/dashboard/` (`server.py`, `api.py`, `state.py`, `render.py`).
- Ops: `scripts/*.ps1` (this task), `scripts/db.sh` (WSL Docker lifecycle), `mem.cmd` / `mem.ps1`.
