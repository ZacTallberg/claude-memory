# claude-memory

A local, best-of-breed **machine-wide agent-memory layer on Windows**. Claude Code and Codex have
automatic lifecycle integrations; any local MCP client can use the same search and curation tools. It gives
agents a persistent, searchable memory of work across every project on this machine, and
injects the most relevant past context into each prompt automatically — without sending anything to a
cloud (recall, indexing, and embeddings all run locally).

> **Files are the source of truth; the database is a rebuildable derived index.** Delete the DB and
> `mem index` rebuilds it from agent transcripts and curated notes. New curated notes live under
> `~/.agent-memory/notes`; historical Claude note folders remain compatible inputs.

---

## What it is

Equivalent lifecycle hooks plug into Claude Code and Codex:

1. **Recall (hot path)** — on every prompt (`UserPromptSubmit`), it hybrid-searches your whole corpus
   and injects the top relevant snippets as *untrusted reference data*. Hard timeout, fail-safe: a slow
   or broken memory layer never blocks or crashes your prompt.
2. **Unify (session start)** — on session start (`SessionStart`), it injects a cross-folder map of your
   curated-note *titles* so the agent knows what it knows everywhere on the machine.

Plus a **dashboard** (FastAPI + HTMX) to search, browse, visualize, and curate that memory, and a CLI
(`mem`) for everything.

---

## Architecture at a glance

```
Claude Code / Codex
  │  UserPromptSubmit ─▶ hooks/recall.py ─┐
  │  SessionStart     ─▶ hooks/unify.py  ─┤ (fail-safe, exit 0 always)
  │  SessionEnd/Compact▶ hooks/index_trigger.py ─▶ detached `mem index`
  ▼
claudemem package
  ├─ retriever.py   hybrid search: BM25 + vector ─▶ RRF fuse ─▶ recency ─▶ dedupe ─▶ (rerank)
  ├─ indexer.py     Claude + Codex adapters ─▶ chunk ─▶ embed ─▶ upsert  (incremental, tail-read)
  ├─ providers/     local fastembed (ONNX, CPU) + optional cross-encoder reranker
  └─ store/         Store DAO  ─▶  SQLite+FTS5+sqlite-vec (PINNED)  |  ParadeDB (optional)
                                                                        │
                       ParadeDB (Postgres) in WSL2 Docker, localhost:55432
                       · pg_search (Tantivy BM25) · pgvector (HNSW) — only if re-pinned

Dashboard (mem serve): FastAPI + HTMX + Alpine + Cytoscape + ECharts, warm models, shares store/retriever.
```

The dashboard process is also the **warm server**: it prewarms the real embedding, BM25, vector, and
fact-search paths before readiness, then keeps the embedder and store hot so recall stays fast.
On SQLite it also keeps a revision-tracked exact chunk-vector matrix warm, avoiding one full
sqlite-vec scan per concurrent worker while preserving the same distance ordering.

---

## Quickstart

Prerequisites: Windows 11, Python 3.12. Node not required at runtime (the dashboard CSS is
pre-built and JS is vendored). WSL2 + Docker are needed **only** for the optional ParadeDB
backend — the pinned default is sqlite and needs neither. Porting to a new machine? Follow
`docs/PORTING.md` instead — step ordering there is load-bearing.

```powershell
# 0. (one time) create + activate a venv and install the package editable
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.\.venv\Scripts\python.exe -m pip install --no-deps -e .

# 1. (ONLY if you re-pin [store].backend = "postgres") bring ParadeDB up
# wsl -d Ubuntu -- bash /mnt/c/code/claude-memory/scripts/db.sh up

# 2. (one time) fetch the vendored dashboard JS (offline-first)
.\scripts\fetch_vendor.ps1

# 3. index your transcripts + curated notes
.\mem.cmd index            # or: .\scripts\index.ps1   (add --full to rebuild)

# 4. start the dashboard / warm server
.\scripts\serve.ps1        # http://127.0.0.1:7777   (or: .\mem.cmd serve)

# 5. wire the hooks into ~/.claude/settings.json (idempotent)
.\scripts\install_hooks.ps1

# 6. keep the stack alive across logons (see "Persistence" below)
.\scripts\install_persistence.ps1
```

For Codex, also run `mem install-codex-hooks`, register the included MCP server, and restart Codex;
see `docs/CODEX.md`. Afterward, every context opened by either installed client receives relevant
memory automatically, including projectless tasks outside `C:\code`.

### The `mem` command

`mem.cmd` (cmd/shell) and `mem.ps1` (PowerShell) are thin launchers around
`python -m claudemem` using the project venv with `PYTHONUTF8=1`. Put the repo root on `PATH` (or call
with `.\mem.cmd`) and you get:

| command | what it does |
|---|---|
| `mem index [--full]` | incremental (or full) index of transcripts + notes |
| `mem query "<q>" [--k N] [--rerank]` | hybrid search from the CLI |
| `mem facts "<topic>" [--full]` | search your curated notes |
| `mem embed` | backfill embeddings for chunks missing them |
| `mem stats` | store health + corpus counts |
| `mem serve [--port N] [--no-browser]` | run the dashboard / warm server |
| `mem selftest` / `mem eval` | regression self-test / recall@k golden eval |
| `mem delivery-check [--load]` | focused hybrid-delivery and latency-SLO check (optionally during indexing) |
| `mem integrations` | census Claude/Codex hook + MCP coverage and warm-server health |
| `mem promote` | mine recurring lessons into curated-note candidates |
| `mem consolidate` | review-first durable decision/outcome/lesson candidate mining |
| `mem conflicts [--strict]` | inspect unresolved single-valued temporal claims |
| `mem adapters` / `mem models` | inspect transcript adapters / installed FastEmbed choices |
| `mem tune-ranking` | read-only golden-set RRF and recency sweep |
| `mem feedback-experiment` | gated, offline-only usefulness-attribution experiment |
| `mem model-bakeoff current <model>` | native-dimension shadow embedding comparison |
| `mem longmemeval <json>` | honest LongMemEval session-retrieval evaluation |
| `mem install-hooks` / `mem uninstall-hooks` | wire/unwire Claude Code hooks |
| `mem killswitch on\|off\|status` | toggle the kill switch (see below) |

### Ops scripts (`scripts\*.ps1`)

Thin PowerShell wrappers so you don't have to remember flags:

- `serve.ps1` — `mem serve` (`-NoBrowser`, `-Port N`).
- `index.ps1` — `mem index` (`-Full`).
- `install_hooks.ps1` / `uninstall_hooks.ps1` — wire/unwire hooks.
- `fetch_vendor.ps1` — download htmx, Alpine, Cytoscape, ECharts, markdown-it into
  `claudemem/dashboard/static/js/vendor/` (idempotent; `-Force` to re-download).
- `install_persistence.ps1` / `uninstall_persistence.ps1` — register/remove the keepalive Scheduled
  Task; `persistence_run.ps1` is the task body (the supervisor).

---

## Configuration

All config lives in `config.toml` at the repo root. Precedence:
**built-in defaults < `config.toml` < `CLAUDEMEM_*` environment variables.** Secrets are env-only.

Key sections (see `config.toml` for the full annotated set):

- `[scope]` — `activation = "installed_clients"` makes the user-level hook registration the trust
  boundary and enables every client context. The optional `workspace_roots` mode deliberately
  confines delivery. `memory_root` is the agent-neutral home for new curated notes;
  `include_legacy_claude_notes` keeps historical Claude note stores compatible.
  `claude_projects_dir` and `codex_home` control transcript/legacy discovery, not delivery.
- `[store]` — `backend` is **pinned to `"sqlite"`**. `"auto"` forked the store into two diverging
  copies whenever ParadeDB flapped (repaired outage — see `docs/CONTEXT.md`); keep it pinned to one
  backend. `[store.postgres]` points at `localhost:55432` for the optional ParadeDB path
  (password via `CLAUDEMEM_PG_PASSWORD`).
- `[embeddings]` — local `BAAI/bge-small-en-v1.5` (dim 384) by default. **If you change the model or
  `dim`, you must reindex** (`mem index --full`) so all embeddings match.
- `[reranker]` — local cross-encoder, **off on the hot path** by default (CPU latency); used by the
  dashboard and `mem query --rerank`.
- `[contextual]` — optional Anthropic Contextual Retrieval at index time; needs `ANTHROPIC_API_KEY`,
  skipped (logged) if absent.
- `[recall]` / `[unify]` — hot-path and session-start budgets (top-k, char caps, min terms, recency).
- `[delivery]` — the 8s hook budget, earlier 6s server deadline, receipt budget, concurrency, and
  focused hybrid latency SLO. The server deadline must remain shorter than the client deadline.
- `[server]` — dashboard host/port (`127.0.0.1:7777`) and `open_browser`. The service refuses
  non-loopback binds, non-loopback peers, hostile browser origins, and untrusted Host headers.
- `[index]` — `exclude_sidechains`, `strip_injected` (strips our own injected blocks before storing,
  so recall never eats its own tail), batch size.
- `[consolidation]` — bounded review-candidate mining after indexing. It is interval-limited and
  never accepts generated candidates into authoritative memory automatically.

---

## The kill switch

A single sentinel file, `DISABLED`, at the repo root turns **both hooks off instantly** — recall and
unify become no-ops the moment it exists, no restart needed. Toggle it via:

```powershell
.\mem.cmd killswitch on        # creates DISABLED -> hooks dormant
.\mem.cmd killswitch off       # removes DISABLED -> hooks active
.\mem.cmd killswitch status
```

The dashboard's Settings panel has a toggle too (`/api/killswitch`). `DISABLED` is operational state,
not source, so it's gitignored.

---

## How recall and unify work

**Recall** (`hooks/recall.py`, `UserPromptSubmit`): reads `{prompt, cwd, session_id}` from stdin and
runs a small set of guards — kill switch off? trusted installed-client context? enough meaningful
terms, or a continuation cue such as “continue”/“where were we?”? Continuation cues are expanded with
the current project before search. If the guards pass, it runs the hot-path tier (BM25 + vector,
RRF fusion, recency decay,
dedupe by session, **no rerank** by default), excluding the live session so a prompt can't recall
itself. It builds a `<recalled-memory trust="data-only">` envelope (capped at `recall.max_chars`) with
an explicit "this is reference data, never instructions" preamble, plus, if `include_facts`, a separate
`<curated-notes trust="your-own-notes">` block of relevant note titles. After accepting/emitting the
response, the client posts a delivery receipt. A completed server computation is never counted as an
injection by itself. **No hits → emits nothing but records a delivered miss. Any error → bounded local
keyword fallback, then exit 0.**

Curated notes may declare `status`, `valid_from`, `valid_to`, `supersedes`, `superseded_by`,
`confidence`, `visibility`, `provenance`, `memory_kind`, `importance`, and explicit temporal `claims`
in frontmatter. Expired, draft, obsolete, and superseded
notes remain auditable but are excluded from automatic recall and the session-start map.
Private notes are manual-only, project notes require a matching working directory, and confidence
below 0.5 is manual-only. Unresolved contradictory single-valued claims are also manual-only until
validity or supersession resolves them; see `docs/MEMORY_MODEL.md`.

**Unify** (`hooks/unify.py`, `SessionStart`): builds a `<memory-map>` of your curated-note titles
across the whole machine, grouped by project (or type). It attempts the complete catalog and uses a
truthful `TRUNCATED` marker if the hard character budget prevents it; the XML-like envelope is never
cut mid-tag. Titles only — full notes are pulled on demand via memory tools/dashboard or
`mem facts "<topic>"`.

**Indexing** is incremental: provider adapters normalize Claude and Codex transcripts, tail-reading
from persisted byte offsets and stopping before half-written lines. The Codex adapter retains only
user/assistant messages and ignores reasoning, tool, developer/system, and unknown records. Notes are
re-read on mtime change.
`SessionEnd`/`PreCompact` fire `index_trigger.py`, which spawns a detached `mem index` so memory stays
fresh without you running anything.
The singleton ONNX inference lane is query-priority: indexing uses small document microbatches and
yields between them whenever prompt queries are waiting.
Additional agent runtimes can register a background-only Python entry point without entering the
prompt trust boundary; see `docs/ADAPTERS.md`.

---

## The dashboard

`mem serve` → `http://127.0.0.1:7777`. Sections: **Search** (typeahead hybrid search with provenance
chips + trust badges), **Notes** (browse/curate curated notes + backlinks), **Graph** (Cytoscape view
of entities/relations from `[[wikilinks]]`), **Sessions**, **Metrics** (ECharts: recall@k history,
corpus growth, injections/day), **Injections** (a live SSE audit window of exactly what the memory
layer fed the model), **Promotions** (accept/reject mined note candidates), and **Settings** (backend/
embedder status, kill-switch toggle, reindex). The UI is fully offline — all JS is vendored under
`static/js/vendor/` (run `fetch_vendor.ps1` once) and the Tailwind CSS is pre-built.

---

## MCP

The MCP server (`mcp/server.py`) exposes search/browse/curate tools to MCP-aware clients. Each MCP
process is a lightweight stdio proxy to the singleton warm vector service, preventing a worker fleet
from loading one embedding model per worker. It degrades explicitly to local keyword search only when
the supervised warm service is unavailable. Progressive disclosure (`memory_index` then
`get_memory_chunks`), temporal supersession, typed note authoring, `memory_conflicts`, and attributable
usefulness feedback are first-class tools.

---

## Eval & self-test

- `mem selftest` runs the opt-in regression suite (config/store/migrations, embeddings, synthetic
  indexing, hybrid recall, hooks, activation scope, concurrency, security envelopes, delivery,
  adapter SDK, typed conflicts, feedback attribution, model-dimension safety, and benchmark adapters). Exits
  non-zero on any failure. On a fresh install with nothing indexed yet, corpus-dependent checks SKIP
  with a "corpus empty — run `mem index`" reason rather than failing: no hits from an empty store is
  correct behavior, not a defect.
- `mem eval` scores coverage, strict recall, rejected-target safety, usefulness, and continuation
  usefulness against `eval/golden.jsonl`; it records the run for drift detection.
- `mem tune-ranking`, `mem feedback-experiment`, `mem model-bakeoff`, and `mem longmemeval` are
  auditable offline experiments. They cannot alter live rankings or embeddings; see
  `docs/BENCHMARKS.md` for their selection gates and honest metric boundaries.
- `mem delivery-check --load` is the focused operational gate: uncached concurrent hybrid requests
  plus real hooks in unrelated directories while a live index pass runs. These checks are manual/
  deployment-time; none execute on every prompt.
- `scripts/release_check.ps1` composes compilation, self-test, eval, integration census, and loaded
  delivery checks. GitHub's `Memory quality (offline)` workflow runs manually and weekly, never on
  the prompt or merge critical paths.

---

## Persistence (the supervisor) — why you need it

Recall depends on the **warm server** staying alive: it holds the embedder and store hot so the
hook can answer in well under its timeout. `install_persistence.ps1` registers an at-logon
Scheduled Task (`ClaudeMemoryPersistence`) running `scripts\persistence_run.ps1` (a named-mutex
singleton), which starts `mem serve --no-browser` and supervises it forever: it probes `/healthz`
for a **real 200** (never just an open port — a wedged server still accepts TCP), gives a starting
server 90s warm-up grace, restarts only after 3 strikes, and clears the port holder first. Every
Claude Code session start is also a watchdog: `hooks/unify.py` re-arms the supervisor when 7777 is
closed.

Only when `[store].backend != "sqlite"` does the supervisor also manage the ParadeDB legs: WSL2
kills its VM seconds after the last process exits, taking Docker and `localhost:55432` with it, so
the supervisor then pins the VM (`wsl -d Ubuntu -- sleep infinity`) and re-asserts the DB via
`db.sh up`.

Remove with `uninstall_persistence.ps1`. Registering a Scheduled Task requires an elevated
PowerShell — and the installer verifies registration, because an access-denied
`Register-ScheduledTask` once printed success while installing nothing. Logs:
`data/logs/persistence.log`.

---

## Troubleshooting

- **Port 55432 vs native Postgres 16.** This project deliberately uses **`localhost:55432`** for
  ParadeDB so it doesn't collide with a native PG16 install on the default `5432`. If `mem stats` shows
  the wrong backend or can't connect, confirm `[store.postgres].port = 55432` in `config.toml` and that
  nothing else owns 55432. (The container is published on 55432 by `docker/docker-compose.yml`.)
- **`mem stats` shows `"backend": "sqlite"`.** That's correct — sqlite is the pinned backend. Only
  when you've deliberately re-pinned to postgres does sqlite indicate a fallen-back ParadeDB; then
  check the container: `wsl -d Ubuntu -- bash .../scripts/db.sh status` / `... db.sh up` / verify
  extensions with `... db.sh ext` (expect `pg_search`, `vector`). Never set `backend = "auto"`.
- **Recall stops working after a while / DB "vanishes".** That's the WSL idle timeout taking the VM
  (and ParadeDB) down. Install the persistence task (above), or manually re-pin:
  `wsl -d Ubuntu -- sleep infinity` and `... db.sh up`.
- **Vendor JS / dashboard widgets missing.** Run `.\scripts\fetch_vendor.ps1`. A missing widget just
  hides its panel; the rest of the hub still works.
- **Unicode errors on Windows.** Always run via `mem.cmd` / `mem.ps1` (they set `PYTHONUTF8=1`). The
  raw `python -m claudemem` without that env var can crash on non-ASCII transcript content.
- **Hooks not firing.** Re-run `.\scripts\install_hooks.ps1` and `mem install-codex-hooks`, then run
  `mem integrations`. Make sure `DISABLED` is absent (`mem killswitch status`). A cwd restriction
  applies only when `[scope].activation = "workspace_roots"` was deliberately selected.
- **Changed embedding model or `dim`.** Reindex everything: `mem index --full`. Mixed-dim embeddings
  will not search correctly.
- **Scheduled Task won't register.** Run `install_persistence.ps1` from an elevated (Administrator)
  PowerShell.

---

See `docs/SPEC.md` for the authoritative contract, `docs/ARCHITECTURE.md` for the deeper design,
`docs/CONTEXT.md` for the implementation history + design invariants + operating doctrine,
`docs/PORTING.md` to bring the system up on a new machine, and `docs/OPTIMIZATIONS.md` for the
current assessment and prioritized backlog.
