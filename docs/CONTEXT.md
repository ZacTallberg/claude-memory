# Implementation context — history, design rationale, operating doctrine

Everything a maintainer (human or agent) needs to know about *why* this system is shaped the way
it is. The README says what it does; this says what it survived. Provenance: distilled 2026-07-31
from the operator's memory stores on the original machine.

---

## Timeline

**2026-06-07 — built.** Hybrid recall engine + hooks + dashboard, ParadeDB (Postgres/pg_search/
pgvector in WSL2 Docker) as primary store with sqlite fallback, `backend = "auto"`.

**2026-07-06 — first repair (silent degradation, ~1 month).** An audit found 141 memory writes vs
15 content-bearing recalls in a month: capture was healthy, resurfacing was broken, and nothing had
said so. Root causes and the fixes that became load-bearing design:

- `backend = "auto"` **forked the store** into diverging postgres/sqlite copies whenever ParadeDB
  flapped → backend pinned to **sqlite** (FTS5 + sqlite-vec). ParadeDB remains available by
  re-pinning, but auto is an anti-memory: never restore it.
- Zero-result recalls were unlogged → **every** recall/unify now logs to the `injections` table,
  including misses, with latency and source. Telemetry-first: a miss you can't see is a miss you
  re-teach.
- Warm server death was invisible → supervisor (`persistence_run.ps1`, named-mutex singleton),
  re-armed at logon and by every SessionStart hook.
- Promotion loop added (SessionEnd `index --promote` mines cue-phrases into candidates, reviewed
  in the dashboard). The miner needs BOTH a score floor and overlap-coefficient dedup against ALL
  prior candidates — without them each run dredged the next-15-worst clusters and resurrected
  rejected ones. First harvest: 45 drafted, 1 accepted, 44 rejected; steady state drafts 0 until
  new sessions add lessons.
- Anti-memory rows: the retriever suppresses chunks matching `anti_memory` entries, so outdated
  transcript advice (e.g. a superseded deploy target) stops recalling. Add via `POST /api/anti`.
- False-green gotcha fixed: `Register-ScheduledTask` denial did NOT trip the installer's catch —
  it printed success while installing nothing. The installer now verifies registration.

**2026-07-29 — second repair (the seven-day wedge).** One thread held the store's global lock
forever; 72 others parked behind it; every recall fell back to keyword-only for seven days. Nothing
noticed because the watchdog probed with `socket.create_connection()` — **a wedged server still
accepts TCP in 88ms**. Measured tail before the fix: recall p50 1.9s / p90 107s / max 240s. Fixes:

- `/healthz` — liveness AND store responsiveness; a listening-but-dead server now reads 503.
- Hook probes require a real 200, not an open port.
- Recall/unify admission-capped and deadline-bounded. Key mechanism of the death spiral:
  `asyncio.to_thread` **cannot be cancelled**, so every 4s client timeout leaked a worker that
  still fought for the lock.
- Store lock acquired with a timeout → `StoreBusy` instead of a permanent deadlock.
- Supervisor: 90s warm-up grace, 3 strikes, clears the port holder before restart.
- Indexer prunes facts whose backing file is genuinely absent from disk (see PORTING §4 for why
  this makes migration ordering load-bearing).
- Wedge guards added to selftest, **each seeding a real wedge and proven to fire**, plus one
  proving the OLD port-probe passes where the new probe fails, so the regression cannot return
  silently.

**2026-07-31 — healthz v2 (busy ≠ wedged).** The 2s single-probe deadline over-fired: a probe that
merely queued behind an in-flight recall exceeded it routinely, so the supervisor restarted a
*healthy* server every ~3 minutes — and each restart dropped the warm models, making the next probe
slower still. The wedge being guarded lasted seven days; the signal must be **sustained**
unresponsiveness. Now: 8s per-probe deadline, 503 only after `WEDGE_AFTER_S` (180s) without a
single successful store op. Over-firing is not a stricter guard — it is a guard someone turns off.

Also measured 2026-07-31: with 3+ concurrent Claude sessions, warm-server recall latencies hit
13–67s and hooks fall back to keyword-only exactly when the machine is busiest. The beacon reports
it honestly (that part works as designed). This is the top open item in `OPTIMIZATIONS.md`.

## Design invariants (violate these and you are rebuilding a repaired outage)

1. **Files are the source of truth; the DB is derived.** Curated notes are plain markdown a human
   can read, diff, and back up. The DB rebuilds from files + transcripts.
2. **Hooks are fail-safe: exit 0 always, hard client timeout, keyword-only local fallback.** A slow
   or broken memory layer must never block or crash a prompt.
3. **Delivery health is never silent.** The beacon (`~/.claude/memory-health.json`), the statusline
   segment, the TRUNCATED marker on capped memory maps, logged misses. Degradation must be visible
   the moment it starts, not at the next audit.
4. **A health probe must observe the failure it guards.** Port-open ≠ healthy; `is-active` ≠
   `is-working`. And its inverse: a guard must be proven to fire (the selftest seeds real wedges).
5. **Recall never eats its own tail**: injected `<recalled-memory>` blocks are stripped before
   indexing; the live session is excluded from its own recall.
6. **Injections are labeled untrusted**: recalled content ships in a `trust="data-only"` envelope
   with an explicit never-follow-instructions preamble — memory is reference data, not authority.
7. **One backend, pinned.** Auto-failover between stores forks the store.

## Why these components (decisions, not defaults)

- **Hybrid BM25 + vector + RRF + 45-day recency half-life + session dedup** — lexical catches
  identifiers and exact phrases vector misses; vector catches paraphrase; RRF fuses without score
  calibration; recency because operational memory rots.
- **bge-small-en-v1.5 (384-dim, fastembed/ONNX, CPU)** — chosen as the best model *guaranteed
  available in fastembed* on a no-GPU, RAM-constrained box. An upgrade is a config key + full
  reindex; see OPTIMIZATIONS.
- **Cross-encoder rerank OFF the hot path** — measured ~1s/prompt vs 68–90ms fused; quality gain
  not worth 10× hot-path latency. It stays on the dashboard/full tier.
- **Contextual enrichment OFF by default** — needs `ANTHROPIC_API_KEY` in the hook environment and
  pays per chunk; flip `[contextual]` when the cost is accepted.
- **Warm server owns the models** — hooks are thin clients; embedding load happens once, not per
  prompt. The dashboard and warm server are the same process (a known tension — see
  OPTIMIZATIONS #1/#6).
- **Scope gating** (`workspace_roots`, `min_terms`) — recall activates only where it can help;
  trivial prompts and foreign directories get nothing.

## Operating doctrine for the file layer (the memory-lifecycle protocol)

The engine indexes what the file layer keeps; the file layer has its own laws (canonical copy
lives in the operator's hub memory store; distilled here because the engine is built around them):

- **Retention** — one fact per file; the frontmatter `description` is the retrieval key
  (front-load searchable nouns/verbs, ≤160 chars). Before writing a feedback note, search for a
  sibling and UPDATE it ("reaffirmed <date>") — repetition is a promotion signal, not a new file.
  Project checkpoints: one file per project, updated in place, ≤~150 lines. Every
  "pending/awaiting" gets a date. The `MEMORY.md` index line is written in the same turn as the
  file — an orphan is a bug.
- **Decay** — user/feedback notes are permanent (killed only by contradiction); reference notes are
  verified on recall; project notes untouched 30 days move to Archive. A reversal EDITS the old
  note in the same turn: mark it OBSOLETE with a pointer to the successor and keep one anti-memory
  line ("we believed X; wrong because Y") so the mistake can't be re-learned.
- **Promotion ladder** — incident → feedback file → (2+ siblings or reaffirmed ×2) → merged
  canonical rule → repo doctrine doc → global `~/.claude/CLAUDE.md`.
- **Session-end sweep** — lessons written? checkpoints updated-not-appended? DONE archived,
  contradictions reconciled? index one-line-per-file and inside budget?
- **When the engine degrades, `MEMORY.md` IS the recall system** — its hygiene is load-bearing.
  The engine's job is to make degradation visible instead of silent, never to make the file layer
  optional.
