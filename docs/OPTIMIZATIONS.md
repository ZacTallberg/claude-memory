# Is this the best way? — assessment + optimization backlog

Written 2026-07-31 against live measurements on the original machine. Verdict first, then the
backlog in priority order. Items are labeled **[measured]** (observed, with numbers) or
**[hypothesis]** (plausible cause/fix — verify before acting; an observation is not a diagnosis).

## Verdict

The architecture is genuinely best-practice for a **local, private, no-GPU** agent-memory layer:
hybrid BM25+vector with RRF is the standard retrieval shape; embeddings/rerank run locally with
nothing leaving the machine; hooks are fail-safe with a keyword fallback; every injection —
including misses — is telemetered; health has a probe that observes the actual failure mode plus a
user-visible beacon; the files-are-truth invariant means the whole index is disposable. Most
"agent memory" products ship less than this. The setup is the right shape.

What keeps it from being the best *running instance* of that shape today is one measured problem
(#1) and a handful of upgrades the design already left doors open for.

---

## 1. Hot-path recall collapse under concurrent sessions — repaired 2026-08-02

**[measured 2026-07-31]** With 3+ Claude Code sessions active, warm-server recall latencies were
13.1s / 37.8s / 67.2s (healthy single-session baseline: 0.29–0.47s). The hooks' client timeout
(~4s) then routes every prompt to the keyword-only fallback — so recall quality is at its worst
exactly when the machine is busiest. The beacon honestly shows `local-fallback` (visibility works);
the server also answered `/healthz` in 56ms once idle, so this is contention, not a wedge.

**[hypothesis — candidate causes, in the order to check]**
a. The store's global lock serializes *reads*. SQLite in WAL mode supports many concurrent
   readers; recall is read-only. Giving the recall path its own read-only connection(s) outside
   the global lock is the biggest structural win if the lock is the culprit.
b. Every recall embeds the query on the same starved CPU; an LRU cache of query embeddings is
   nearly free to add.
c. Memory pressure: the box runs down to ~200MB free with agent fleets open, and the server's
   working set was observed collapsing 323MB → 21MB — the ONNX model gets paged out, and the next
   recall pays seconds of page-in. An int8-quantized embedding model would shrink the working set;
   so would splitting the dashboard off the hot path (#6).
d. Admission caps sized for one session, not a fleet.

**Also worth doing regardless of root cause:** raise the hook client timeout modestly (4s → ~8s).
The fallback still guards the prompt; today the timeout amputates server answers that were 5–10s
away, and the telemetry shows those answers do eventually arrive and get logged.

**Repair:** SQLite hot reads now use per-thread query-only connections instead of the writer's global
Python lock; ONNX query inference is serialized with a bounded LRU query cache; hook work runs in a
fixed executor whose admission slot remains occupied after a caller timeout until the real thread
exits; and the client timeout is 8s. The dead-supervisor gap now has a heartbeat checked at every
SessionStart. Per-client MCP processes proxy to the singleton warm service rather than multiplying
models.

**Measured proof 2026-08-02:** 12 distinct recalls at four-way concurrency returned 12/12 hybrid
contexts with zero shed/timeouts/fallbacks; p90 wall latency 908ms, max 1.054s. A vector-only query
ranked a Codex-sourced session first. The self-test includes guards proving reads bypass the writer
lock and timed-out work retains admission until it actually exits.

**Follow-up audit and second repair 2026-08-02:** live 24-hour telemetry still showed 24 hybrid
computations finishing after the 8s caller timeout and 28 delivered keyword fallbacks versus only 21
timely hybrid responses. The missing variable was live indexing: document and query embedding shared
one serialized ONNX lock, and a 64-document batch could hold it for 15–40s. The server deadline was
also 12s, longer than its 8s caller, so it knowingly produced abandoned answers. Repaired by strict
query priority between four-document microbatches, a 6s server/8s client budget, client delivery
receipts, and a focused `mem delivery-check --load` gate. Completion is no longer success telemetry.

## 2. Backup floor for the DB after porting — cheap, do immediately post-port

On the original machine the DB is rebuildable (transcripts + notes exist). **After a port it is
not**: the old transcripts stay behind, so `data/claudemem.db` becomes the only copy of
transcript-derived memory. Nightly copy of the DB + note stores, restore rehearsed once. (Same
doctrine as every other single-copy store on the old machine.)

## 3. Embedding model upgrade — the design's built-in door, never yet walked through

**[hypothesis]** `bge-small-en-v1.5` (2023) was picked as the best model guaranteed in fastembed on
a CPU box, not the best available today. Candidates that still fit no-GPU/low-RAM: EmbeddingGemma
(300M, 2025), snowflake-arctic-embed-s, bge-m3 (heavier). The harness for deciding **already
exists**: `mem eval` scores recall@k against `eval/golden.jsonl` and prints drift. Process: branch
config → `mem index --full` (44k chunks ≈ hours on CPU, schedule it) → `mem eval` delta → keep or
revert. Don't switch on reputation; switch on the golden-set delta. Note interaction with #1c: a
bigger model worsens the paging problem — quantized variants preferred.

## 4. Contextual enrichment is OFF — a known-value flag gated on spend

`[contextual]` (Anthropic contextual retrieval at index time) is implemented, tested, and off
because the hook environment has no `ANTHROPIC_API_KEY`. A key exists on the machine (separate
paid API pool). Enabling it for **curated notes only** (`enrich_notes = true`, transcripts off) is
low-volume and materially improves note recall. This costs real money per indexed chunk —
**operator decision, recorded here as recommended-with-cost, deliberately not flipped by an
agent.**

## 5. Retrieval constants have never been re-tuned

`recency_half_life_days = 45`, `rrf_k = 60`, `bm25_k/vector_k = 40`, `top_k = 6` are launch values.
The golden eval exists; a one-off parameter sweep (`mem eval` per variant) either improves recall@k
or confirms the defaults. Zero risk: it's all read-only measurement.

## 6. Dashboard and hot path share one process — mitigated, keep measuring

A heavy dashboard action still shares RAM/GIL with recall, but the measured dominant contention—the
ONNX lane—is now query-priority and indexing is interruptible between microbatches. Keep the single
resident model unless `delivery-check --load` or delivery receipts regress; process separation is a
last resort because a second model worsens memory pressure on this host.

## 7. Small coherence items

- `CLAUDE.md` says "run the test suite in `tests/`" but `tests/` is empty — the suite is
  `mem selftest` (30 checks). Fixed in the same commit as this document.
- README described ParadeDB as the primary store; the pinned reality is sqlite. Fixed alongside.
- Selftest count drifts upward as guards are added — docs should say "30" only where a number is
  unavoidable; prefer "all checks pass".
- The seven pending auto-mined candidates were reviewed on 2026-08-02 and rejected as contextless
  progress fragments or duplicates. The miner now requires recurrence for assistant-authored lessons;
  explicit one-off user corrections remain eligible for human review. A fresh run drafted zero.

## Non-goals (deliberate, not neglect)

- **Cross-platform support.** Windows-only launchers, PowerShell supervisor, Scheduled Task — this
  ports Windows→Windows. A Linux port is a different piece of work (systemd unit replaces the
  Scheduled Task; `mem.cmd` → shell alias) and nothing in the Python core blocks it.
- **Cloud sync of memory.** The privacy stance (nothing leaves the machine) is a feature. Moving
  memory between machines is an explicit, operator-driven copy (PORTING §4), not a sync service.
- **Hot-path reranking.** Measured at ~1s/prompt on this CPU vs 68–90ms without — permanently on
  the dashboard tier unless the hardware changes.
