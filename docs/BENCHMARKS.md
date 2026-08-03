# Quality and model-selection harness

The critical prompt path contains only bounded production guards. Broader tests and experiments are
manual, release-time, or scheduled offline, where they add evidence without delaying every agent
turn. No command in this document activates an experimental ranking or mutates live embeddings.

## Gates

- `mem selftest` checks behavioral invariants, migrations, adapters, typed conflicts, attribution,
  model-dimension handling, and benchmark adapters.
- `mem eval` evaluates the live retriever on the local golden set: coverage, strict recall, session
  and fact MRR, negative-target safety, and continuation usefulness.
- `mem delivery-check --load` validates the actual client delivery boundary and latency under
  concurrent prompts while indexing.
- `scripts/release_check.ps1` composes the release gate. The weekly/manual GitHub workflow remains
  off the merge and prompt critical paths.

## Read-only experiments

- `mem tune-ranking` sweeps RRF and recency settings. A candidate must preserve coverage, strict
  recall, and negative safety; ties prefer the live configuration.
- `mem feedback-experiment` joins explicit outcomes to delivery receipts. Explicit item attribution
  counts more than request-level attribution, evidence is shrunk toward neutral, per-item effects are
  capped, and the golden set must improve without a safety regression. It refuses to evaluate below
  the configured evidence floor.
- `mem model-bakeoff current <fastembed-model>` embeds the same bounded evidence pool for each model.
  It uses each model's query/passage semantics and native dimension, refuses arbitrary truncation,
  and requires coverage, strict-recall, and negative-safety parity before MRR can select a winner. A winning
  candidate is evidence for a separate full reindex and release gate—not an automatic cutover.
- `mem longmemeval <dataset.json>` adapts the official LongMemEval V1/cleaned JSON format into a
  session-retrieval test and reports Recall@k/MRR by question category. It does **not** claim the
  benchmark's end-to-end answer-generation accuracy; retrieval and answer synthesis are different
  measurements.

Reports are timestamped under `data/experiments` or `data/benchmarks` and include
`live_change_applied: false`. Model bake-offs are CPU-intensive and should be run when latency-
sensitive game workers are idle.

## Current evidence (2026-08-03)

The 22-query live golden evaluation achieved useful recall@6 = 1.000, negative-safety@6 = 1.000,
strict recall@6 = 0.955, session MRR = 0.505, and fact MRR = 0.886. A bounded dense bake-off compared
`BAAI/bge-small-en-v1.5` with `snowflake/snowflake-arctic-embed-s`; BGE preserved 1.000 coverage and
0.955 strict recall and achieved mean MRR 0.855, while Arctic-S achieved 0.955 coverage, 0.818 strict
recall, and mean MRR 0.816. The deployed BGE model therefore remains the evidence-backed choice among
the tested local candidates. This conclusion is local to this corpus and harness, not a universal
embedding leaderboard.
