# Changelog

## 0.3.0 — 2026-08-03

- Added semantic, episodic, and procedural memory kinds plus bounded importance metadata.
- Added explicit temporal claims, deterministic contradiction detection, conflict inspection, and
  fail-closed automatic recall until a conflict is resolved by validity or supersession.
- Added client-neutral transcript adapters through a small protocol and Python entry points; adapter
  extensions load only in the background indexer, never in prompt hooks.
- Added review-first background consolidation for durable decisions, outcomes, and recurring lessons.
- Added delivery-level chunk/fact attribution and a conservative, offline-only feedback learner with
  Bayesian shrinkage, bounded weights, golden-set gates, and no automatic production activation.
- Added read-only rank sweeps, native-dimension embedding bake-offs, and a LongMemEval session-
  retrieval adapter. All experiments write auditable reports and cannot mutate the live index.
- Corrected local embedding calls to use model-specific query and passage semantics.
- Replaced concurrent SQLite flat-vector SQL scans on the warm path with an exact, revision-tracked
  in-memory matrix that updates incrementally during indexing; four-way loaded delivery fell from
  about 3.49s to at most 1.91s across the final repeated gates without changing the 3.0s SLO.
- Extended graph storage with typed node/edge metadata and expanded the regression suite to 53 checks.

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
