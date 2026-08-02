# Release gates

V2 does not replace the live service because it starts, looks polished, or retrieves
one anecdote. Every exactness gate is all-or-nothing; aggregate metrics cannot hide a
security, deletion, attribution, or durability failure.

## Gold task bed

The first sealed set contains 64 real cases, eight in each category:

1. direct user facts, preferences, and values;
2. assistant decisions and commitments;
3. cross-session and cross-project synthesis;
4. temporal updates, corrections, reversions, and historical state;
5. procedures, environment gotchas, and false-premise awareness;
6. provider, project, session, role, and agent attribution;
7. abstention and hard semantic negatives;
8. contamination, forgetting, secret purge, and collateral retention.

Evidence groups are ANDed; equivalent evidence inside a group is ORed. Labels point to
stable provider/source/message/span hashes, never mutable database row IDs. Related
paraphrases and updates stay in one split. Twenty-four cases are development data and
forty remain sealed until finalists are selected.

## Quality gates

| Dimension | Gate |
|---|---:|
| Strict evidence-group Recall@6 | at least 90%, lower 95% CI at least 85% |
| nDCG@6 | at least 0.85 |
| MRR@6 | at least 0.80 |
| Category floor | no category below 80% |
| False context on negative queries | at most 5% |
| Superseded evidence in current-state top six | at most 2% |
| Provider/project/session/role metadata | 100% exact |
| Explicitly forgotten or quarantined evidence | 0 retrievals |
| Ambient/injected/system payload contamination | 0 cases |
| Irrelevant returned tokens | at most 35% |

## Operational gates

| Dimension | Gate |
|---|---:|
| Hook-to-cross-provider lexical freshness | p95 15 seconds, maximum 30 seconds |
| Vector freshness | p95 120 seconds in steady state |
| Warm fast recall | p50 1 second, p95 2.5 seconds, p99 5 seconds |
| Five concurrent agents | p95 4 seconds, p99 7 seconds |
| Silent fallback | zero |
| Fallback, timeout, and shed rate | at most 1% steady state |
| Steady/peak working set | at most 750 MB / 1.5 GB in five-agent soak |
| Crash to usable recall | at most 60 seconds |
| Cursor advance without matching canonical commit | zero |
| Unknown-process termination | zero |
| Verified backup/restore | 100%, zero task-bed score delta |

## Failure injection

Run only on disposable snapshots:

- embedder and reranker unavailable;
- vector extension unavailable;
- warm daemon stopped or wedged;
- writer lock held;
- all request slots saturated;
- background indexing during five simultaneous queries;
- partial transcript/event payload;
- duplicate lifecycle delivery;
- process killed at every ingest and backup transaction boundary;
- source deleted or truncated after indexing;
- model revision or dimension changed across restart;
- malformed ambient and injected-memory blocks;
- clock/time-zone skew during temporal retrieval;
- concurrent contradictory claims;
- latest backup restored into an empty installation.

## Model bakeoff

Keep the corpus, chunking, labels, and storage fixed. Compare BM25-only, vector-only,
hybrid, and reranked hybrid separately. BGE-small is the control. Candidate embedding
generations may include Arctic Embed S, GTE ModernBERT, EmbeddingGemma, Nomic Embed
v1.5/v2, and Qwen3 Embedding at hardware-feasible sizes. Candidate rerankers include
the current MiniLM control, Mixedbread xsmall, and GTE reranker.

A candidate is promoted only if it improves strict Recall@6 by at least three points
or nDCG@6 by at least two points, has no category regression above two points, and
passes every latency and resource gate. Each run records corpus hash, code commit,
lockfile, exact model revision/checksum, quantization, dimension, prefixes, hardware,
configuration, and per-query output.

