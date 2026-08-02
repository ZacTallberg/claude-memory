# Gold-64 task bed

This directory holds adjudicated retrieval cases, not generated placeholder tasks.
Do not create a case until its exact canonical evidence has been located in the V2
store and independently checked.

The first release set must contain 64 real cases: eight in each category defined in
`docs/RELEASE-GATES.md`, with 24 development cases and 40 sealed test cases. Evidence
groups are ANDed; alternative items inside one group are ORed. Labels use canonical
event/claim/procedure IDs and content hashes, never SQLite row numbers or search-index
document IDs.

Generate the current schema with:

```powershell
system-memory eval-schema eval/gold64.schema.json
```

Validate the complete case file without running it by loading it through the model
and `validate_gold64`. Run development cases freely. Running the sealed split requires
the explicit `--allow-sealed` flag and should happen only for a finalist recorded with
its corpus hash, code revision, dependency lock hash, and exact model manifests.

Evaluation reports contain case IDs, stable result references, scores, modes, and
latencies. They intentionally omit query text and memory bodies.
