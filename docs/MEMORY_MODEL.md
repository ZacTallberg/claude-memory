# Memory model

The memory layer is machine-wide and client-neutral. Claude Code and Codex are currently built-in
sources and delivery clients; neither owns the memory model. Other local agent runtimes can join via
the transcript-adapter contract or MCP without changing the core.

## Three useful kinds of memory

- **Semantic** — durable facts, preferences, architectural decisions, and relationships.
- **Episodic** — an observed event or outcome: what happened, in what context, and how it ended.
- **Procedural** — a reusable way of working, including corrections and verified practices.

Every curated Markdown note can carry `memory_kind` and an `importance` from 0 to 1. Older notes are
compatible and default conservatively. These fields influence ranking only slightly; lexical/vector
relevance and lifecycle eligibility remain dominant.

```yaml
---
name: memory-model-selection
description: The embedding model currently approved for production memory.
metadata:
  type: reference
  memory_kind: semantic
  importance: 0.9
  claims:
    - subject: memory.embedding
      predicate: model
      object: BAAI/bge-small-en-v1.5
      cardinality: one
      confidence: 1.0
      valid_from: 2026-08-03T00:00:00Z
      provenance: model-bakeoff-20260803-055032
---
```

## Temporal claims and contradictions

Claims are optional and explicit. A claim has a subject, predicate, object, cardinality (`one` or
`many`), confidence, optional validity interval, status, and provenance. The system does not pretend
that arbitrary prose has been perfectly converted into a knowledge graph.

Two active, overlapping, single-valued claims with the same subject and predicate but different
objects are a conflict. Both notes remain visible for audit and manual search, but both are withheld
from automatic recall. Resolve the conflict by superseding one note, ending a validity interval, or
marking a genuinely multi-valued relation as `many`. Inspect conflicts with `mem conflicts` or the
MCP `memory_conflicts` tool. This is deliberately fail-closed: unresolved truth is not injected as if
it were settled.

## Consolidation

`mem consolidate` mines durable candidate memories from transcripts. The indexer may run the same
bounded scan at most once per configured interval. It creates review candidates only; it never
silently promotes generated text into authoritative notes. Decisions, verified outcomes, explicit
user corrections, and recurring lessons receive stronger evidence than generic progress narration.

Markdown files remain the source of truth. Types, graph metadata, conflict indexes, embeddings, and
ranking signals are all rebuildable derived state.
