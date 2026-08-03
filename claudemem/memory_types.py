"""Typed memory metadata and deterministic temporal-claim analysis.

Markdown remains authoritative.  This module interprets optional structured metadata without
trying to infer facts on the prompt path.  A note may declare ``memory_kind`` and ``claims`` in
top-level frontmatter or inside ``metadata``::

    metadata:
      memory_kind: semantic
      importance: 0.9
      claims:
        - subject: memory.embedding
          predicate: model
          object: BAAI/bge-small-en-v1.5
          cardinality: one

Only claims explicitly marked ``cardinality: one`` can conflict.  This avoids treating legitimate
many-valued relationships as contradictions.  Conflicts are derived diagnostics; no note is
silently rewritten or deleted.
"""
from __future__ import annotations

import hashlib
import re
from datetime import datetime, timezone
from typing import Any, Iterable

from .lifecycle import ACTIVE_STATUSES, parse_datetime

MEMORY_KINDS = {"semantic", "episodic", "procedural"}
CLAIM_CARDINALITIES = {"one", "many"}


def _nested(meta: dict) -> dict:
    value = meta.get("metadata")
    return value if isinstance(value, dict) else {}


def _pick(meta: dict, *keys: str, default=None):
    nested = _nested(meta)
    for key in keys:
        if key in meta and meta[key] not in (None, ""):
            return meta[key]
        if key in nested and nested[key] not in (None, ""):
            return nested[key]
    return default


def _clean(value: Any, *, limit: int = 500) -> str:
    return " ".join(str(value or "").split())[:limit]


def _norm(value: Any) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", str(value or "").casefold()))


def default_memory_kind(note_type: str) -> str:
    """Map the older note taxonomy onto the cognitive-memory taxonomy."""
    return "procedural" if str(note_type).casefold() == "feedback" else "semantic"


def _normalize_claim(raw: dict, *, note_name: str, lifecycle: dict, index: int) -> dict | None:
    subject = _clean(raw.get("subject"), limit=200)
    predicate = _clean(raw.get("predicate") or raw.get("relation"), limit=120)
    obj = _clean(raw.get("object") if "object" in raw else raw.get("value"), limit=500)
    if not subject or not predicate or not obj:
        return None
    cardinality = str(raw.get("cardinality") or "many").strip().casefold()
    if cardinality not in CLAIM_CARDINALITIES:
        cardinality = "many"
    status = str(raw.get("status") or lifecycle.get("status") or "active").strip().casefold()
    try:
        confidence = max(0.0, min(1.0, float(raw.get("confidence", lifecycle.get("confidence", 1.0)))))
    except (TypeError, ValueError):
        confidence = float(lifecycle.get("confidence", 1.0) or 1.0)
    valid_from = parse_datetime(raw.get("valid_from") or raw.get("validFrom")
                                or lifecycle.get("valid_from"))
    valid_to = parse_datetime(raw.get("valid_to") or raw.get("validTo")
                              or lifecycle.get("valid_to"))
    provenance = _clean(raw.get("provenance") or lifecycle.get("provenance"), limit=500) or None
    stable = "\x1f".join((_norm(note_name), _norm(subject), _norm(predicate), _norm(obj), str(index)))
    claim_id = _clean(raw.get("id"), limit=128) or hashlib.sha256(stable.encode("utf-8")).hexdigest()[:20]
    return {
        "id": claim_id,
        "subject": subject,
        "predicate": predicate,
        "object": obj,
        "cardinality": cardinality,
        "status": status,
        "confidence": confidence,
        "valid_from": valid_from.isoformat() if valid_from else None,
        "valid_to": valid_to.isoformat() if valid_to else None,
        "provenance": provenance,
        "note": note_name,
    }


def normalize_memory_metadata(meta: dict, *, note_name: str, note_type: str,
                              lifecycle: dict) -> dict:
    """Normalize cognitive kind, importance, and explicit temporal claims."""
    kind = str(_pick(meta, "memory_kind", "memoryKind",
                     default=default_memory_kind(note_type))).strip().casefold()
    if kind not in MEMORY_KINDS:
        kind = default_memory_kind(note_type)
    try:
        importance = max(0.0, min(1.0, float(_pick(meta, "importance", default=0.7))))
    except (TypeError, ValueError):
        importance = 0.7
    raw_claims = _pick(meta, "claims", default=[])
    if isinstance(raw_claims, dict):
        raw_claims = [raw_claims]
    if not isinstance(raw_claims, list):
        raw_claims = []
    claims: list[dict] = []
    for index, raw in enumerate(raw_claims):
        if not isinstance(raw, dict):
            continue
        claim = _normalize_claim(raw, note_name=note_name, lifecycle=lifecycle, index=index)
        if claim:
            claims.append(claim)
    return {"memory_kind": kind, "importance": importance, "claims": claims}


def _claim_active(claim: dict, *, now: datetime) -> bool:
    if str(claim.get("status") or "active").casefold() not in ACTIVE_STATUSES:
        return False
    valid_from = parse_datetime(claim.get("valid_from"))
    valid_to = parse_datetime(claim.get("valid_to"))
    return not ((valid_from and now < valid_from) or (valid_to and now >= valid_to))


def _intervals_overlap(a: dict, b: dict) -> bool:
    floor = datetime.min.replace(tzinfo=timezone.utc)
    ceiling = datetime.max.replace(tzinfo=timezone.utc)
    a_from = parse_datetime(a.get("valid_from")) or floor
    b_from = parse_datetime(b.get("valid_from")) or floor
    a_to = parse_datetime(a.get("valid_to")) or ceiling
    b_to = parse_datetime(b.get("valid_to")) or ceiling
    return max(a_from, b_from) < min(a_to, b_to)


def find_claim_conflicts(notes: Iterable, *, now: datetime | None = None) -> list[dict]:
    """Return unresolved, overlapping single-valued claims with different values.

    A declared note-level supersession resolves the historical pair.  The result is stable and
    suitable for the CLI, MCP, dashboard, and derived fact metadata.
    """
    current = now or datetime.now(timezone.utc)
    notes = list(notes)
    groups: dict[tuple[str, str], list[tuple[Any, dict]]] = {}
    for note in notes:
        if str(note.lifecycle.get("status") or "active").casefold() not in ACTIVE_STATUSES:
            continue
        note_from = parse_datetime(note.lifecycle.get("valid_from"))
        note_to = parse_datetime(note.lifecycle.get("valid_to"))
        if (note_from and current < note_from) or (note_to and current >= note_to):
            continue
        for claim in note.claims:
            if claim.get("cardinality") != "one" or not _claim_active(claim, now=current):
                continue
            groups.setdefault((_norm(claim["subject"]), _norm(claim["predicate"])), []).append(
                (note, claim)
            )

    conflicts: list[dict] = []
    seen: set[tuple[str, str, str, str]] = set()
    for (_subject_key, _predicate_key), items in groups.items():
        for left_index, (left_note, left) in enumerate(items):
            for right_note, right in items[left_index + 1:]:
                if _norm(left["object"]) == _norm(right["object"]):
                    continue
                if not _intervals_overlap(left, right):
                    continue
                # Paths are the authoritative note identity. Names are human labels and may be
                # duplicated across projects or between canonical and compatible legacy roots.
                pair = tuple(sorted((str(left_note.path), str(right_note.path))))
                pair_key = (_subject_key, _predicate_key, *pair)
                if pair_key in seen:
                    continue
                seen.add(pair_key)
                left_supersedes = str(left_note.lifecycle.get("supersedes") or "")
                right_supersedes = str(right_note.lifecycle.get("supersedes") or "")
                if (left_supersedes == right_note.name or right_supersedes == left_note.name
                        or str(left_note.lifecycle.get("superseded_by") or "") == right_note.name
                        or str(right_note.lifecycle.get("superseded_by") or "") == left_note.name):
                    continue
                stable = "\x1f".join((left["subject"], left["predicate"], *pair))
                conflicts.append({
                    "id": hashlib.sha256(stable.encode("utf-8")).hexdigest()[:20],
                    "subject": left["subject"],
                    "predicate": left["predicate"],
                    "left": {"note": left_note.name, "project": left_note.project,
                             "path": str(left_note.path), "value": left["object"],
                             "claim_id": left["id"], "confidence": left["confidence"]},
                    "right": {"note": right_note.name, "project": right_note.project,
                              "path": str(right_note.path), "value": right["object"],
                              "claim_id": right["id"], "confidence": right["confidence"]},
                    "severity": "high" if min(left["confidence"], right["confidence"]) >= 0.8
                    else "medium",
                    "resolution": "supersede one note, bound its validity interval, or mark the claim many-valued",
                })
    return sorted(conflicts, key=lambda c: (c["subject"].casefold(), c["predicate"].casefold(), c["id"]))


def conflict_index(conflicts: Iterable[dict]) -> dict[str, list[str]]:
    """Map authoritative note paths to conflict ids for derived indexing metadata."""
    out: dict[str, list[str]] = {}
    for conflict in conflicts:
        for side in ("left", "right"):
            path = str(conflict[side].get("path") or conflict[side]["note"])
            out.setdefault(path, []).append(str(conflict["id"]))
    return out
