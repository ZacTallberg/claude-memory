"""Temporal lifecycle rules for curated notes.

Markdown remains authoritative. These helpers interpret optional lifecycle metadata stored
in frontmatter and mirrored into the derived database ``meta`` column.
"""
from __future__ import annotations

from datetime import datetime, timezone

ACTIVE_STATUSES = {"active", "current"}
VALID_STATUSES = ACTIVE_STATUSES | {"superseded", "obsolete", "draft"}
VALID_VISIBILITIES = {"machine", "project", "private"}


def parse_datetime(value) -> datetime | None:
    if not value:
        return None
    if isinstance(value, datetime):
        dt = value
    else:
        text = str(value).strip().replace("Z", "+00:00")
        try:
            dt = datetime.fromisoformat(text)
        except ValueError:
            return None
    return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt.astimezone(timezone.utc)


def normalize_lifecycle(meta: dict) -> dict:
    nested = meta.get("metadata") if isinstance(meta.get("metadata"), dict) else {}

    def pick(*keys, default=None):
        for key in keys:
            if key in meta and meta[key] not in (None, ""):
                return meta[key]
            if key in nested and nested[key] not in (None, ""):
                return nested[key]
        return default

    status = str(pick("status", default="active")).strip().lower()
    if status not in VALID_STATUSES:
        status = "active"
    visibility = str(pick("visibility", default="machine")).strip().lower()
    if visibility not in VALID_VISIBILITIES:
        visibility = "machine"
    try:
        confidence = max(0.0, min(1.0, float(pick("confidence", default=1.0))))
    except (TypeError, ValueError):
        confidence = 1.0
    valid_from = parse_datetime(pick("valid_from", "validFrom"))
    valid_to = parse_datetime(pick("valid_to", "validTo"))
    supersedes = pick("supersedes")
    superseded_by = pick("superseded_by", "supersededBy")
    provenance = pick("provenance", "source")
    return {
        "status": status,
        "visibility": visibility,
        "confidence": confidence,
        "valid_from": valid_from.isoformat() if valid_from else None,
        "valid_to": valid_to.isoformat() if valid_to else None,
        "supersedes": str(supersedes).strip() or None if supersedes else None,
        "superseded_by": str(superseded_by).strip() or None if superseded_by else None,
        "provenance": str(provenance).strip() or None if provenance else None,
    }


def fact_is_active(fact, *, now: datetime | None = None) -> bool:
    """Whether a fact is eligible for automatic recall and the session-start map."""
    meta = fact.meta or {}
    status = str(meta.get("status") or "active").lower()
    if status not in ACTIVE_STATUSES:
        return False
    current = now or datetime.now(timezone.utc)
    valid_from = parse_datetime(meta.get("valid_from"))
    valid_to = parse_datetime(meta.get("valid_to"))
    if valid_from and current < valid_from:
        return False
    if valid_to and current >= valid_to:
        return False
    return True


def fact_is_recallable(fact, *, project: str | None = None,
                       now: datetime | None = None) -> bool:
    """Whether a fact may enter automatic context for this project.

    Low-confidence and ``private`` notes are manual-only. ``project`` is delivered only to a
    matching cwd-derived project label. ``machine`` is available in every installed context.
    """
    if not fact_is_active(fact, now=now):
        return False
    # A single-valued structured claim that conflicts with another currently valid note is
    # manual-review material, not safe automatic context. The notes remain searchable and
    # auditable; resolving the validity/supersession metadata restores automatic delivery.
    if (fact.meta or {}).get("conflict_ids"):
        return False
    try:
        if float((fact.meta or {}).get("confidence", 1.0)) < 0.5:
            return False
    except (TypeError, ValueError):
        return False
    visibility = str((fact.meta or {}).get("visibility") or "machine").lower()
    if visibility == "machine":
        return True
    if visibility == "project" and project:
        return str(fact.project or "").casefold() == str(project).casefold()
    return False
