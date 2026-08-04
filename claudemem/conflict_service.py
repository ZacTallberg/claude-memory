"""Synchronize derived contradiction metadata after out-of-band note writes."""
from __future__ import annotations

from pathlib import Path

from .config import Config
from .facts import load_notes
from .memory_types import conflict_index, find_claim_conflicts
from .store.base import Store


def _canon(path) -> str:
    """One canonical spelling for a path used as a JOIN KEY.

    Notes loaded from the cfg roots and fact paths stored by the writer can spell the same
    file differently — on Windows an 8.3 short segment (%TEMP% under e.g. ZACHAR~1.OBE)
    survives in one origin and not the other, and the join then silently misses, leaving
    contradictions unquarantined. Resolve both sides before comparing.
    """
    try:
        return str(Path(path).resolve())
    except OSError:
        return str(path)


def synchronize_fact_conflicts(cfg: Config, store: Store) -> list[dict]:
    """Recompute conflicts and quarantine both indexed sides immediately.

    The normal index pass does this while upserting notes. MCP/dashboard promotion writes can happen
    between index passes, so this small metadata-only synchronization closes the unsafe interval
    without re-embedding unchanged content.
    """
    conflicts = find_claim_conflicts(load_notes(cfg))
    by_path = {_canon(k): v for k, v in conflict_index(conflicts).items()}
    for fact in store.list_facts():
        expected = by_path.get(_canon(fact.path), [])
        meta = dict(fact.meta or {})
        if list(meta.get("conflict_ids") or []) == expected:
            continue
        meta["conflict_ids"] = expected
        store.update_fact_meta(fact.id, meta)
    return conflicts
