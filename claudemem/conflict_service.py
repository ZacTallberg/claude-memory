"""Synchronize derived contradiction metadata after out-of-band note writes."""
from __future__ import annotations

from .config import Config
from .facts import load_notes
from .memory_types import conflict_index, find_claim_conflicts
from .store.base import Store


def synchronize_fact_conflicts(cfg: Config, store: Store) -> list[dict]:
    """Recompute conflicts and quarantine both indexed sides immediately.

    The normal index pass does this while upserting notes. MCP/dashboard promotion writes can happen
    between index passes, so this small metadata-only synchronization closes the unsafe interval
    without re-embedding unchanged content.
    """
    conflicts = find_claim_conflicts(load_notes(cfg))
    by_path = conflict_index(conflicts)
    for fact in store.list_facts():
        expected = by_path.get(str(fact.path), [])
        meta = dict(fact.meta or {})
        if list(meta.get("conflict_ids") or []) == expected:
            continue
        meta["conflict_ids"] = expected
        store.update_fact_meta(fact.id, meta)
    return conflicts
