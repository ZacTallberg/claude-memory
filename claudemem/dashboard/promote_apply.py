"""Apply an accepted promotion candidate: write a curated note into a project memory dir and index it."""
from __future__ import annotations

import re
from pathlib import Path

from claudemem.config import Config
from claudemem.facts import load_note
from claudemem.paths import iter_memory_dirs
from claudemem.store.base import Store


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", (s or "note").lower()).strip("_")[:48] or "note"


def accept_candidate(cfg: Config, store: Store, pid: int, project: str | None = None) -> None:
    cand = next((c for c in store.list_promotions() if c["id"] == pid), None)
    if not cand:
        return
    dirs = iter_memory_dirs(cfg)
    target = None
    if project:
        target = next((d.path for d in dirs if d.project == project), None)
    if target is None and dirs:
        target = dirs[0].path
    if target is None:
        store.update_promotion(pid, "accepted")
        return
    fp = Path(target) / f"promoted_{_slug(cand['title'])}.md"
    fp.write_text(
        f"---\nname: {cand['title']}\ndescription: {cand['title']}\n"
        f"metadata:\n  type: {cand.get('type', 'reference')}\n  source: claude-memory-promotion\n---\n\n"
        f"{cand['body']}\n", encoding="utf-8")
    nd = load_note(fp, project or "code")
    if nd:
        store.upsert_fact(path=nd.path, project=nd.project, name=nd.name, title=nd.title,
                          description=nd.description, type=nd.type, tags=nd.tags,
                          origin_session_id=nd.origin_session_id, body=nd.body, embedding=None,
                          mtime=nd.mtime, meta={"wikilinks": nd.wikilinks})
    store.update_promotion(pid, "accepted")
