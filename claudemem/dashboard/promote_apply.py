"""Apply an accepted promotion candidate: write a curated note into a project memory dir and index it."""
from __future__ import annotations

import re
from pathlib import Path

import yaml

from claudemem.config import Config
from claudemem.facts import load_note
from claudemem.note_io import atomic_write_text
from claudemem.paths import canonical_memory_root, is_curated_note_path, iter_memory_dirs
from claudemem.security import redact_secrets
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
    if target is None:
        root = canonical_memory_root(cfg)
        target = root if not project or project == "global" else root / _slug(project)
    target.mkdir(parents=True, exist_ok=True)
    safe_title = redact_secrets(str(cand["title"]))[0]
    safe_body = redact_secrets(str(cand["body"]))[0]
    fp = Path(target) / f"promoted_{_slug(safe_title)}.md"
    if not is_curated_note_path(fp, cfg):
        return
    front = {"name": safe_title, "title": safe_title, "description": safe_title,
             "metadata": {"node_type": "memory", "type": cand.get("type", "reference"),
                          "status": "active", "visibility": "machine", "confidence": 0.8,
                          "provenance": "promotion-review"}}
    frontmatter = yaml.safe_dump(front, sort_keys=False, allow_unicode=True,
                                 default_flow_style=False)
    atomic_write_text(fp, f"---\n{frontmatter}---\n\n{safe_body}\n")
    nd = load_note(fp, project or "code")
    if nd:
        store.upsert_fact(path=nd.path, project=nd.project, name=nd.name, title=nd.title,
                          description=nd.description, type=nd.type, tags=nd.tags,
                          origin_session_id=nd.origin_session_id, body=nd.body, embedding=None,
                          mtime=nd.mtime, meta={"wikilinks": nd.wikilinks,
                                               "lifecycle_schema": 1, **nd.lifecycle})
    store.update_promotion(pid, "accepted")
