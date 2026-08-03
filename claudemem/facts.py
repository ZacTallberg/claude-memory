"""Curated note (markdown + frontmatter) discovery + parsing, and graph extraction."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .config import Config
from .lifecycle import normalize_lifecycle
from .log import get_logger
from .memory_types import normalize_memory_metadata
from .paths import iter_note_files
from .security import redact_secrets
from .text import (collapse_ws, extract_wikilinks, normalize_note_type,
                   note_origin_session, parse_frontmatter)

log = get_logger(__name__)


@dataclass
class NoteData:
    path: str
    project: str
    name: str
    title: str
    description: str
    type: str
    tags: list[str]
    origin_session_id: str | None
    body: str
    wikilinks: list[str]
    lifecycle: dict
    memory_kind: str
    importance: float
    claims: list[dict]
    mtime: float


def _title_from(meta: dict, fallback_filename: str) -> str:
    # `name` is the stable graph/filename identifier; `title` is display prose.
    for key in ("title", "name"):
        v = meta.get(key)
        if v:
            return collapse_ws(str(v))
    stem = Path(fallback_filename).stem
    return stem.replace("_", " ").replace("-", " ").strip().title()


def load_note(path: Path, project: str) -> NoteData | None:
    try:
        raw = path.read_text(encoding="utf-8")
    except Exception as e:
        log.warning("cannot read note %s: %s", path, e)
        return None
    meta, body = parse_frontmatter(raw)
    tags = meta.get("tags") or []
    nested = meta.get("metadata")
    if not tags and isinstance(nested, dict):
        tags = nested.get("tags") or []
    if isinstance(tags, str):
        tags = [t.strip() for t in tags.split(",") if t.strip()]
    elif not isinstance(tags, list):
        tags = []
    lifecycle = normalize_lifecycle(meta)
    memory_meta = normalize_memory_metadata(meta, note_name=str(meta.get("name") or path.stem),
                                            note_type=normalize_note_type(meta),
                                            lifecycle=lifecycle)
    for claim in memory_meta["claims"]:
        for key in ("subject", "predicate", "object", "provenance"):
            if claim.get(key):
                claim[key] = redact_secrets(str(claim[key]))[0]
    return NoteData(
        path=str(path),
        project=project,
        name=str(meta.get("name") or path.stem),
        title=_title_from(meta, path.name),
        description=collapse_ws(redact_secrets(str(meta.get("description", "")))[0]),
        type=normalize_note_type(meta),
        tags=[str(t) for t in tags],
        origin_session_id=note_origin_session(meta),
        body=redact_secrets(body)[0].strip(),
        wikilinks=extract_wikilinks(body),
        lifecycle=lifecycle,
        memory_kind=memory_meta["memory_kind"],
        importance=memory_meta["importance"],
        claims=memory_meta["claims"],
        mtime=path.stat().st_mtime,
    )


def load_notes(cfg: Config) -> list[NoteData]:
    out: list[NoteData] = []
    for nf, project in iter_note_files(cfg):
        nd = load_note(nf, project)
        if nd:
            out.append(nd)
    return out


def _entity_id(value: str) -> str:
    import hashlib
    return "entity:" + hashlib.sha256(value.casefold().encode("utf-8")).hexdigest()[:20]


def build_graph(notes: list[NoteData]) -> tuple[list[dict], list[dict]]:
    """Build the inspectable note graph plus explicit typed temporal claims.

    Wikilinks remain lightweight note-to-note edges. Structured claims add entity nodes and
    predicate edges carrying validity, confidence, provenance, and source-note metadata.
    """
    nodes: dict[str, dict] = {}
    for n in notes:
        nodes[n.name] = {"id": n.name, "label": n.title, "type": n.type, "group": n.project,
                         "meta": {"node_kind": "note", "memory_kind": n.memory_kind,
                                  "importance": n.importance,
                                  "status": n.lifecycle.get("status")}}
    edges: list[dict] = []
    for n in notes:
        for target in n.wikilinks:
            if target not in nodes:
                nodes[target] = {"id": target, "label": target, "type": "missing", "group": n.project}
            edges.append({"source": n.name, "target": target, "kind": "links",
                          "meta": {"source_note": n.name}})
        supersedes = n.lifecycle.get("supersedes")
        if supersedes:
            if supersedes not in nodes:
                nodes[supersedes] = {"id": supersedes, "label": supersedes,
                                     "type": "missing", "group": n.project}
            edges.append({"source": n.name, "target": supersedes, "kind": "supersedes",
                          "meta": {"source_note": n.name,
                                   "valid_from": n.lifecycle.get("valid_from")}})
        for claim in n.claims:
            subject_id = _entity_id(claim["subject"])
            object_id = _entity_id(claim["object"])
            nodes.setdefault(subject_id, {"id": subject_id, "label": claim["subject"],
                                          "type": "entity", "group": n.project,
                                          "meta": {"node_kind": "entity"}})
            nodes.setdefault(object_id, {"id": object_id, "label": claim["object"],
                                         "type": "entity", "group": n.project,
                                         "meta": {"node_kind": "entity"}})
            edges.append({"source": n.name, "target": subject_id, "kind": "asserts",
                          "meta": {"source_note": n.name, "claim_id": claim["id"]}})
            edges.append({"source": subject_id, "target": object_id,
                          "kind": claim["predicate"],
                          "meta": {"source_note": n.name, "claim_id": claim["id"],
                                   "cardinality": claim["cardinality"],
                                   "status": claim["status"],
                                   "confidence": claim["confidence"],
                                   "valid_from": claim["valid_from"],
                                   "valid_to": claim["valid_to"],
                                   "provenance": claim["provenance"]}})
    return list(nodes.values()), edges
