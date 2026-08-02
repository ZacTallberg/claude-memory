"""Curated note (markdown + frontmatter) discovery + parsing, and graph extraction."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .config import Config
from .log import get_logger
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
        mtime=path.stat().st_mtime,
    )


def load_notes(cfg: Config) -> list[NoteData]:
    out: list[NoteData] = []
    for nf, project in iter_note_files(cfg):
        nd = load_note(nf, project)
        if nd:
            out.append(nd)
    return out


def build_graph(notes: list[NoteData]) -> tuple[list[dict], list[dict]]:
    """Nodes = notes (+ referenced-but-missing targets); edges = [[wikilink]] relations."""
    nodes: dict[str, dict] = {}
    for n in notes:
        nodes[n.name] = {"id": n.name, "label": n.title, "type": n.type, "group": n.project}
    edges: list[dict] = []
    for n in notes:
        for target in n.wikilinks:
            if target not in nodes:
                nodes[target] = {"id": target, "label": target, "type": "missing", "group": n.project}
            edges.append({"source": n.name, "target": target, "kind": "links"})
    return list(nodes.values()), edges
