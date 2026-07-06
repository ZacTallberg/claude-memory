"""Claude Code JSONL transcript reader: incremental tail-indexing by byte offset,
live-file-safe (stops at the first partial line), with corpus cleaning.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from .config import Config
from .log import get_logger
from .text import collapse_ws, strip_injected_blocks

log = get_logger(__name__)

_MESSAGE_TYPES = {"user", "assistant"}


@dataclass
class Unit:
    role: str            # user | assistant
    kind: str            # text | thinking | tool
    text: str
    session_id: str | None
    cwd: str | None
    ts: datetime | None
    ordinal: int


def _parse_ts(s) -> datetime | None:
    if not s:
        return None
    try:
        return datetime.fromisoformat(str(s).replace("Z", "+00:00")).astimezone(timezone.utc)
    except Exception:
        return None


def _tool_summary(block: dict) -> str:
    t = block.get("type")
    if t == "tool_use":
        name = block.get("name", "tool")
        inp = block.get("input") or {}
        keys = ", ".join(list(inp.keys())[:6])
        return f"[tool_use {name}] {keys}".strip()
    if t == "tool_result":
        c = block.get("content")
        if isinstance(c, list):
            c = " ".join(b.get("text", "") for b in c if isinstance(b, dict))
        head = collapse_ws(str(c or ""))[:200]
        return f"[tool_result] {head}".strip()
    return ""


def _record_units(rec: dict, cfg: Config, ordinal_start: int) -> tuple[list[Unit], int]:
    if rec.get("type") not in _MESSAGE_TYPES:
        return [], ordinal_start
    if rec.get("isMeta"):
        return [], ordinal_start
    if rec.get("isSidechain") and cfg.index.exclude_sidechains:
        return [], ordinal_start

    msg = rec.get("message") or {}
    role = msg.get("role") or rec.get("type")
    session_id = rec.get("sessionId")
    cwd = rec.get("cwd")
    ts = _parse_ts(rec.get("timestamp"))
    content = msg.get("content")
    units: list[Unit] = []
    ordinal = ordinal_start

    def add(kind: str, raw: str):
        nonlocal ordinal
        cleaned = collapse_ws(strip_injected_blocks(raw or ""))
        if len(cleaned) < 3:
            return
        units.append(Unit(role=role, kind=kind, text=cleaned, session_id=session_id,
                          cwd=cwd, ts=ts, ordinal=ordinal))
        ordinal += 1

    if isinstance(content, str):
        add("text", content)
    elif isinstance(content, list):
        for block in content:
            if not isinstance(block, dict):
                continue
            bt = block.get("type")
            if bt == "text":
                add("text", block.get("text", ""))
            elif bt == "thinking":
                add("thinking", block.get("thinking", "") or block.get("text", ""))
            elif bt in ("tool_use", "tool_result"):
                if cfg.index.tool_blobs:
                    add("tool", _tool_summary(block))
    return units, ordinal


def parse_new(path: Path, start_byte: int, cfg: Config) -> tuple[list[Unit], int]:
    """Read only the new tail of a JSONL transcript.
    Returns (units, new_byte_offset). Stops before any half-written final line.
    Detects shrink/rotation (start_byte > size) and reindexes from 0.
    """
    try:
        size = path.stat().st_size
    except OSError:
        return [], start_byte
    if start_byte > size:        # rotated / truncated
        start_byte = 0
    if start_byte == size:
        return [], start_byte

    units: list[Unit] = []
    ordinal = 0
    consumed = start_byte
    with open(path, "rb") as f:
        f.seek(start_byte)
        data = f.read()
    # Split keeping track of complete lines (ending in \n). A trailing fragment is left for next run.
    parts = data.split(b"\n")
    complete = parts[:-1]  # everything before the last (possibly partial) fragment
    for raw in complete:
        line_len = len(raw) + 1  # include the newline
        consumed += line_len
        s = raw.strip()
        if not s:
            continue
        try:
            rec = json.loads(s.decode("utf-8", "replace"))
        except Exception:
            continue
        u, ordinal = _record_units(rec, cfg, ordinal)
        units.extend(u)
    return units, consumed
