"""Codex rollout adapter for the shared transcript index.

Codex documents that rollout JSONL is not a stable public hook interface, so this parser is
deliberately narrow and fail-safe: it accepts only the canonical user/assistant message records,
ignores tools, reasoning, developer/system context, and unknown shapes, and preserves the same
partial-line and injected-context protections as the Claude adapter.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from .config import Config
from .text import collapse_ws, strip_injected_blocks
from .transcripts import Unit, _parse_ts


def _unwrap_user_payload(text: str, outer: str, inner: str) -> str:
    """Replace a Codex transport wrapper with only its user-authored payload."""
    outer_re = re.compile(fr"<{outer}\b.*?</{outer}>", re.DOTALL | re.IGNORECASE)
    inner_re = re.compile(fr"<{inner}\b[^>]*>(.*?)</{inner}>", re.DOTALL | re.IGNORECASE)

    def repl(match: re.Match) -> str:
        nested = inner_re.search(match.group(0))
        return nested.group(1) if nested else " "

    return outer_re.sub(repl, text)


def clean_codex_text(text: str) -> str:
    """Keep authored/delegated task text while removing Codex transport and ambient UI state."""
    out = text or ""
    # A delegated main-worker task is valuable project memory; the routing/thread wrapper is not.
    out = _unwrap_user_payload(out, "codex_delegation", "input")
    # Goal continuations repeat a user-provided objective inside generic system prose. Retain the
    # objective, not the instruction boilerplate that Codex adds around it.
    out = _unwrap_user_payload(out, "codex_internal_context", "objective")
    return collapse_ws(strip_injected_blocks(out))


def _meta(path: Path) -> dict:
    """Read the first complete session_meta record without scanning a large rollout."""
    try:
        with path.open("rb") as f:
            raw = f.readline()
        rec = json.loads(raw.decode("utf-8-sig", "replace"))
        if rec.get("type") == "session_meta" and isinstance(rec.get("payload"), dict):
            return rec["payload"]
    except Exception:
        pass
    return {}


def parse_new(path: Path, start_byte: int, cfg: Config) -> tuple[list[Unit], int]:
    try:
        size = path.stat().st_size
    except OSError:
        return [], start_byte
    if start_byte > size:
        start_byte = 0
    if start_byte == size:
        return [], start_byte

    meta = _meta(path)
    if cfg.index.exclude_sidechains and (meta.get("agent_path") or meta.get("parent_thread_id")):
        return [], size
    session_id = meta.get("session_id") or meta.get("id") or path.stem
    cwd = meta.get("cwd")

    with path.open("rb") as f:
        f.seek(start_byte)
        data = f.read()
    parts = data.split(b"\n")
    complete = parts[:-1]
    consumed = start_byte
    units: list[Unit] = []
    ordinal = 0

    for raw in complete:
        consumed += len(raw) + 1
        try:
            rec = json.loads(raw.decode("utf-8", "replace"))
        except Exception:
            continue
        if rec.get("type") != "response_item":
            continue
        payload = rec.get("payload")
        if not isinstance(payload, dict) or payload.get("type") != "message":
            continue
        role = payload.get("role")
        if role not in ("user", "assistant"):
            continue
        content = payload.get("content")
        if not isinstance(content, list):
            continue
        texts: list[str] = []
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") in ("input_text", "output_text", "text"):
                texts.append(str(block.get("text") or ""))
        cleaned = clean_codex_text("\n".join(texts))
        if len(cleaned) < 3:
            continue
        units.append(Unit(role=role, kind="text", text=cleaned, session_id=session_id,
                          cwd=cwd, ts=_parse_ts(rec.get("timestamp")), ordinal=ordinal))
        ordinal += 1
    return units, consumed
