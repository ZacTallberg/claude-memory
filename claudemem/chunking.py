"""Chunk transcript text on paragraph / code-block boundaries with light overlap.
Curated notes are indexed whole (one chunk) — they're already small and curated.
"""
from __future__ import annotations

import re

_FENCE = re.compile(r"```.*?```", re.DOTALL)


def estimate_tokens(s: str) -> int:
    return max(1, len(s) // 4)


def _segments(text: str) -> list[str]:
    """Split into segments, keeping fenced code blocks intact, then paragraphs."""
    segs: list[str] = []
    last = 0
    for m in _FENCE.finditer(text):
        pre = text[last:m.start()]
        if pre.strip():
            segs.extend(p for p in re.split(r"\n\s*\n", pre) if p.strip())
        segs.append(m.group(0))
        last = m.end()
    tail = text[last:]
    if tail.strip():
        segs.extend(p for p in re.split(r"\n\s*\n", tail) if p.strip())
    return segs


def chunk_text(content: str, *, max_tokens: int = 400, overlap_frac: float = 0.15) -> list[str]:
    content = (content or "").strip()
    if not content:
        return []
    max_chars = max_tokens * 4
    if len(content) <= max_chars:
        return [content]

    overlap_chars = int(max_chars * overlap_frac)
    segs = _segments(content)
    chunks: list[str] = []
    buf = ""
    for seg in segs:
        # Hard-split any single oversized segment (e.g. a huge code block).
        while len(seg) > max_chars:
            if buf:
                chunks.append(buf)
                buf = ""
            chunks.append(seg[:max_chars])
            seg = seg[max_chars - overlap_chars:]
        if not buf:
            buf = seg
        elif len(buf) + 2 + len(seg) <= max_chars:
            buf = buf + "\n\n" + seg
        else:
            chunks.append(buf)
            # start next buffer with a tail overlap of the previous one
            tail = buf[-overlap_chars:] if overlap_chars else ""
            buf = (tail + "\n\n" + seg) if tail else seg
    if buf.strip():
        chunks.append(buf)
    return chunks
