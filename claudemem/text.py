"""Text utilities: corpus cleaning, term extraction, frontmatter parsing, snippets, age."""
from __future__ import annotations

import re
from datetime import datetime, timezone

import yaml

from .security import redact_secrets

# Blocks WE (or the harness) inject — must be stripped before storing transcript text,
# else recall recursively indexes its own past output ("eats its own tail").
_INJECTED_BLOCKS = [
    (r"<recalled-memory\b.*?</recalled-memory>", re.DOTALL | re.IGNORECASE),
    (r"<curated-notes\b.*?</curated-notes>", re.DOTALL | re.IGNORECASE),
    (r"<memory-map\b.*?</memory-map>", re.DOTALL | re.IGNORECASE),
    (r"<system-reminder\b.*?</system-reminder>", re.DOTALL | re.IGNORECASE),
    (r"<local-command-[a-z\-]+>.*?</local-command-[a-z\-]+>", re.DOTALL | re.IGNORECASE),
    (r"<command-(name|message|args)>.*?</command-(name|message|args)>", re.DOTALL | re.IGNORECASE),
    (r"<task-notification\b.*?</task-notification>", re.DOTALL | re.IGNORECASE),
    (r"<local-command-caveat\b.*?</local-command-caveat>", re.DOTALL | re.IGNORECASE),
    # Codex desktop supplies these as ambient/system context inside user-shaped rollout
    # messages. They are not authored conversation and must never become semantic memory.
    (r"<in-app-browser-context\b.*?</in-app-browser-context>", re.DOTALL | re.IGNORECASE),
    (r"<environment_context\b.*?</environment_context>", re.DOTALL | re.IGNORECASE),
    (r"<recommended_plugins\b.*?</recommended_plugins>", re.DOTALL | re.IGNORECASE),
    (r"<app-context\b.*?</app-context>", re.DOTALL | re.IGNORECASE),
]

_STOPWORDS = set("""
a an the and or but if then else for to of in on at by with from into over under again further
is are was were be been being do does did doing have has had having i you he she it we they me him
her us them my your his its our their this that these those as so not no yes can will just dont don't
should would could what which who whom how when where why all any both each few more most other some
such only own same than too very s t can't cannot about up down out off
""".split())

_WORD = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_\-/.]*")
_CONTINUATION = re.compile(
    r"\b(continue|resume|proceed|carry\s+on|keep\s+going|pick\s+up|what(?:'s|\s+is)\s+next|"
    r"finish\s+(?:it|this|the\s+work)|where\s+were\s+we)\b",
    re.IGNORECASE,
)


def strip_injected_blocks(text: str) -> str:
    if not text:
        return text
    out = text
    for pat, flags in _INJECTED_BLOCKS:
        out = re.sub(pat, " ", out, flags=flags)
    return redact_secrets(out)[0]


def extract_terms(text: str) -> list[str]:
    """Meaningful, lowercased terms (stopwords + 1-char tokens removed)."""
    if not text:
        return []
    terms = []
    for m in _WORD.findall(text):
        w = m.lower()
        if len(w) < 2 or w in _STOPWORDS:
            continue
        terms.append(w)
    return terms


def meaningful_term_count(text: str) -> int:
    return len(set(extract_terms(text)))


def is_continuation_prompt(text: str) -> bool:
    """True for a short cue whose meaning depends on durable prior-work context."""
    return bool(_CONTINUATION.search(text or ""))


def should_recall(text: str, min_terms: int) -> bool:
    """Gate noise while never dropping a genuine continuation/resumption cue."""
    return meaningful_term_count(text) >= min_terms or is_continuation_prompt(text)


def recall_query(text: str, cwd: str | None = None) -> str:
    """Expand context-dependent continuation cues into a useful semantic retrieval query."""
    if not is_continuation_prompt(text):
        return text
    project = ""
    if cwd:
        try:
            from pathlib import Path
            project = Path(cwd).name
        except Exception:
            project = ""
    suffix = "prior work current task pending decisions unfinished steps next actions"
    if project:
        suffix += f" project {project}"
    return f"{text} {suffix}".strip()


def collapse_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def snippet(text: str, max_chars: int) -> str:
    s = collapse_ws(strip_injected_blocks(text or ""))
    if len(s) <= max_chars:
        return s
    return s[: max_chars - 1].rstrip() + "…"


def parse_frontmatter(md_text: str) -> tuple[dict, str]:
    """Split YAML frontmatter from a markdown note. Returns (meta_dict, body)."""
    if md_text.startswith("﻿"):
        md_text = md_text.lstrip("﻿")
    m = re.match(r"^---\s*\n(.*?)\n---\s*\n?(.*)$", md_text, re.DOTALL)
    if not m:
        return {}, md_text
    try:
        meta = yaml.safe_load(m.group(1)) or {}
        if not isinstance(meta, dict):
            meta = {}
    except Exception:
        meta = {}
    return meta, m.group(2)


def normalize_note_type(meta: dict) -> str:
    """Resolve the note type from either top-level `type` or nested `metadata.type`."""
    t = meta.get("type")
    if not t and isinstance(meta.get("metadata"), dict):
        t = meta["metadata"].get("type")
    t = (t or "reference").strip().lower()
    return t if t in ("user", "feedback", "project", "reference") else "reference"


def note_origin_session(meta: dict) -> str | None:
    if meta.get("originSessionId"):
        return str(meta["originSessionId"])
    md = meta.get("metadata")
    if isinstance(md, dict) and md.get("originSessionId"):
        return str(md["originSessionId"])
    return None


_WIKILINK = re.compile(r"\[\[([^\]]+)\]\]")


def extract_wikilinks(body: str) -> list[str]:
    return [collapse_ws(x) for x in _WIKILINK.findall(body or "")]


def human_age(ts: datetime | None) -> str:
    if ts is None:
        return "unknown date"
    now = datetime.now(timezone.utc)
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    days = (now - ts).total_seconds() / 86400.0
    if days < 1:
        hrs = max(1, int(days * 24))
        return f"{hrs}h ago"
    if days < 30:
        return f"{int(days)}d ago"
    if days < 365:
        return f"{int(days / 30)}mo ago"
    return f"{days / 365:.1f}y ago"


def budget_join(parts: list[str], max_chars: int, sep: str = "\n") -> str:
    """Join parts in order, stopping before exceeding max_chars."""
    out: list[str] = []
    total = 0
    for p in parts:
        add = len(p) + (len(sep) if out else 0)
        if total + add > max_chars:
            break
        out.append(p)
        total += add
    return sep.join(out)
