"""Provider-neutral normalization for memory-eligible authored content."""

from __future__ import annotations

import re
from dataclasses import dataclass

from .security import SecretFinding, redact_secrets

NORMALIZER_VERSION = "2.0.0"

_DROP_BLOCKS = (
    "recalled-memory",
    "memory-map",
    "in-app-browser-context",
    "environment_context",
    "recommended_plugins",
    "app-context",
    "heartbeat",
)
_BLOCK = re.compile(
    rf"<(?P<tag>{'|'.join(re.escape(item) for item in _DROP_BLOCKS)})\b[^>]*>.*?"
    rf"</(?P=tag)\s*>",
    re.DOTALL | re.IGNORECASE,
)
_TRANSPORT_LINE = re.compile(
    r"^\s*(?:Message Type|Task name|Sender|Payload|Tool call|Tool result)\s*:\s*.*$",
    re.MULTILINE,
)


@dataclass(frozen=True)
class NormalizedText:
    text: str
    secret_findings: tuple[SecretFinding, ...]
    dropped: bool


def strip_injected_blocks(text: str, *, strip_transport: bool = False) -> str:
    """Remove known transport injections without touching authored formatting or secrets."""
    value = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    previous = None
    while value != previous:
        previous = value
        value = _BLOCK.sub("\n", value)
    if strip_transport:
        value = _TRANSPORT_LINE.sub("", value)
    return "\n".join(line.rstrip() for line in value.split("\n")).strip()


def normalize_authored_text(text: str, *, strip_transport: bool = False) -> NormalizedText:
    """Preserve paragraphs/code while removing known injected transport and secrets."""
    value = strip_injected_blocks(text, strip_transport=strip_transport)
    safe, findings = redact_secrets(value)
    return NormalizedText(text=safe, secret_findings=findings, dropped=not bool(safe.strip()))
