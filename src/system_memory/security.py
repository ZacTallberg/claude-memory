"""Sanitize credential-shaped material without ever returning matched plaintext."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SecretFinding:
    kind: str
    fingerprint: str


def _fingerprint(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8", "replace")).hexdigest()[:20]


_MARKER = re.compile(r"\[REDACTED_SECRET:[a-z0-9-]+\]", re.IGNORECASE)
_WHOLE: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "private-key",
        re.compile(
            r"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----.*?"
            r"-----END (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----",
            re.DOTALL | re.IGNORECASE,
        ),
    ),
    ("anthropic-token", re.compile(r"\bsk-ant-[A-Za-z0-9_-]{20,}\b")),
    ("openai-token", re.compile(r"\bsk-(?:proj-|svcacct-)?[A-Za-z0-9_-]{20,}\b")),
    (
        "github-token",
        re.compile(r"\b(?:github_pat_[A-Za-z0-9_]{20,}|gh[pousr]_[A-Za-z0-9]{20,})\b"),
    ),
    ("aws-access-key", re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b")),
    ("google-api-key", re.compile(r"\bAIza[0-9A-Za-z_-]{30,}\b")),
    ("slack-token", re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{16,}\b")),
)
_PREFIXED: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "bearer-token",
        re.compile(r"(?i)(\bAuthorization\s*:\s*Bearer\s+)([A-Za-z0-9._~+/=-]{12,})"),
    ),
    (
        "url-password",
        re.compile(r"(?i)(\b[a-z][a-z0-9+.-]*://[^\s:/@]+:)([^\s/@]{4,})(@)"),
    ),
    (
        "password-literal",
        re.compile(
            r"(?i)(\b(?:password|passwd|pwd)\b\s*(?:(?:is|was)\s+|[:=]\s*)[\"']?)"
            r"([^\s\"'`,;<>]{4,})([\"']?)"
        ),
    ),
    (
        "assigned-secret",
        re.compile(
            r"(?i)(\b(?:password|passwd|pwd|api[_-]?key|access[_-]?token|auth[_-]?token|"
            r"secret(?:[_-]?key)?|client[_-]?secret)\b\s*[:=]\s*[\"']?)"
            r"([A-Za-z0-9_./+=-]{8,})([\"']?)"
        ),
    ),
)


def redact_secrets(text: str) -> tuple[str, tuple[SecretFinding, ...]]:
    """Return safe text and irreversible findings. Existing markers are idempotent."""
    value = text or ""
    markers: list[str] = []

    def shield(match: re.Match[str]) -> str:
        markers.append(match.group(0))
        return f"\ue000{len(markers) - 1}\ue001"

    value = _MARKER.sub(shield, value)
    findings: list[SecretFinding] = []

    for kind, pattern in _WHOLE:

        def replace_whole(match: re.Match[str], *, label: str = kind) -> str:
            findings.append(SecretFinding(label, _fingerprint(match.group(0))))
            return f"[REDACTED_SECRET:{label}]"

        value = pattern.sub(replace_whole, value)

    for kind, pattern in _PREFIXED:

        def replace_prefixed(match: re.Match[str], *, label: str = kind) -> str:
            findings.append(SecretFinding(label, _fingerprint(match.group(2))))
            suffix = match.group(3) if match.lastindex and match.lastindex >= 3 else ""
            return f"{match.group(1)}[REDACTED_SECRET:{label}]{suffix}"

        value = pattern.sub(replace_prefixed, value)

    for index, marker in enumerate(markers):
        value = value.replace(f"\ue000{index}\ue001", marker)
    return value, tuple(findings)


def sanitize_structure(value: Any) -> tuple[Any, tuple[SecretFinding, ...]]:
    """Recursively redact strings in JSON-compatible provenance and metadata."""
    if isinstance(value, str):
        return redact_secrets(value)
    if isinstance(value, list | tuple):
        safe_items = []
        findings: list[SecretFinding] = []
        for item in value:
            safe, found = sanitize_structure(item)
            safe_items.append(safe)
            findings.extend(found)
        return safe_items, tuple(findings)
    if isinstance(value, dict):
        safe_mapping = {}
        findings = []
        for key, item in value.items():
            safe_key, key_findings = redact_secrets(str(key))
            safe_value, value_findings = sanitize_structure(item)
            safe_mapping[safe_key] = safe_value
            findings.extend(key_findings)
            findings.extend(value_findings)
        return safe_mapping, tuple(findings)
    return value, ()
