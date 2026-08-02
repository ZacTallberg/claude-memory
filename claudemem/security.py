"""Deterministic secret detection and redaction for every memory ingress/egress path.

The scanner never logs or returns matched plaintext. Findings contain only a type and a
short one-way fingerprint so duplicate exposure can be counted without reproducing it.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass


@dataclass(frozen=True)
class SecretFinding:
    kind: str
    fingerprint: str


def _fingerprint(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8", "replace")).hexdigest()[:16]


_WHOLE_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("private-key", re.compile(
        r"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----.*?"
        r"-----END (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----",
        re.DOTALL | re.IGNORECASE)),
    ("anthropic-token", re.compile(r"\bsk-ant-[A-Za-z0-9_-]{20,}\b")),
    ("openai-token", re.compile(r"\bsk-(?:proj-|svcacct-)?[A-Za-z0-9_-]{20,}\b")),
    ("github-token", re.compile(
        r"\b(?:github_pat_[A-Za-z0-9_]{20,}|gh[pousr]_[A-Za-z0-9]{20,})\b")),
    ("aws-access-key", re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b")),
    ("google-api-key", re.compile(r"\bAIza[0-9A-Za-z_-]{30,}\b")),
    ("slack-token", re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{16,}\b")),
)

_PREFIX_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("bearer-token", re.compile(
        r"(?i)(\bAuthorization\s*:\s*Bearer\s+)([A-Za-z0-9._~+/=-]{12,})")),
    ("url-password", re.compile(
        r"(?i)(\b[a-z][a-z0-9+.-]*://[^\s:/@]+:)([^\s/@]{4,})(@)")),
    ("password-literal", re.compile(
        r"(?i)(\b(?:password|passwd|pwd)\b\s*(?:(?:is|was)\s+|[:=]\s*)[\"']?)"
        r"([^\s\"'`,;<>]{4,})([\"']?)")),
    ("assigned-secret", re.compile(
        r"(?i)(\b(?:password|passwd|pwd|api[_-]?key|access[_-]?token|auth[_-]?token|"
        r"secret(?:[_-]?key)?|client[_-]?secret)\b\s*[:=]\s*[\"']?)"
        r"([A-Za-z0-9_./+=-]{8,})([\"']?)")),
)

_TRIGGERS = (
    "sk-", "github_pat_", "ghp_", "gho_", "ghu_", "ghs_", "ghr_",
    "akia", "asia", "aiza", "xox", "private key-----", "authorization",
    "://", "password", "passwd", "pwd", "api_key", "api-key", "apikey",
    "access_token", "access-token", "auth_token", "auth-token", "client_secret",
    "client-secret", "secret_key", "secret-key",
)
_REDACTION_MARKER = re.compile(r"\[REDACTED_SECRET:[a-z0-9-]+\]", re.IGNORECASE)


def redact_secrets(text: str) -> tuple[str, list[SecretFinding]]:
    """Return redacted text and non-reversible findings; never expose matched values."""
    out = text or ""
    markers: list[str] = []

    def shield(match: re.Match[str]) -> str:
        markers.append(match.group(0))
        return f"\ue000{len(markers) - 1}\ue001"

    out = _REDACTION_MARKER.sub(shield, out)

    def restore(value: str) -> str:
        for index, marker in enumerate(markers):
            value = value.replace(f"\ue000{index}\ue001", marker)
        return value

    lower = out.lower()
    if not any(trigger in lower for trigger in _TRIGGERS):
        return restore(out), []
    findings: list[SecretFinding] = []

    for kind, pattern in _WHOLE_PATTERNS:
        def replace_whole(match: re.Match[str], *, _kind: str = kind) -> str:
            findings.append(SecretFinding(_kind, _fingerprint(match.group(0))))
            return f"[REDACTED_SECRET:{_kind}]"
        out = pattern.sub(replace_whole, out)

    for kind, pattern in _PREFIX_PATTERNS:
        def replace_prefixed(match: re.Match[str], *, _kind: str = kind) -> str:
            secret = match.group(2)
            findings.append(SecretFinding(_kind, _fingerprint(secret)))
            if _kind == "url-password":
                return match.group(1) + f"[REDACTED_SECRET:{_kind}]" + match.group(3)
            suffix = match.group(3) if match.lastindex and match.lastindex >= 3 else ""
            return match.group(1) + f"[REDACTED_SECRET:{_kind}]" + suffix
        out = pattern.sub(replace_prefixed, out)

    return restore(out), findings


def contains_secret(text: str) -> bool:
    return bool(redact_secrets(text)[1])
