"""Stable public identifiers.

IDs are derived from canonical evidence identity, never database row numbers. The
prefix keeps accidental cross-entity comparisons visible while the full SHA-256
digest makes collisions operationally irrelevant.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", "replace")).hexdigest()


def stable_id(prefix: str, *parts: Any) -> str:
    """Create a deterministic identifier from typed canonical parts."""
    encoded: list[Any] = []
    for part in parts:
        if isinstance(part, Mapping):
            encoded.append({str(key): part[key] for key in sorted(part, key=str)})
        elif isinstance(part, Sequence) and not isinstance(part, str | bytes | bytearray):
            encoded.append(list(part))
        else:
            encoded.append(part)
    digest = hashlib.sha256(canonical_json(encoded).encode("utf-8")).hexdigest()
    return f"{prefix}_{digest}"


def source_id(kind: str, provider: str, locator: str) -> str:
    return stable_id("src", kind, provider, locator)


def episode_id(provider: str, agent_id: str, session_id: str, sequence: int = 0) -> str:
    return stable_id("ep", provider, agent_id, session_id, sequence)


def event_id(
    provider: str,
    source: str,
    provider_event_id: str | None,
    role: str,
    kind: str,
    occurred_at: str,
    body_hash: str,
) -> str:
    identity = provider_event_id or body_hash
    return stable_id("evt", provider, source, identity, role, kind, occurred_at)


def document_id(memory_type: str, ref_id: str, rendering_hash: str) -> str:
    return stable_id("doc", memory_type, ref_id, rendering_hash)
