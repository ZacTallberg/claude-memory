"""Content-addressed sanitized canonical archive."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from .ids import canonical_json, content_hash


class ArchiveConflict(RuntimeError):
    pass


class CanonicalArchive:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()

    def event_path(self, event_id: str, payload_hash: str) -> Path:
        digest = content_hash(f"{event_id}:{payload_hash}")
        return self.root / "events" / digest[:2] / f"{digest}.json"

    def put_event(self, event_id: str, payload: dict[str, Any]) -> Path:
        encoded = (canonical_json(payload) + "\n").encode("utf-8")
        payload_hash = content_hash(payload["content"])
        target = self.event_path(event_id, payload_hash)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if target.read_bytes() != encoded:
                raise ArchiveConflict(f"archive identity conflict for {event_id}")
            return target

        descriptor, temp_name = tempfile.mkstemp(prefix=".building-", dir=target.parent)
        temp = Path(temp_name)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            try:
                # The hard link publishes a fully flushed inode only if the target is
                # absent. Concurrent identical deliveries verify the winner instead of
                # overwriting it; removing the temporary name leaves the target intact.
                os.link(temp, target)
            except FileExistsError as error:
                if target.read_bytes() != encoded:
                    raise ArchiveConflict(f"archive identity conflict for {event_id}") from error
        finally:
            if temp.exists():
                temp.unlink()
        return target

    def verify_event(self, path: Path, event_id: str, payload_hash: str) -> bool:
        resolved = path.resolve()
        if self.root not in resolved.parents or not resolved.is_file():
            return False
        try:
            payload = json.loads(resolved.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
        return (
            payload.get("event_id") == event_id
            and content_hash(payload.get("content", "")) == payload_hash
        )
