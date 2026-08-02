"""Verified, rotating backups for the irreplaceable local-memory state."""
from __future__ import annotations

import hashlib
import json
import shutil
import sqlite3
import tempfile
import zipfile
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .config import Config
from .paths import iter_memory_dirs, safe_under
from .security import redact_secrets


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_snapshot(snapshot_dir: Path) -> dict:
    snapshot_dir = snapshot_dir.resolve()
    manifest_path = snapshot_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for name, expected in manifest["sha256"].items():
        path = snapshot_dir / name
        if not path.is_file() or _sha256(path) != expected:
            raise RuntimeError(f"backup checksum mismatch: {name}")
    db_path = snapshot_dir / "claudemem.db"
    conn = sqlite3.connect(str(db_path))
    try:
        result = conn.execute("PRAGMA quick_check").fetchone()[0]
        if result != "ok":
            raise RuntimeError(f"backup SQLite quick_check failed: {result}")
        counts = {
            "sources": conn.execute("SELECT count(*) FROM sources").fetchone()[0],
            "chunks": conn.execute("SELECT count(*) FROM chunks").fetchone()[0],
            "facts": conn.execute("SELECT count(*) FROM facts").fetchone()[0],
        }
    finally:
        conn.close()
    return {"ok": True, "snapshot": str(snapshot_dir), "counts": counts,
            "notes": manifest.get("notes", 0), "created_at": manifest.get("created_at")}


def create_backup(cfg: Config, *, if_due: bool = False, retention: int = 14,
                  due_hours: float = 20.0) -> dict:
    if cfg.store.backend != "sqlite":
        raise RuntimeError("verified local backup currently requires the pinned sqlite backend")
    source = Path(cfg.store.sqlite.path)
    if not source.is_absolute():
        source = cfg.root / source
    source = source.resolve()
    if not source.is_file():
        raise FileNotFoundError(source)

    backup_root = (cfg.data_dir / "backups").resolve()
    backup_root.mkdir(parents=True, exist_ok=True)
    # A machine shutdown or externally killed process can leave an incomplete staging
    # directory. It is never a usable snapshot (there is no final manifest), so reap only
    # old, exactly-scoped staging directories before starting another atomic backup.
    stale_before = datetime.now(timezone.utc) - timedelta(hours=1)
    for candidate in backup_root.glob(".building-*"):
        resolved = candidate.resolve()
        if (not candidate.is_dir() or resolved.parent != backup_root
                or not safe_under(resolved, [backup_root])):
            continue
        modified = datetime.fromtimestamp(candidate.stat().st_mtime, tz=timezone.utc)
        if modified < stale_before:
            shutil.rmtree(resolved)
    existing = sorted((p for p in backup_root.iterdir()
                       if p.is_dir() and (p / "manifest.json").is_file()), reverse=True)
    if if_due and existing:
        age_hours = ((datetime.now(timezone.utc).timestamp()
                      - (existing[0] / "manifest.json").stat().st_mtime) / 3600.0)
        if age_hours < due_hours:
            result = verify_snapshot(existing[0])
            result.update(created=False, reason=f"latest backup is {age_hours:.1f}h old")
            return result

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    temp_dir = Path(tempfile.mkdtemp(prefix=".building-", dir=str(backup_root))).resolve()
    final_dir = (backup_root / stamp).resolve()
    if not safe_under(temp_dir, [backup_root]) or not safe_under(final_dir, [backup_root]):
        raise RuntimeError("backup path escaped backup root")
    try:
        db_copy = temp_dir / "claudemem.db"
        src = sqlite3.connect(str(source), timeout=30)
        dst = sqlite3.connect(str(db_copy))
        try:
            src.backup(dst)
            dst.commit()
        finally:
            dst.close()
            src.close()

        notes_zip = temp_dir / "curated-notes.zip"
        note_count = 0
        note_redactions: Counter[str] = Counter()
        with zipfile.ZipFile(notes_zip, "w", compression=zipfile.ZIP_DEFLATED,
                             compresslevel=6) as archive:
            for memory_dir in iter_memory_dirs(cfg):
                for note in sorted(memory_dir.path.glob("*.md")):
                    raw = note.read_text(encoding="utf-8", errors="replace")
                    sanitized, findings = redact_secrets(raw)
                    note_redactions.update(finding.kind for finding in findings)
                    archive.writestr(f"{memory_dir.encoded_dir}/memory/{note.name}", sanitized)
                    note_count += 1

        manifest = {
            "format": 1,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source_db": str(source),
            "embedding_model": cfg.embeddings.model,
            "embedding_dim": cfg.embeddings.dim,
            "notes": note_count,
            "notes_sanitized": True,
            "secret_redactions": dict(sorted(note_redactions.items())),
            "sha256": {"claudemem.db": _sha256(db_copy),
                       "curated-notes.zip": _sha256(notes_zip)},
        }
        (temp_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        verify_snapshot(temp_dir)
        temp_dir.rename(final_dir)
    except Exception:
        if temp_dir.exists() and safe_under(temp_dir, [backup_root]):
            shutil.rmtree(temp_dir)
        raise

    snapshots = sorted((p for p in backup_root.iterdir()
                        if p.is_dir() and (p / "manifest.json").is_file()), reverse=True)
    for old in snapshots[max(1, retention):]:
        resolved = old.resolve()
        if safe_under(resolved, [backup_root]) and resolved.parent == backup_root:
            shutil.rmtree(resolved)
    result = verify_snapshot(final_dir)
    result.update(created=True, retained=min(len(snapshots), max(1, retention)))
    return result
