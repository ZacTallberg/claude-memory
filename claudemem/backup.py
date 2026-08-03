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
from .paths import canonical_memory_root, iter_memory_dirs, safe_under
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
    declared = set(manifest.get("sha256") or {})
    required = {"claudemem.db", "curated-notes.zip"}
    if declared != required:
        raise RuntimeError(
            f"backup manifest payload set differs: expected={sorted(required)} "
            f"declared={sorted(declared)}")
    actual_files = {path.name for path in snapshot_dir.iterdir() if path.is_file()}
    expected_files = declared | {"manifest.json"}
    # A read-only SQLite viewer may create coordination sidecars beside an otherwise immutable
    # snapshot. They contain no backup payload: accept only these exact names, and only while WAL
    # is empty. Every declared payload is still hash-verified below; unknown extras remain fatal.
    allowed_sidecars = {"claudemem.db-shm", "claudemem.db-wal"}
    unknown_files = actual_files - expected_files - allowed_sidecars
    missing_files = expected_files - actual_files
    wal_path = snapshot_dir / "claudemem.db-wal"
    if wal_path.exists() and wal_path.stat().st_size:
        raise RuntimeError("backup snapshot has a non-empty SQLite WAL")
    if unknown_files or missing_files:
        raise RuntimeError(
            f"backup directory payload set differs: missing={sorted(missing_files)} "
            f"unknown={sorted(unknown_files)}")
    for name, expected in manifest["sha256"].items():
        path = snapshot_dir / name
        if not path.is_file() or path.stat().st_size <= 0 or _sha256(path) != expected:
            raise RuntimeError(f"backup checksum mismatch: {name}")
    db_path = snapshot_dir / "claudemem.db"
    conn = sqlite3.connect(f"file:{db_path.as_posix()}?mode=ro&immutable=1", uri=True)
    try:
        result = conn.execute("PRAGMA quick_check").fetchone()[0]
        if result != "ok":
            raise RuntimeError(f"backup SQLite quick_check failed: {result}")
        counts = {
            "sources": conn.execute("SELECT count(*) FROM sources").fetchone()[0],
            "chunks": conn.execute("SELECT count(*) FROM chunks").fetchone()[0],
            "facts": conn.execute("SELECT count(*) FROM facts").fetchone()[0],
        }
        secret_counts: Counter[str] = Counter()
        for table_row in conn.execute(
                "SELECT name,sql FROM sqlite_master "
                "WHERE type='table' AND name NOT LIKE 'sqlite_%'"):
            table, schema_sql = table_row
            if "using vec0" in (schema_sql or "").casefold():
                continue
            try:
                columns = [(row[1], (row[2] or "").upper())
                           for row in conn.execute(f'PRAGMA table_info("{table}")')]
            except sqlite3.DatabaseError:
                continue
            text_columns = [name for name, kind in columns
                            if kind in ("TEXT", "") or "CHAR" in kind or "CLOB" in kind]
            if not text_columns:
                continue
            selected = ",".join('"' + name.replace('"', '""') + '"'
                                for name in text_columns)
            try:
                rows = conn.execute(f'SELECT {selected} FROM "{table}"')
                for row in rows:
                    for value in row:
                        if isinstance(value, str):
                            _safe, findings = redact_secrets(value)
                            secret_counts.update(finding.kind for finding in findings)
            except sqlite3.DatabaseError:
                # FTS/vector shadow tables may not be queryable without their extensions;
                # canonical text tables were scanned independently above.
                continue
    finally:
        conn.close()
    with zipfile.ZipFile(snapshot_dir / "curated-notes.zip") as archive:
        for member in archive.namelist():
            if member.endswith("/"):
                continue
            text = archive.read(member).decode("utf-8", errors="replace")
            _safe, findings = redact_secrets(text)
            secret_counts.update(finding.kind for finding in findings)
    if secret_counts:
        raise RuntimeError(
            "backup secret scan failed: "
            + json.dumps(dict(sorted(secret_counts.items())), sort_keys=True))
    return {"ok": True, "snapshot": str(snapshot_dir), "counts": counts,
            "notes": manifest.get("notes", 0), "created_at": manifest.get("created_at"),
            "secret_scan": "clean",
            "benign_sidecars": sorted(actual_files & allowed_sidecars)}


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
                family = ("canonical" if safe_under(memory_dir.path, [canonical_memory_root(cfg)])
                          else "legacy-claude")
                for note in sorted(memory_dir.path.glob("*.md")):
                    raw = note.read_text(encoding="utf-8", errors="replace")
                    sanitized, findings = redact_secrets(raw)
                    note_redactions.update(finding.kind for finding in findings)
                    archive.writestr(
                        f"{family}/{memory_dir.encoded_dir}/memory/{note.name}", sanitized)
                    note_count += 1

        manifest = {
            "format": 2,
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
