from __future__ import annotations

import hashlib
import json
import os
import secrets
import shutil
import sqlite3
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from .archive import CanonicalArchive
from .security import redact_secrets
from .store import MemoryStore


class BackupError(RuntimeError):
    pass


class FileReceipt(BaseModel):
    model_config = ConfigDict(extra="forbid")

    size: int = Field(ge=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class BackupManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: int = 1
    snapshot_id: str
    created_at: datetime
    database_file: str = "data/system-memory.db"
    archive_root: str = "data/archive"
    event_count: int
    source_count: int
    archive_file_count: int
    database_integrity: str
    secret_findings: dict[str, int]
    code_revision: str | None = None
    dependency_lock_sha256: str | None = None
    files: dict[str, FileReceipt]


class BackupManager:
    def __init__(self, store: MemoryStore, *, repository_root: Path | None = None) -> None:
        self.store = store
        self.repository_root = repository_root.resolve() if repository_root else None

    def create(self, destination_root: Path) -> Path:
        destination = destination_root.resolve()
        if destination == self.store.archive.root or self.store.archive.root in destination.parents:
            raise BackupError("backup destination cannot be inside the canonical archive")
        destination.mkdir(parents=True, exist_ok=True)
        self.store.backfill_archive_references()
        snapshot_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ") + f"-{secrets.token_hex(4)}"
        staging = destination / f".building-{snapshot_id}"
        final = destination / snapshot_id
        if staging.exists() or final.exists():
            raise BackupError("backup snapshot identity collision")
        staging.mkdir()
        try:
            database_target = staging / "data" / "system-memory.db"
            database_target.parent.mkdir(parents=True)
            self._backup_database(database_target)
            counts = self._copy_referenced_archives(database_target, staging / "data" / "archive")
            receipts = staging / "receipts"
            receipts.mkdir()
            self._write_source_inventory(database_target, receipts / "source-inventory.json")
            code_revision = repository_revision(self.repository_root)
            lock_sha256 = self._copy_lockfile(receipts)
            runtime_receipt = {
                "schema_version": counts["schema_version"],
                "code_revision": code_revision,
                "dependency_lock_sha256": lock_sha256,
                "created_at": datetime.now(UTC).isoformat(),
            }
            (receipts / "runtime.json").write_text(
                json.dumps(runtime_receipt, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            integrity = self._database_integrity(database_target)
            findings = self._scan_snapshot_secrets(database_target, staging / "data" / "archive")
            if findings:
                raise BackupError(
                    "backup failed closed because sanitized canonical storage still "
                    f"contains credential-shaped material: {dict(sorted(findings.items()))}"
                )
            manifest = BackupManifest(
                snapshot_id=snapshot_id,
                created_at=datetime.now(UTC),
                event_count=counts["events"],
                source_count=counts["sources"],
                archive_file_count=counts["archives"],
                database_integrity=integrity,
                secret_findings={},
                code_revision=code_revision,
                dependency_lock_sha256=lock_sha256,
                files=self._file_receipts(staging),
            )
            (staging / "manifest.json").write_text(
                manifest.model_dump_json(indent=2) + "\n", encoding="utf-8"
            )
            self.verify_snapshot(staging)
            os.replace(staging, final)
            return final
        except Exception:
            if staging.exists() and staging.parent == destination:
                shutil.rmtree(staging)
            raise

    def _backup_database(self, destination: Path) -> None:
        source = self.store.database.connect(read_only=True)
        target = sqlite3.connect(destination)
        try:
            source.backup(target, pages=1_024, sleep=0.01)
            target.commit()
        finally:
            target.close()
            source.close()

    def _copy_referenced_archives(self, database: Path, destination: Path) -> dict[str, int]:
        connection = sqlite3.connect(f"file:{database.as_posix()}?mode=ro&immutable=1", uri=True)
        connection.row_factory = sqlite3.Row
        try:
            rows = connection.execute(
                "SELECT id,content_sha256,archive_ref FROM memory_events ORDER BY id"
            ).fetchall()
            source_count = int(connection.execute("SELECT COUNT(*) FROM sources").fetchone()[0])
            schema_version = int(connection.execute("PRAGMA user_version").fetchone()[0])
        finally:
            connection.close()
        copied: set[str] = set()
        for row in rows:
            reference = row["archive_ref"]
            if not reference:
                raise BackupError(f"event is missing portable archive reference: {row['id']}")
            source = self.store.archive.resolve_reference(reference)
            if not self.store.archive.verify_event(source, row["id"], row["content_sha256"]):
                raise BackupError(f"event archive failed verification: {row['id']}")
            relative = Path(reference)
            if relative.as_posix() in copied:
                continue
            target = (destination / relative).resolve()
            if destination.resolve() not in target.parents:
                raise BackupError("archive reference escapes backup destination")
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            copied.add(relative.as_posix())
        return {
            "events": len(rows),
            "sources": source_count,
            "archives": len(copied),
            "schema_version": schema_version,
        }

    @staticmethod
    def _write_source_inventory(database: Path, destination: Path) -> None:
        connection = sqlite3.connect(f"file:{database.as_posix()}?mode=ro&immutable=1", uri=True)
        connection.row_factory = sqlite3.Row
        try:
            rows = connection.execute(
                """SELECT id,kind,provider,locator_hash,content_hash,cursor,loss_flags,
                          first_seen_at,last_seen_at FROM sources ORDER BY id"""
            ).fetchall()
        finally:
            connection.close()
        inventory = [dict(row) for row in rows]
        destination.write_text(
            json.dumps(inventory, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def _copy_lockfile(self, receipts: Path) -> str | None:
        if not self.repository_root:
            return None
        source = self.repository_root / "uv.lock"
        if not source.is_file():
            return None
        destination = receipts / "uv.lock"
        shutil.copy2(source, destination)
        return _sha256(destination)

    @staticmethod
    def _file_receipts(root: Path) -> dict[str, FileReceipt]:
        return {
            path.relative_to(root).as_posix(): FileReceipt(
                size=path.stat().st_size,
                sha256=_sha256(path),
            )
            for path in sorted(root.rglob("*"))
            if path.is_file() and path.name != "manifest.json"
        }

    @staticmethod
    def _database_integrity(database: Path) -> str:
        connection = sqlite3.connect(f"file:{database.as_posix()}?mode=ro&immutable=1", uri=True)
        try:
            rows = connection.execute("PRAGMA integrity_check").fetchall()
            foreign = connection.execute("PRAGMA foreign_key_check").fetchall()
        finally:
            connection.close()
        if rows != [("ok",)] or foreign:
            raise BackupError("snapshot database failed integrity or foreign-key validation")
        return "ok"

    @classmethod
    def _scan_snapshot_secrets(cls, database: Path, archive: Path) -> Counter[str]:
        findings: Counter[str] = Counter()
        connection = sqlite3.connect(f"file:{database.as_posix()}?mode=ro&immutable=1", uri=True)
        try:
            tables = [
                row[0]
                for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
                )
            ]
            for table in tables:
                columns = [
                    row[1]
                    for row in connection.execute(f'PRAGMA table_info("{table}")')
                    if "TEXT" in str(row[2]).upper()
                ]
                for column in columns:
                    query = f'SELECT "{column}" FROM "{table}" WHERE "{column}" IS NOT NULL'
                    for (value,) in connection.execute(query):
                        if not isinstance(value, str):
                            continue
                        findings.update(item.kind for item in redact_secrets(value)[1])
        finally:
            connection.close()
        for path in archive.rglob("*.json"):
            findings.update(
                item.kind for item in redact_secrets(path.read_text(encoding="utf-8"))[1]
            )
        return findings

    @classmethod
    def verify_snapshot(cls, snapshot_path: Path) -> BackupManifest:
        snapshot = snapshot_path.resolve()
        manifest_path = snapshot / "manifest.json"
        if not manifest_path.is_file():
            raise BackupError("backup manifest is missing")
        try:
            manifest = BackupManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
        except Exception as error:
            raise BackupError("backup manifest is invalid") from error
        actual_files = {
            path.relative_to(snapshot).as_posix()
            for path in snapshot.rglob("*")
            if path.is_file() and path.name != "manifest.json"
        }
        if actual_files != set(manifest.files):
            raise BackupError("backup payload file set differs from its manifest")
        for relative, receipt in manifest.files.items():
            path = (snapshot / relative).resolve()
            if snapshot not in path.parents or not path.is_file():
                raise BackupError("manifest path escapes or is missing from the snapshot")
            if path.stat().st_size != receipt.size or _sha256(path) != receipt.sha256:
                raise BackupError(f"backup payload hash mismatch: {relative}")
        database = snapshot / manifest.database_file
        if cls._database_integrity(database) != manifest.database_integrity:
            raise BackupError("database integrity receipt differs from verification")
        cls._verify_archive_set(snapshot, manifest, database)
        findings = cls._scan_snapshot_secrets(database, snapshot / manifest.archive_root)
        if findings or manifest.secret_findings:
            raise BackupError("backup secret scan is not clean")
        return manifest

    @classmethod
    def _verify_archive_set(cls, snapshot: Path, manifest: BackupManifest, database: Path) -> None:
        archive_root = (snapshot / manifest.archive_root).resolve()
        archive = CanonicalArchive(archive_root)
        connection = sqlite3.connect(f"file:{database.as_posix()}?mode=ro&immutable=1", uri=True)
        connection.row_factory = sqlite3.Row
        try:
            events = connection.execute(
                "SELECT id,content_sha256,archive_ref FROM memory_events ORDER BY id"
            ).fetchall()
            sources = int(connection.execute("SELECT COUNT(*) FROM sources").fetchone()[0])
        finally:
            connection.close()
        if len(events) != manifest.event_count or sources != manifest.source_count:
            raise BackupError("backup canonical counts differ from its manifest")
        references: set[str] = set()
        for event in events:
            reference = event["archive_ref"]
            if not reference or not archive.verify_event(
                reference, event["id"], event["content_sha256"]
            ):
                raise BackupError(f"backup archive verification failed: {event['id']}")
            references.add(Path(reference).as_posix())
        actual = {
            path.relative_to(archive_root).as_posix() for path in archive_root.rglob("*.json")
        }
        if references != actual or len(actual) != manifest.archive_file_count:
            raise BackupError("backup archive set has missing or unreferenced payloads")

    @classmethod
    def restore(cls, snapshot_path: Path, target_root: Path) -> Path:
        snapshot = snapshot_path.resolve()
        manifest = cls.verify_snapshot(snapshot)
        target = target_root.resolve()
        if target.exists():
            raise BackupError("restore target already exists; restore requires an isolated path")
        target.parent.mkdir(parents=True, exist_ok=True)
        staging = target.parent / f".restoring-{target.name}-{secrets.token_hex(4)}"
        if staging.exists():
            raise BackupError("restore staging identity collision")
        try:
            staging.mkdir()
            shutil.copytree(snapshot / "data", staging / "data")
            receipt = {
                "restored_at": datetime.now(UTC).isoformat(),
                "snapshot_id": manifest.snapshot_id,
                "snapshot_manifest_sha256": _sha256(snapshot / "manifest.json"),
                "database_sha256": _sha256(staging / manifest.database_file),
                "event_count": manifest.event_count,
                "archive_file_count": manifest.archive_file_count,
            }
            (staging / "restore-receipt.json").write_text(
                json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            cls._database_integrity(staging / manifest.database_file)
            cls._verify_archive_set(staging, manifest, staging / manifest.database_file)
            os.replace(staging, target)
            return target
        except Exception:
            if staging.exists() and staging.parent == target.parent:
                shutil.rmtree(staging)
            raise


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def repository_revision(repository_root: Path | None) -> str | None:
    if not repository_root:
        return None
    git = repository_root.resolve() / ".git"
    head = git / "HEAD"
    if not head.is_file():
        return None
    value = head.read_text(encoding="utf-8").strip()
    if value.startswith("ref: "):
        reference = git / value.removeprefix("ref: ")
        if not reference.is_file():
            return None
        value = reference.read_text(encoding="utf-8").strip()
    return value if len(value) == 40 and all(char in "0123456789abcdef" for char in value) else None
