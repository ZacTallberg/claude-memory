from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from pathlib import Path

from .clock import utc_iso
from .migrations import MIGRATIONS


class MigrationError(RuntimeError):
    pass


class Database:
    """SQLite connection policy and ordered migrations.

    The API service is the sole normal writer. SQLite still enforces serialization for
    repair CLIs and crash recovery, so correctness does not depend on an in-process lock.
    """

    def __init__(self, path: Path, *, busy_timeout_ms: int = 5_000) -> None:
        self.path = path.resolve()
        self.busy_timeout_ms = busy_timeout_ms

    def connect(self, *, read_only: bool = False) -> sqlite3.Connection:
        if not read_only:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            target = str(self.path)
        else:
            target = f"file:{self.path.as_posix()}?mode=ro"
        connection = sqlite3.connect(
            target,
            timeout=self.busy_timeout_ms / 1000,
            isolation_level=None,
            uri=read_only,
            check_same_thread=False,
        )
        connection.row_factory = sqlite3.Row
        connection.execute(f"PRAGMA busy_timeout={int(self.busy_timeout_ms)}")
        connection.execute("PRAGMA foreign_keys=ON")
        if not read_only:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("PRAGMA synchronous=FULL")
            connection.execute("PRAGMA wal_autocheckpoint=1000")
        return connection

    @contextmanager
    def read(self) -> Iterator[sqlite3.Connection]:
        connection = self.connect(read_only=True)
        try:
            yield connection
        finally:
            connection.close()

    @contextmanager
    def write(self, *, immediate: bool = True) -> Iterator[sqlite3.Connection]:
        connection = self.connect()
        try:
            connection.execute("BEGIN IMMEDIATE" if immediate else "BEGIN")
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def migrate(self) -> int:
        with self.write() as connection:
            current = int(connection.execute("PRAGMA user_version").fetchone()[0])
            for migration in MIGRATIONS:
                existing = None
                with suppress(sqlite3.OperationalError):
                    existing = connection.execute(
                        "SELECT checksum FROM schema_migrations WHERE version=?",
                        (migration.version,),
                    ).fetchone()
                if existing and existing["checksum"] != migration.checksum:
                    raise MigrationError(
                        f"migration {migration.version} checksum differs from installed schema"
                    )
                if migration.version <= current:
                    continue
                if migration.version != current + 1:
                    raise MigrationError(
                        f"migration sequence gap: installed={current}, next={migration.version}"
                    )
                self._execute_script(connection, migration.sql)
                connection.execute(
                    "INSERT OR REPLACE INTO schema_migrations(version,name,checksum,applied_at) "
                    "VALUES (?,?,?,?)",
                    (migration.version, migration.name, migration.checksum, utc_iso()),
                )
                connection.execute(f"PRAGMA user_version={migration.version}")
                current = migration.version
        return current

    @staticmethod
    def _execute_script(connection: sqlite3.Connection, script: str) -> None:
        """Execute a SQL script without sqlite3.executescript's implicit COMMIT."""
        pending = ""
        for line in script.splitlines(keepends=True):
            pending += line
            if not sqlite3.complete_statement(pending):
                continue
            statement = pending.strip()
            pending = ""
            if statement:
                connection.execute(statement)
        if pending.strip():
            raise MigrationError("incomplete SQL statement at end of migration")

    def health(self) -> dict[str, object]:
        with self.read() as connection:
            quick = connection.execute("PRAGMA quick_check").fetchone()[0]
            version = int(connection.execute("PRAGMA user_version").fetchone()[0])
            events = int(connection.execute("SELECT COUNT(*) FROM memory_events").fetchone()[0])
        return {
            "ok": quick == "ok",
            "quick_check": quick,
            "schema_version": version,
            "events": events,
            "path": str(self.path),
        }
