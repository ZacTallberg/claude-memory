"""Pick a storage backend from config, with auto-fallback to SQLite."""
from __future__ import annotations

from ..config import Config
from ..log import get_logger
from .base import Store

log = get_logger(__name__)

_singleton: Store | None = None


def _make(cfg: Config, backend: str) -> Store:
    if backend == "postgres":
        from .postgres_store import PostgresStore
        s: Store = PostgresStore(cfg)
    else:
        from .sqlite_store import SqliteStore
        s = SqliteStore(cfg)
    s.connect()
    s.migrate()
    return s


def get_store(cfg: Config, *, cached: bool = True) -> Store:
    """auto: try Postgres, fall back to SQLite. Cached singleton by default."""
    global _singleton
    if cached and _singleton is not None:
        return _singleton

    backend = cfg.store.backend
    if backend in ("postgres", "sqlite"):
        store = _make(cfg, backend)
    else:  # auto
        try:
            store = _make(cfg, "postgres")
            log.info("using postgres/paradedb backend")
        except Exception as e:
            log.warning("postgres unavailable (%s); falling back to sqlite", e)
            store = _make(cfg, "sqlite")

    if cached:
        _singleton = store
    return store
