"""Rotating file logger. Hooks must NEVER print logs to stdout (stdout is the
hook contract channel) — everything goes to data/logs/claudemem.log."""
from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler

from .config import DATA_DIR

_LOG_DIR = DATA_DIR / "logs"
_configured = False


def get_logger(name: str = "claudemem") -> logging.Logger:
    global _configured
    logger = logging.getLogger(name)
    if not _configured:
        try:
            _LOG_DIR.mkdir(parents=True, exist_ok=True)
            handler = RotatingFileHandler(
                _LOG_DIR / "claudemem.log", maxBytes=2_000_000, backupCount=3, encoding="utf-8"
            )
            handler.setFormatter(logging.Formatter(
                "%(asctime)s %(levelname)s %(name)s %(message)s"))
            root = logging.getLogger("claudemem")
            root.addHandler(handler)
            root.setLevel(logging.INFO)
            root.propagate = False
        except Exception:
            # Never let logging setup break a hook.
            pass
        _configured = True
    return logger
