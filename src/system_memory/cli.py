from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from .api import create_app
from .archive import CanonicalArchive
from .auth import CredentialStore
from .database import Database
from .settings import Settings
from .store import MemoryStore


def _settings(args: argparse.Namespace) -> Settings:
    values = {}
    if getattr(args, "root", None):
        values["root"] = Path(args.root)
    if getattr(args, "port", None):
        values["port"] = args.port
    return Settings(**values)


def _store(settings: Settings) -> MemoryStore:
    memory = MemoryStore(
        Database(settings.resolved_database_path, busy_timeout_ms=settings.busy_timeout_ms),
        CanonicalArchive(settings.resolved_archive_path),
    )
    memory.initialize()
    return memory


def _write_private(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(content)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        if path.exists():
            path.unlink()
        raise


def command_init(args: argparse.Namespace) -> int:
    settings = _settings(args)
    memory = _store(settings)
    credentials = CredentialStore(memory.database)
    token_path = settings.resolved_token_path
    if token_path.exists():
        token = token_path.read_text(encoding="utf-8").strip()
        if credentials.authenticate(token):
            print(json.dumps({"ok": True, "initialized": False, "token_path": str(token_path)}))
            return 0
        raise RuntimeError("token file exists but does not authenticate; refusing to overwrite")
    credential, token = credentials.create(
        actor_id="local-admin",
        label="local administrator",
        scopes={"*"},
    )
    _write_private(token_path, token)
    print(
        json.dumps(
            {
                "ok": True,
                "initialized": True,
                "credential_id": credential.credential_id,
                "token_path": str(token_path),
            }
        )
    )
    return 0


def command_health(args: argparse.Namespace) -> int:
    settings = _settings(args)
    memory = _store(settings)
    result = {**memory.database.health(), "counts": memory.counts()}
    print(json.dumps(result, indent=2))
    return 0 if result["ok"] else 1


def command_serve(args: argparse.Namespace) -> int:
    import uvicorn

    settings = _settings(args)
    app = create_app(settings)
    uvicorn.run(app, host=settings.host, port=settings.port, log_level="info")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="system-memory")
    parser.add_argument("--root", help="installation root (defaults to current directory)")
    subparsers = parser.add_subparsers(dest="command", required=True)

    initialize = subparsers.add_parser("init", help="initialize storage and a local admin token")
    initialize.set_defaults(handler=command_init)

    health = subparsers.add_parser("health", help="inspect local database integrity")
    health.set_defaults(handler=command_health)

    serve = subparsers.add_parser("serve", help="run the authenticated loopback API")
    serve.add_argument("--port", type=int, default=None)
    serve.set_defaults(handler=command_serve)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except Exception as error:
        print(
            json.dumps({"ok": False, "error": type(error).__name__, "detail": str(error)}),
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
