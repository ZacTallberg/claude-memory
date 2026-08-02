from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from .api import create_app
from .archive import CanonicalArchive
from .auth import CredentialStore
from .backup import BackupManager
from .database import Database
from .evaluation import load_cases, run_evaluation, validate_gold64, write_schema
from .legacy_v1 import LegacyV1Importer
from .raw_transcripts import RawTranscriptImporter
from .recall import RecallEngine
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


def command_import_v1(args: argparse.Namespace) -> int:
    settings = _settings(args)
    memory = _store(settings)

    def progress(report):
        print(
            json.dumps(
                {
                    "progress": True,
                    "chunk_rows_seen": report.chunk_rows_seen,
                    "inserted": report.chunk_events_inserted,
                    "duplicates": report.exact_duplicates_skipped,
                }
            ),
            file=sys.stderr,
            flush=True,
        )

    report = LegacyV1Importer(memory).import_database(
        Path(args.database),
        only_missing_sources=args.only_missing_sources,
        only_facts=args.only_facts,
        include_facts=not args.skip_facts,
        progress_every=args.progress_every,
        on_progress=progress,
    )
    print(report.model_dump_json(indent=2))
    return 0


def command_import_raw(args: argparse.Namespace) -> int:
    settings = _settings(args)
    memory = _store(settings)
    report = RawTranscriptImporter(memory).import_sources(Path(args.database))
    print(report.model_dump_json(indent=2))
    return 0


def command_build_lexical(args: argparse.Namespace) -> int:
    settings = _settings(args)
    memory = _store(settings)
    corpus_sha256 = memory.event_corpus_sha256()
    generation = memory.create_search_generation(
        corpus_sha256=corpus_sha256,
        chunker_version="event-v1",
        lexical_config={"tokenizer": "unicode61 remove_diacritics 2"},
        code_revision=args.code_revision,
        lock_sha256=args.lock_sha256,
    )

    def progress(indexed: int) -> None:
        print(
            json.dumps({"progress": True, "generation_id": generation, "indexed": indexed}),
            file=sys.stderr,
            flush=True,
        )

    indexed = memory.index_all_events(
        generation,
        batch_size=args.batch_size,
        on_progress=progress,
    )
    memory.activate_generation(generation, expected_event_corpus_sha256=corpus_sha256)
    status = memory.embedding_status(generation)
    print(
        json.dumps(
            {
                "generation_id": generation,
                "corpus_sha256": corpus_sha256,
                "indexed_now": indexed,
                **status,
            },
            indent=2,
        )
    )
    return 0


def command_evaluate(args: argparse.Namespace) -> int:
    settings = _settings(args)
    memory = _store(settings)
    cases = load_cases(Path(args.cases))
    if args.validate_gold64:
        validate_gold64(cases)
    selected = [case for case in cases if case.split == args.split]
    if args.split == "test" and not args.allow_sealed:
        raise ValueError("sealed test cases require --allow-sealed")
    if not selected:
        raise ValueError(f"case file contains no {args.split} cases")
    summary = run_evaluation(RecallEngine(memory), selected)
    print(summary.model_dump_json(indent=2))
    return 0


def command_eval_schema(args: argparse.Namespace) -> int:
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    write_schema(output)
    print(json.dumps({"ok": True, "schema": str(output)}))
    return 0


def command_backup(args: argparse.Namespace) -> int:
    settings = _settings(args)
    memory = _store(settings)
    snapshot = BackupManager(memory, repository_root=Path(__file__).resolve().parents[2]).create(
        Path(args.destination)
    )
    print(json.dumps({"ok": True, "snapshot": str(snapshot)}))
    return 0


def command_verify_backup(args: argparse.Namespace) -> int:
    manifest = BackupManager.verify_snapshot(Path(args.snapshot))
    print(manifest.model_dump_json(indent=2))
    return 0


def command_restore(args: argparse.Namespace) -> int:
    target = BackupManager.restore(Path(args.snapshot), Path(args.target))
    print(json.dumps({"ok": True, "restored": str(target)}))
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

    importer = subparsers.add_parser(
        "import-v1", help="import a stabilized claude-memory SQLite snapshot"
    )
    importer.add_argument("database")
    import_mode = importer.add_mutually_exclusive_group()
    import_mode.add_argument("--only-missing-sources", action="store_true")
    import_mode.add_argument("--only-facts", action="store_true")
    import_mode.add_argument("--skip-facts", action="store_true")
    importer.add_argument("--progress-every", type=int, default=1_000)
    importer.set_defaults(handler=command_import_v1)

    raw_importer = subparsers.add_parser(
        "import-raw", help="reparse available Claude and Codex source files from a v1 inventory"
    )
    raw_importer.add_argument("database")
    raw_importer.set_defaults(handler=command_import_raw)

    lexical = subparsers.add_parser(
        "build-lexical", help="build and atomically activate a lexical event index generation"
    )
    lexical.add_argument("--batch-size", type=int, default=1_000)
    lexical.add_argument("--code-revision")
    lexical.add_argument("--lock-sha256")
    lexical.set_defaults(handler=command_build_lexical)

    evaluate = subparsers.add_parser(
        "evaluate", help="run evidence-group retrieval evaluation against an active generation"
    )
    evaluate.add_argument("cases")
    evaluate.add_argument("--split", choices=("dev", "test"), default="dev")
    evaluate.add_argument("--allow-sealed", action="store_true")
    evaluate.add_argument("--validate-gold64", action="store_true")
    evaluate.set_defaults(handler=command_evaluate)

    schema = subparsers.add_parser("eval-schema", help="write the evaluation case JSON schema")
    schema.add_argument("output")
    schema.set_defaults(handler=command_eval_schema)

    backup = subparsers.add_parser("backup", help="create a verified canonical snapshot")
    backup.add_argument("destination")
    backup.set_defaults(handler=command_backup)

    verify_backup = subparsers.add_parser(
        "verify-backup", help="verify every declared backup payload and archive reference"
    )
    verify_backup.add_argument("snapshot")
    verify_backup.set_defaults(handler=command_verify_backup)

    restore = subparsers.add_parser(
        "restore", help="restore a verified snapshot into a new isolated root"
    )
    restore.add_argument("snapshot")
    restore.add_argument("target")
    restore.set_defaults(handler=command_restore)
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
