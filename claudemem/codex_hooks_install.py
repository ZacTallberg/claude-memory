"""Install/remove shared-memory lifecycle hooks in ~/.codex/hooks.json.

The hook programs themselves are provider-neutral. This module only translates their lifecycle
registration into Codex's user-level hook file while preserving all unrelated hooks.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

from .config import ROOT, load_config

HOOKS_FILE = Path.home() / ".codex" / "hooks.json"
_HOOKS_DIR = ROOT / "hooks"
_OUR_SCRIPTS = ("recall.py", "unify.py", "index_trigger.py")


def _py() -> str:
    venv = ROOT / ".venv" / "Scripts" / "python.exe"
    return str(venv if venv.exists() else Path("python")).replace("\\", "/")


def _cmd(script: str) -> str:
    return f'"{_py()}" "{str(_HOOKS_DIR / script).replace(chr(92), "/")}"'


def _is_ours(entry: dict) -> bool:
    for hook in entry.get("hooks", []):
        command = (hook.get("commandWindows") or hook.get("command_windows")
                   or hook.get("command") or "").replace("\\", "/").rstrip('"')
        if any(command.endswith(f"/hooks/{script}") for script in _OUR_SCRIPTS):
            return True
    return False


def _handler(script: str, timeout: int, status: str, *, context_limit: int | None = None) -> dict:
    handler = {
        "type": "command",
        "command": _cmd(script),
        "commandWindows": _cmd(script),
        "timeout": timeout,
        "statusMessage": status,
    }
    if context_limit is not None:
        handler["additionalContextLimit"] = context_limit
    return handler


def _desired() -> dict:
    # The recall envelope is capped at recall.max_chars (recall_format), which machine-local
    # config overlays may raise — the registered limit must always cover it.
    recall_limit = load_config().recall.max_chars + 500
    return {
        "UserPromptSubmit": [{
            "hooks": [_handler("recall.py", 15, "Recalling shared local memory",
                               context_limit=recall_limit)]
        }],
        "SessionStart": [{
            "matcher": "startup|resume|clear|compact",
            "hooks": [_handler("unify.py", 30, "Loading the shared memory map",
                               context_limit=10000)]
        }],
        "SessionEnd": [{
            "hooks": [_handler("index_trigger.py", 3, "Updating shared local memory")]
        }],
        "PreCompact": [{
            "hooks": [_handler("index_trigger.py", 10, "Preserving memory before compaction")]
        }],
    }


def _load() -> dict:
    if not HOOKS_FILE.exists():
        return {"description": "User-level lifecycle hooks."}
    try:
        return json.loads(HOOKS_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {"description": "User-level lifecycle hooks."}


def _save(data: dict) -> None:
    HOOKS_FILE.parent.mkdir(parents=True, exist_ok=True)
    if HOOKS_FILE.exists():
        shutil.copy2(HOOKS_FILE, HOOKS_FILE.with_suffix(".json.bak"))
    HOOKS_FILE.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def install() -> str:
    data = _load()
    hooks = data.setdefault("hooks", {})
    for event, desired in _desired().items():
        hooks[event] = [entry for entry in hooks.get(event, []) if not _is_ours(entry)] + desired
    _save(data)
    return f"installed Codex hooks into {HOOKS_FILE}"


def uninstall() -> str:
    data = _load()
    hooks = data.get("hooks", {})
    for event in list(hooks):
        hooks[event] = [entry for entry in hooks[event] if not _is_ours(entry)]
        if not hooks[event]:
            del hooks[event]
    _save(data)
    return f"removed shared-memory hooks from {HOOKS_FILE}"


def status() -> dict:
    hooks = _load().get("hooks", {})
    return {event: [h.get("commandWindows") or h.get("command")
                    for entry in entries if _is_ours(entry) for h in entry.get("hooks", [])]
            for event, entries in hooks.items() if any(_is_ours(entry) for entry in entries)}
