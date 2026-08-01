"""Wire/unwire our hooks into ~/.claude/settings.json. Idempotent; preserves all other settings.

Windows command form: uses the venv python with forward-slash paths (backslashes misparse in
the shell-form command).
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

from .config import ROOT

SETTINGS = Path.home() / ".claude" / "settings.json"
_HOOKS_DIR = (ROOT / "hooks")


def _py() -> str:
    venv = ROOT / ".venv" / "Scripts" / "python.exe"
    p = venv if venv.exists() else Path("python")
    return str(p).replace("\\", "/")


def _cmd(script: str) -> str:
    return f'"{_py()}" "{str(_HOOKS_DIR / script).replace(chr(92), "/")}"'


_OUR_SCRIPTS = ("recall.py", "unify.py", "index_trigger.py", "fleet_autoenroll.py")


def _is_ours(entry: dict) -> bool:
    # Recognition must survive the repo living under ANY directory name. Matching the literal
    # substring "claude-memory/hooks/" meant a clone under another name never recognized its own
    # entries and stacked a duplicate on every install.
    for h in entry.get("hooks", []):
        cmd = (h.get("command", "")).replace("\\", "/").rstrip('"')
        if any(cmd.endswith(f"/hooks/{s}") for s in _OUR_SCRIPTS):
            return True
    return False


def _desired() -> dict:
    return {
        "UserPromptSubmit": [
            {"matcher": "", "hooks": [{"type": "command", "command": _cmd("recall.py"), "timeout": 20}]}
        ],
        "SessionStart": [
            {"matcher": m, "hooks": [{"type": "command", "command": _cmd("unify.py"), "timeout": 30}]}
            for m in ("startup", "resume", "clear")
        ] + [
            # Fleet carrier (isolated, stdlib, fail-safe): enrolls an un-enrolled Vigor
            # fleet machine from the hub's open bootstrap; instant no-op everywhere else.
            {"matcher": "startup", "hooks": [{"type": "command", "command": _cmd("fleet_autoenroll.py"), "timeout": 90}]}
        ],
        "SessionEnd": [
            {"matcher": "", "hooks": [{"type": "command", "command": _cmd("index_trigger.py"), "timeout": 10}]}
        ],
        "PreCompact": [
            {"matcher": "", "hooks": [{"type": "command", "command": _cmd("index_trigger.py"), "timeout": 10}]}
        ],
    }


def _load() -> dict:
    if SETTINGS.exists():
        try:
            return json.loads(SETTINGS.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _save(data: dict) -> None:
    SETTINGS.parent.mkdir(parents=True, exist_ok=True)
    if SETTINGS.exists():
        shutil.copy2(SETTINGS, SETTINGS.with_suffix(".json.bak"))
    SETTINGS.write_text(json.dumps(data, indent=2), encoding="utf-8")


def install() -> str:
    # A wired-looking dead layer is the failure mode fail-safe hooks cannot surface: if the
    # interpreter the hook commands reference cannot EXECUTE (e.g. an exe-blocking group
    # policy on venvs outside an allowlisted root), every hook dies silently forever.
    # Prove execution once, loudly, at install time.
    import subprocess
    try:
        proc = subprocess.run([_py().replace("/", "\\"), "-c", "pass"],
                              capture_output=True, timeout=15)
        if proc.returncode != 0:
            raise OSError(proc.stderr.decode(errors="replace")[:200])
    except Exception as exc:
        raise SystemExit(
            f"REFUSING to install hooks: the hook interpreter {_py()} cannot execute "
            f"({exc}). If your machine blocks executables outside approved roots "
            f"(corp group policy), move this clone + venv under an approved path "
            f"(e.g. C:\\code) and re-run.") from exc
    data = _load()
    hooks = data.setdefault("hooks", {})
    desired = _desired()
    for event, entries in desired.items():
        existing = [e for e in hooks.get(event, []) if not _is_ours(e)]  # drop our stale ones
        hooks[event] = existing + entries
    _save(data)
    return f"installed hooks into {SETTINGS}"


def uninstall() -> str:
    data = _load()
    hooks = data.get("hooks", {})
    for event in list(hooks.keys()):
        hooks[event] = [e for e in hooks[event] if not _is_ours(e)]
        if not hooks[event]:
            del hooks[event]
    _save(data)
    return f"removed our hooks from {SETTINGS}"


def status() -> dict:
    data = _load()
    hooks = data.get("hooks", {})
    out = {}
    for event, entries in hooks.items():
        ours = [e for e in entries if _is_ours(e)]
        if ours:
            out[event] = [h["command"] for e in ours for h in e["hooks"]]
    return out
