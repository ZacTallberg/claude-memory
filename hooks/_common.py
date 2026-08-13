"""Shared hook plumbing: stdin event read, fail-safe runner, warm-server call, context emit.

Hooks must ALWAYS exit 0 and never print anything except the (optional) JSON context line.
"""
from __future__ import annotations

import json
import os
import sys
import urllib.request

# Make the claudemem package importable regardless of how the hook is invoked.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def read_event() -> dict:
    try:
        data = sys.stdin.buffer.read()
        return json.loads(data.decode("utf-8-sig")) if data else {}  # tolerate a BOM
    except Exception:
        return {}


# Held while the context envelope is being written, so the watchdog can never tear a
# half-written JSON line (malformed hook output is worse than no output).
_emitting = __import__("threading").Lock()


def emit_context(text: str, event_name: str) -> None:
    text = (text or "")[:9500]
    if not text.strip():
        return
    with _emitting:
        sys.stdout.write(json.dumps({
            "hookSpecificOutput": {"hookEventName": event_name, "additionalContext": text}
        }))
        sys.stdout.flush()


def call_server(path: str, payload: dict, timeout: float) -> dict | None:
    """POST to the warm dashboard server. Returns parsed JSON or None on any failure."""
    from claudemem.config import load_config
    cfg = load_config()
    url = f"http://{cfg.server.host}:{cfg.server.port}{path}"
    try:
        req = urllib.request.Request(
            url, data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception:
        return None


def write_health(**fields) -> None:
    """Drop the tiny health beacon the statusline renders — memory delivery must never degrade silently
    (the 2026-07-06 harness audit found both engines down with zero signal). Written by BOTH unify
    (SessionStart) AND recall (every UserPromptSubmit) so a long-lived session keeps it fresh — the beacon
    reflects that memory was actually DELIVERED recently, not merely that a session once started."""
    import time
    try:
        path = os.path.join(os.path.expanduser("~"), ".claude", "memory-health.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"ts": int(time.time()), **fields}, f)
    except Exception:
        pass


def _budget_seconds() -> float:
    """Hard ceiling for this hook process. Env wins so a harness can tighten it without config."""
    raw = os.environ.get("CLAUDEMEM_HOOK_BUDGET_S")
    if raw:
        try:
            return max(1.0, float(raw))
        except ValueError:
            pass
    try:
        from claudemem.config import load_config
        return load_config().delivery.hook_budget_seconds
    except Exception:
        return 20.0  # loading config is itself work; never let that failure remove the ceiling


def _start_watchdog(budget: float) -> None:
    """Force-exit at `budget` seconds, whatever the hook is doing.

    Every other timeout in this system is a request someone else may overrun: the server's own
    deadline, a socket timeout, a store lock. Those bound the COMMON case, and a transient stall
    (a 1.9 GB store checkpointing, a model lane held by indexing, a wedged FS) slips past all of
    them — one measured recall spent 38.1s against a 20s harness limit. A timer that owns the
    process is the only bound that cannot be overrun, so the harness-visible failure ("hook timed
    out") becomes structurally impossible rather than merely unlikely.
    """
    import threading

    def bail() -> None:
        # If the envelope is mid-write, let it finish; a torn JSON line is worse than a late one.
        _emitting.acquire(timeout=0.5)
        try:
            sys.stdout.flush()
        except Exception:
            pass
        os._exit(0)

    t = threading.Timer(budget, bail)
    t.daemon = True
    t.start()


def run_failsafe(fn) -> None:
    """Run a hook body; swallow everything; always exit 0 (never block the prompt).

    A bootstrap ceiling is armed FIRST, before the config read that resolves the real one, so a
    stall inside setup is bounded exactly like a stall inside retrieval. Reading config is itself
    I/O and must not be the one unguarded step.
    """
    import threading
    boot = threading.Timer(45.0, lambda: os._exit(0))
    boot.daemon = True
    boot.start()
    try:
        budget = _budget_seconds()
    finally:
        boot.cancel()
    _start_watchdog(budget)
    try:
        fn()
    except Exception:
        pass
    finally:
        try:
            sys.stdout.flush()
        except Exception:
            pass
        os._exit(0)
