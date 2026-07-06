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


def emit_context(text: str, event_name: str) -> None:
    text = (text or "")[:9500]
    if not text.strip():
        return
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


def run_failsafe(fn) -> None:
    """Run a hook body; swallow everything; always exit 0 (never block the prompt)."""
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
