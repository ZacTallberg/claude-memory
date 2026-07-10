"""SessionStart hook — UNIFY. Injects a cross-folder map of curated-fact titles so the agent
knows what it knows everywhere. Warm server first, local fallback otherwise. Fail-safe.
"""
from __future__ import annotations

from _common import call_server, emit_context, read_event, run_failsafe, write_health

EVENT = "SessionStart"


def main() -> None:
    ev = read_event()
    cwd = ev.get("cwd")
    session_id = ev.get("session_id")
    source = ev.get("source")

    from claudemem.config import load_config
    from claudemem.paths import in_scope, killed

    cfg = load_config()
    if killed():
        return
    if not in_scope(cwd, cfg):
        return

    _ensure_server(cfg)
    res = call_server("/api/unify", {"cwd": cwd, "session_id": session_id, "source": source},
                      timeout=4.0)
    if res and res.get("additionalContext"):
        emit_context(res["additionalContext"], EVENT)
        write_health(source="server", chars=len(res["additionalContext"]))
        return

    _local(cfg, session_id)


def _ensure_server(cfg) -> None:
    """Every session start doubles as a watchdog: if the warm server is down, re-arm the
    persistence supervisor (detached; the supervisor is a singleton via a named mutex).
    The current session still uses the local fallback — the server warms for the next one."""
    import os
    import socket
    import subprocess

    try:
        with socket.create_connection((cfg.server.host, cfg.server.port), timeout=0.3):
            return
    except OSError:
        pass
    from claudemem.config import ROOT
    runner = str(ROOT / "scripts" / "persistence_run.ps1")
    flags = (0x00000008 | 0x00000200) if os.name == "nt" else 0  # DETACHED | NEW_PROCESS_GROUP
    try:
        subprocess.Popen(
            ["powershell.exe", "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass",
             "-WindowStyle", "Hidden", "-File", runner],
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            creationflags=flags, close_fds=True)
    except Exception:
        pass


def _local(cfg, session_id: str | None) -> None:
    from claudemem.recall_format import format_unify
    from claudemem.store.factory import get_store

    store = get_store(cfg)
    tmap = store.facts_titles_map()
    try:
        pending = len(store.list_promotions("pending"))
    except Exception:
        pending = 0
    text = format_unify(tmap, cfg, pending_promotions=pending)
    total = sum(len(v) for v in tmap.values())
    write_health(source="local-fallback", backend=type(store).__name__,
                 facts_total=total, facts_shown=min(total, cfg.unify.max_facts),
                 chars=len(text))
    if not text:
        return
    try:
        store.log_injection(hook="unify-fallback", session_id=session_id, prompt_excerpt="",
                            n_recalled=0, n_facts=sum(len(v) for v in tmap.values()),
                            chars=len(text), latency_ms=0)
    except Exception:
        pass
    emit_context(text, EVENT)


if __name__ == "__main__":
    run_failsafe(main)
