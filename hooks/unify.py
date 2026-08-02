"""SessionStart hook — UNIFY. Injects a cross-folder map of curated-fact titles so the agent
knows what it knows everywhere. Warm server first, local fallback otherwise. Fail-safe.
"""
from __future__ import annotations

from _common import call_server, emit_context, read_event, run_failsafe, write_health

EVENT = "SessionStart"

# Kept short: this runs on the session-start path, and a slow health probe would tax every
# session to diagnose a server that the supervisor is going to restart anyway.
_HEALTH_TIMEOUT_S = 2.0


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
    """Every session start doubles as a watchdog: if the warm server is unhealthy, re-arm the
    persistence supervisor (detached; the supervisor is a singleton via a named mutex).
    The current session still uses the local fallback — the server warms for the next one.

    This MUST probe for a real answer, not an open port. The original check was
    `socket.create_connection(...)`, which returns as soon as something is listening. On
    2026-07-29 a server that had been wedged since 2026-07-22 — store lock held, 73 threads
    parked, /api/stats not answering in 300s — still accepted TCP in 88ms, so this watchdog
    reported it healthy on every single session start for seven days while every recall
    silently degraded to keyword-only. A guard that cannot observe the failure it exists to
    catch is worse than no guard: it manufactures confidence.
    """
    import os
    import subprocess
    import urllib.request

    import json
    import time
    from claudemem.config import ROOT

    heartbeat = ROOT / "data" / "persistence-heartbeat.json"
    supervisor_alive = False
    try:
        data = json.loads(heartbeat.read_text(encoding="utf-8"))
        supervisor_alive = time.time() - float(data.get("ts", 0)) < 60
    except Exception:
        pass

    # Session-start recovery is a liveness decision. /healthz also checks readiness and may
    # be slow while the shared writer is legitimately indexing; spawning a replacement then
    # can destroy useful work.
    url = f"http://{cfg.server.host}:{cfg.server.port}/livez"
    try:
        with urllib.request.urlopen(url, timeout=_HEALTH_TIMEOUT_S) as r:
            if r.status == 200 and supervisor_alive:
                return
    except Exception:
        pass  # unreachable, wedged, 503, or too slow -> all mean "re-arm the supervisor"
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
