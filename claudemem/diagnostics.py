"""Focused operational checks for machine-wide agent-memory delivery.

Unlike the full self-test and golden relevance eval, these checks are intentionally narrow:
client coverage, hot-path hybrid delivery, real hook output, and the latency SLO.
"""
from __future__ import annotations

import json
import os
import subprocess
import time
import urllib.request
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .codex_hooks_install import status as codex_hook_status
from .config import ROOT, load_config
from .hooks_install import status as claude_hook_status
from .paths import in_scope
from .store.factory import get_store


def _json_url(path: str, *, body: dict | None = None, timeout: float = 10.0) -> dict | None:
    cfg = load_config()
    url = f"http://{cfg.server.host}:{cfg.server.port}{path}"
    try:
        if body is None:
            req = urllib.request.Request(url, method="GET")
        else:
            req = urllib.request.Request(
                url, data=json.dumps(body).encode("utf-8"),
                headers={"Content-Type": "application/json"}, method="POST",
            )
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except Exception:
        return None


def integration_status() -> dict:
    """Machine-readable census of every runtime this installation knows how to wire."""
    cfg = load_config()
    codex_config = Path.home() / ".codex" / "config.toml"
    mcp_registered = False
    try:
        text = codex_config.read_text(encoding="utf-8")
        mcp_registered = "[mcp_servers.claude-memory]" in text
    except Exception:
        pass
    health = _json_url("/healthz", timeout=2.0)
    claude_hooks = claude_hook_status()
    codex_hooks = codex_hook_status()
    clients = {
        "claude-code": {
            "automatic_hooks": sorted(claude_hooks),
            "complete": all(name in claude_hooks for name in
                            ("UserPromptSubmit", "SessionStart", "SessionEnd", "PreCompact")),
        },
        "codex": {
            "automatic_hooks": sorted(codex_hooks),
            "mcp_registered": mcp_registered,
            "complete": (all(name in codex_hooks for name in
                             ("UserPromptSubmit", "SessionStart", "SessionEnd", "PreCompact"))
                         and mcp_registered),
        },
    }
    all_clients = all(c["complete"] for c in clients.values())
    server_ok = bool(health and health.get("ok") and health.get("retrieval_mode") == "hybrid")
    machine_wide = cfg.scope.activation == "installed_clients"
    return {
        "ok": all_clients and server_ok and machine_wide,
        "activation": cfg.scope.activation,
        "machine_wide": machine_wide,
        "indexed_transcript_providers": list(cfg.index.transcript_providers),
        "clients": clients,
        "all_known_clients_complete": all_clients,
        "warm_server_hybrid": server_ok,
        "warm_server": health,
        "note": ("Coverage means every supported/configured local client. A new agent runtime needs "
                 "an explicit lifecycle-hook or MCP adapter before it can consume this memory."),
    }


def _hook_probe(cwd: str, token: str) -> dict:
    cfg = load_config()
    python = ROOT / ".venv" / "Scripts" / "python.exe"
    event = {
        "prompt": f"memory recall hook retriever embedding continuation {token}",
        "cwd": cwd,
        "session_id": f"delivery-check-{token}",
    }
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    started = time.perf_counter()
    completed = subprocess.run(
        [str(python), str(ROOT / "hooks" / "recall.py")],
        input=json.dumps(event), text=True, capture_output=True, env=env, cwd=str(ROOT),
        timeout=cfg.delivery.client_timeout_seconds + 5,
    )
    latency_ms = int((time.perf_counter() - started) * 1000)
    context = ""
    try:
        context = json.loads(completed.stdout).get("hookSpecificOutput", {}).get(
            "additionalContext", "")
    except Exception:
        pass
    receipt = None
    for row in get_store(cfg).recent_injections(100):
        details = row.get("details") or {}
        if isinstance(details, str):
            try:
                details = json.loads(details)
            except Exception:
                details = {}
        if details.get("request_id") and token in str(row.get("prompt_excerpt") or ""):
            receipt = {**dict(row), "details": details}
            break
    return {
        "cwd": cwd, "in_scope": in_scope(cwd, cfg), "exit_code": completed.returncode,
        "chars": len(context), "latency_ms": latency_ms,
        "receipt_hook": receipt.get("hook") if receipt else None,
        "retrieval_mode": (receipt or {}).get("details", {}).get("retrieval_mode"),
        "ok": (completed.returncode == 0 and bool(context) and receipt is not None
               and receipt.get("hook") == "hook-recall-delivered"
               and receipt.get("details", {}).get("retrieval_mode") == "hybrid"),
    }


def delivery_check(*, concurrency: int = 4, under_index_load: bool = False,
                   cwd: str | None = None) -> dict:
    """Exercise uncached hybrid requests plus real hooks in unrelated directories."""
    cfg = load_config()
    index_load = None
    if under_index_load:
        index_load = _json_url("/api/index", body={"reason": "delivery-check"}, timeout=2.0)
    load_accepted = (not under_index_load or bool(index_load and
                     (index_load.get("started") or index_load.get("reason") == "already running")))

    concurrency = max(1, min(int(concurrency), 16))
    probe_id = uuid.uuid4().hex[:10]

    def one(i: int) -> dict:
        query = (f"shared agent memory recall index retriever continuation delivery "
                 f"{probe_id}-{i}")
        started = time.perf_counter()
        result = _json_url(
            "/api/recall",
            body={"prompt": query, "session_id": f"delivery-direct-{probe_id}-{i}",
                  "request_id": f"direct-{probe_id}-{i}", "cwd": cwd or str(ROOT)},
            timeout=cfg.delivery.client_timeout_seconds + 1,
        )
        wall_ms = int((time.perf_counter() - started) * 1000)
        return {"wall_ms": wall_ms, "response": result}

    direct = []
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(one, i) for i in range(concurrency)]
        for future in as_completed(futures):
            direct.append(future.result())
    latencies = sorted(row["wall_ms"] for row in direct)
    direct_ok = all(
        row["response"] is not None
        and row["response"].get("retrieval_mode") == "hybrid"
        and not any(row["response"].get(k) for k in ("timeout", "shed", "error"))
        and row["wall_ms"] <= cfg.delivery.hybrid_slo_ms
        for row in direct
    )

    sample_cwds = [cwd or str(ROOT), str(Path.home() / "Documents" / "Codex")]
    if os.name == "nt":
        sample_cwds.append("C:/Windows/Temp")
    hooks = [_hook_probe(sample, f"{probe_id}-hook-{i}")
             for i, sample in enumerate(dict.fromkeys(sample_cwds))]
    p95_index = min(len(latencies) - 1, max(0, int(0.95 * len(latencies))))
    result = {
        "ok": load_accepted and direct_ok and all(row["ok"] for row in hooks),
        "activation": cfg.scope.activation,
        "under_index_load": under_index_load,
        "index_load": index_load,
        "concurrency": concurrency,
        "hybrid_slo_ms": cfg.delivery.hybrid_slo_ms,
        "direct": {
            "passed": sum(1 for row in direct if row["response"] is not None
                          and row["response"].get("retrieval_mode") == "hybrid"
                          and row["wall_ms"] <= cfg.delivery.hybrid_slo_ms),
            "total": len(direct),
            "p50_ms": latencies[len(latencies) // 2],
            "p95_ms": latencies[p95_index],
            "max_ms": max(latencies),
        },
        "hooks": hooks,
    }
    return result
