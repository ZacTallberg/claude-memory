"""UserPromptSubmit hook — RECALL. Injects relevant past snippets as untrusted reference
data. Prefers the warm dashboard server (hybrid + warm models); falls back to a fast local
keyword-only search. Fully fail-safe: any problem -> inject nothing, exit 0.
"""
from __future__ import annotations

import time
import uuid

from _common import call_server, emit_context, read_event, run_failsafe, write_health

EVENT = "UserPromptSubmit"


def main() -> None:
    ev = read_event()
    prompt = ev.get("prompt") or ""
    cwd = ev.get("cwd")
    session_id = ev.get("session_id")

    from claudemem.config import load_config
    from claudemem.paths import killed, in_scope
    from claudemem.text import should_recall

    cfg = load_config()
    if killed():
        return
    if not in_scope(cwd, cfg):
        return
    if not should_recall(prompt, cfg.recall.min_terms):
        return

    request_id = uuid.uuid4().hex

    # 1) Warm server path (full hybrid; this client records the delivery receipt).
    res = call_server("/api/recall", {"prompt": prompt, "cwd": cwd, "session_id": session_id,
                                      "request_id": request_id},
                      timeout=cfg.delivery.client_timeout_seconds)
    if res is not None:
        degraded = any(res.get(key) for key in ("timeout", "shed", "error"))
        if degraded:
            reason = next((key for key in ("timeout", "shed", "error") if res.get(key)),
                          "degraded")
            _local_keyword(cfg, prompt, session_id, cwd=cwd,
                           request_id=request_id, reason=reason)
            return
        # The server answered — memory is healthy and delivering. Refresh the beacon EVERY prompt so a
        # long-lived session never shows a false "mem stale" (the beacon used to age only from SessionStart).
        mode = res.get("retrieval_mode") or "unknown"
        write_health(source=("server" if mode == "hybrid" else "server-keyword-only"),
                     retrieval_mode=mode, vector_used=bool(res.get("vector_used")),
                     chars=len(res.get("additionalContext") or ""))
        text = res.get("additionalContext") or ""
        if text:
            emit_context(text, EVENT)
        _receipt(cfg, request_id=request_id, session_id=session_id, prompt=prompt,
                 status=("delivered" if text else "miss"), res=res)
        # A healthy hybrid miss is complete. Do not repeat it with BM25 or mislabel it fallback.
        return

    # 2) Fallback: local keyword-only (no embedding model load).
    _local_keyword(cfg, prompt, session_id, cwd=cwd, request_id=request_id,
                   reason="server-unavailable")


def _local_keyword(cfg, prompt: str, session_id: str | None, *, cwd: str | None,
                   request_id: str, reason: str) -> None:
    from claudemem.providers.embeddings import NullEmbeddingProvider
    from claudemem.providers.reranker import NoopReranker
    from claudemem.recall_format import format_recall
    from claudemem.retriever import Retriever
    from claudemem.store.factory import get_store
    from claudemem.text import recall_query

    t0 = time.time()
    store = get_store(cfg)
    r = Retriever(cfg, store,
                  embedder=NullEmbeddingProvider(cfg.embeddings.dim),
                  reranker=NoopReranker())
    query = recall_query(prompt, cwd)
    results = r.search(query, tier="hot", exclude_session=session_id,
                       k=cfg.recall.top_k, do_rerank=False)
    from claudemem.paths import project_from_cwd
    facts = (r.search_facts(query, cfg.recall.facts_k, automatic=True,
                            project=project_from_cwd(cwd))
             if cfg.recall.include_facts else [])
    # Log even a zero-result miss — recall hit-rate is unmeasurable otherwise.
    text = format_recall(results, facts, cfg, request_id=request_id) if (results or facts) else ""
    try:
        store.log_injection(hook="recall-fallback", session_id=session_id,
                            prompt_excerpt=prompt[:200], n_recalled=len(results),
                            n_facts=len(facts), chars=len(text),
                            latency_ms=int((time.time() - t0) * 1000),
                            details={"request_id": request_id, "delivery_status": "fallback",
                                     "failure_reason": reason, "retrieval_mode": "keyword-only",
                                     "vector_used": False,
                                     "chunk_ids": [result.chunk.id for result in results],
                                     "fact_ids": [fact.id for fact in facts]})
    except Exception:
        pass
    # Fell back to the local engine — still a healthy delivery; refresh the beacon (yellow "mem sq" state).
    write_health(source="local-fallback", backend=type(store).__name__)
    if not text:
        return
    emit_context(text, EVENT)


def _receipt(cfg, *, request_id: str, session_id: str | None, prompt: str,
             status: str, res: dict) -> None:
    """Confirm what the hook accepted, instead of what the server merely computed."""
    call_server("/api/delivery", {
        "request_id": request_id,
        "client": "hook",
        "status": status,
        "session_id": session_id,
        "prompt_excerpt": prompt[:200],
        "n_recalled": int(res.get("n_recalled") or 0),
        "n_facts": int(res.get("n_facts") or 0),
        "chars": len(res.get("additionalContext") or ""),
        "latency_ms": int(res.get("latency_ms") or 0),
        "retrieval_mode": res.get("retrieval_mode") or "unknown",
        "vector_used": bool(res.get("vector_used")),
        "chunk_ids": list(res.get("chunk_ids") or []),
        "fact_ids": list(res.get("fact_ids") or []),
    }, timeout=cfg.delivery.receipt_timeout_seconds)


if __name__ == "__main__":
    run_failsafe(main)
