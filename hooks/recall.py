"""UserPromptSubmit hook — RECALL. Injects relevant past snippets as untrusted reference
data. Prefers the warm dashboard server (hybrid + warm models); falls back to a fast local
keyword-only search. Fully fail-safe: any problem -> inject nothing, exit 0.
"""
from __future__ import annotations

import time

from _common import call_server, emit_context, read_event, run_failsafe, write_health

EVENT = "UserPromptSubmit"


def main() -> None:
    ev = read_event()
    prompt = ev.get("prompt") or ""
    cwd = ev.get("cwd")
    session_id = ev.get("session_id")

    from claudemem.config import load_config
    from claudemem.paths import killed, in_scope
    from claudemem.text import meaningful_term_count

    cfg = load_config()
    if killed():
        return
    if not in_scope(cwd, cfg):
        return
    if meaningful_term_count(prompt) < cfg.recall.min_terms:
        return

    # 1) Warm server path (full hybrid; server logs the injection).
    res = call_server("/api/recall", {"prompt": prompt, "cwd": cwd, "session_id": session_id},
                      timeout=8.0)
    if res is not None:
        degraded = any(res.get(key) for key in ("timeout", "shed", "error"))
        if degraded:
            _local_keyword(cfg, prompt, session_id)
            return
        # The server answered — memory is healthy and delivering. Refresh the beacon EVERY prompt so a
        # long-lived session never shows a false "mem stale" (the beacon used to age only from SessionStart).
        mode = res.get("retrieval_mode") or "unknown"
        write_health(source=("server" if mode == "hybrid" else "server-keyword-only"),
                     retrieval_mode=mode, vector_used=bool(res.get("vector_used")),
                     chars=len(res.get("additionalContext") or ""))
        if res.get("additionalContext"):
            emit_context(res["additionalContext"], EVENT)
        # A healthy hybrid miss is complete. Do not repeat it with BM25 or mislabel it fallback.
        return

    # 2) Fallback: local keyword-only (no embedding model load).
    _local_keyword(cfg, prompt, session_id)


def _local_keyword(cfg, prompt: str, session_id: str | None) -> None:
    from claudemem.providers.embeddings import NullEmbeddingProvider
    from claudemem.providers.reranker import NoopReranker
    from claudemem.recall_format import format_recall
    from claudemem.retriever import Retriever
    from claudemem.store.factory import get_store

    t0 = time.time()
    store = get_store(cfg)
    r = Retriever(cfg, store,
                  embedder=NullEmbeddingProvider(cfg.embeddings.dim),
                  reranker=NoopReranker())
    results = r.search(prompt, tier="hot", exclude_session=session_id,
                       k=cfg.recall.top_k, do_rerank=False)
    facts = r.search_facts(prompt, cfg.recall.facts_k) if cfg.recall.include_facts else []
    # Log even a zero-result miss — recall hit-rate is unmeasurable otherwise.
    text = format_recall(results, facts, cfg) if (results or facts) else ""
    try:
        store.log_injection(hook="recall-fallback", session_id=session_id,
                            prompt_excerpt=prompt[:200], n_recalled=len(results),
                            n_facts=len(facts), chars=len(text),
                            latency_ms=int((time.time() - t0) * 1000))
    except Exception:
        pass
    # Fell back to the local engine — still a healthy delivery; refresh the beacon (yellow "mem sq" state).
    write_health(source="local-fallback", backend=type(store).__name__)
    if not text:
        return
    emit_context(text, EVENT)


if __name__ == "__main__":
    run_failsafe(main)
