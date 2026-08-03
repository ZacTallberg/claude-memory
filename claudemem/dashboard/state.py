"""Warm application state: one store connection + warm embedder/reranker shared across requests.
This is what makes the recall hot path fast — the model is loaded once, not per prompt."""
from __future__ import annotations

import threading

from claudemem.config import load_config
from claudemem.log import get_logger
from claudemem.providers.embeddings import get_embedding_provider
from claudemem.providers.reranker import get_reranker
from claudemem.retriever import Retriever
from claudemem.store.factory import get_store

log = get_logger(__name__)


class AppState:
    def __init__(self):
        self.cfg = load_config()
        self.store = get_store(self.cfg)
        # Warm the embedder eagerly so the first recall is fast.
        self.embedder = get_embedding_provider(self.cfg)
        self.reranker = get_reranker(self.cfg)  # cached; cross-encoder loads on first use
        self.retriever = Retriever(self.cfg, self.store, self.embedder, self.reranker)
        # Touch the real query + SQLite vector/FTS paths before readiness. The model constructor's
        # tiny probe warms ONNX itself but not the database pages; without this, the first four-way
        # worker burst after a restart paid paging costs and narrowly missed the delivery SLO.
        try:
            warm_query = "shared agent memory retrieval continuity"
            qvec = self.retriever.embed_query(warm_query)
            self.retriever.search(warm_query, tier="hot", k=1, do_rerank=False, qvec=qvec)
            self.retriever.search_facts(warm_query, k=1, qvec=qvec)
        except Exception as exc:
            log.warning("AppState retrieval prewarm degraded: %s", exc)
        log.info("AppState warm: backend=%s embedder=%s avail=%s",
                 self.store.name, getattr(self.embedder, "name", "?"), self.embedder.available())


_state: AppState | None = None
_lock = threading.Lock()


def get_state() -> AppState:
    global _state
    if _state is None:
        with _lock:
            if _state is None:
                _state = AppState()
    return _state
