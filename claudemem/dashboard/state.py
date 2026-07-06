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
