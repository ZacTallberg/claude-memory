"""Local cross-encoder reranker via fastembed (ONNX, CPU, no torch)."""
from __future__ import annotations

from ..config import Config
from ..log import get_logger
from .reranker import Reranker

log = get_logger(__name__)


class FastEmbedReranker(Reranker):
    def __init__(self, cfg: Config):
        from fastembed.rerank.cross_encoder import TextCrossEncoder  # lazy

        self.cfg = cfg
        self._model = TextCrossEncoder(model_name=cfg.reranker.model)
        self.name = f"fastembed-rerank:{cfg.reranker.model}"
        log.info("FastEmbedReranker ready model=%s", cfg.reranker.model)

    def rerank(self, query: str, items: list[tuple[int, str]], top_n: int) -> list[tuple[int, float]]:
        if not items:
            return []
        ids = [cid for cid, _ in items]
        docs = [txt for _, txt in items]
        scores = list(self._model.rerank(query, docs))
        ranked = sorted(zip(ids, scores), key=lambda x: x[1], reverse=True)
        return [(int(cid), float(s)) for cid, s in ranked[:top_n]]
