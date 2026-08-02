"""Small deterministic ranking helpers shared by storage backends."""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable


def reciprocal_rank_order(ranked_ids: Iterable[Iterable[int]], *, rrf_k: int = 60) -> list[int]:
    """Fuse ranked id lists with Reciprocal Rank Fusion and deterministic tie breaking."""
    scores: dict[int, float] = defaultdict(float)
    best_rank: dict[int, int] = {}
    first_seen: dict[int, int] = {}
    seen_order = 0
    for ranked in ranked_ids:
        for rank, item_id in enumerate(ranked, 1):
            scores[item_id] += 1.0 / (rrf_k + rank)
            best_rank[item_id] = min(best_rank.get(item_id, rank), rank)
            if item_id not in first_seen:
                first_seen[item_id] = seen_order
                seen_order += 1
    return sorted(scores, key=lambda item_id: (-scores[item_id], best_rank[item_id],
                                                first_seen[item_id]))
