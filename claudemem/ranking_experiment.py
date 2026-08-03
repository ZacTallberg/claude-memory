"""Read-only golden-set sweep for retrieval constants."""
from __future__ import annotations

import json
import time
from dataclasses import replace
from itertools import product
from pathlib import Path

from .config import load_config
from .feedback_learning import _eval
from .note_io import atomic_write_text
from .retriever import Retriever
from .store.factory import get_store


def run_sweep(*, rrf_values: list[int] | None = None,
              half_life_values: list[float] | None = None,
              output: Path | None = None) -> dict:
    cfg = load_config()
    store = get_store(cfg)
    rrf_values = sorted({cfg.recall.rrf_k,
                         *(max(1, int(value)) for value in (rrf_values or [30, 90]))})
    half_life_values = sorted({cfg.recall.recency_half_life_days,
                               *(max(1.0, float(value))
                                 for value in (half_life_values or [21.0, 90.0]))})
    rows: list[dict] = []
    for rrf_k, half_life in product(rrf_values, half_life_values):
        candidate_cfg = replace(cfg, recall=replace(
            cfg.recall, rrf_k=max(1, int(rrf_k)), recency_half_life_days=max(1.0, float(half_life))))
        metrics = _eval(Retriever(candidate_cfg, store), k=cfg.recall.top_k)
        rows.append({"rrf_k": rrf_k, "recency_half_life_days": half_life, **metrics})
    baseline = next(row for row in rows
                    if row["rrf_k"] == cfg.recall.rrf_k
                    and row["recency_half_life_days"] == cfg.recall.recency_half_life_days)
    eligible = [row for row in rows if row["coverage_at_k"] >= baseline["coverage_at_k"]
                and row["strict_at_k"] >= baseline["strict_at_k"]
                and row["negative_safety_at_k"] >= baseline["negative_safety_at_k"]]
    # Prefer the deployed setting when every quality metric ties. A sweep must not recommend
    # operational churn merely because an equally scoring candidate was enumerated first.
    best = max(eligible, key=lambda row: (
        row["mean_mrr"], row["fact_mrr"] or 0.0, row["session_mrr"] or 0.0,
        int(row["rrf_k"] == baseline["rrf_k"]
            and row["recency_half_life_days"] == baseline["recency_half_life_days"]),
    ), default=baseline)
    changed = (best["rrf_k"] != baseline["rrf_k"]
               or best["recency_half_life_days"] != baseline["recency_half_life_days"])
    report = {"schema": 1, "created_at": time.time(), "baseline": baseline,
              "candidates": rows, "recommendation": best,
              "recommend_change": changed,
              "selection_rule": "preserve coverage, strict recall, and negative safety; maximize mean MRR",
              "live_change_applied": False}
    if output is None:
        stamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
        output = cfg.data_dir / "experiments" / f"ranking-sweep-{stamp}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(output, json.dumps(report, indent=2, sort_keys=True) + "\n")
    report["output"] = str(output)
    return report
