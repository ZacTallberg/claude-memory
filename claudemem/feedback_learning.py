"""Conservative, offline-only ranking experiments from usefulness feedback.

Delivery receipts identify what reached a client. Feedback identifies outcomes. This module joins
the two by request id, applies stronger credit only when the caller explicitly attributes item ids,
and produces bounded Bayesian weights. Nothing is activated on the prompt path automatically.
"""
from __future__ import annotations

import json
import statistics
import time
from dataclasses import dataclass
from pathlib import Path

from .config import load_config
from .eval import (_fact_hit, _first_fact_rank, _first_session_rank, _score_query,
                   load_golden)
from .note_io import atomic_write_text
from .retriever import Retriever
from .store.base import Chunk, Fact, Store
from .store.factory import get_store
from .text import recall_query

OUTCOME_VALUE = {"helpful": 1.0, "neutral": 0.0, "harmful": -1.0, "stale": -0.75}


def _details(row: dict) -> dict:
    value = row.get("details") or {}
    if isinstance(value, dict):
        return value
    try:
        parsed = json.loads(value)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {}


def _valid_ids(values) -> list[int]:
    """Accept only positive integer evidence ids from untrusted receipt/feedback details."""
    out: list[int] = []
    for value in values if isinstance(values, list) else []:
        try:
            item_id = int(value)
        except (TypeError, ValueError):
            continue
        if item_id > 0 and item_id not in out:
            out.append(item_id)
    return out


@dataclass
class FeedbackProfile:
    chunk_weights: dict[int, float]
    fact_weights: dict[int, float]
    feedback_events: int
    attributed_events: int

    def chunk_multiplier(self, chunk: Chunk) -> float:
        return self.chunk_weights.get(chunk.id, 1.0)

    def fact_multiplier(self, fact: Fact) -> float:
        return self.fact_weights.get(fact.id, 1.0)


def build_profile(store: Store, *, limit: int = 10000) -> tuple[FeedbackProfile, dict]:
    injections: dict[str, dict] = {}
    for row in store.recent_injections(limit):
        details = _details(row)
        request_id = str(details.get("request_id") or "")
        if request_id and request_id not in injections:
            injections[request_id] = details

    observations: dict[tuple[str, int], list[tuple[float, float]]] = {}
    events = attributed = 0
    outcomes = {key: 0 for key in OUTCOME_VALUE}
    for row in store.recent_memory_feedback(limit):
        outcome = str(row.get("outcome") or "")
        if outcome not in OUTCOME_VALUE:
            continue
        events += 1
        outcomes[outcome] += 1
        feedback_details = _details(row)
        delivered = injections.get(str(row.get("request_id") or ""), {})
        explicit_chunks = _valid_ids(feedback_details.get("chunk_ids"))
        explicit_facts = _valid_ids(feedback_details.get("fact_ids"))
        is_explicit = bool(explicit_chunks or explicit_facts)
        chunk_ids = explicit_chunks or _valid_ids(delivered.get("chunk_ids"))
        fact_ids = explicit_facts or _valid_ids(delivered.get("fact_ids"))
        if not (chunk_ids or fact_ids):
            continue
        attributed += 1
        evidence_weight = 1.0 if is_explicit else 0.25
        value = OUTCOME_VALUE[outcome]
        for kind, ids in (("chunk", chunk_ids), ("fact", fact_ids)):
            for item_id in ids:
                observations.setdefault((kind, item_id), []).append((value, evidence_weight))

    item_rows: list[dict] = []
    chunk_weights: dict[int, float] = {}
    fact_weights: dict[int, float] = {}
    for (kind, item_id), values in observations.items():
        weighted_n = sum(weight for _value, weight in values)
        weighted_sum = sum(value * weight for value, weight in values)
        # Zero-centred prior equivalent to three strong observations prevents tiny feedback
        # samples from swinging rank. Final multipliers are capped to +/-15%.
        posterior = weighted_sum / (weighted_n + 3.0)
        multiplier = max(0.85, min(1.15, 1.0 + 0.15 * posterior))
        target = chunk_weights if kind == "chunk" else fact_weights
        target[item_id] = multiplier
        item_rows.append({"kind": kind, "id": item_id, "observations": len(values),
                          "effective_evidence": round(weighted_n, 3),
                          "posterior_utility": round(posterior, 5),
                          "experimental_multiplier": round(multiplier, 5)})

    profile = FeedbackProfile(chunk_weights, fact_weights, events, attributed)
    report = {"feedback_events": events, "attributed_events": attributed,
              "outcomes": outcomes, "items": sorted(item_rows,
                  key=lambda row: (-row["effective_evidence"], row["kind"], row["id"])),
              "activation": "offline-experiment-only"}
    return profile, report


def _eval(retriever: Retriever, *, k: int) -> dict:
    golden = load_golden()
    coverage = strict = violations = negatives = 0
    session_rr: list[float] = []
    fact_rr: list[float] = []
    for row in golden:
        query = recall_query(row.q, row.cwd)
        qvec = retriever.embed_query(query)
        results = retriever.search(query, tier="hot", k=k, qvec=qvec)
        sessions = [result.chunk.session_id for result in results]
        facts = retriever.search_facts(query, k=k, qvec=qvec)
        titles = [f"{fact.title} {fact.name}" for fact in facts]
        coverage += int(_score_query(row, sessions, titles, [k])[k])
        strict += int(_score_query(row, sessions, titles, [k], require_all=True)[k])
        if row.expect_sessions:
            rank = _first_session_rank(row.expect_sessions, sessions)
            session_rr.append(1.0 / rank if rank else 0.0)
        if row.expect_facts:
            rank = _first_fact_rank(row.expect_facts, titles)
            fact_rr.append(1.0 / rank if rank else 0.0)
        if row.reject_sessions or row.reject_facts:
            negatives += 1
            violations += int(any(value in set(row.reject_sessions) for value in sessions)
                              or _fact_hit(row.reject_facts, titles, k))
    n = len(golden) or 1
    rr = session_rr + fact_rr
    return {"n": len(golden), "coverage_at_k": coverage / n, "strict_at_k": strict / n,
            "negative_safety_at_k": 1.0 - (violations / negatives if negatives else 0.0),
            "session_mrr": statistics.fmean(session_rr) if session_rr else None,
            "fact_mrr": statistics.fmean(fact_rr) if fact_rr else None,
            "mean_mrr": statistics.fmean(rr) if rr else 0.0}


def run_experiment(*, min_feedback: int = 5, output: Path | None = None) -> dict:
    cfg = load_config()
    store = get_store(cfg)
    profile, evidence = build_profile(store)
    report = {"schema": 1, "created_at": time.time(), "evidence": evidence,
              "minimum_feedback": min_feedback, "live_change_applied": False}
    if profile.attributed_events < min_feedback:
        report.update({"status": "insufficient-data", "gate_passed": False,
                       "reason": f"need {min_feedback} attributed feedback events"})
    else:
        baseline = _eval(Retriever(cfg, store), k=cfg.recall.top_k)
        candidate = _eval(Retriever(
            cfg, store, chunk_rank_multiplier=profile.chunk_multiplier,
            fact_rank_multiplier=profile.fact_multiplier), k=cfg.recall.top_k)
        gate = (candidate["coverage_at_k"] >= baseline["coverage_at_k"]
                and candidate["strict_at_k"] >= baseline["strict_at_k"]
                and candidate["negative_safety_at_k"] >= baseline["negative_safety_at_k"]
                and candidate["mean_mrr"] > baseline["mean_mrr"])
        report.update({"status": "evaluated", "baseline": baseline, "candidate": candidate,
                       "gate_passed": gate,
                       "gate": "no coverage/strict/safety regression and mean MRR improves"})
    if output is None:
        stamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
        output = cfg.data_dir / "experiments" / f"feedback-ranking-{stamp}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(output, json.dumps(report, indent=2, sort_keys=True) + "\n")
    report["output"] = str(output)
    return report
