"""LongMemEval session-retrieval adapter.

Supports the official V1/cleaned JSON shape. It evaluates the memory layer's retrieval stage using
the benchmark-provided ``answer_session_ids`` evidence labels. It does not claim answer-generation
accuracy: that requires a separately chosen reader LLM and the official evaluator.
"""
from __future__ import annotations

import json
import math
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np

from ..chunking import chunk_text
from ..config import load_config
from ..note_io import atomic_write_text
from ..providers.embeddings import get_embedding_provider
from ..ranking import reciprocal_rank_order
from ..text import collapse_ws, extract_terms


def _turn_text(turn) -> str:
    if isinstance(turn, str):
        return turn
    if not isinstance(turn, dict):
        return ""
    content = turn.get("content") or ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(str(item.get("text") or "") for item in content
                         if isinstance(item, dict) and item.get("text"))
    return str(content)


def _session_chunks(session) -> list[str]:
    turns = session if isinstance(session, list) else [session]
    text = "\n".join(f"{str(turn.get('role') or 'unknown')}: {_turn_text(turn)}"
                     if isinstance(turn, dict) else _turn_text(turn) for turn in turns)
    return [collapse_ws(piece) for piece in chunk_text(text) if collapse_ws(piece)] or [collapse_ws(text)]


def _load(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("LongMemEval input must be a JSON array")
    required = {"question_id", "question", "haystack_sessions", "haystack_session_ids"}
    bad = next((index for index, row in enumerate(data)
                if not isinstance(row, dict) or not required.issubset(row)), None)
    if bad is not None:
        raise ValueError(f"LongMemEval row {bad} is missing required fields")
    return data


def _bm25_rank(query: str, docs: list[str]) -> list[int]:
    tokenized = [extract_terms(doc) for doc in docs]
    qterms = extract_terms(query)
    if not qterms or not docs:
        return list(range(len(docs)))
    df = Counter(term for terms in tokenized for term in set(terms))
    avgdl = statistics.fmean(len(terms) for terms in tokenized) or 1.0
    n_docs = len(docs)
    scores: list[tuple[int, float]] = []
    for index, terms in enumerate(tokenized):
        tf = Counter(terms)
        dl = max(1, len(terms))
        score = 0.0
        for term in qterms:
            freq = tf.get(term, 0)
            if not freq:
                continue
            idf = math.log(1.0 + (n_docs - df[term] + 0.5) / (df[term] + 0.5))
            score += idf * (freq * 2.2) / (freq + 1.2 * (0.25 + 0.75 * dl / avgdl))
        scores.append((index, score))
    return [index for index, _score in sorted(scores, key=lambda item: (-item[1], item[0]))]


def _session_order(chunk_order: Iterable[int], owners: list[str]) -> list[str]:
    out: list[str] = []
    for index in chunk_order:
        session = owners[index]
        if session not in out:
            out.append(session)
    return out


def run(path: Path, *, mode: str = "hybrid", k: int = 10, limit: int | None = None,
        output: Path | None = None) -> dict:
    if mode not in {"bm25", "vector", "hybrid"}:
        raise ValueError("mode must be bm25, vector, or hybrid")
    rows = _load(path)
    if limit is not None:
        rows = rows[:max(0, limit)]
    cfg = load_config()
    provider = get_embedding_provider(cfg) if mode in {"vector", "hybrid"} else None
    if mode in {"vector", "hybrid"} and (provider is None or not provider.available()):
        raise RuntimeError("configured embedding provider is unavailable")

    results: list[dict] = []
    reciprocal_ranks: list[float] = []
    hits = 0
    skipped_abstention = 0
    category_rr: dict[str, list[float]] = defaultdict(list)
    started = time.perf_counter()
    for ordinal, row in enumerate(rows, 1):
        question_id = str(row["question_id"])
        expected = [str(value) for value in row.get("answer_session_ids") or []]
        if not expected:
            skipped_abstention += 1
            results.append({"question_id": question_id, "question_type": row.get("question_type"),
                            "retrieved_session_ids": [], "expected_session_ids": [],
                            "retrieval_scored": False})
            continue
        session_ids = [str(value) for value in row["haystack_session_ids"]]
        sessions = row["haystack_sessions"]
        if len(session_ids) != len(sessions):
            raise ValueError(f"{question_id}: session id/history lengths differ")
        docs: list[str] = []
        owners: list[str] = []
        for session_id, session in zip(session_ids, sessions):
            pieces = _session_chunks(session)
            docs.extend(pieces)
            owners.extend([session_id] * len(pieces))

        rankings: list[list[str]] = []
        if mode in {"bm25", "hybrid"}:
            rankings.append(_session_order(_bm25_rank(row["question"], docs), owners))
        if mode in {"vector", "hybrid"}:
            doc_vectors = np.asarray(provider.embed_documents(docs), dtype=np.float32)
            query_vector = np.asarray(provider.embed_query(row["question"]), dtype=np.float32)
            rankings.append(_session_order(np.argsort(-(doc_vectors @ query_vector)), owners))
        retrieved = (reciprocal_rank_order(rankings, rrf_k=cfg.recall.rrf_k)
                     if mode == "hybrid" else rankings[0])[:k]
        rank = next((index for index, value in enumerate(retrieved, 1) if value in expected), None)
        hit = rank is not None
        hits += int(hit)
        rr = 1.0 / rank if rank else 0.0
        reciprocal_ranks.append(rr)
        category_rr[str(row.get("question_type") or "unknown")].append(rr)
        results.append({"question_id": question_id, "question_type": row.get("question_type"),
                        "retrieved_session_ids": retrieved, "expected_session_ids": expected,
                        "retrieval_scored": True, "hit_at_k": hit, "first_evidence_rank": rank})
        if ordinal % 10 == 0:
            print(f"LongMemEval retrieval: {ordinal}/{len(rows)}")

    scored = len(reciprocal_ranks)
    report = {
        "schema": 1, "benchmark": "LongMemEval", "metric_scope": "session retrieval only",
        "input": str(path), "mode": mode, "k": k, "rows": len(rows), "scored": scored,
        "abstention_rows_unscored": skipped_abstention,
        "recall_at_k": hits / scored if scored else 0.0,
        "mrr_at_k": statistics.fmean(reciprocal_ranks) if reciprocal_ranks else 0.0,
        "by_question_type": {
            category: {"scored": len(values),
                       "recall_at_k": sum(value > 0 for value in values) / len(values),
                       "mrr_at_k": statistics.fmean(values)}
            for category, values in sorted(category_rr.items())
        },
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "embedding_model": provider.name if provider else None,
        "results": results,
        "disclaimer": "Retrieval metrics are not LongMemEval answer-generation accuracy.",
    }
    if output is None:
        stamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
        output = cfg.data_dir / "benchmarks" / f"longmemeval-retrieval-{stamp}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(output, json.dumps(report, indent=2, sort_keys=True) + "\n")
    report["output"] = str(output)
    return report
