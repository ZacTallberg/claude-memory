"""Shadow embedding-model bake-off over the local golden set.

The live database is never mutated. Each model embeds the same bounded pool of positive evidence,
lexical hard negatives, and all curated facts. This is a selection instrument, not a leaderboard:
the winner must preserve strict recall and negative safety before rank quality can decide.
"""
from __future__ import annotations

import json
import statistics
import time
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from .config import Config, load_config
from .eval import (_fact_hit, _first_fact_rank, _first_session_rank, _score_query,
                   load_golden)
from .lifecycle import fact_is_recallable
from .note_io import atomic_write_text
from .providers.embeddings import get_embedding_provider
from .store.base import Chunk, Fact, Store
from .store.factory import get_store
from .text import recall_query
from .paths import project_from_cwd


@dataclass(frozen=True)
class ModelSpec:
    model: str
    dim: int
    size_gb: float | None = None
    description: str = ""


def supported_model_specs() -> dict[str, ModelSpec]:
    from fastembed import TextEmbedding
    out: dict[str, ModelSpec] = {}
    for raw in TextEmbedding.list_supported_models():
        name = str(raw.get("model") or "")
        if not name:
            continue
        out[name] = ModelSpec(model=name, dim=int(raw.get("dim") or 0),
                              size_gb=float(raw["size_in_GB"]) if raw.get("size_in_GB") else None,
                              description=str(raw.get("description") or ""))
    return out


def parse_model_spec(value: str, cfg: Config) -> ModelSpec:
    value = str(value or "").strip()
    if value in {"", "current"}:
        native = supported_model_specs().get(cfg.embeddings.model)
        return replace(native, dim=cfg.embeddings.dim) if native else ModelSpec(
            cfg.embeddings.model, cfg.embeddings.dim)
    name, sep, raw_dim = value.rpartition("@")
    supported = supported_model_specs()
    model = name if sep else value
    if model not in supported:
        raise ValueError(f"model {model!r} is not supported by installed FastEmbed")
    native = supported[model]
    dim = int(raw_dim) if sep else native.dim
    if dim != native.dim:
        raise ValueError(
            f"{model!r} native dim is {native.dim}; this harness refuses arbitrary truncation "
            "until validated Matryoshka dimensions are explicitly supported"
        )
    return replace(native, dim=dim)


def _fact_text(fact: Fact) -> str:
    return " ".join((fact.name or "", fact.title or "", fact.project or "",
                     " ".join(fact.tags or []), fact.description or "", fact.body or ""))


def _chunk_text(chunk: Chunk) -> str:
    return ((chunk.context_blurb + " ") if chunk.context_blurb else "") + chunk.content


def _candidate_pool(store: Store, *, pool_k: int) -> tuple[dict[int, Chunk], dict[int, Fact], list]:
    golden = load_golden()
    chunks: dict[int, Chunk] = {}
    expected_sessions = sorted({session for row in golden for session in row.expect_sessions})
    for chunk in store.chunks_for_sessions(expected_sessions, limit_per_session=30):
        chunks[chunk.id] = chunk
    for row in golden:
        query = recall_query(row.q, row.cwd)
        ids = [candidate.chunk_id for candidate in store.search_bm25(query, pool_k)]
        for chunk in store.get_chunks(ids):
            chunks[chunk.id] = chunk
    facts = {fact.id: fact for fact in store.list_facts()}
    return chunks, facts, golden


def _embed(provider, texts: list[str], *, batch: int = 64) -> np.ndarray:
    vectors: list[list[float]] = []
    for start in range(0, len(texts), batch):
        vectors.extend(provider.embed_documents(texts[start:start + batch]))
    return np.asarray(vectors, dtype=np.float32)


def _evaluate_model(cfg: Config, spec: ModelSpec, chunks: dict[int, Chunk],
                    facts: dict[int, Fact], golden: list, *, k: int) -> dict:
    offline_cfg = replace(
        cfg,
        embeddings=replace(cfg.embeddings, provider="local", model=spec.model, dim=spec.dim,
                           document_microbatch_size=128),
    )
    provider = get_embedding_provider(offline_cfg)
    if not provider.available():
        return {"model": spec.model, "dim": spec.dim, "ok": False,
                "error": getattr(provider, "reason", "provider unavailable")}

    chunk_ids = sorted(chunks)
    fact_ids = sorted(facts)
    started = time.perf_counter()
    chunk_vectors = _embed(provider, [_chunk_text(chunks[item]) for item in chunk_ids])
    fact_vectors = _embed(provider, [_fact_text(facts[item]) for item in fact_ids])
    index_ms = int((time.perf_counter() - started) * 1000)

    hits = strict_hits = negative_violations = 0
    negative_total = 0
    session_rr: list[float] = []
    fact_rr: list[float] = []
    query_ms: list[float] = []
    for row in golden:
        query = recall_query(row.q, row.cwd)
        q0 = time.perf_counter()
        qvec = np.asarray(provider.embed_query(query), dtype=np.float32)
        session_scores: dict[str, float] = {}
        if len(chunk_ids):
            scores = chunk_vectors @ qvec
            for index in np.argsort(-scores):
                session = chunks[chunk_ids[int(index)]].session_id
                if session and session not in session_scores:
                    session_scores[session] = float(scores[int(index)])
        sessions = list(session_scores)[:k]
        ranked_facts: list[Fact] = []
        if len(fact_ids):
            fact_scores = fact_vectors @ qvec
            project = project_from_cwd(row.cwd)
            ranked_facts = [facts[fact_ids[int(index)]] for index in np.argsort(-fact_scores)
                            if fact_is_recallable(facts[fact_ids[int(index)]], project=project)][:k]
        titles = [f"{fact.title} {fact.name}" for fact in ranked_facts]
        query_ms.append((time.perf_counter() - q0) * 1000)

        hits += int(_score_query(row, sessions, titles, [k])[k])
        strict_hits += int(_score_query(row, sessions, titles, [k], require_all=True)[k])
        if row.expect_sessions:
            rank = _first_session_rank(row.expect_sessions, sessions)
            session_rr.append(1.0 / rank if rank else 0.0)
        if row.expect_facts:
            rank = _first_fact_rank(row.expect_facts, titles)
            fact_rr.append(1.0 / rank if rank else 0.0)
        if row.reject_sessions or row.reject_facts:
            negative_total += 1
            negative_violations += int(
                any(session in set(row.reject_sessions) for session in sessions)
                or _fact_hit(row.reject_facts, titles, k)
            )

    n = len(golden) or 1
    return {
        "model": spec.model, "dim": spec.dim, "size_gb": spec.size_gb, "ok": True,
        "candidate_chunks": len(chunk_ids), "candidate_facts": len(fact_ids),
        "coverage_at_k": hits / n, "strict_at_k": strict_hits / n,
        "negative_safety_at_k": 1.0 - (negative_violations / negative_total if negative_total else 0.0),
        "session_mrr": statistics.fmean(session_rr) if session_rr else None,
        "fact_mrr": statistics.fmean(fact_rr) if fact_rr else None,
        "mean_mrr": statistics.fmean(session_rr + fact_rr) if session_rr or fact_rr else 0.0,
        "index_ms": index_ms,
        "query_p50_ms": statistics.median(query_ms) if query_ms else None,
        "query_p95_ms": sorted(query_ms)[min(len(query_ms) - 1, int(len(query_ms) * 0.95))]
        if query_ms else None,
    }


def run_bakeoff(models: list[str], *, pool_k: int = 50, k: int | None = None,
                output: Path | None = None) -> dict:
    cfg = load_config()
    store = get_store(cfg)
    k = k or cfg.recall.top_k
    chunks, facts, golden = _candidate_pool(store, pool_k=max(10, pool_k))
    specs: list[ModelSpec] = []
    # A candidate is meaningless without the deployed control, so always include it even if the
    # caller lists only proposed models.
    for value in ["current", *(models or [])]:
        spec = parse_model_spec(value, cfg)
        if (spec.model, spec.dim) not in {(item.model, item.dim) for item in specs}:
            specs.append(spec)
    results: list[dict] = []
    for spec in specs:
        try:
            results.append(_evaluate_model(cfg, spec, chunks, facts, golden, k=k))
        except Exception as exc:
            # One unavailable/corrupt candidate must not discard the deployed control's report.
            results.append({"model": spec.model, "dim": spec.dim, "size_gb": spec.size_gb,
                            "ok": False, "error": f"{type(exc).__name__}: {exc}"})
    healthy = [row for row in results if row.get("ok")]
    baseline = next((row for row in healthy
                     if row["model"] == cfg.embeddings.model and row["dim"] == cfg.embeddings.dim),
                    healthy[0] if healthy else None)
    eligible = [row for row in healthy if baseline is None or (
        row["coverage_at_k"] >= baseline["coverage_at_k"]
        and row["strict_at_k"] >= baseline["strict_at_k"]
        and row["negative_safety_at_k"] >= baseline["negative_safety_at_k"])]
    winner = max(eligible, key=lambda row: (row["mean_mrr"], row["coverage_at_k"],
                                            int(row["model"] == cfg.embeddings.model
                                                and row["dim"] == cfg.embeddings.dim),
                                            -float(row["query_p50_ms"] or 1e12)),
                 default=None)
    report = {
        "schema": 1, "created_at": time.time(), "live_model": cfg.embeddings.model,
        "live_dim": cfg.embeddings.dim, "k": k, "pool_k": pool_k,
        "golden_queries": len(golden), "results": results,
        "selection_rule": "preserve baseline coverage, strict recall, and negative safety, then maximize mean MRR",
        "winner": {key: winner[key] for key in ("model", "dim", "mean_mrr")}
        if winner else None,
        "recommend_change": bool(winner and (winner["model"] != cfg.embeddings.model
                                              or winner["dim"] != cfg.embeddings.dim)),
        "live_change_applied": False,
    }
    if output is None:
        stamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
        output = cfg.data_dir / "experiments" / f"model-bakeoff-{stamp}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(output, json.dumps(report, indent=2, sort_keys=True) + "\n")
    report["output"] = str(output)
    return report
