"""Recall@k golden scorer + drift snapshot.

Loads `eval/golden.jsonl` (one golden query per line), runs the live retriever
(`Retriever.search` + `Retriever.search_facts`) for each query, and scores how
often the expected sessions / curated facts appear in the top-k results.

A golden line is::

    {"q": "<natural query>",
     "expect_sessions": ["<session_id>", ...],   # any-of, may be empty
     "expect_facts": ["<title/name substring>", ...],  # any-of (case-insensitive), may be empty
     "note": "..."}

For each query we compute, at several cutoffs (@1, @3, @k where k=config.recall.top_k):
  - session recall: did any expected session_id land in the top-N chunk results?
  - fact recall:    did any expected fact substring match a top-N curated fact?
  - overall hit:    session-hit OR fact-hit (only the dimensions that are specified count).

The headline metric `recall_at_k` is the mean overall-hit at the full k cutoff.
It is recorded via `store.record_metric` and the run prints the delta vs the
previous recorded run (a cheap drift detector).
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass

from .config import ROOT, load_config
from .log import get_logger
from .retriever import Retriever
from .store.factory import get_store

log = get_logger(__name__)

GOLDEN_PATH = ROOT / "eval" / "golden.jsonl"


@dataclass
class Golden:
    q: str
    expect_sessions: list[str]
    expect_facts: list[str]
    note: str = ""


def load_golden(path=GOLDEN_PATH) -> list[Golden]:
    """Parse golden.jsonl, skipping blank lines and tolerating bad rows."""
    out: list[Golden] = []
    if not path.exists():
        return out
    for ln, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        raw = raw.strip()
        if not raw or raw.startswith("#"):
            continue
        try:
            d = json.loads(raw)
            out.append(Golden(
                q=d["q"],
                expect_sessions=[str(s) for s in d.get("expect_sessions", [])],
                expect_facts=[str(s) for s in d.get("expect_facts", [])],
                note=d.get("note", ""),
            ))
        except Exception as e:  # noqa: BLE001 - one bad line shouldn't abort the run
            log.warning("golden.jsonl line %d skipped: %s", ln, e)
    return out


def _session_hit(expected: list[str], sessions: list[str | None], n: int) -> bool:
    top = {s for s in sessions[:n] if s}
    return any(e in top for e in expected)


def _fact_hit(expected: list[str], titles: list[str], n: int) -> bool:
    top = [t.lower() for t in titles[:n]]
    return any(any(e.lower() in t for t in top) for e in expected)


def _score_query(g: Golden, sessions: list[str | None], facts: list[str],
                 cutoffs: list[int]) -> dict[int, bool]:
    """Per-cutoff overall hit (session-OR-fact, counting only specified dims)."""
    res: dict[int, bool] = {}
    for n in cutoffs:
        hits: list[bool] = []
        if g.expect_sessions:
            hits.append(_session_hit(g.expect_sessions, sessions, n))
        if g.expect_facts:
            hits.append(_fact_hit(g.expect_facts, facts, n))
        # A query with no expectations can't fail; treat as miss-neutral (False).
        res[n] = any(hits) if hits else False
    return res


def run_eval() -> dict:
    cfg = load_config()
    store = get_store(cfg)
    retr = Retriever(cfg, store)
    golden = load_golden()

    if not golden:
        print(f"no golden queries found at {GOLDEN_PATH}")
        return {"n": 0, "recall_at_k": 0.0}

    k = cfg.recall.top_k
    cutoffs = sorted({1, 3, k})
    run_id = f"eval-{int(time.time())}"

    agg = {n: 0 for n in cutoffs}          # overall hits per cutoff
    sess_agg = {n: 0 for n in cutoffs}     # session-only hits (over queries that specify sessions)
    fact_agg = {n: 0 for n in cutoffs}     # fact-only hits (over queries that specify facts)
    sess_total = sum(1 for g in golden if g.expect_sessions)
    fact_total = sum(1 for g in golden if g.expect_facts)

    rows: list[tuple[str, dict[int, bool]]] = []
    t0 = time.time()
    for g in golden:
        results = retr.search(g.q, tier="hot", k=max(cutoffs))
        sessions = [r.chunk.session_id for r in results]
        fact_objs = retr.search_facts(g.q, k=max(cutoffs))
        facts = [f"{f.title} {f.name}" for f in fact_objs]

        per = _score_query(g, sessions, facts, cutoffs)
        rows.append((g.q, per))
        for n in cutoffs:
            if per[n]:
                agg[n] += 1
            if g.expect_sessions and _session_hit(g.expect_sessions, sessions, n):
                sess_agg[n] += 1
            if g.expect_facts and _fact_hit(g.expect_facts, facts, n):
                fact_agg[n] += 1
    elapsed_ms = int((time.time() - t0) * 1000)

    n = len(golden)
    overall = {nn: agg[nn] / n for nn in cutoffs}

    # ---- print table ----
    print(f"\nrecall@k eval — {n} golden queries, k={k}, backend={store.name}, {elapsed_ms} ms\n")
    hdr_cuts = "  ".join(f"@{nn:<4}" for nn in cutoffs)
    print(f"{'query':<52} {hdr_cuts}")
    print("-" * (52 + 1 + len(hdr_cuts)))
    for q, per in rows:
        marks = "  ".join(f"{('  Y ' if per[nn] else '  . '):<5}" for nn in cutoffs)
        print(f"{q[:52]:<52} {marks}")
    print("-" * (52 + 1 + len(hdr_cuts)))
    tot = "  ".join(f"{overall[nn]*100:4.0f}%" for nn in cutoffs)
    print(f"{'OVERALL recall':<52} {tot}")
    if sess_total:
        srow = "  ".join(f"{(sess_agg[nn]/sess_total)*100:4.0f}%" for nn in cutoffs)
        print(f"{f'  sessions (n={sess_total})':<52} {srow}")
    if fact_total:
        frow = "  ".join(f"{(fact_agg[nn]/fact_total)*100:4.0f}%" for nn in cutoffs)
        print(f"{f'  facts (n={fact_total})':<52} {frow}")

    recall_at_k = overall[k]

    # ---- drift vs previous recorded run ----
    prev = store.metric_series("recall_at_k")
    prev_val = prev[-1]["value"] if prev else None

    details = {
        "run_id": run_id, "n": n, "k": k,
        "recall_at_1": overall.get(1, recall_at_k),
        "recall_at_3": overall.get(3, recall_at_k),
        "recall_at_k": recall_at_k,
        "sessions": {str(nn): sess_agg[nn] for nn in cutoffs},
        "facts": {str(nn): fact_agg[nn] for nn in cutoffs},
        "elapsed_ms": elapsed_ms,
    }
    try:
        store.record_metric("recall_at_k", recall_at_k, run_id=run_id, details=details)
    except Exception as e:  # noqa: BLE001 - eval should still report even if persistence fails
        log.warning("record_metric failed: %s", e)

    print()
    if prev_val is None:
        print(f"recall@{k} = {recall_at_k:.3f}  (first recorded run — no prior baseline)")
    else:
        delta = recall_at_k - prev_val
        arrow = "+" if delta >= 0 else ""
        flag = "" if delta >= 0 else "  <-- REGRESSION"
        print(f"recall@{k} = {recall_at_k:.3f}  (prev {prev_val:.3f}, drift {arrow}{delta:.3f}){flag}")

    return details


if __name__ == "__main__":
    run_eval()
