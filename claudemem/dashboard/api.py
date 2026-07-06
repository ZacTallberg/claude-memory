"""All dashboard + hook-server endpoints. JSON API for the UI, plus the warm
/api/recall and /api/unify the hooks call, plus a few HTML fragments for HTMX."""
from __future__ import annotations

import asyncio
import threading
import time
from collections import deque

from fastapi import APIRouter, Query, Request
from fastapi.responses import HTMLResponse
from sse_starlette.sse import EventSourceResponse

from claudemem.indexer import index as run_index
from claudemem.paths import killed, set_killed
from claudemem.recall_format import format_recall, format_unify
from claudemem.text import human_age, meaningful_term_count, snippet

from .render import md_to_html
from .state import get_state

router = APIRouter()

# ---- index progress broker (for SSE) ----
_progress: deque[str] = deque(maxlen=1000)
_index_running = False
_index_lock = threading.Lock()


def _push(msg: str) -> None:
    _progress.append(f"{time.strftime('%H:%M:%S')} {msg}")


# ---- serializers ----
def _result_json(x) -> dict:
    c = x.chunk
    return {
        "id": c.id, "kind": "transcript", "project": c.project, "session": c.session_id,
        "role": c.role, "block": c.kind, "ts": c.ts.isoformat() if c.ts else None,
        "age": human_age(c.ts), "score": round(float(x.score), 4), "reranked": x.reranked,
        "title": f"{c.project} · {c.role}/{c.kind}", "snippet": snippet(c.content, 400),
    }


def _fact_brief(f) -> dict:
    return {"id": f.id, "kind": "fact", "type": f.type, "title": f.title,
            "description": f.description, "project": f.project, "path": f.path,
            "tags": f.tags}


# ================= warm hook endpoints =================
@router.post("/api/recall")
async def api_recall(req: Request):
    body = await req.json()
    st = get_state()
    cfg = st.cfg
    prompt = (body.get("prompt") or "")
    session_id = body.get("session_id")
    if meaningful_term_count(prompt) < cfg.recall.min_terms:
        return {"additionalContext": "", "n_recalled": 0, "n_facts": 0}

    def work():
        t0 = time.time()
        tier = "full" if cfg.reranker.hot_path else "hot"
        qv = st.retriever.embed_query(prompt)  # embed once, reuse for chunks + facts
        results = st.retriever.search(prompt, tier=tier, exclude_session=session_id,
                                      k=cfg.recall.top_k, qvec=qv)
        facts = (st.retriever.search_facts(prompt, cfg.recall.facts_k, qvec=qv)
                 if cfg.recall.include_facts else [])
        text = format_recall(results, facts, cfg)
        latency = int((time.time() - t0) * 1000)
        # Log zero-result misses too — hit-rate is unmeasurable otherwise.
        try:
            st.store.log_injection(hook="recall", session_id=session_id, prompt_excerpt=prompt[:200],
                                   n_recalled=len(results), n_facts=len(facts), chars=len(text),
                                   latency_ms=latency)
        except Exception:
            pass
        return {"additionalContext": text, "n_recalled": len(results), "n_facts": len(facts),
                "latency_ms": latency}

    return await asyncio.to_thread(work)


@router.post("/api/unify")
async def api_unify(req: Request):
    body = await req.json()
    st = get_state()
    cfg = st.cfg
    session_id = body.get("session_id")

    def work():
        tmap = st.store.facts_titles_map()
        try:
            pending = len(st.store.list_promotions("pending"))
        except Exception:
            pending = 0
        text = format_unify(tmap, cfg, pending_promotions=pending)
        if text:
            try:
                st.store.log_injection(hook="unify", session_id=session_id, prompt_excerpt="",
                                       n_recalled=0, n_facts=sum(len(v) for v in tmap.values()),
                                       chars=len(text), latency_ms=0)
            except Exception:
                pass
        return {"additionalContext": text}

    return await asyncio.to_thread(work)


# ================= search =================
@router.get("/api/search")
async def api_search(q: str = Query(""), kind: str = Query("all"),
                     k: int = Query(10), rerank: bool = Query(True)):
    st = get_state()
    if not q.strip():
        return {"results": [], "facts": [], "timing_ms": 0}

    def work():
        t0 = time.time()
        results, facts = [], []
        qv = st.retriever.embed_query(q)  # embed once, reuse for chunks + facts
        if kind in ("all", "transcripts"):
            res = st.retriever.search(q, tier=("full" if rerank else "hot"), k=k, do_rerank=rerank, qvec=qv)
            results = [_result_json(x) for x in res]
        if kind in ("all", "facts"):
            facts = [_fact_brief(f) for f in st.retriever.search_facts(q, k, qvec=qv)]
        return {"results": results, "facts": facts, "timing_ms": int((time.time() - t0) * 1000)}

    return await asyncio.to_thread(work)


@router.get("/ui/search", response_class=HTMLResponse)
async def ui_search(q: str = Query(""), rerank: bool = Query(True)):
    if not q.strip():
        return HTMLResponse("")
    data = await api_search(q=q, kind="all", k=10, rerank=rerank)
    rows = []
    for f in data["facts"]:
        rows.append(f'<div class="hit fact"><span class="badge {f["type"]}">{f["type"]}</span>'
                    f'<a class="t" href="#" hx-get="/ui/fact/{f["id"]}" hx-target="#detail">{f["title"]}</a>'
                    f'<div class="d">{f["description"]}</div><div class="m">{f["project"]}</div></div>')
    for r in data["results"]:
        rows.append(f'<div class="hit"><span class="badge data">recall {r["score"]}</span>'
                    f'<div class="t">{r["title"]}</div><div class="d">{r["snippet"]}</div>'
                    f'<div class="m">{r["project"]} · {r["session"]} · {r["age"]}</div></div>')
    return HTMLResponse(f'<div class="meta">{data["timing_ms"]} ms · {len(data["results"])} recalls · '
                        f'{len(data["facts"])} notes</div>' + "".join(rows) or "<div class='meta'>no hits</div>")


# ================= facts / notes =================
@router.get("/api/facts")
async def api_facts(project: str | None = None, type: str | None = None):
    st = get_state()
    return {"facts": [_fact_brief(f) for f in st.store.list_facts(project=project, type=type)]}


@router.get("/api/fact/{fid}")
async def api_fact(fid: int):
    st = get_state()
    f = st.store.get_fact(fid)
    if not f:
        return {"error": "not found"}
    d = _fact_brief(f)
    d["body"] = f.body
    d["body_html"] = md_to_html(f.body)
    d["origin_session_id"] = f.origin_session_id
    return d


@router.get("/ui/fact/{fid}", response_class=HTMLResponse)
async def ui_fact(fid: int):
    st = get_state()
    f = st.store.get_fact(fid)
    if not f:
        return HTMLResponse("<div class='meta'>not found</div>")
    return HTMLResponse(
        f'<div class="fact-detail"><div class="badge {f.type}">{f.type}</div>'
        f'<h2>{f.title}</h2><div class="m">{f.project} · {f.path}</div>'
        f'<div class="body">{md_to_html(f.body)}</div></div>')


# ================= graph / metrics / stats =================
@router.get("/api/graph")
async def api_graph():
    return get_state().store.graph()


@router.get("/api/metrics")
async def api_metrics():
    st = get_state()
    out = {}
    for m in ("chunks", "chunks_embedded", "facts", "sources", "recall_at_k"):
        out[m] = [{"ts": str(r["ts"]), "value": r["value"]} for r in st.store.metric_series(m)]
    return out


@router.get("/api/stats")
async def api_stats():
    st = get_state()
    h = st.store.health()
    h.update({"embedder": getattr(st.embedder, "name", "?"),
              "embedder_available": st.embedder.available(),
              "reranker": getattr(st.reranker, "name", "noop"),
              "reranker_available": st.reranker.available(),
              "killswitch": killed()})
    return h


# ================= injections (audit window) =================
@router.get("/api/injections")
async def api_injections(limit: int = 50):
    return {"injections": [dict(r, ts=str(r.get("ts"))) for r in
                           get_state().store.recent_injections(limit)]}


@router.get("/api/injections/stream")
async def api_injections_stream():
    async def gen():
        last = None
        while True:
            rows = get_state().store.recent_injections(1)
            if rows and rows[0].get("id") != last:
                last = rows[0].get("id")
                import json as _j
                yield {"event": "injection", "data": _j.dumps(dict(rows[0], ts=str(rows[0].get("ts"))))}
            await asyncio.sleep(2)
    return EventSourceResponse(gen())


# ================= promotions / anti-memory =================
@router.get("/api/promotions")
async def api_promotions(status: str | None = None):
    return {"promotions": get_state().store.list_promotions(status)}


@router.post("/api/promotions/{pid}")
async def api_promotion_act(pid: int, req: Request):
    body = await req.json()
    action = body.get("action")
    st = get_state()
    if action == "accept":
        # accepting writes a draft note into the chosen memory dir then re-indexes it
        from .promote_apply import accept_candidate
        accept_candidate(st.cfg, st.store, pid, body.get("project"))
    elif action == "reject":
        st.store.update_promotion(pid, "rejected")
    return {"ok": True}


@router.get("/api/anti")
async def api_anti():
    return {"anti": get_state().store.list_anti_memory()}


@router.post("/api/anti")
async def api_anti_add(req: Request):
    body = await req.json()
    get_state().store.add_anti_memory(key=body.get("key", ""), reason=body.get("reason", ""),
                                      chunk_id=body.get("chunk_id"))
    return {"ok": True}


# ================= kill switch =================
@router.get("/api/killswitch")
async def api_killswitch_get():
    return {"on": killed()}


@router.post("/api/killswitch")
async def api_killswitch_set(req: Request):
    body = await req.json()
    set_killed(bool(body.get("on")))
    return {"on": killed()}


# ================= index control =================
@router.post("/api/index")
async def api_index(req: Request):
    global _index_running
    body = await req.json() if req.headers.get("content-type", "").startswith("application/json") else {}
    full = bool(body.get("full"))
    with _index_lock:
        if _index_running:
            return {"started": False, "reason": "already running"}
        _index_running = True

    def job():
        global _index_running
        st = get_state()
        try:
            _push("index started" + (" (full)" if full else ""))
            run_index(st.cfg, st.store, full=full, progress=_push)
            _push("index complete")
        except Exception as e:
            _push(f"index error: {e}")
        finally:
            _index_running = False

    threading.Thread(target=job, daemon=True).start()
    return {"started": True}


@router.get("/api/index/stream")
async def api_index_stream():
    async def gen():
        idx = 0
        while True:
            while idx < len(_progress):
                yield {"event": "progress", "data": _progress[idx]}
                idx += 1
            if not _index_running and idx >= len(_progress):
                yield {"event": "done", "data": "idle"}
            await asyncio.sleep(1)
    return EventSourceResponse(gen())
