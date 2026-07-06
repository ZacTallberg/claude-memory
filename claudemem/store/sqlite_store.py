"""SQLite + FTS5 + sqlite-vec fallback store (graceful degradation when Postgres is down).

If sqlite-vec can't load, vector search returns [] and the retriever runs keyword-only.
"""
from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

from ..config import Config
from ..log import get_logger
from ..text import extract_terms
from .base import Candidate, Chunk, Fact, Store

log = get_logger(__name__)


def _iso(ts: datetime | None) -> str | None:
    return ts.isoformat() if ts else None


def _parse_ts(s) -> datetime | None:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s)
    except Exception:
        return None


def _fts_query(text: str) -> str:
    """Build a safe FTS5 MATCH expr from arbitrary text: quoted terms OR'd together."""
    terms = extract_terms(text)[:24]
    if not terms:
        return ""
    return " OR ".join(f'"{t}"' for t in terms)


class SqliteStore(Store):
    name = "sqlite"

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.dim = cfg.embeddings.dim
        p = Path(cfg.store.sqlite.path)
        self.path = p if p.is_absolute() else (cfg.root / p)
        self._conn: sqlite3.Connection | None = None
        self._vec = False
        self._lock = threading.RLock()

    def connect(self) -> None:
        if self._conn is not None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.execute("PRAGMA busy_timeout=5000;")
        self._conn.execute("PRAGMA synchronous=NORMAL;")
        try:
            import sqlite_vec
            self._conn.enable_load_extension(True)
            sqlite_vec.load(self._conn)
            self._conn.enable_load_extension(False)
            self._vec = True
        except Exception as e:
            log.warning("sqlite-vec unavailable (keyword-only): %s", e)
            self._vec = False

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None

    def migrate(self) -> None:
        with self._lock:
            c = self._conn
            c.executescript("""
                CREATE TABLE IF NOT EXISTS sources(id INTEGER PRIMARY KEY, path TEXT UNIQUE, kind TEXT,
                    project TEXT, session_id TEXT, bytes_indexed INTEGER DEFAULT 0, mtime REAL,
                    last_indexed TEXT, meta TEXT DEFAULT '{}');
                CREATE TABLE IF NOT EXISTS chunks(id INTEGER PRIMARY KEY, source_id INTEGER, ordinal INTEGER,
                    kind TEXT, role TEXT, session_id TEXT, project TEXT, cwd TEXT, ts TEXT, content TEXT,
                    context_blurb TEXT, search_text TEXT, token_est INTEGER, meta TEXT DEFAULT '{}');
                CREATE TABLE IF NOT EXISTS facts(id INTEGER PRIMARY KEY, path TEXT UNIQUE, project TEXT,
                    name TEXT, title TEXT, description TEXT, type TEXT, tags TEXT, origin_session_id TEXT,
                    body TEXT, search_text TEXT, mtime REAL, meta TEXT DEFAULT '{}');
                CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(search_text, tokenize='porter');
                CREATE VIRTUAL TABLE IF NOT EXISTS facts_fts USING fts5(search_text, tokenize='porter');
                CREATE TABLE IF NOT EXISTS graph_nodes(id TEXT PRIMARY KEY, label TEXT, type TEXT, grp TEXT);
                CREATE TABLE IF NOT EXISTS graph_edges(id INTEGER PRIMARY KEY, source TEXT, target TEXT, kind TEXT);
                CREATE TABLE IF NOT EXISTS injections(id INTEGER PRIMARY KEY, ts TEXT DEFAULT (datetime('now')),
                    hook TEXT, session_id TEXT, prompt_excerpt TEXT, n_recalled INTEGER, n_facts INTEGER,
                    chars INTEGER, latency_ms INTEGER, details TEXT);
                CREATE TABLE IF NOT EXISTS metrics(id INTEGER PRIMARY KEY, ts TEXT DEFAULT (datetime('now')),
                    metric TEXT, value REAL, run_id TEXT, details TEXT);
                CREATE TABLE IF NOT EXISTS promotion_candidates(id INTEGER PRIMARY KEY,
                    ts TEXT DEFAULT (datetime('now')), title TEXT, body TEXT, type TEXT, support TEXT,
                    score REAL, status TEXT DEFAULT 'pending');
                CREATE TABLE IF NOT EXISTS anti_memory(id INTEGER PRIMARY KEY, ts TEXT DEFAULT (datetime('now')),
                    key TEXT, reason TEXT, chunk_id INTEGER);
                CREATE TABLE IF NOT EXISTS kv(key TEXT PRIMARY KEY, value TEXT);
                CREATE INDEX IF NOT EXISTS chunks_session ON chunks(session_id);
            """)
            if self._vec:
                c.execute(f"CREATE VIRTUAL TABLE IF NOT EXISTS chunks_vec USING vec0(embedding float[{self.dim}]);")
                c.execute(f"CREATE VIRTUAL TABLE IF NOT EXISTS facts_vec USING vec0(embedding float[{self.dim}]);")
            c.commit()

    def health(self) -> dict:
        try:
            return {"backend": self.name, "ok": True, "vector": self._vec, "bm25": True,
                    "dim": self.dim, **self.counts()}
        except Exception as e:
            return {"backend": self.name, "ok": False, "error": str(e)}

    # ---- sources & indexing ----
    def get_source(self, path: str) -> dict | None:
        with self._lock:
            r = self._conn.execute("SELECT id,bytes_indexed,mtime,session_id FROM sources WHERE path=?",
                                   (path,)).fetchone()
            return dict(r) if r else None

    def upsert_source(self, *, path, kind, project, session_id, bytes_indexed, mtime, meta=None) -> int:
        with self._lock:
            cur = self._conn.execute("""INSERT INTO sources(path,kind,project,session_id,bytes_indexed,
                    mtime,last_indexed,meta) VALUES (?,?,?,?,?,?,datetime('now'),?)
                    ON CONFLICT(path) DO UPDATE SET kind=excluded.kind, project=excluded.project,
                    session_id=excluded.session_id, bytes_indexed=excluded.bytes_indexed,
                    mtime=excluded.mtime, last_indexed=datetime('now'), meta=excluded.meta""",
                    (path, kind, project, session_id, bytes_indexed, mtime, json.dumps(meta or {})))
            self._conn.commit()
            r = self._conn.execute("SELECT id FROM sources WHERE path=?", (path,)).fetchone()
            return r["id"]

    def set_bytes_indexed(self, source_id: int, n: int) -> None:
        with self._lock:
            self._conn.execute("UPDATE sources SET bytes_indexed=?, last_indexed=datetime('now') WHERE id=?",
                               (n, source_id))
            self._conn.commit()

    def add_chunks(self, source_id: int, chunks: list[dict]) -> list[int]:
        ids: list[int] = []
        with self._lock:
            for ch in chunks:
                blurb = ch.get("context_blurb")
                search_text = ((blurb + " ") if blurb else "") + ch["content"]
                cur = self._conn.execute("""INSERT INTO chunks(source_id,ordinal,kind,role,session_id,
                        project,cwd,ts,content,context_blurb,search_text,token_est,meta)
                        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                        (source_id, ch.get("ordinal", 0), ch.get("kind"), ch.get("role"),
                         ch.get("session_id"), ch.get("project"), ch.get("cwd"), _iso(ch.get("ts")),
                         ch["content"], blurb, search_text, ch.get("token_est"),
                         json.dumps(ch.get("meta") or {})))
                cid = cur.lastrowid
                self._conn.execute("INSERT INTO chunks_fts(rowid,search_text) VALUES (?,?)", (cid, search_text))
                ids.append(cid)
            self._conn.commit()
        return ids

    def set_embeddings(self, rows: list[tuple[int, list[float]]]) -> None:
        if not rows or not self._vec:
            return
        import sqlite_vec
        with self._lock:
            for cid, vec in rows:
                self._conn.execute("DELETE FROM chunks_vec WHERE rowid=?", (cid,))
                self._conn.execute("INSERT INTO chunks_vec(rowid,embedding) VALUES (?,?)",
                                   (cid, sqlite_vec.serialize_float32(vec)))
            self._conn.commit()

    def chunks_missing_embeddings(self, limit: int) -> list[Chunk]:
        if not self._vec:
            return []
        with self._lock:
            rows = self._conn.execute("""SELECT * FROM chunks WHERE id NOT IN
                    (SELECT rowid FROM chunks_vec) ORDER BY id LIMIT ?""", (limit,)).fetchall()
            return [self._row_to_chunk(r) for r in rows]

    def delete_source(self, path: str) -> None:
        with self._lock:
            r = self._conn.execute("SELECT id FROM sources WHERE path=?", (path,)).fetchone()
            if r:
                sid = r["id"]
                cids = [x["id"] for x in self._conn.execute("SELECT id FROM chunks WHERE source_id=?", (sid,))]
                for cid in cids:
                    self._conn.execute("DELETE FROM chunks_fts WHERE rowid=?", (cid,))
                    if self._vec:
                        self._conn.execute("DELETE FROM chunks_vec WHERE rowid=?", (cid,))
                self._conn.execute("DELETE FROM chunks WHERE source_id=?", (sid,))
                self._conn.execute("DELETE FROM sources WHERE id=?", (sid,))
                self._conn.commit()

    # ---- retrieval ----
    def _filter_ids(self, ids: list[int], exclude_session, kinds) -> set[int]:
        if not ids:
            return set()
        ph = ",".join("?" * len(ids))
        q = f"SELECT id FROM chunks WHERE id IN ({ph})"
        params: list = list(ids)
        if exclude_session:
            q += " AND (session_id IS NULL OR session_id<>?)"
            params.append(exclude_session)
        if kinds:
            q += f" AND kind IN ({','.join('?' * len(kinds))})"
            params.extend(kinds)
        return {r["id"] for r in self._conn.execute(q, params)}

    def search_bm25(self, query, k, *, exclude_session=None, kinds=None) -> list[Candidate]:
        expr = _fts_query(query)
        if not expr:
            return []
        with self._lock:
            rows = self._conn.execute(
                "SELECT rowid, bm25(chunks_fts) AS r FROM chunks_fts WHERE chunks_fts MATCH ? "
                "ORDER BY r LIMIT ?", (expr, k * 4)).fetchall()
            ok = self._filter_ids([r["rowid"] for r in rows], exclude_session, kinds)
            out, rank = [], 0
            for r in rows:
                if r["rowid"] in ok:
                    rank += 1
                    out.append(Candidate(chunk_id=r["rowid"], rank=rank, score=-float(r["r"])))
                    if rank >= k:
                        break
            return out

    def search_vector(self, qvec, k, *, exclude_session=None, kinds=None) -> list[Candidate]:
        if not qvec or not self._vec:
            return []
        import sqlite_vec
        with self._lock:
            rows = self._conn.execute(
                "SELECT rowid, distance FROM chunks_vec WHERE embedding MATCH ? ORDER BY distance LIMIT ?",
                (sqlite_vec.serialize_float32(qvec), k * 4)).fetchall()
            ok = self._filter_ids([r["rowid"] for r in rows], exclude_session, kinds)
            out, rank = [], 0
            for r in rows:
                if r["rowid"] in ok:
                    rank += 1
                    out.append(Candidate(chunk_id=r["rowid"], rank=rank, score=1.0 - float(r["distance"])))
                    if rank >= k:
                        break
            return out

    @staticmethod
    def _row_to_chunk(r) -> Chunk:
        return Chunk(id=r["id"], source_id=r["source_id"], kind=r["kind"], role=r["role"],
                     session_id=r["session_id"], project=r["project"], cwd=r["cwd"],
                     ts=_parse_ts(r["ts"]), content=r["content"], context_blurb=r["context_blurb"],
                     ordinal=r["ordinal"] or 0, meta=json.loads(r["meta"] or "{}"))

    def get_chunks(self, ids: Iterable[int]) -> list[Chunk]:
        ids = list(ids)
        if not ids:
            return []
        ph = ",".join("?" * len(ids))
        with self._lock:
            rows = self._conn.execute(f"SELECT * FROM chunks WHERE id IN ({ph})", ids).fetchall()
            return [self._row_to_chunk(r) for r in rows]

    # ---- curated facts ----
    def upsert_fact(self, *, path, project, name, title, description, type, tags, origin_session_id,
                    body, embedding, mtime, meta=None) -> int:
        search_text = " ".join([title or "", description or "", body or ""])
        with self._lock:
            self._conn.execute("""INSERT INTO facts(path,project,name,title,description,type,tags,
                    origin_session_id,body,search_text,mtime,meta) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
                    ON CONFLICT(path) DO UPDATE SET project=excluded.project, name=excluded.name,
                    title=excluded.title, description=excluded.description, type=excluded.type,
                    tags=excluded.tags, origin_session_id=excluded.origin_session_id, body=excluded.body,
                    search_text=excluded.search_text, mtime=excluded.mtime, meta=excluded.meta""",
                    (path, project, name, title, description, type, json.dumps(tags), origin_session_id,
                     body, search_text, mtime, json.dumps(meta or {})))
            fid = self._conn.execute("SELECT id FROM facts WHERE path=?", (path,)).fetchone()["id"]
            self._conn.execute("DELETE FROM facts_fts WHERE rowid=?", (fid,))
            self._conn.execute("INSERT INTO facts_fts(rowid,search_text) VALUES (?,?)", (fid, search_text))
            if embedding and self._vec:
                import sqlite_vec
                self._conn.execute("DELETE FROM facts_vec WHERE rowid=?", (fid,))
                self._conn.execute("INSERT INTO facts_vec(rowid,embedding) VALUES (?,?)",
                                   (fid, sqlite_vec.serialize_float32(embedding)))
            self._conn.commit()
            return fid

    @staticmethod
    def _row_to_fact(r) -> Fact:
        return Fact(id=r["id"], path=r["path"], project=r["project"], name=r["name"], title=r["title"],
                    description=r["description"], type=r["type"], tags=json.loads(r["tags"] or "[]"),
                    origin_session_id=r["origin_session_id"], body=r["body"] or "", mtime=r["mtime"],
                    meta=json.loads(r["meta"] or "{}"))

    def list_facts(self, *, project=None, type=None) -> list[Fact]:
        q, params = "SELECT * FROM facts", []
        cl = []
        if project:
            cl.append("project=?"); params.append(project)
        if type:
            cl.append("type=?"); params.append(type)
        if cl:
            q += " WHERE " + " AND ".join(cl)
        q += " ORDER BY project, type, title"
        with self._lock:
            return [self._row_to_fact(r) for r in self._conn.execute(q, params)]

    def search_facts(self, query, k, *, qvec=None) -> list[Fact]:
        expr = _fts_query(query)
        out: dict[int, Fact] = {}
        with self._lock:
            if expr:
                rows = self._conn.execute("SELECT rowid FROM facts_fts WHERE facts_fts MATCH ? "
                                          "ORDER BY bm25(facts_fts) LIMIT ?", (expr, k)).fetchall()
                for r in rows:
                    fr = self._conn.execute("SELECT * FROM facts WHERE id=?", (r["rowid"],)).fetchone()
                    if fr:
                        out[fr["id"]] = self._row_to_fact(fr)
            if qvec and self._vec:
                import sqlite_vec
                rows = self._conn.execute("SELECT rowid FROM facts_vec WHERE embedding MATCH ? "
                                          "ORDER BY distance LIMIT ?",
                                          (sqlite_vec.serialize_float32(qvec), k)).fetchall()
                for r in rows:
                    if r["rowid"] not in out:
                        fr = self._conn.execute("SELECT * FROM facts WHERE id=?", (r["rowid"],)).fetchone()
                        if fr:
                            out[fr["id"]] = self._row_to_fact(fr)
        return list(out.values())

    def get_fact(self, fact_id: int) -> Fact | None:
        with self._lock:
            r = self._conn.execute("SELECT * FROM facts WHERE id=?", (fact_id,)).fetchone()
            return self._row_to_fact(r) if r else None

    def delete_fact(self, path: str) -> None:
        with self._lock:
            r = self._conn.execute("SELECT id FROM facts WHERE path=?", (path,)).fetchone()
            if r:
                self._conn.execute("DELETE FROM facts_fts WHERE rowid=?", (r["id"],))
                if self._vec:
                    self._conn.execute("DELETE FROM facts_vec WHERE rowid=?", (r["id"],))
                self._conn.execute("DELETE FROM facts WHERE id=?", (r["id"],))
                self._conn.commit()

    def facts_titles_map(self) -> dict[str, list[Fact]]:
        out: dict[str, list[Fact]] = {}
        for f in self.list_facts():
            out.setdefault(f.project, []).append(f)
        return out

    # ---- graph ----
    def replace_graph(self, nodes, edges) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM graph_edges;")
            self._conn.execute("DELETE FROM graph_nodes;")
            for n in nodes:
                self._conn.execute("INSERT OR REPLACE INTO graph_nodes(id,label,type,grp) VALUES (?,?,?,?)",
                                   (n["id"], n.get("label"), n.get("type"), n.get("group")))
            for e in edges:
                self._conn.execute("INSERT INTO graph_edges(source,target,kind) VALUES (?,?,?)",
                                   (e["source"], e["target"], e.get("kind")))
            self._conn.commit()

    def graph(self) -> dict:
        with self._lock:
            nodes = [{"id": r["id"], "label": r["label"], "type": r["type"], "group": r["grp"]}
                     for r in self._conn.execute("SELECT * FROM graph_nodes")]
            edges = [{"source": r["source"], "target": r["target"], "kind": r["kind"]}
                     for r in self._conn.execute("SELECT * FROM graph_edges")]
        return {"nodes": nodes, "edges": edges}

    # ---- promotion ----
    def add_promotion_candidate(self, *, title, body, type, support, score) -> int:
        with self._lock:
            cur = self._conn.execute("INSERT INTO promotion_candidates(title,body,type,support,score) "
                                     "VALUES (?,?,?,?,?)", (title, body, type, json.dumps(support), score))
            self._conn.commit()
            return cur.lastrowid

    def list_promotions(self, status=None) -> list[dict]:
        with self._lock:
            if status:
                rows = self._conn.execute("SELECT * FROM promotion_candidates WHERE status=? ORDER BY score DESC",
                                          (status,)).fetchall()
            else:
                rows = self._conn.execute("SELECT * FROM promotion_candidates ORDER BY ts DESC").fetchall()
            return [dict(r) for r in rows]

    def update_promotion(self, pid, status) -> None:
        with self._lock:
            self._conn.execute("UPDATE promotion_candidates SET status=? WHERE id=?", (status, pid))
            self._conn.commit()

    # ---- anti-memory ----
    def add_anti_memory(self, *, key, reason, chunk_id=None) -> int:
        with self._lock:
            cur = self._conn.execute("INSERT INTO anti_memory(key,reason,chunk_id) VALUES (?,?,?)",
                                     (key, reason, chunk_id))
            self._conn.commit()
            return cur.lastrowid

    def list_anti_memory(self) -> list[dict]:
        with self._lock:
            return [dict(r) for r in self._conn.execute("SELECT * FROM anti_memory ORDER BY ts DESC")]

    # ---- observability ----
    def log_injection(self, *, hook, session_id, prompt_excerpt, n_recalled, n_facts, chars,
                      latency_ms, details=None) -> None:
        with self._lock:
            self._conn.execute("""INSERT INTO injections(hook,session_id,prompt_excerpt,n_recalled,
                    n_facts,chars,latency_ms,details) VALUES (?,?,?,?,?,?,?,?)""",
                    (hook, session_id, prompt_excerpt, n_recalled, n_facts, chars, latency_ms,
                     json.dumps(details or {})))
            self._conn.commit()

    def recent_injections(self, limit=50) -> list[dict]:
        with self._lock:
            return [dict(r) for r in self._conn.execute(
                "SELECT * FROM injections ORDER BY ts DESC LIMIT ?", (limit,))]

    def record_metric(self, metric, value, *, run_id=None, details=None) -> None:
        with self._lock:
            self._conn.execute("INSERT INTO metrics(metric,value,run_id,details) VALUES (?,?,?,?)",
                               (metric, value, run_id, json.dumps(details or {})))
            self._conn.commit()

    def metric_series(self, metric) -> list[dict]:
        with self._lock:
            return [dict(r) for r in self._conn.execute(
                "SELECT ts,value,run_id,details FROM metrics WHERE metric=? ORDER BY ts", (metric,))]

    # ---- misc ----
    def counts(self) -> dict:
        with self._lock:
            def n(q):
                return self._conn.execute(q).fetchone()[0]
            return {"sources": n("SELECT count(*) FROM sources"),
                    "chunks": n("SELECT count(*) FROM chunks"),
                    "chunks_embedded": n("SELECT count(*) FROM chunks_vec") if self._vec else 0,
                    "facts": n("SELECT count(*) FROM facts"),
                    "injections": n("SELECT count(*) FROM injections")}

    def kv_get(self, key: str) -> dict | None:
        with self._lock:
            r = self._conn.execute("SELECT value FROM kv WHERE key=?", (key,)).fetchone()
            return json.loads(r["value"]) if r else None

    def kv_set(self, key: str, value: dict) -> None:
        with self._lock:
            self._conn.execute("INSERT OR REPLACE INTO kv(key,value) VALUES (?,?)", (key, json.dumps(value)))
            self._conn.commit()
