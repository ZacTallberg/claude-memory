"""ParadeDB (PostgreSQL 18 + pg_search Tantivy BM25 + pgvector) store — the primary backend.

Embeddings are passed as '[...]'::vector literals (no pgvector-python dependency).
BM25 uses the validated programmatic form: `id @@@ paradedb.match('search_text', %s)`
with `paradedb.score(id)` ranking — safe for arbitrary prompt text.
"""
from __future__ import annotations

import threading
from datetime import datetime
from typing import Iterable, Sequence

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Json

from ..config import Config
from ..log import get_logger
from .base import Candidate, Chunk, Fact, Store

log = get_logger(__name__)


def _vec_literal(vec: list[float]) -> str:
    return "[" + ",".join(f"{x:.7g}" for x in vec) + "]"


class PostgresStore(Store):
    name = "postgres"

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.dim = cfg.embeddings.dim
        pg = cfg.store.postgres
        self._conninfo = (f"host={pg.host} port={pg.port} dbname={pg.dbname} "
                          f"user={pg.user} password={pg.password}")
        self._conn: psycopg.Connection | None = None
        self._lock = threading.RLock()

    # ---- lifecycle ----
    def connect(self) -> None:
        if self._conn is None or self._conn.closed:
            self._conn = psycopg.connect(self._conninfo, autocommit=True, connect_timeout=10)

    def _cur(self):
        self.connect()
        return self._conn.cursor(row_factory=dict_row)

    def close(self) -> None:
        if self._conn and not self._conn.closed:
            self._conn.close()

    def migrate(self) -> None:
        with self._lock, self._cur() as c:
            c.execute("CREATE EXTENSION IF NOT EXISTS pg_search;")
            c.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            try:
                c.execute("CREATE EXTENSION IF NOT EXISTS vectorscale CASCADE;")
            except Exception as e:
                log.info("vectorscale not available (using pgvector hnsw): %s", e)
            d = self.dim
            c.execute("""
                CREATE TABLE IF NOT EXISTS sources(
                    id bigserial PRIMARY KEY, path text UNIQUE NOT NULL, kind text NOT NULL,
                    project text, session_id text, bytes_indexed bigint DEFAULT 0,
                    mtime double precision, first_seen timestamptz DEFAULT now(),
                    last_indexed timestamptz, meta jsonb DEFAULT '{}');""")
            c.execute(f"""
                CREATE TABLE IF NOT EXISTS chunks(
                    id bigserial PRIMARY KEY,
                    source_id bigint REFERENCES sources(id) ON DELETE CASCADE,
                    ordinal int DEFAULT 0, kind text, role text, session_id text, project text,
                    cwd text, ts timestamptz, content text NOT NULL, context_blurb text,
                    search_text text NOT NULL, embedding vector({d}), token_est int,
                    meta jsonb DEFAULT '{{}}');""")
            c.execute(f"""
                CREATE TABLE IF NOT EXISTS facts(
                    id bigserial PRIMARY KEY, path text UNIQUE NOT NULL, project text, name text,
                    title text, description text, type text, tags text[], origin_session_id text,
                    body text, search_text text NOT NULL, embedding vector({d}),
                    mtime double precision, meta jsonb DEFAULT '{{}}');""")
            c.execute("""CREATE TABLE IF NOT EXISTS graph_nodes(id text PRIMARY KEY, label text,
                         type text, grp text);""")
            c.execute("""CREATE TABLE IF NOT EXISTS graph_edges(id bigserial PRIMARY KEY,
                         source text, target text, kind text);""")
            c.execute("""CREATE TABLE IF NOT EXISTS injections(id bigserial PRIMARY KEY,
                         ts timestamptz DEFAULT now(), hook text, session_id text, prompt_excerpt text,
                         n_recalled int, n_facts int, chars int, latency_ms int, details jsonb);""")
            c.execute("""CREATE TABLE IF NOT EXISTS metrics(id bigserial PRIMARY KEY,
                         ts timestamptz DEFAULT now(), metric text, value double precision,
                         run_id text, details jsonb);""")
            c.execute("""CREATE TABLE IF NOT EXISTS promotion_candidates(id bigserial PRIMARY KEY,
                         ts timestamptz DEFAULT now(), title text, body text, type text,
                         support jsonb, score double precision, status text DEFAULT 'pending');""")
            c.execute("""CREATE TABLE IF NOT EXISTS anti_memory(id bigserial PRIMARY KEY,
                         ts timestamptz DEFAULT now(), key text, reason text, chunk_id bigint);""")
            c.execute("""CREATE TABLE IF NOT EXISTS kv(key text PRIMARY KEY, value jsonb);""")
            # BM25 (Tantivy) indexes
            c.execute("""CREATE INDEX IF NOT EXISTS chunks_bm25 ON chunks
                         USING bm25 (id, search_text) WITH (key_field='id');""")
            c.execute("""CREATE INDEX IF NOT EXISTS facts_bm25 ON facts
                         USING bm25 (id, search_text) WITH (key_field='id');""")
            # Vector (pgvector HNSW, cosine)
            c.execute("""CREATE INDEX IF NOT EXISTS chunks_vec ON chunks
                         USING hnsw (embedding vector_cosine_ops);""")
            c.execute("""CREATE INDEX IF NOT EXISTS facts_vec ON facts
                         USING hnsw (embedding vector_cosine_ops);""")
            c.execute("CREATE INDEX IF NOT EXISTS chunks_session ON chunks(session_id);")

    def health(self) -> dict:
        try:
            cnt = self.counts()
            return {"backend": self.name, "ok": True, "vector": True, "bm25": True,
                    "dim": self.dim, **cnt}
        except Exception as e:
            return {"backend": self.name, "ok": False, "error": str(e)}

    # ---- sources & indexing ----
    def get_source(self, path: str) -> dict | None:
        with self._lock, self._cur() as c:
            c.execute("SELECT id, bytes_indexed, mtime, session_id, project FROM sources WHERE path=%s", (path,))
            return c.fetchone()

    def upsert_source(self, *, path, kind, project, session_id, bytes_indexed, mtime, meta=None) -> int:
        with self._lock, self._cur() as c:
            c.execute("""INSERT INTO sources(path, kind, project, session_id, bytes_indexed, mtime,
                              last_indexed, meta) VALUES (%s,%s,%s,%s,%s,%s, now(), %s)
                         ON CONFLICT (path) DO UPDATE SET kind=EXCLUDED.kind, project=EXCLUDED.project,
                              session_id=EXCLUDED.session_id, bytes_indexed=EXCLUDED.bytes_indexed,
                              mtime=EXCLUDED.mtime, last_indexed=now(), meta=EXCLUDED.meta
                         RETURNING id""",
                      (path, kind, project, session_id, bytes_indexed, mtime, Json(meta or {})))
            return c.fetchone()["id"]

    def set_bytes_indexed(self, source_id: int, n: int) -> None:
        with self._lock, self._cur() as c:
            c.execute("UPDATE sources SET bytes_indexed=%s, last_indexed=now() WHERE id=%s", (n, source_id))

    def add_chunks(self, source_id: int, chunks: list[dict]) -> list[int]:
        if not chunks:
            return []
        cols = ("source_id", "ordinal", "kind", "role", "session_id", "project", "cwd", "ts",
                "content", "context_blurb", "search_text", "token_est", "meta")
        rows = []
        params: list = []
        for ch in chunks:
            blurb = ch.get("context_blurb")
            search_text = ((blurb + " ") if blurb else "") + ch["content"]
            params.extend([source_id, ch.get("ordinal", 0), ch.get("kind"), ch.get("role"),
                           ch.get("session_id"), ch.get("project"), ch.get("cwd"), ch.get("ts"),
                           ch["content"], blurb, search_text, ch.get("token_est"),
                           Json(ch.get("meta") or {})])
            rows.append("(" + ",".join(["%s"] * len(cols)) + ")")
        sql = f"INSERT INTO chunks ({','.join(cols)}) VALUES {','.join(rows)} RETURNING id"
        with self._lock, self._cur() as c:
            c.execute(sql, params)
            return [r["id"] for r in c.fetchall()]

    def set_embeddings(self, rows: list[tuple[int, list[float]]]) -> None:
        if not rows:
            return
        # One statement per batch (UPDATE ... FROM VALUES) instead of N round-trips.
        placeholders = ",".join(["(%s,%s)"] * len(rows))
        params: list = []
        for cid, v in rows:
            params.extend([cid, _vec_literal(v)])
        sql = (f"UPDATE chunks AS c SET embedding = d.emb::vector "
               f"FROM (VALUES {placeholders}) AS d(id, emb) WHERE c.id = d.id::bigint")
        with self._lock, self._cur() as c:
            c.execute(sql, params)

    def chunks_missing_embeddings(self, limit: int) -> list[Chunk]:
        with self._lock, self._cur() as c:
            c.execute("""SELECT id, source_id, kind, role, session_id, project, cwd, ts, content,
                                context_blurb, ordinal, meta FROM chunks
                         WHERE embedding IS NULL ORDER BY id LIMIT %s""", (limit,))
            return [self._row_to_chunk(r) for r in c.fetchall()]

    def delete_source(self, path: str) -> None:
        with self._lock, self._cur() as c:
            c.execute("DELETE FROM sources WHERE path=%s", (path,))

    # ---- retrieval ----
    @staticmethod
    def _filter_sql(exclude_session, kinds) -> tuple[str, list]:
        clauses, params = [], []
        if exclude_session:
            clauses.append("(session_id IS NULL OR session_id <> %s)")
            params.append(exclude_session)
        if kinds:
            clauses.append("kind = ANY(%s)")
            params.append(list(kinds))
        return ((" AND " + " AND ".join(clauses)) if clauses else ""), params

    def search_bm25(self, query, k, *, exclude_session=None, kinds=None) -> list[Candidate]:
        if not query.strip():
            return []
        extra, params = self._filter_sql(exclude_session, kinds)
        sql = (f"SELECT id, paradedb.score(id) AS s FROM chunks "
               f"WHERE id @@@ paradedb.match('search_text', %s){extra} "
               f"ORDER BY s DESC LIMIT %s")
        with self._lock, self._cur() as c:
            c.execute(sql, [query, *params, k])
            return [Candidate(chunk_id=r["id"], rank=i + 1, score=float(r["s"]))
                    for i, r in enumerate(c.fetchall())]

    def search_vector(self, qvec, k, *, exclude_session=None, kinds=None) -> list[Candidate]:
        if not qvec:
            return []
        extra, params = self._filter_sql(exclude_session, kinds)
        lit = _vec_literal(qvec)
        sql = (f"SELECT id, 1 - (embedding <=> %s::vector) AS sim FROM chunks "
               f"WHERE embedding IS NOT NULL{extra} "
               f"ORDER BY embedding <=> %s::vector LIMIT %s")
        with self._lock, self._cur() as c:
            c.execute(sql, [lit, *params, lit, k])
            return [Candidate(chunk_id=r["id"], rank=i + 1, score=float(r["sim"]))
                    for i, r in enumerate(c.fetchall())]

    @staticmethod
    def _row_to_chunk(r: dict) -> Chunk:
        return Chunk(id=r["id"], source_id=r["source_id"], kind=r["kind"], role=r["role"],
                     session_id=r["session_id"], project=r["project"], cwd=r["cwd"], ts=r["ts"],
                     content=r["content"], context_blurb=r.get("context_blurb"),
                     ordinal=r.get("ordinal", 0), meta=r.get("meta") or {})

    def get_chunks(self, ids: Iterable[int]) -> list[Chunk]:
        ids = list(ids)
        if not ids:
            return []
        with self._lock, self._cur() as c:
            c.execute("""SELECT id, source_id, kind, role, session_id, project, cwd, ts, content,
                                context_blurb, ordinal, meta FROM chunks WHERE id = ANY(%s)""", (ids,))
            return [self._row_to_chunk(r) for r in c.fetchall()]

    # ---- curated facts ----
    def upsert_fact(self, *, path, project, name, title, description, type, tags, origin_session_id,
                    body, embedding, mtime, meta=None) -> int:
        search_text = " ".join([title or "", description or "", body or ""])
        emb = _vec_literal(embedding) if embedding else None
        with self._lock, self._cur() as c:
            c.execute(f"""INSERT INTO facts(path, project, name, title, description, type, tags,
                              origin_session_id, body, search_text, embedding, mtime, meta)
                          VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,{'%s::vector' if emb else 'NULL'},%s,%s)
                          ON CONFLICT (path) DO UPDATE SET project=EXCLUDED.project, name=EXCLUDED.name,
                              title=EXCLUDED.title, description=EXCLUDED.description, type=EXCLUDED.type,
                              tags=EXCLUDED.tags, origin_session_id=EXCLUDED.origin_session_id,
                              body=EXCLUDED.body, search_text=EXCLUDED.search_text,
                              embedding=EXCLUDED.embedding, mtime=EXCLUDED.mtime, meta=EXCLUDED.meta
                          RETURNING id""",
                      ([path, project, name, title, description, type, tags, origin_session_id,
                        body, search_text] + ([emb] if emb else []) + [mtime, Json(meta or {})]))
            return c.fetchone()["id"]

    @staticmethod
    def _row_to_fact(r: dict) -> Fact:
        return Fact(id=r["id"], path=r["path"], project=r["project"], name=r["name"], title=r["title"],
                    description=r["description"], type=r["type"], tags=list(r.get("tags") or []),
                    origin_session_id=r.get("origin_session_id"), body=r.get("body") or "",
                    mtime=r.get("mtime"), meta=r.get("meta") or {})

    def list_facts(self, *, project=None, type=None) -> list[Fact]:
        clauses, params = [], []
        if project:
            clauses.append("project=%s"); params.append(project)
        if type:
            clauses.append("type=%s"); params.append(type)
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        with self._lock, self._cur() as c:
            c.execute(f"SELECT * FROM facts{where} ORDER BY project, type, title", params)
            return [self._row_to_fact(r) for r in c.fetchall()]

    def search_facts(self, query, k, *, qvec=None) -> list[Fact]:
        out: dict[int, Fact] = {}
        with self._lock, self._cur() as c:
            if query.strip():
                c.execute("""SELECT * FROM facts WHERE id @@@ paradedb.match('search_text', %s)
                             ORDER BY paradedb.score(id) DESC LIMIT %s""", (query, k))
                for r in c.fetchall():
                    out[r["id"]] = self._row_to_fact(r)
            if qvec:
                lit = _vec_literal(qvec)
                c.execute("""SELECT * FROM facts WHERE embedding IS NOT NULL
                             ORDER BY embedding <=> %s::vector LIMIT %s""", (lit, k))
                for r in c.fetchall():
                    out.setdefault(r["id"], self._row_to_fact(r))
        return list(out.values())[:k] if not qvec else list(out.values())[: max(k, len(out))]

    def get_fact(self, fact_id: int) -> Fact | None:
        with self._lock, self._cur() as c:
            c.execute("SELECT * FROM facts WHERE id=%s", (fact_id,))
            r = c.fetchone()
            return self._row_to_fact(r) if r else None

    def delete_fact(self, path: str) -> None:
        with self._lock, self._cur() as c:
            c.execute("DELETE FROM facts WHERE path=%s", (path,))

    def facts_titles_map(self) -> dict[str, list[Fact]]:
        out: dict[str, list[Fact]] = {}
        for f in self.list_facts():
            out.setdefault(f.project, []).append(f)
        return out

    # ---- graph ----
    def replace_graph(self, nodes: list[dict], edges: list[dict]) -> None:
        with self._lock, self._cur() as c:
            c.execute("DELETE FROM graph_edges;")
            c.execute("DELETE FROM graph_nodes;")
            for n in nodes:
                c.execute("""INSERT INTO graph_nodes(id,label,type,grp) VALUES (%s,%s,%s,%s)
                             ON CONFLICT (id) DO UPDATE SET label=EXCLUDED.label, type=EXCLUDED.type,
                             grp=EXCLUDED.grp""", (n["id"], n.get("label"), n.get("type"), n.get("group")))
            for e in edges:
                c.execute("INSERT INTO graph_edges(source,target,kind) VALUES (%s,%s,%s)",
                          (e["source"], e["target"], e.get("kind")))

    def graph(self) -> dict:
        with self._lock, self._cur() as c:
            c.execute("SELECT id,label,type,grp FROM graph_nodes")
            nodes = [{"id": r["id"], "label": r["label"], "type": r["type"], "group": r["grp"]}
                     for r in c.fetchall()]
            c.execute("SELECT source,target,kind FROM graph_edges")
            edges = [dict(r) for r in c.fetchall()]
        return {"nodes": nodes, "edges": edges}

    # ---- promotion ----
    def add_promotion_candidate(self, *, title, body, type, support, score) -> int:
        with self._lock, self._cur() as c:
            c.execute("""INSERT INTO promotion_candidates(title,body,type,support,score)
                         VALUES (%s,%s,%s,%s,%s) RETURNING id""",
                      (title, body, type, Json(support), score))
            return c.fetchone()["id"]

    def list_promotions(self, status=None) -> list[dict]:
        with self._lock, self._cur() as c:
            if status:
                c.execute("SELECT * FROM promotion_candidates WHERE status=%s ORDER BY score DESC", (status,))
            else:
                c.execute("SELECT * FROM promotion_candidates ORDER BY ts DESC")
            return [dict(r) for r in c.fetchall()]

    def update_promotion(self, pid, status) -> None:
        with self._lock, self._cur() as c:
            c.execute("UPDATE promotion_candidates SET status=%s WHERE id=%s", (status, pid))

    # ---- anti-memory ----
    def add_anti_memory(self, *, key, reason, chunk_id=None) -> int:
        with self._lock, self._cur() as c:
            c.execute("INSERT INTO anti_memory(key,reason,chunk_id) VALUES (%s,%s,%s) RETURNING id",
                      (key, reason, chunk_id))
            return c.fetchone()["id"]

    def list_anti_memory(self) -> list[dict]:
        with self._lock, self._cur() as c:
            c.execute("SELECT * FROM anti_memory ORDER BY ts DESC")
            return [dict(r) for r in c.fetchall()]

    # ---- observability ----
    def log_injection(self, *, hook, session_id, prompt_excerpt, n_recalled, n_facts, chars,
                      latency_ms, details=None) -> None:
        with self._lock, self._cur() as c:
            c.execute("""INSERT INTO injections(hook,session_id,prompt_excerpt,n_recalled,n_facts,
                              chars,latency_ms,details) VALUES (%s,%s,%s,%s,%s,%s,%s,%s)""",
                      (hook, session_id, prompt_excerpt, n_recalled, n_facts, chars, latency_ms,
                       Json(details or {})))

    def recent_injections(self, limit=50) -> list[dict]:
        with self._lock, self._cur() as c:
            c.execute("SELECT * FROM injections ORDER BY ts DESC LIMIT %s", (limit,))
            return [dict(r) for r in c.fetchall()]

    def record_metric(self, metric, value, *, run_id=None, details=None) -> None:
        with self._lock, self._cur() as c:
            c.execute("INSERT INTO metrics(metric,value,run_id,details) VALUES (%s,%s,%s,%s)",
                      (metric, value, run_id, Json(details or {})))

    def metric_series(self, metric) -> list[dict]:
        with self._lock, self._cur() as c:
            c.execute("SELECT ts, value, run_id, details FROM metrics WHERE metric=%s ORDER BY ts", (metric,))
            return [dict(r) for r in c.fetchall()]

    # ---- misc ----
    def counts(self) -> dict:
        with self._lock, self._cur() as c:
            c.execute("""SELECT (SELECT count(*) FROM sources) AS sources,
                                (SELECT count(*) FROM chunks) AS chunks,
                                (SELECT count(*) FROM chunks WHERE embedding IS NOT NULL) AS chunks_embedded,
                                (SELECT count(*) FROM facts) AS facts,
                                (SELECT count(*) FROM injections) AS injections""")
            return dict(c.fetchone())

    def kv_get(self, key: str) -> dict | None:
        with self._lock, self._cur() as c:
            c.execute("SELECT value FROM kv WHERE key=%s", (key,))
            r = c.fetchone()
            return r["value"] if r else None

    def kv_set(self, key: str, value: dict) -> None:
        with self._lock, self._cur() as c:
            c.execute("""INSERT INTO kv(key,value) VALUES (%s,%s)
                         ON CONFLICT (key) DO UPDATE SET value=EXCLUDED.value""", (key, Json(value)))
