"""Configuration loading: built-in defaults < config.toml < CLAUDEMEM_* env vars.

Resolved into a frozen Config dataclass. Secrets (DB password, API keys) only via env.
"""
from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

# Project root = parent of the claudemem package directory.
PKG_DIR = Path(__file__).resolve().parent
ROOT = PKG_DIR.parent
CONFIG_PATH = ROOT / "config.toml"
DATA_DIR = ROOT / "data"


@dataclass(frozen=True)
class ScopeCfg:
    workspace_roots: tuple[str, ...]
    claude_projects_dir: str
    codex_home: str


@dataclass(frozen=True)
class PgCfg:
    host: str
    port: int
    dbname: str
    user: str
    password: str


@dataclass(frozen=True)
class SqliteCfg:
    path: str


@dataclass(frozen=True)
class StoreCfg:
    backend: str
    postgres: PgCfg
    sqlite: SqliteCfg


@dataclass(frozen=True)
class EmbeddingsCfg:
    provider: str
    model: str
    dim: int
    query_prefix: str
    doc_prefix: str


@dataclass(frozen=True)
class RerankerCfg:
    enabled: bool
    provider: str
    model: str
    hot_path: bool
    candidates: int


@dataclass(frozen=True)
class ContextualCfg:
    enabled: bool
    model: str
    enrich_notes: bool
    enrich_transcripts: bool
    max_doc_chars: int


@dataclass(frozen=True)
class RecallCfg:
    top_k: int
    bm25_k: int
    vector_k: int
    rrf_k: int
    min_terms: int
    max_chars: int
    recency_half_life_days: float
    snippet_chars: int
    include_facts: bool
    facts_k: int


@dataclass(frozen=True)
class UnifyCfg:
    max_facts: int
    group_by: str


@dataclass(frozen=True)
class ServerCfg:
    host: str
    port: int
    open_browser: bool


@dataclass(frozen=True)
class IndexCfg:
    exclude_sidechains: bool
    strip_injected: bool
    tool_blobs: bool
    batch_size: int
    transcript_providers: tuple[str, ...]
    live_interval_seconds: int


@dataclass(frozen=True)
class Config:
    root: Path
    data_dir: Path
    scope: ScopeCfg
    store: StoreCfg
    embeddings: EmbeddingsCfg
    reranker: RerankerCfg
    contextual: ContextualCfg
    recall: RecallCfg
    unify: UnifyCfg
    server: ServerCfg
    index: IndexCfg

    @property
    def anthropic_api_key(self) -> str | None:
        return os.environ.get("ANTHROPIC_API_KEY") or None


_DEFAULTS: dict = {
    "scope": {
        "workspace_roots": ["C:/code"],
        "claude_projects_dir": str(Path.home() / ".claude" / "projects").replace("\\", "/"),
        "codex_home": str(Path.home() / ".codex").replace("\\", "/"),
    },
    "store": {
        # "auto" forks the store into diverging copies when the primary flaps (repaired outage —
        # docs/CONTEXT.md); the safe default is one pinned backend, even with no config.toml present.
        "backend": "sqlite",
        "postgres": {"host": "localhost", "port": 55432, "dbname": "claudemem", "user": "claudemem"},
        "sqlite": {"path": "data/claudemem.db"},
    },
    "embeddings": {"provider": "local", "model": "BAAI/bge-small-en-v1.5", "dim": 384,
                   "query_prefix": "", "doc_prefix": ""},
    "reranker": {"enabled": True, "provider": "local", "model": "Xenova/ms-marco-MiniLM-L-6-v2",
                 "hot_path": False, "candidates": 30},
    "contextual": {"enabled": True, "model": "claude-haiku-4-5", "enrich_notes": True,
                   "enrich_transcripts": False, "max_doc_chars": 60000},
    "recall": {"top_k": 6, "bm25_k": 40, "vector_k": 40, "rrf_k": 60, "min_terms": 3,
               "max_chars": 8000, "recency_half_life_days": 45.0, "snippet_chars": 600,
               "include_facts": True, "facts_k": 4},
    "unify": {"max_facts": 120, "group_by": "project"},
    "server": {"host": "127.0.0.1", "port": 7777, "open_browser": True},
    "index": {"exclude_sidechains": True, "strip_injected": True, "tool_blobs": False,
              "batch_size": 64, "transcript_providers": ["claude", "codex"],
              "live_interval_seconds": 60},
}


def _deep_merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _b(v) -> bool:
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in ("1", "true", "yes", "on")


@lru_cache(maxsize=1)
def load_config() -> Config:
    raw = dict(_DEFAULTS)
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, "rb") as f:
            raw = _deep_merge(raw, tomllib.load(f))

    # Environment overrides (highest precedence) for the keys most likely to vary.
    env = os.environ
    if "CLAUDEMEM_BACKEND" in env:
        raw["store"]["backend"] = env["CLAUDEMEM_BACKEND"]
    if "CLAUDEMEM_PG_HOST" in env:
        raw["store"]["postgres"]["host"] = env["CLAUDEMEM_PG_HOST"]
    if "CLAUDEMEM_PG_PORT" in env:
        raw["store"]["postgres"]["port"] = int(env["CLAUDEMEM_PG_PORT"])
    if "CLAUDEMEM_EMBED_MODEL" in env:
        raw["embeddings"]["model"] = env["CLAUDEMEM_EMBED_MODEL"]
    if "CLAUDEMEM_EMBED_DIM" in env:
        raw["embeddings"]["dim"] = int(env["CLAUDEMEM_EMBED_DIM"])
    if "CLAUDEMEM_CODEX_HOME" in env:
        raw["scope"]["codex_home"] = env["CLAUDEMEM_CODEX_HOME"]

    pg_pw = env.get("CLAUDEMEM_PG_PASSWORD", "claudemem")

    s = raw["scope"]
    st = raw["store"]
    em = raw["embeddings"]
    rr = raw["reranker"]
    ct = raw["contextual"]
    rc = raw["recall"]
    un = raw["unify"]
    sv = raw["server"]
    ix = raw["index"]

    return Config(
        root=ROOT,
        data_dir=DATA_DIR,
        scope=ScopeCfg(
            workspace_roots=tuple(s["workspace_roots"]),
            claude_projects_dir=s["claude_projects_dir"],
            codex_home=s["codex_home"],
        ),
        store=StoreCfg(
            backend=st["backend"],
            postgres=PgCfg(host=st["postgres"]["host"], port=int(st["postgres"]["port"]),
                           dbname=st["postgres"]["dbname"], user=st["postgres"]["user"], password=pg_pw),
            sqlite=SqliteCfg(path=st["sqlite"]["path"]),
        ),
        embeddings=EmbeddingsCfg(provider=em["provider"], model=em["model"], dim=int(em["dim"]),
                                 query_prefix=em["query_prefix"], doc_prefix=em["doc_prefix"]),
        reranker=RerankerCfg(enabled=_b(rr["enabled"]), provider=rr["provider"], model=rr["model"],
                             hot_path=_b(rr["hot_path"]), candidates=int(rr["candidates"])),
        contextual=ContextualCfg(enabled=_b(ct["enabled"]), model=ct["model"],
                                 enrich_notes=_b(ct["enrich_notes"]),
                                 enrich_transcripts=_b(ct["enrich_transcripts"]),
                                 max_doc_chars=int(ct["max_doc_chars"])),
        recall=RecallCfg(top_k=int(rc["top_k"]), bm25_k=int(rc["bm25_k"]), vector_k=int(rc["vector_k"]),
                         rrf_k=int(rc["rrf_k"]), min_terms=int(rc["min_terms"]),
                         max_chars=int(rc["max_chars"]),
                         recency_half_life_days=float(rc["recency_half_life_days"]),
                         snippet_chars=int(rc["snippet_chars"]), include_facts=_b(rc["include_facts"]),
                         facts_k=int(rc["facts_k"])),
        unify=UnifyCfg(max_facts=int(un["max_facts"]), group_by=un["group_by"]),
        server=ServerCfg(host=sv["host"], port=int(sv["port"]), open_browser=_b(sv["open_browser"])),
        index=IndexCfg(exclude_sidechains=_b(ix["exclude_sidechains"]), strip_injected=_b(ix["strip_injected"]),
                       tool_blobs=_b(ix["tool_blobs"]), batch_size=int(ix["batch_size"]),
                       transcript_providers=tuple(ix["transcript_providers"]),
                       live_interval_seconds=max(0, int(ix["live_interval_seconds"]))),
    )
