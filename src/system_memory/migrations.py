"""Ordered SQLite schema migrations for canonical memory and disposable indexes."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass


@dataclass(frozen=True)
class Migration:
    version: int
    name: str
    sql: str

    @property
    def checksum(self) -> str:
        return hashlib.sha256(self.sql.encode("utf-8")).hexdigest()


MIGRATIONS: tuple[Migration, ...] = (
    Migration(
        1,
        "canonical_memory",
        r"""
CREATE TABLE IF NOT EXISTS schema_migrations (
    version INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    checksum TEXT NOT NULL,
    applied_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS sources (
    id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    provider TEXT NOT NULL,
    locator TEXT NOT NULL,
    locator_hash TEXT NOT NULL,
    content_hash TEXT,
    cursor INTEGER NOT NULL DEFAULT 0 CHECK(cursor >= 0),
    loss_flags TEXT NOT NULL DEFAULT '[]',
    metadata TEXT NOT NULL DEFAULT '{}',
    first_seen_at TEXT NOT NULL,
    last_seen_at TEXT NOT NULL,
    UNIQUE(kind, provider, locator)
);

CREATE TABLE IF NOT EXISTS episodes (
    id TEXT PRIMARY KEY,
    provider TEXT NOT NULL,
    agent_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    parent_session_id TEXT,
    sequence INTEGER NOT NULL DEFAULT 0 CHECK(sequence >= 0),
    project_id TEXT,
    task_id TEXT,
    hub_instance_id TEXT,
    status TEXT NOT NULL DEFAULT 'open' CHECK(status IN ('open','closed','repaired')),
    started_at TEXT NOT NULL,
    ended_at TEXT,
    summary TEXT,
    summary_model TEXT,
    metadata TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    UNIQUE(provider, agent_id, session_id, sequence)
);

CREATE TABLE IF NOT EXISTS memory_events (
    id TEXT PRIMARY KEY,
    source_id TEXT NOT NULL REFERENCES sources(id) ON DELETE RESTRICT,
    episode_id TEXT NOT NULL REFERENCES episodes(id) ON DELETE RESTRICT,
    provider TEXT NOT NULL,
    provider_event_id TEXT,
    agent_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    parent_session_id TEXT,
    project_id TEXT,
    task_id TEXT,
    hub_instance_id TEXT,
    worktree TEXT,
    commit_sha TEXT,
    role TEXT NOT NULL,
    authority TEXT NOT NULL,
    kind TEXT NOT NULL,
    occurred_at TEXT NOT NULL,
    ingested_at TEXT NOT NULL,
    content TEXT NOT NULL,
    content_sha256 TEXT NOT NULL,
    source_offset_start INTEGER,
    source_offset_end INTEGER,
    visibility TEXT NOT NULL,
    trust TEXT NOT NULL,
    normalizer_version TEXT NOT NULL,
    loss_flags TEXT NOT NULL DEFAULT '[]',
    metadata TEXT NOT NULL DEFAULT '{}',
    UNIQUE(source_id, provider_event_id)
);
CREATE INDEX IF NOT EXISTS memory_events_episode_idx ON memory_events(episode_id, occurred_at);
CREATE INDEX IF NOT EXISTS memory_events_session_idx
    ON memory_events(provider, session_id, occurred_at);
CREATE INDEX IF NOT EXISTS memory_events_project_idx ON memory_events(project_id, occurred_at);
CREATE INDEX IF NOT EXISTS memory_events_hash_idx ON memory_events(content_sha256);

CREATE TABLE IF NOT EXISTS secret_tombstones (
    fingerprint TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    source_id TEXT,
    created_at TEXT NOT NULL,
    reason TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS claim_series (
    id TEXT PRIMARY KEY,
    subject_key TEXT NOT NULL,
    predicate_key TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(subject_key, predicate_key)
);

CREATE TABLE IF NOT EXISTS claim_revisions (
    id TEXT PRIMARY KEY,
    series_id TEXT NOT NULL REFERENCES claim_series(id) ON DELETE RESTRICT,
    revision_no INTEGER NOT NULL CHECK(revision_no > 0),
    operation TEXT NOT NULL CHECK(operation IN ('ADD','MERGE','SUPERSEDE','RETRACT','DISPUTE')),
    state TEXT NOT NULL CHECK(
        state IN ('proposed','accepted','disputed','superseded','retracted','forgotten')
    ),
    subject TEXT NOT NULL,
    predicate TEXT NOT NULL,
    value_json TEXT NOT NULL,
    rendering TEXT NOT NULL,
    authority TEXT NOT NULL,
    confidence REAL NOT NULL CHECK(confidence >= 0 AND confidence <= 1),
    valid_from TEXT,
    valid_to TEXT,
    transaction_at TEXT NOT NULL,
    predecessor_revision_id TEXT REFERENCES claim_revisions(id) ON DELETE RESTRICT,
    created_by TEXT NOT NULL,
    reviewed_by TEXT,
    reviewed_at TEXT,
    content_sha256 TEXT NOT NULL,
    UNIQUE(series_id, revision_no)
);
CREATE INDEX IF NOT EXISTS claim_revisions_state_idx
    ON claim_revisions(series_id, state, transaction_at);

CREATE TABLE IF NOT EXISTS claim_evidence (
    revision_id TEXT NOT NULL REFERENCES claim_revisions(id) ON DELETE CASCADE,
    event_id TEXT NOT NULL REFERENCES memory_events(id) ON DELETE RESTRICT,
    span_start INTEGER NOT NULL DEFAULT 0 CHECK(span_start >= 0),
    span_end INTEGER,
    relation TEXT NOT NULL CHECK(relation IN ('supports','contradicts','context')),
    PRIMARY KEY(revision_id, event_id, span_start, relation)
);

CREATE TABLE IF NOT EXISTS procedure_series (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    scope TEXT NOT NULL DEFAULT 'global',
    created_at TEXT NOT NULL,
    UNIQUE(name, scope)
);

CREATE TABLE IF NOT EXISTS procedure_revisions (
    id TEXT PRIMARY KEY,
    series_id TEXT NOT NULL REFERENCES procedure_series(id) ON DELETE RESTRICT,
    revision_no INTEGER NOT NULL CHECK(revision_no > 0),
    state TEXT NOT NULL CHECK(state IN ('proposed','verified','superseded','retired','forgotten')),
    rendering TEXT NOT NULL,
    preconditions_json TEXT NOT NULL DEFAULT '[]',
    steps_json TEXT NOT NULL DEFAULT '[]',
    expected_outcome TEXT NOT NULL,
    failure_history_json TEXT NOT NULL DEFAULT '[]',
    last_verified_at TEXT,
    verified_by TEXT,
    predecessor_revision_id TEXT REFERENCES procedure_revisions(id) ON DELETE RESTRICT,
    created_at TEXT NOT NULL,
    content_sha256 TEXT NOT NULL,
    UNIQUE(series_id, revision_no)
);

CREATE TABLE IF NOT EXISTS procedure_evidence (
    revision_id TEXT NOT NULL REFERENCES procedure_revisions(id) ON DELETE CASCADE,
    event_id TEXT NOT NULL REFERENCES memory_events(id) ON DELETE RESTRICT,
    span_start INTEGER NOT NULL DEFAULT 0,
    span_end INTEGER,
    relation TEXT NOT NULL DEFAULT 'supports',
    PRIMARY KEY(revision_id, event_id, span_start)
);

CREATE TABLE IF NOT EXISTS core_compilations (
    id TEXT PRIMARY KEY,
    generated_at TEXT NOT NULL,
    token_budget INTEGER NOT NULL CHECK(token_budget > 0),
    content TEXT NOT NULL,
    content_sha256 TEXT NOT NULL,
    receipt_json TEXT NOT NULL,
    active INTEGER NOT NULL DEFAULT 0 CHECK(active IN (0,1))
);
CREATE UNIQUE INDEX IF NOT EXISTS one_active_core_compilation
    ON core_compilations(active) WHERE active = 1;

CREATE TABLE IF NOT EXISTS search_generations (
    id TEXT PRIMARY KEY,
    status TEXT NOT NULL CHECK(status IN ('building','active','retired','failed')),
    created_at TEXT NOT NULL,
    activated_at TEXT,
    corpus_sha256 TEXT NOT NULL,
    normalizer_version TEXT NOT NULL,
    chunker_version TEXT NOT NULL,
    lexical_config_json TEXT NOT NULL,
    embedding_manifest_json TEXT,
    reranker_manifest_json TEXT,
    code_revision TEXT,
    lock_sha256 TEXT,
    receipt_json TEXT NOT NULL DEFAULT '{}'
);
CREATE UNIQUE INDEX IF NOT EXISTS one_active_search_generation
    ON search_generations(status) WHERE status = 'active';

CREATE TABLE IF NOT EXISTS search_documents (
    row_id INTEGER PRIMARY KEY AUTOINCREMENT,
    id TEXT NOT NULL UNIQUE,
    generation_id TEXT NOT NULL REFERENCES search_generations(id) ON DELETE CASCADE,
    memory_type TEXT NOT NULL,
    ref_id TEXT NOT NULL,
    provider TEXT,
    project_id TEXT,
    task_id TEXT,
    session_id TEXT,
    role TEXT,
    authority TEXT NOT NULL,
    occurred_at TEXT,
    valid_from TEXT,
    valid_to TEXT,
    title TEXT NOT NULL,
    body TEXT NOT NULL,
    search_text TEXT NOT NULL,
    content_sha256 TEXT NOT NULL,
    metadata TEXT NOT NULL DEFAULT '{}',
    UNIQUE(generation_id, memory_type, ref_id, content_sha256)
);
CREATE INDEX IF NOT EXISTS search_documents_generation_idx
    ON search_documents(generation_id, memory_type);
CREATE INDEX IF NOT EXISTS search_documents_project_idx
    ON search_documents(generation_id, project_id);

CREATE VIRTUAL TABLE IF NOT EXISTS search_documents_fts USING fts5(
    title,
    search_text,
    content='search_documents',
    content_rowid='row_id',
    tokenize='unicode61 remove_diacritics 2'
);
CREATE TRIGGER IF NOT EXISTS search_documents_ai AFTER INSERT ON search_documents BEGIN
    INSERT INTO search_documents_fts(rowid, title, search_text)
    VALUES (new.row_id, new.title, new.search_text);
END;
CREATE TRIGGER IF NOT EXISTS search_documents_ad AFTER DELETE ON search_documents BEGIN
    INSERT INTO search_documents_fts(search_documents_fts, rowid, title, search_text)
    VALUES ('delete', old.row_id, old.title, old.search_text);
END;
CREATE TRIGGER IF NOT EXISTS search_documents_au AFTER UPDATE ON search_documents BEGIN
    INSERT INTO search_documents_fts(search_documents_fts, rowid, title, search_text)
    VALUES ('delete', old.row_id, old.title, old.search_text);
    INSERT INTO search_documents_fts(rowid, title, search_text)
    VALUES (new.row_id, new.title, new.search_text);
END;

CREATE TABLE IF NOT EXISTS embedding_queue (
    document_id TEXT NOT NULL REFERENCES search_documents(id) ON DELETE CASCADE,
    generation_id TEXT NOT NULL REFERENCES search_generations(id) ON DELETE CASCADE,
    priority INTEGER NOT NULL DEFAULT 100,
    attempts INTEGER NOT NULL DEFAULT 0,
    available_at TEXT NOT NULL,
    claimed_at TEXT,
    last_error_code TEXT,
    PRIMARY KEY(document_id, generation_id)
);

CREATE TABLE IF NOT EXISTS retrieval_receipts (
    request_id TEXT PRIMARY KEY,
    query_sha256 TEXT NOT NULL,
    requested_at TEXT NOT NULL,
    delivered_at TEXT,
    mode TEXT NOT NULL CHECK(mode IN ('hybrid','keyword_only','empty','timeout','shed','error')),
    computed_late INTEGER NOT NULL DEFAULT 0 CHECK(computed_late IN (0,1)),
    generation_id TEXT,
    current_project_id TEXT,
    stage_latency_json TEXT NOT NULL DEFAULT '{}',
    result_ids_json TEXT NOT NULL DEFAULT '[]',
    fallback_reason TEXT,
    index_age_seconds REAL,
    config_sha256 TEXT
);

CREATE TABLE IF NOT EXISTS idempotency (
    scope TEXT NOT NULL,
    key TEXT NOT NULL,
    request_sha256 TEXT NOT NULL,
    response_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    expires_at TEXT NOT NULL,
    PRIMARY KEY(scope, key)
);
""",
    ),
    Migration(
        2,
        "scoped_credentials",
        r"""
CREATE TABLE IF NOT EXISTS credentials (
    id TEXT PRIMARY KEY,
    actor_id TEXT NOT NULL,
    label TEXT NOT NULL,
    token_sha256 TEXT NOT NULL UNIQUE,
    scopes_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    expires_at TEXT,
    revoked_at TEXT,
    last_used_at TEXT,
    metadata TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS credentials_actor_idx ON credentials(actor_id, revoked_at);
""",
    ),
    Migration(
        3,
        "embedding_generations",
        r"""
CREATE TABLE IF NOT EXISTS embedding_vectors (
    generation_id TEXT NOT NULL REFERENCES search_generations(id) ON DELETE CASCADE,
    document_id TEXT NOT NULL REFERENCES search_documents(id) ON DELETE CASCADE,
    dimension INTEGER NOT NULL CHECK(dimension > 0),
    vector BLOB NOT NULL,
    vector_sha256 TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY(generation_id, document_id)
);
CREATE INDEX IF NOT EXISTS embedding_vectors_generation_idx
    ON embedding_vectors(generation_id);
""",
    ),
)
