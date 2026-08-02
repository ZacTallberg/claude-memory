from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np

from .archive import CanonicalArchive
from .clock import utc_iso
from .database import Database
from .ids import content_hash, document_id, episode_id, event_id, source_id, stable_id
from .models import ClaimOperation, ClaimProposal, EmbeddingManifest, IngestEvent, IngestResult
from .normalize import NORMALIZER_VERSION, normalize_authored_text
from .security import redact_secrets, sanitize_structure


class IdentityConflict(RuntimeError):
    pass


class EvidenceError(RuntimeError):
    pass


@dataclass(frozen=True)
class SearchHit:
    document_id: str
    memory_type: str
    ref_id: str
    provider: str | None
    project_id: str | None
    task_id: str | None
    session_id: str | None
    role: str | None
    authority: str
    occurred_at: str | None
    title: str
    body: str
    content_sha256: str
    lexical_score: float
    exact_score: float
    project_boost: float
    vector_score: float = 0.0
    fusion_score: float = 0.0

    @property
    def score(self) -> float:
        base = self.fusion_score if self.fusion_score else self.lexical_score + self.vector_score
        return base + self.exact_score + self.project_boost


class MemoryStore:
    def __init__(self, database: Database, archive: CanonicalArchive) -> None:
        self.database = database
        self.archive = archive

    def initialize(self) -> int:
        return self.database.migrate()

    def ingest(self, incoming: IngestEvent) -> IngestResult:
        normalized = normalize_authored_text(incoming.content)
        if normalized.dropped:
            raise ValueError("event contains no memory-eligible authored content")

        safe_locator, locator_findings = redact_secrets(incoming.source_locator)
        safe_metadata, metadata_findings = sanitize_structure(incoming.metadata)
        safe_worktree, worktree_findings = redact_secrets(incoming.worktree or "")
        all_findings = (
            normalized.secret_findings + locator_findings + metadata_findings + worktree_findings
        )
        body_hash = content_hash(normalized.text)
        src_id = source_id(incoming.source_kind, incoming.provider, safe_locator)
        ep_id = episode_id(
            incoming.provider,
            incoming.agent_id,
            incoming.session_id,
            incoming.episode_sequence,
        )
        occurred = utc_iso(incoming.occurred_at)
        evt_id = event_id(
            incoming.provider,
            src_id,
            incoming.provider_event_id,
            incoming.role.value,
            incoming.kind.value,
            occurred,
            body_hash,
        )
        now = utc_iso()
        archive_payload = {
            "schema_version": 1,
            "event_id": evt_id,
            "source_id": src_id,
            "episode_id": ep_id,
            "provider": incoming.provider,
            "provider_event_id": incoming.provider_event_id,
            "agent_id": incoming.agent_id,
            "session_id": incoming.session_id,
            "parent_session_id": incoming.parent_session_id,
            "project_id": incoming.project_id,
            "task_id": incoming.task_id,
            "hub_instance_id": incoming.hub_instance_id,
            "role": incoming.role.value,
            "authority": incoming.authority.value,
            "kind": incoming.kind.value,
            "occurred_at": occurred,
            "content": normalized.text,
            "content_sha256": body_hash,
            "normalizer_version": NORMALIZER_VERSION,
            "loss_flags": list(incoming.loss_flags),
            "metadata": safe_metadata,
        }
        # Archive first. A crash can leave an unreferenced immutable blob, but can never
        # commit a canonical event whose archive payload does not already exist.
        archive_path = self.archive.put_event(evt_id, archive_payload)
        metadata = dict(safe_metadata)
        metadata["archive_path"] = str(archive_path)

        inserted = False
        with self.database.write() as connection:
            connection.execute(
                """INSERT INTO sources(
                       id,kind,provider,locator,locator_hash,content_hash,cursor,loss_flags,
                       metadata,first_seen_at,last_seen_at
                   ) VALUES (?,?,?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(id) DO UPDATE SET
                       content_hash=COALESCE(excluded.content_hash,sources.content_hash),
                       cursor=MAX(sources.cursor,excluded.cursor),
                       loss_flags=excluded.loss_flags,
                       metadata=excluded.metadata,
                       last_seen_at=excluded.last_seen_at""",
                (
                    src_id,
                    incoming.source_kind,
                    incoming.provider,
                    safe_locator,
                    content_hash(safe_locator),
                    None,
                    incoming.source_offset_end or 0,
                    json.dumps(incoming.loss_flags),
                    json.dumps({"latest_archive": str(archive_path)}, sort_keys=True),
                    now,
                    now,
                ),
            )
            connection.execute(
                """INSERT INTO episodes(
                       id,provider,agent_id,session_id,parent_session_id,sequence,project_id,
                       task_id,hub_instance_id,status,started_at,metadata,created_at
                   ) VALUES (?,?,?,?,?,?,?,?,?,'open',?,?,?)
                   ON CONFLICT(id) DO UPDATE SET
                       parent_session_id=COALESCE(episodes.parent_session_id,excluded.parent_session_id),
                       project_id=COALESCE(episodes.project_id,excluded.project_id),
                       task_id=COALESCE(episodes.task_id,excluded.task_id),
                       hub_instance_id=COALESCE(episodes.hub_instance_id,excluded.hub_instance_id)""",
                (
                    ep_id,
                    incoming.provider,
                    incoming.agent_id,
                    incoming.session_id,
                    incoming.parent_session_id,
                    incoming.episode_sequence,
                    incoming.project_id,
                    incoming.task_id,
                    incoming.hub_instance_id,
                    occurred,
                    json.dumps({}, sort_keys=True),
                    now,
                ),
            )
            existing = connection.execute(
                "SELECT content_sha256 FROM memory_events WHERE id=?", (evt_id,)
            ).fetchone()
            if existing:
                if existing["content_sha256"] != body_hash:
                    raise IdentityConflict(f"event identity conflict: {evt_id}")
            else:
                connection.execute(
                    """INSERT INTO memory_events(
                           id,source_id,episode_id,provider,provider_event_id,agent_id,session_id,
                           parent_session_id,project_id,task_id,hub_instance_id,worktree,commit_sha,
                           role,authority,kind,occurred_at,ingested_at,content,content_sha256,
                           source_offset_start,source_offset_end,visibility,trust,normalizer_version,
                           loss_flags,metadata
                       ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (
                        evt_id,
                        src_id,
                        ep_id,
                        incoming.provider,
                        incoming.provider_event_id,
                        incoming.agent_id,
                        incoming.session_id,
                        incoming.parent_session_id,
                        incoming.project_id,
                        incoming.task_id,
                        incoming.hub_instance_id,
                        safe_worktree or None,
                        incoming.commit_sha,
                        incoming.role.value,
                        incoming.authority.value,
                        incoming.kind.value,
                        occurred,
                        now,
                        normalized.text,
                        body_hash,
                        incoming.source_offset_start,
                        incoming.source_offset_end,
                        incoming.visibility,
                        incoming.trust,
                        NORMALIZER_VERSION,
                        json.dumps(incoming.loss_flags),
                        json.dumps(metadata, ensure_ascii=False, sort_keys=True),
                    ),
                )
                inserted = True
            for finding in all_findings:
                connection.execute(
                    """INSERT OR IGNORE INTO secret_tombstones(
                           fingerprint,kind,source_id,created_at,reason
                       ) VALUES (?,?,?,?,?)""",
                    (finding.fingerprint, finding.kind, src_id, now, "redacted during ingestion"),
                )

        return IngestResult(
            event_id=evt_id,
            source_id=src_id,
            episode_id=ep_id,
            inserted=inserted,
            redaction_count=len(all_findings),
            content_sha256=body_hash,
        )

    def close_episode(self, episode: str, *, ended_at: datetime | None = None) -> bool:
        with self.database.write() as connection:
            changed = connection.execute(
                "UPDATE episodes SET status='closed',ended_at=? WHERE id=? AND status='open'",
                (utc_iso(ended_at), episode),
            ).rowcount
        return changed == 1

    def propose_claim(self, proposal: ClaimProposal) -> str | None:
        if proposal.operation == ClaimOperation.NOOP:
            return None
        subject_key = self._key(proposal.subject)
        predicate_key = self._key(proposal.predicate)
        series = proposal.series_id or stable_id("claim", subject_key, predicate_key)
        now = utc_iso()
        value_json = json.dumps(proposal.value, ensure_ascii=False, sort_keys=True)
        rendering = normalize_authored_text(proposal.rendering).text
        with self.database.write() as connection:
            connection.execute(
                "INSERT OR IGNORE INTO claim_series(id,subject_key,predicate_key,created_at) "
                "VALUES (?,?,?,?)",
                (series, subject_key, predicate_key, now),
            )
            row = connection.execute(
                "SELECT COALESCE(MAX(revision_no),0)+1 AS next "
                "FROM claim_revisions WHERE series_id=?",
                (series,),
            ).fetchone()
            revision_no = int(row["next"])
            revision_id = stable_id(
                "claimrev",
                series,
                revision_no,
                proposal.operation.value,
                content_hash(rendering),
            )
            if proposal.predecessor_revision_id:
                predecessor = connection.execute(
                    "SELECT series_id FROM claim_revisions WHERE id=?",
                    (proposal.predecessor_revision_id,),
                ).fetchone()
                if not predecessor or predecessor["series_id"] != series:
                    raise EvidenceError("claim predecessor is absent or belongs to another series")
            evidence_rows = []
            for evidence in proposal.evidence:
                event = connection.execute(
                    "SELECT length(content) AS size FROM memory_events WHERE id=?",
                    (evidence.event_id,),
                ).fetchone()
                if not event:
                    raise EvidenceError(f"missing evidence event {evidence.event_id}")
                span_end = (
                    evidence.span_end if evidence.span_end is not None else int(event["size"])
                )
                if evidence.span_start > span_end or span_end > int(event["size"]):
                    raise EvidenceError(f"invalid evidence span for {evidence.event_id}")
                evidence_rows.append(
                    (
                        revision_id,
                        evidence.event_id,
                        evidence.span_start,
                        span_end,
                        evidence.relation,
                    )
                )
            connection.execute(
                """INSERT INTO claim_revisions(
                       id,series_id,revision_no,operation,state,subject,predicate,value_json,
                       rendering,authority,confidence,valid_from,valid_to,transaction_at,
                       predecessor_revision_id,created_by,content_sha256
                   ) VALUES (?,?,?,?,'proposed',?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    revision_id,
                    series,
                    revision_no,
                    proposal.operation.value,
                    proposal.subject,
                    proposal.predicate,
                    value_json,
                    rendering,
                    proposal.authority.value,
                    proposal.confidence,
                    utc_iso(proposal.valid_from) if proposal.valid_from else None,
                    utc_iso(proposal.valid_to) if proposal.valid_to else None,
                    now,
                    proposal.predecessor_revision_id,
                    proposal.created_by,
                    content_hash(rendering),
                ),
            )
            connection.executemany(
                """INSERT INTO claim_evidence(
                       revision_id,event_id,span_start,span_end,relation
                   ) VALUES (?,?,?,?,?)""",
                evidence_rows,
            )
        return revision_id

    def accept_claim(self, revision_id: str, *, reviewer: str) -> None:
        if not reviewer.strip():
            raise ValueError("reviewer is required")
        with self.database.write() as connection:
            revision = connection.execute(
                "SELECT * FROM claim_revisions WHERE id=?", (revision_id,)
            ).fetchone()
            if not revision:
                raise EvidenceError("claim revision does not exist")
            if revision["state"] == "accepted":
                return
            if revision["state"] != "proposed":
                raise EvidenceError(f"cannot accept claim in state {revision['state']}")
            operation = revision["operation"]
            predecessor_id = revision["predecessor_revision_id"]
            if operation in (ClaimOperation.SUPERSEDE.value, ClaimOperation.RETRACT.value):
                if not predecessor_id:
                    raise EvidenceError(f"{operation} requires a predecessor")
                predecessor = connection.execute(
                    "SELECT state FROM claim_revisions WHERE id=?", (predecessor_id,)
                ).fetchone()
                if not predecessor or predecessor["state"] != "accepted":
                    raise EvidenceError("predecessor must be an accepted revision")
                predecessor_state = "superseded" if operation == "SUPERSEDE" else "retracted"
                connection.execute(
                    "UPDATE claim_revisions SET state=?,valid_to=COALESCE(valid_to,?) WHERE id=?",
                    (
                        predecessor_state,
                        revision["valid_from"] or revision["transaction_at"],
                        predecessor_id,
                    ),
                )
            connection.execute(
                "UPDATE claim_revisions SET state='accepted',reviewed_by=?,reviewed_at=? "
                "WHERE id=?",
                (reviewer, utc_iso(), revision_id),
            )

    def create_search_generation(
        self,
        *,
        corpus_sha256: str,
        chunker_version: str,
        lexical_config: dict[str, Any] | None = None,
        embedding_manifest: dict[str, Any] | None = None,
        reranker_manifest: dict[str, Any] | None = None,
        code_revision: str | None = None,
        lock_sha256: str | None = None,
    ) -> str:
        parsed_embedding = (
            EmbeddingManifest.model_validate(embedding_manifest).model_dump(mode="json")
            if embedding_manifest
            else None
        )
        manifest = {
            "corpus_sha256": corpus_sha256,
            "chunker_version": chunker_version,
            "normalizer_version": NORMALIZER_VERSION,
            "lexical": lexical_config or {},
            "embedding": parsed_embedding,
            "reranker": reranker_manifest,
            "code_revision": code_revision,
            "lock_sha256": lock_sha256,
        }
        generation = stable_id("gen", manifest)
        with self.database.write() as connection:
            connection.execute(
                """INSERT OR IGNORE INTO search_generations(
                       id,status,created_at,corpus_sha256,normalizer_version,chunker_version,
                       lexical_config_json,embedding_manifest_json,reranker_manifest_json,
                       code_revision,lock_sha256,receipt_json
                   ) VALUES (?,'building',?,?,?,?,?,?,?,?,?,?)""",
                (
                    generation,
                    utc_iso(),
                    corpus_sha256,
                    NORMALIZER_VERSION,
                    chunker_version,
                    json.dumps(lexical_config or {}, sort_keys=True),
                    json.dumps(parsed_embedding, sort_keys=True) if parsed_embedding else None,
                    json.dumps(reranker_manifest, sort_keys=True) if reranker_manifest else None,
                    code_revision,
                    lock_sha256,
                    json.dumps(manifest, sort_keys=True),
                ),
            )
        return generation

    def index_event(self, event: str, generation: str) -> str:
        with self.database.write() as connection:
            target = connection.execute(
                "SELECT * FROM search_generations WHERE id=?", (generation,)
            ).fetchone()
            if not target or target["status"] != "building":
                raise ValueError("event documents can only be added to a building generation")
            row = connection.execute("SELECT * FROM memory_events WHERE id=?", (event,)).fetchone()
            if not row:
                raise ValueError("event does not exist")
            doc_id = document_id("event", event, row["content_sha256"])
            title = f"{row['provider']} {row['role']} · {row['occurred_at']}"
            search_text = self._search_text(title, row["content"], row)
            connection.execute(
                """INSERT OR IGNORE INTO search_documents(
                       id,generation_id,memory_type,ref_id,provider,project_id,task_id,session_id,
                       role,authority,occurred_at,title,body,search_text,content_sha256,metadata
                   ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    doc_id,
                    generation,
                    "event",
                    event,
                    row["provider"],
                    row["project_id"],
                    row["task_id"],
                    row["session_id"],
                    row["role"],
                    row["authority"],
                    row["occurred_at"],
                    title,
                    row["content"],
                    search_text,
                    row["content_sha256"],
                    row["metadata"],
                ),
            )
            if target["embedding_manifest_json"]:
                connection.execute(
                    """INSERT OR IGNORE INTO embedding_queue(
                           document_id,generation_id,priority,attempts,available_at
                       ) VALUES (?,?,100,0,?)""",
                    (doc_id, generation, utc_iso()),
                )
        return doc_id

    def embedding_manifest(self, generation: str) -> EmbeddingManifest | None:
        with self.database.read() as connection:
            row = connection.execute(
                "SELECT embedding_manifest_json FROM search_generations WHERE id=?",
                (generation,),
            ).fetchone()
        if not row or not row["embedding_manifest_json"]:
            return None
        return EmbeddingManifest.model_validate(json.loads(row["embedding_manifest_json"]))

    def pending_embedding_batch(self, generation: str, *, limit: int = 8) -> list[tuple[str, str]]:
        with self.database.read() as connection:
            rows = connection.execute(
                """SELECT d.id,d.body
                   FROM embedding_queue q
                   JOIN search_documents d ON d.id=q.document_id
                   WHERE q.generation_id=? AND q.available_at<=?
                   ORDER BY q.priority,q.available_at,q.document_id
                   LIMIT ?""",
                (generation, utc_iso(), max(1, min(limit, 8))),
            ).fetchall()
        return [(row["id"], row["body"]) for row in rows]

    def put_embeddings(self, generation: str, vectors: dict[str, list[float]]) -> int:
        manifest = self.embedding_manifest(generation)
        if not manifest:
            raise ValueError("generation has no embedding manifest")
        prepared: list[tuple[str, bytes, str]] = []
        for document, values in vectors.items():
            vector = np.asarray(values, dtype="<f4")
            if vector.ndim != 1 or int(vector.shape[0]) != manifest.dimension:
                raise ValueError(
                    f"embedding dimension mismatch for {document}: "
                    f"got={vector.shape} expected={manifest.dimension}"
                )
            if not np.isfinite(vector).all():
                raise ValueError(f"embedding contains non-finite values for {document}")
            norm = float(np.linalg.norm(vector))
            if norm <= 0:
                raise ValueError(f"embedding has zero norm for {document}")
            if manifest.normalized:
                vector = vector / norm
            payload = vector.astype("<f4", copy=False).tobytes()
            prepared.append((document, payload, hashlib.sha256(payload).hexdigest()))
        with self.database.write() as connection:
            for document, payload, digest in prepared:
                exists = connection.execute(
                    "SELECT 1 FROM search_documents WHERE id=? AND generation_id=?",
                    (document, generation),
                ).fetchone()
                if not exists:
                    raise ValueError(f"document {document} is not in generation {generation}")
                connection.execute(
                    """INSERT INTO embedding_vectors(
                           generation_id,document_id,dimension,vector,vector_sha256,created_at
                       ) VALUES (?,?,?,?,?,?)
                       ON CONFLICT(generation_id,document_id) DO UPDATE SET
                           dimension=excluded.dimension,vector=excluded.vector,
                           vector_sha256=excluded.vector_sha256,created_at=excluded.created_at""",
                    (
                        generation,
                        document,
                        manifest.dimension,
                        payload,
                        digest,
                        utc_iso(),
                    ),
                )
                connection.execute(
                    "DELETE FROM embedding_queue WHERE generation_id=? AND document_id=?",
                    (generation, document),
                )
        return len(prepared)

    def vector_records(self, generation: str) -> list[dict[str, Any]]:
        with self.database.read() as connection:
            rows = connection.execute(
                """SELECT d.*,v.dimension,v.vector,v.vector_sha256
                   FROM embedding_vectors v
                   JOIN search_documents d ON d.id=v.document_id
                   WHERE v.generation_id=? AND d.generation_id=?
                   ORDER BY d.id""",
                (generation, generation),
            ).fetchall()
        return [dict(row) for row in rows]

    def embedding_status(self, generation: str) -> dict[str, int]:
        with self.database.read() as connection:
            documents = int(
                connection.execute(
                    "SELECT COUNT(*) FROM search_documents WHERE generation_id=?", (generation,)
                ).fetchone()[0]
            )
            vectors = int(
                connection.execute(
                    "SELECT COUNT(*) FROM embedding_vectors WHERE generation_id=?", (generation,)
                ).fetchone()[0]
            )
            pending = int(
                connection.execute(
                    "SELECT COUNT(*) FROM embedding_queue WHERE generation_id=?", (generation,)
                ).fetchone()[0]
            )
        return {"documents": documents, "vectors": vectors, "pending": pending}

    def activate_generation(self, generation: str) -> None:
        with self.database.write() as connection:
            row = connection.execute(
                "SELECT status,embedding_manifest_json FROM search_generations WHERE id=?",
                (generation,),
            ).fetchone()
            if not row or row["status"] not in ("building", "active"):
                raise ValueError("generation is not activatable")
            count = int(
                connection.execute(
                    "SELECT COUNT(*) FROM search_documents WHERE generation_id=?", (generation,)
                ).fetchone()[0]
            )
            if count == 0:
                raise ValueError("cannot activate an empty search generation")
            manifest = row["embedding_manifest_json"]
            if manifest:
                vectors = int(
                    connection.execute(
                        "SELECT COUNT(*) FROM embedding_vectors WHERE generation_id=?",
                        (generation,),
                    ).fetchone()[0]
                )
                pending = int(
                    connection.execute(
                        "SELECT COUNT(*) FROM embedding_queue WHERE generation_id=?",
                        (generation,),
                    ).fetchone()[0]
                )
                if vectors != count or pending:
                    raise ValueError(
                        "cannot activate an incomplete embedding generation: "
                        f"documents={count} vectors={vectors} pending={pending}"
                    )
            connection.execute(
                "UPDATE search_generations SET status='retired' WHERE status='active' AND id<>?",
                (generation,),
            )
            connection.execute(
                "UPDATE search_generations SET status='active',activated_at=? WHERE id=?",
                (utc_iso(), generation),
            )

    def lexical_search(
        self,
        query: str,
        *,
        limit: int = 20,
        current_project_id: str | None = None,
        hard_project_ids: tuple[str, ...] = (),
        hard_providers: tuple[str, ...] = (),
        hard_session_ids: tuple[str, ...] = (),
        hard_roles: tuple[str, ...] = (),
    ) -> list[SearchHit]:
        fts_query, exact_terms = self._fts_query(query)
        if not fts_query:
            return []
        with self.database.read() as connection:
            active = connection.execute(
                "SELECT id FROM search_generations WHERE status='active'"
            ).fetchone()
            if not active:
                return []
            parameters: list[Any] = [fts_query, active["id"]]
            where = "search_documents_fts MATCH ? AND d.generation_id=?"
            if hard_project_ids:
                placeholders = ",".join("?" for _ in hard_project_ids)
                where += f" AND d.project_id IN ({placeholders})"
                parameters.extend(hard_project_ids)
            if hard_providers:
                placeholders = ",".join("?" for _ in hard_providers)
                where += f" AND d.provider IN ({placeholders})"
                parameters.extend(hard_providers)
            if hard_session_ids:
                placeholders = ",".join("?" for _ in hard_session_ids)
                where += f" AND d.session_id IN ({placeholders})"
                parameters.extend(hard_session_ids)
            if hard_roles:
                placeholders = ",".join("?" for _ in hard_roles)
                where += f" AND d.role IN ({placeholders})"
                parameters.extend(hard_roles)
            parameters.append(max(limit * 4, limit))
            rows = connection.execute(
                f"""SELECT d.*, bm25(search_documents_fts,5.0,1.0) AS rank
                    FROM search_documents_fts
                    JOIN search_documents d ON d.row_id=search_documents_fts.rowid
                    WHERE {where}
                    ORDER BY rank
                    LIMIT ?""",
                parameters,
            ).fetchall()

        lowered_query = query.casefold()
        hits: list[SearchHit] = []
        for row in rows:
            exact = sum(
                0.12 for term in exact_terms if term.casefold() in row["search_text"].casefold()
            )
            if lowered_query in row["search_text"].casefold():
                exact += 0.5
            project_boost = (
                0.15 if current_project_id and row["project_id"] == current_project_id else 0.0
            )
            lexical = 1.0 / (1.0 + max(0.0, float(row["rank"])))
            hits.append(
                SearchHit(
                    document_id=row["id"],
                    memory_type=row["memory_type"],
                    ref_id=row["ref_id"],
                    provider=row["provider"],
                    project_id=row["project_id"],
                    task_id=row["task_id"],
                    session_id=row["session_id"],
                    role=row["role"],
                    authority=row["authority"],
                    occurred_at=row["occurred_at"],
                    title=row["title"],
                    body=row["body"],
                    content_sha256=row["content_sha256"],
                    lexical_score=lexical,
                    exact_score=exact,
                    project_boost=project_boost,
                )
            )
        hits.sort(key=lambda item: (-item.score, item.document_id))
        return hits[:limit]

    def active_generation_id(self) -> str | None:
        with self.database.read() as connection:
            row = connection.execute(
                "SELECT id FROM search_generations WHERE status='active'"
            ).fetchone()
        return row["id"] if row else None

    def record_retrieval(
        self,
        *,
        request_id: str,
        query_sha256: str,
        requested_at: str,
        delivered_at: str,
        mode: str,
        generation_id: str | None,
        current_project_id: str | None,
        stage_latency: dict[str, float],
        result_ids: list[str],
        fallback_reason: str | None,
        index_age_seconds: float | None = None,
    ) -> None:
        with self.database.write() as connection:
            connection.execute(
                """INSERT INTO retrieval_receipts(
                       request_id,query_sha256,requested_at,delivered_at,mode,generation_id,
                       current_project_id,stage_latency_json,result_ids_json,fallback_reason,
                       index_age_seconds
                   ) VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    request_id,
                    query_sha256,
                    requested_at,
                    delivered_at,
                    mode,
                    generation_id,
                    current_project_id,
                    json.dumps(stage_latency, sort_keys=True),
                    json.dumps(result_ids),
                    fallback_reason,
                    index_age_seconds,
                ),
            )

    def counts(self) -> dict[str, int]:
        with self.database.read() as connection:
            names = ("sources", "episodes", "memory_events", "claim_revisions", "search_documents")
            return {
                name: int(connection.execute(f"SELECT COUNT(*) FROM {name}").fetchone()[0])
                for name in names
            }

    @staticmethod
    def _key(value: str) -> str:
        return " ".join(value.casefold().split())

    @staticmethod
    def _search_text(title: str, body: str, row: sqlite3.Row) -> str:
        fields = [
            title,
            body,
            row["provider"] or "",
            row["project_id"] or "",
            row["task_id"] or "",
            row["session_id"] or "",
            row["role"] or "",
            row["authority"] or "",
            row["worktree"] or "",
            row["commit_sha"] or "",
        ]
        return "\n".join(item for item in fields if item)

    @staticmethod
    def _fts_query(query: str) -> tuple[str, tuple[str, ...]]:
        quoted = re.findall(r'"([^"\n]{2,})"', query)
        stripped = re.sub(r'"[^"\n]*"', " ", query)
        tokens = re.findall(r"[\w./:\\-]{2,}", stripped, flags=re.UNICODE)
        stop = {
            "the",
            "and",
            "that",
            "this",
            "with",
            "from",
            "what",
            "when",
            "where",
            "which",
            "would",
            "could",
            "should",
            "about",
            "into",
            "have",
            "has",
            "had",
            "was",
            "were",
            "are",
            "for",
            "you",
            "our",
        }
        terms: list[str] = []
        for value in [*quoted, *tokens]:
            cleaned = value.strip()
            if not cleaned or cleaned.casefold() in stop:
                continue
            if cleaned.casefold() not in {term.casefold() for term in terms}:
                terms.append(cleaned)
            if len(terms) >= 24:
                break
        if not terms:
            return "", ()

        def quote(term: str) -> str:
            return '"' + term.replace('"', '""') + '"'

        return " OR ".join(quote(term) for term in terms), tuple(terms)
