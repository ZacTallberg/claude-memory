"""Incremental indexer: scan transcripts (tail by byte offset) + curated notes -> chunk ->
(optional Contextual Retrieval) -> store -> embed pending. Zero LLM required by default.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Callable

from . import facts as facts_mod
from .chunking import chunk_text, estimate_tokens
from .config import Config
from .log import get_logger
from .memory_types import conflict_index, find_claim_conflicts
from .providers.contextual import Contextualizer
from .providers.embeddings import EmbeddingProvider, get_embedding_provider
from .store.base import Store
from .transcript_adapters import discover_transcripts, get_adapter

log = get_logger(__name__)
Progress = Callable[[str], None] | None


@dataclass
class IndexStats:
    transcripts_scanned: int = 0
    units: int = 0
    chunks_added: int = 0
    notes: int = 0
    notes_unchanged: int = 0
    notes_pruned: int = 0
    embedded: int = 0
    candidates_drafted: int = 0
    errors: int = 0

    def as_dict(self) -> dict:
        return self.__dict__.copy()


def _emit(progress: Progress, msg: str) -> None:
    log.info(msg)
    if progress:
        try:
            progress(msg)
        except Exception:
            pass


def index(cfg: Config, store: Store, *, full: bool = False, progress: Progress = None,
          only_provider: str | None = None) -> IndexStats:
    provider = get_embedding_provider(cfg)
    ctx = Contextualizer(cfg)
    stats = IndexStats()

    # --- transcripts (incremental) ---
    tfiles = discover_transcripts(cfg, only_provider=only_provider)
    _emit(progress, f"scanning {len(tfiles)} transcripts...")
    for tf in tfiles:
        try:
            src = store.get_source(str(tf.path))
            start = 0 if full else (src["bytes_indexed"] if src else 0)
            adapter = get_adapter(tf.provider)
            units, new_off = adapter.parse_new(tf.path, start, cfg)
            if full and src:
                store.delete_source(str(tf.path))
            session_id = (units[0].session_id if units
                          else (src.get("session_id") if src else None) or tf.path.stem)
            project = tf.project
            if tf.provider == "codex" and units and units[0].cwd:
                project = Path(units[0].cwd).name or "codex"
            elif src and not units:
                project = src.get("project", project)
            sid = store.upsert_source(path=str(tf.path), kind="transcript", project=project,
                                      session_id=session_id, bytes_indexed=new_off,
                                      mtime=tf.path.stat().st_mtime,
                                      meta={"encoded_dir": tf.encoded_dir, "provider": tf.provider})
            chunk_dicts = []
            for u in units:
                for piece in chunk_text(u.text):
                    blurb = None
                    if cfg.contextual.enrich_transcripts and ctx.available():
                        blurb = ctx.contextualize(u.text, piece)
                    chunk_dicts.append({
                        "ordinal": u.ordinal, "kind": u.kind, "role": u.role,
                        "session_id": u.session_id, "project": project, "cwd": u.cwd,
                        "ts": u.ts, "content": piece, "context_blurb": blurb,
                        "token_est": estimate_tokens(piece), "meta": {},
                    })
            if chunk_dicts:
                store.add_chunks(sid, chunk_dicts)
                stats.chunks_added += len(chunk_dicts)
            stats.units += len(units)
            stats.transcripts_scanned += 1
            if units:
                _emit(progress, f"  {tf.provider}:{project}/{tf.path.name}: +{len(chunk_dicts)} chunks")
        except Exception as e:
            stats.errors += 1
            log.exception("transcript index failed: %s (%s)", tf.path, e)

    # --- curated notes (facts) ---
    notes = []
    if only_provider is None:
        notes = facts_mod.load_notes(cfg)
        _emit(progress, f"indexing {len(notes)} curated notes...")
        conflicts = find_claim_conflicts(notes)
        conflicts_by_note = conflict_index(conflicts)
        if conflicts:
            _emit(progress, f"  {len(conflicts)} unresolved structured-claim conflicts held from automatic recall")
        try:
            existing_facts = {f.path: f for f in store.list_facts()}
        except Exception:
            existing_facts = {}
        embedding_model = getattr(provider, "name", None) if provider.available() else None
        for nd in notes:
            try:
                existing = existing_facts.get(nd.path)
                unchanged = (not full and existing is not None and existing.mtime is not None
                             and abs(float(existing.mtime) - float(nd.mtime)) < 1e-6
                             and embedding_model
                             and existing.meta.get("embedding_model") == embedding_model
                             and existing.meta.get("lifecycle_schema") == 2
                             and existing.meta.get("conflict_ids", []) == conflicts_by_note.get(nd.path, []))
                if unchanged:
                    stats.notes_unchanged += 1
                    continue
                emb = None
                if provider.available():
                    text = " ".join([nd.name, nd.title, nd.project, " ".join(nd.tags),
                                     nd.description, nd.body])
                    try:
                        emb = provider.embed_documents([text])[0]
                    except Exception:
                        emb = None
                store.upsert_fact(path=nd.path, project=nd.project, name=nd.name, title=nd.title,
                                  description=nd.description, type=nd.type, tags=nd.tags,
                                  origin_session_id=nd.origin_session_id, body=nd.body, embedding=emb,
                                  mtime=nd.mtime, meta={"wikilinks": nd.wikilinks,
                                                       "embedding_model": embedding_model,
                                                       "lifecycle_schema": 2,
                                                       "memory_kind": nd.memory_kind,
                                                       "importance": nd.importance,
                                                       "claims": nd.claims,
                                                       "conflict_ids": conflicts_by_note.get(nd.path, []),
                                                       **nd.lifecycle})
                stats.notes += 1
            except Exception as e:
                stats.errors += 1
                log.exception("note index failed: %s (%s)", nd.path, e)

    # --- prune facts whose note file is gone ---
    # upsert_fact never removes anything, so deleted/renamed notes lingered in the index forever
    # and kept appearing in <memory-map> and <curated-notes> (found 2026-07-29: 201 files on disk
    # vs 206 rows). Deleting purely on "absent from `notes`" would be dangerous - a config or
    # scope change shrinks that set without a single file being removed - so absence from the
    # scan is only a CANDIDATE, and the row is dropped only when the path is genuinely missing
    # from disk. Deterministic signal, not inference.
    try:
        if only_provider is None and notes:  # provider-only rebuilds never touch curated notes
            seen = {nd.path for nd in notes}
            for f in store.list_facts():
                if f.path in seen:
                    continue
                if Path(f.path).exists():
                    log.warning("fact %s is out of scan scope but still on disk; keeping", f.path)
                    continue
                store.delete_fact(f.path)
                stats.notes_pruned += 1
            if stats.notes_pruned:
                _emit(progress, f"pruned {stats.notes_pruned} facts whose note file was deleted")
    except Exception as e:
        log.exception("fact prune failed: %s", e)

    # --- graph from wikilinks ---
    if only_provider is None:
        try:
            nodes, edges = facts_mod.build_graph(notes)
            store.replace_graph(nodes, edges)
        except Exception as e:
            log.exception("graph build failed: %s", e)

    # --- embed pending chunks ---
    stats.embedded = embed_pending(cfg, store, provider, progress=progress)

    # --- metrics snapshot ---
    try:
        cnt = store.counts()
        for m in ("sources", "chunks", "chunks_embedded", "facts"):
            store.record_metric(m, float(cnt.get(m, 0)))
    except Exception:
        pass

    # Review-first consolidation is deliberately off the prompt path. It may draft typed
    # semantic/episodic/procedural candidates after an index pass, but it never writes or activates
    # a curated note. A durable KV timestamp bounds work across the frequent live-sync passes.
    if (only_provider is None and cfg.consolidation.enabled
            and cfg.consolidation.auto_after_index
            and (stats.units or stats.notes or stats.chunks_added)):
        try:
            state = store.kv_get("consolidation:last_run") or {}
            last = float(state.get("epoch") or 0.0)
            due_s = cfg.consolidation.min_interval_hours * 3600.0
            if time.time() - last >= due_s:
                from .promote import mine_candidates
                stats.candidates_drafted = mine_candidates(
                    cfg, store, cap=cfg.consolidation.candidate_cap)
                store.kv_set("consolidation:last_run", {
                    "epoch": time.time(), "drafted": stats.candidates_drafted,
                    "mode": "review-first", "memory_schema": 2,
                })
                _emit(progress, f"drafted {stats.candidates_drafted} typed memory candidates for review")
        except Exception as e:
            # Consolidation is nonessential background work; indexing remains successful.
            log.exception("background consolidation failed: %s", e)

    _emit(progress, f"done: {stats.as_dict()}")
    return stats


def embed_pending(cfg: Config, store: Store, provider: EmbeddingProvider | None = None,
                  *, progress: Progress = None) -> int:
    provider = provider or get_embedding_provider(cfg)
    if not provider.available():
        _emit(progress, "embeddings unavailable; keyword-only index")
        return 0
    total = 0
    while True:
        batch = store.chunks_missing_embeddings(cfg.index.batch_size)
        if not batch:
            break
        texts = [((c.context_blurb + " ") if c.context_blurb else "") + c.content for c in batch]
        try:
            vecs = provider.embed_documents(texts)
        except Exception as e:
            log.exception("embedding batch failed: %s", e)
            break
        store.set_embeddings(list(zip([c.id for c in batch], vecs)))
        total += len(batch)
        _emit(progress, f"  embedded {total} chunks")
    return total
