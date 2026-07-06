"""Incremental indexer: scan transcripts (tail by byte offset) + curated notes -> chunk ->
(optional Contextual Retrieval) -> store -> embed pending. Zero LLM required by default.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from . import facts as facts_mod
from . import paths, transcripts
from .chunking import chunk_text, estimate_tokens
from .config import Config
from .log import get_logger
from .providers.contextual import Contextualizer
from .providers.embeddings import EmbeddingProvider, get_embedding_provider
from .store.base import Store

log = get_logger(__name__)
Progress = Callable[[str], None] | None


@dataclass
class IndexStats:
    transcripts_scanned: int = 0
    units: int = 0
    chunks_added: int = 0
    notes: int = 0
    embedded: int = 0
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


def index(cfg: Config, store: Store, *, full: bool = False, progress: Progress = None) -> IndexStats:
    provider = get_embedding_provider(cfg)
    ctx = Contextualizer(cfg)
    stats = IndexStats()

    # --- transcripts (incremental) ---
    tfiles = paths.iter_transcript_files(cfg)
    _emit(progress, f"scanning {len(tfiles)} transcripts...")
    for tf in tfiles:
        try:
            src = store.get_source(str(tf.path))
            start = 0 if full else (src["bytes_indexed"] if src else 0)
            units, new_off = transcripts.parse_new(tf.path, start, cfg)
            if full and src:
                store.delete_source(str(tf.path))
            session_id = (units[0].session_id if units
                          else (src.get("session_id") if src else None) or tf.path.stem)
            sid = store.upsert_source(path=str(tf.path), kind="transcript", project=tf.project,
                                      session_id=session_id, bytes_indexed=new_off,
                                      mtime=tf.path.stat().st_mtime,
                                      meta={"encoded_dir": tf.encoded_dir})
            chunk_dicts = []
            for u in units:
                for piece in chunk_text(u.text):
                    blurb = None
                    if cfg.contextual.enrich_transcripts and ctx.available():
                        blurb = ctx.contextualize(u.text, piece)
                    chunk_dicts.append({
                        "ordinal": u.ordinal, "kind": u.kind, "role": u.role,
                        "session_id": u.session_id, "project": tf.project, "cwd": u.cwd,
                        "ts": u.ts, "content": piece, "context_blurb": blurb,
                        "token_est": estimate_tokens(piece), "meta": {},
                    })
            if chunk_dicts:
                store.add_chunks(sid, chunk_dicts)
                stats.chunks_added += len(chunk_dicts)
            stats.units += len(units)
            stats.transcripts_scanned += 1
            if units:
                _emit(progress, f"  {tf.project}/{tf.path.name}: +{len(chunk_dicts)} chunks")
        except Exception as e:
            stats.errors += 1
            log.exception("transcript index failed: %s (%s)", tf.path, e)

    # --- curated notes (facts) ---
    notes = facts_mod.load_notes(cfg)
    _emit(progress, f"indexing {len(notes)} curated notes...")
    for nd in notes:
        try:
            emb = None
            if provider.available():
                text = " ".join([nd.title, nd.description, nd.body])
                try:
                    emb = provider.embed_documents([text])[0]
                except Exception:
                    emb = None
            store.upsert_fact(path=nd.path, project=nd.project, name=nd.name, title=nd.title,
                              description=nd.description, type=nd.type, tags=nd.tags,
                              origin_session_id=nd.origin_session_id, body=nd.body, embedding=emb,
                              mtime=nd.mtime, meta={"wikilinks": nd.wikilinks})
            stats.notes += 1
        except Exception as e:
            stats.errors += 1
            log.exception("note index failed: %s (%s)", nd.path, e)

    # --- graph from wikilinks ---
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
