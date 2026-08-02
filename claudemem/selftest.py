"""Regression self-test (SPEC §9). A flat list of independent checks; each prints
PASS / FAIL / SKIP and the suite returns False if any check FAILs (skips don't fail).

Design rules:
  - READ-ONLY against the real corpus. We never write to the live store, never touch the
    real ~/.claude/settings.json, and restore the kill switch to its prior state.
  - Synthetic inputs exercise the pure logic (chunking, transcript/fact parsing, retriever
    fusion, envelope formatting). A single real query against the populated DB proves the
    end-to-end retrieval path returns hits.
  - Anything that genuinely depends on a missing dependency SKIPs with a clear reason; the
    core checks must PASS.

Run:  python -m claudemem selftest      (cli.cmd_selftest exits 1 on any failure)
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import traceback
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .config import ROOT, load_config


class _Ctx:
    """Collects results and prints a PASS/FAIL/SKIP line per check."""

    def __init__(self, verbose: bool):
        self.verbose = verbose
        self.passed = 0
        self.failed = 0
        self.skipped = 0
        self._n = 0

    def _line(self, tag: str, name: str, detail: str = "") -> None:
        if self.verbose:
            suffix = f"  ({detail})" if detail else ""
            print(f"  [{tag:>4}] {self._n:>2}. {name}{suffix}")

    def check(self, name: str, fn) -> None:
        """fn() -> (ok: bool, detail: str) | bool. Raising counts as FAIL."""
        self._n += 1
        try:
            res = fn()
            ok, detail = res if isinstance(res, tuple) else (bool(res), "")
        except Exception as e:  # a check that blows up is a failure, never fatal to the suite
            self.failed += 1
            self._line("FAIL", name, f"{type(e).__name__}: {e}")
            if self.verbose:
                traceback.print_exc()
            return
        if ok:
            self.passed += 1
            self._line("PASS", name, detail)
        else:
            self.failed += 1
            self._line("FAIL", name, detail)

    def skip(self, name: str, reason: str) -> None:
        self._n += 1
        self.skipped += 1
        self._line("SKIP", name, reason)


# --------------------------------------------------------------------------- #
# Lightweight synthetic fixtures (no I/O, no DB writes)
# --------------------------------------------------------------------------- #

def _synthetic_transcript_lines(session_id: str) -> list[str]:
    """A tiny in-memory JSONL transcript: user + assistant (with thinking) + a sidechain
    line that MUST be excluded + an injected-block that MUST be stripped."""
    now = datetime.now(timezone.utc).isoformat()
    recs = [
        {"type": "user", "uuid": "u1", "timestamp": now, "sessionId": session_id,
         "cwd": "C:/code/claude-memory", "isSidechain": False, "isMeta": False,
         "message": {"role": "user", "content": "How does the recall hook budget characters?"}},
        {"type": "assistant", "uuid": "a1", "timestamp": now, "sessionId": session_id,
         "cwd": "C:/code/claude-memory", "isSidechain": False, "isMeta": False,
         "message": {"role": "assistant", "content": [
             {"type": "thinking", "thinking": "The recall hook caps additionalContext under 10k chars."},
             {"type": "text", "text": "It truncates the envelope to recall.max_chars before emitting. "
                                      "<recalled-memory trust=\"data-only\">leftover injected junk"
                                      "</recalled-memory> trailing text."},
         ]}},
        {"type": "assistant", "uuid": "sc1", "timestamp": now, "sessionId": session_id,
         "cwd": "C:/code/claude-memory", "isSidechain": True, "isMeta": False,
         "message": {"role": "assistant", "content": "subagent noise that must be excluded"}},
    ]
    return [json.dumps(r) for r in recs]


def _synthetic_codex_lines(session_id: str) -> list[str]:
    """Narrow Codex rollout fixture: canonical messages plus records that must be ignored."""
    now = datetime.now(timezone.utc).isoformat()
    recs = [
        {"timestamp": now, "type": "session_meta", "payload": {
            "id": session_id, "session_id": session_id, "cwd": "C:/code/claude-memory"}},
        {"timestamp": now, "type": "response_item", "payload": {
            "type": "message", "role": "user", "content": [
                {"type": "input_text", "text": "How does shared vector memory work?"}]}},
        {"timestamp": now, "type": "response_item", "payload": {
            "type": "message", "role": "assistant", "content": [
                {"type": "output_text", "text": "It fuses lexical and vector ranks. "
                 "<recalled-memory trust=\"data-only\">do not reindex me</recalled-memory>"}]}},
        {"timestamp": now, "type": "response_item", "payload": {
            "type": "message", "role": "developer", "content": [
                {"type": "input_text", "text": "private developer instructions"}]}},
        {"timestamp": now, "type": "response_item", "payload": {
            "type": "function_call_output", "output": "large tool output"}},
        {"timestamp": now, "type": "response_item", "payload": {
            "type": "reasoning", "summary": "private chain of thought"}},
    ]
    return [json.dumps(r) for r in recs]


def _synthetic_note(tmpdir: Path) -> Path:
    """Write a synthetic curated note (with frontmatter) to a throwaway dir for parse testing."""
    p = tmpdir / "synthetic_note.md"
    p.write_text(
        "---\n"
        "name: synthetic-char-cap\n"
        "title: Synthetic Character Cap\n"
        "description: How the recall envelope respects the character cap.\n"
        "metadata:\n"
        "  type: feedback\n"
        "  originSessionId: sess-xyz\n"
        "  tags: [recall, budget]\n"
        "---\n"
        "The envelope is truncated to max_chars. See [[recall-hook]] for details.\n",
        encoding="utf-8",
    )
    return p


# A minimal Chunk-like + Result-like fixture for pure retriever/formatter testing.
def _fake_results(cfg, n: int = 3):
    from .store.base import Chunk
    from .retriever import Result
    out = []
    for i in range(n):
        ch = Chunk(id=i + 1, source_id=1, kind="text", role="assistant",
                   session_id=f"s{i}", project="claude-memory", cwd="C:/code/claude-memory",
                   ts=datetime.now(timezone.utc) - timedelta(days=i),
                   content=f"Synthetic recalled snippet number {i} about memory recall.",
                   context_blurb=None, ordinal=i, meta={})
        out.append(Result(chunk=ch, score=0.9 - 0.1 * i, fused=0.5 - 0.05 * i))
    return out


# --------------------------------------------------------------------------- #
# The suite
# --------------------------------------------------------------------------- #

def run_selftest(verbose: bool = True) -> bool:
    ctx = _Ctx(verbose)
    if verbose:
        print("claude-memory self-test")
        print("=" * 60)

    cfg = load_config()

    # -- 1. config loads with the required shape -----------------------------
    def c_config():
        ok = (cfg.scope.workspace_roots and cfg.embeddings.dim > 0
              and cfg.recall.max_chars <= 10000 and cfg.store.backend in ("auto", "postgres", "sqlite"))
        return ok, f"backend={cfg.store.backend} dim={cfg.embeddings.dim} max_chars={cfg.recall.max_chars}"
    ctx.check("config loads (required sections + sane values)", c_config)

    # -- store: connect + migrate + health ----------------------------------
    store = None
    try:
        from .store.factory import get_store
        store = get_store(cfg)
    except Exception as e:
        # surface as an explicit failed check below; many checks will then skip
        if verbose:
            print(f"  !! store unavailable: {e}")

    def c_store_health():
        if store is None:
            return False, "get_store raised"
        h = store.health()
        return bool(h.get("ok")), f"backend={h.get('backend')} ok={h.get('ok')} bm25={h.get('bm25')} vector={h.get('vector')}"
    ctx.check("store connects + migrates (health ok)", c_store_health)

    # Corpus-dependent checks skip (not fail) on an empty store: a fresh install has indexed
    # nothing yet, and "no hits" is then correct behavior, not a defect.
    _counts: dict = {}
    if store is not None:
        try:
            _counts = store.counts()
        except Exception:
            pass
    n_chunks, n_facts = _counts.get("chunks", 0), _counts.get("facts", 0)
    empty_skip = "corpus empty (install is fine — run `mem index` to populate)"

    # -- embedder loads with correct dim ------------------------------------
    from .providers.embeddings import get_embedding_provider, NullEmbeddingProvider
    embedder = get_embedding_provider(cfg)
    embed_ok = embedder.available()
    if embed_ok:
        def c_embed():
            v = embedder.embed_query("memory recall hook test")
            return (len(v) == cfg.embeddings.dim and embedder.dim == cfg.embeddings.dim,
                    f"name={embedder.name} dim={embedder.dim} |vec|={len(v)}")
        ctx.check("embedder loads with correct dim", c_embed)
    else:
        ctx.skip("embedder loads with correct dim", f"embedder unavailable ({embedder.name})")

    # -- chunking on a synthetic oversized turn -----------------------------
    def c_chunking():
        from .chunking import chunk_text, estimate_tokens
        small = chunk_text("one short paragraph")
        big = chunk_text(("para number %d with some words. " % 0) * 400)
        return (len(small) == 1 and len(big) > 1 and estimate_tokens("abcd") >= 1,
                f"small={len(small)} big={len(big)}")
    ctx.check("chunking: small->1 chunk, oversized->windowed", c_chunking)

    # -- transcript parsing: synthetic JSONL --------------------------------
    sid = "selftest-synth-session"
    def c_transcript_parse():
        from . import transcripts
        lines = _synthetic_transcript_lines(sid)
        with tempfile.TemporaryDirectory() as d:
            tf = Path(d) / f"{sid}.jsonl"
            tf.write_text("\n".join(lines) + "\n", encoding="utf-8")
            units, off = transcripts.parse_new(tf, 0, cfg)
        kinds = {u.kind for u in units}
        roles = {u.role for u in units}
        # sidechain line excluded -> no unit references "subagent noise"
        no_sidechain = all("subagent noise" not in u.text for u in units)
        # injected <recalled-memory> block stripped from stored text
        no_injected = all("leftover injected junk" not in u.text for u in units)
        return (len(units) >= 2 and "text" in kinds and "thinking" in kinds
                and "user" in roles and "assistant" in roles and no_sidechain
                and no_injected and off > 0,
                f"units={len(units)} kinds={sorted(kinds)} sidechain_excluded={no_sidechain}")
    ctx.check("transcript parse: text+thinking, sidechain excluded, injected stripped", c_transcript_parse)

    # -- incremental tail: half-written final line is held back -------------
    def c_transcript_tail():
        from . import transcripts
        lines = _synthetic_transcript_lines(sid)
        with tempfile.TemporaryDirectory() as d:
            tf = Path(d) / f"{sid}.jsonl"
            # complete lines + a trailing partial (no newline) that must NOT be consumed
            tf.write_text("\n".join(lines) + "\n" + '{"type":"user","message"', encoding="utf-8")
            _u, off = transcripts.parse_new(tf, 0, cfg)
            size = tf.stat().st_size
        return off < size, f"consumed={off} < size={size} (partial line held)"
    ctx.check("transcript incremental tail: partial final line held back", c_transcript_tail)

    # -- Codex rollout adapter: messages only, fail-safe on unknown records -
    def c_codex_transcript_parse():
        from . import codex_transcripts
        lines = _synthetic_codex_lines(sid)
        with tempfile.TemporaryDirectory() as d:
            tf = Path(d) / f"rollout-{sid}.jsonl"
            tf.write_text("\n".join(lines) + "\n" + '{"type":"response_item"', encoding="utf-8")
            units, off = codex_transcripts.parse_new(tf, 0, cfg)
            size = tf.stat().st_size
        combined = " ".join(u.text for u in units)
        return (len(units) == 2 and {u.role for u in units} == {"user", "assistant"}
                and all(u.session_id == sid for u in units)
                and "private developer" not in combined and "large tool" not in combined
                and "chain of thought" not in combined and "do not reindex" not in combined
                and off < size,
                f"units={len(units)} roles={sorted({u.role for u in units})} partial_held={off < size}")
    ctx.check("Codex transcript adapter keeps only clean user/assistant messages", c_codex_transcript_parse)

    def c_codex_ambient_cleaning():
        from .codex_transcripts import clean_codex_text
        ambient = ("<recommended_plugins>catalog noise</recommended_plugins> "
                   "<environment_context><cwd>C:/secret</cwd></environment_context> "
                   "<in-app-browser-context>tab noise</in-app-browser-context> "
                   "Please preserve the authored request.")
        delegated = ("<codex_delegation><source_thread_id>abc</source_thread_id>"
                     "<input>Build the actual worker feature.</input></codex_delegation>")
        goal = ("<codex_internal_context source=\"goal\">boilerplate "
                "<objective>Keep live memory synchronized.</objective></codex_internal_context>")
        a, d, g = map(clean_codex_text, (ambient, delegated, goal))
        ok = (a == "Please preserve the authored request."
              and d == "Build the actual worker feature."
              and g == "Keep live memory synchronized.")
        return ok, f"ambient={a!r} delegated={d!r} goal={g!r}"
    ctx.check("Codex corpus cleaning drops ambient state but preserves authored payloads",
              c_codex_ambient_cleaning)

    # -- curated note parse: frontmatter + nested type + wikilinks ----------
    def c_note_parse():
        from .facts import load_note
        with tempfile.TemporaryDirectory() as d:
            p = _synthetic_note(Path(d))
            nd = load_note(p, "synthetic-project")
        return (nd is not None and nd.title == "Synthetic Character Cap"
                and nd.type == "feedback" and nd.origin_session_id == "sess-xyz"
                and "recall-hook" in nd.wikilinks and "recall" in nd.tags,
                f"type={nd.type if nd else None} links={nd.wikilinks if nd else None}")
    ctx.check("curated note parse: nested type, origin, wikilinks, tags", c_note_parse)

    # -- meaningful_term_count gating ---------------------------------------
    def c_terms():
        from .text import meaningful_term_count
        trivial = meaningful_term_count("hi the a")          # all stopwords/short
        real = meaningful_term_count("postgres paradedb embedding retriever fusion")
        return (trivial < cfg.recall.min_terms <= real,
                f"trivial={trivial} real={real} min_terms={cfg.recall.min_terms}")
    ctx.check("meaningful_term_count gating (trivial < min_terms <= real)", c_terms)

    # -- strip_injected_blocks removes a <recalled-memory> block ------------
    def c_strip():
        from .text import strip_injected_blocks
        raw = ('before <recalled-memory trust="data-only">SECRET INJECTED'
               "</recalled-memory> after")
        out = strip_injected_blocks(raw)
        return ("recalled-memory" not in out and "SECRET INJECTED" not in out
                and "before" in out and "after" in out, "block removed, surrounding text kept")
    ctx.check("strip_injected_blocks removes <recalled-memory>", c_strip)

    def c_secret_redaction():
        from .security import redact_secrets
        openai = "sk-proj-" + "A1b2" * 8
        github = "github_pat_" + "Z9y8" * 8
        raw = f"openai={openai} github={github} password: Correct-Horse-123"
        cleaned, findings = redact_secrets(raw)
        kinds = {finding.kind for finding in findings}
        ok = (openai not in cleaned and github not in cleaned and "Correct-Horse-123" not in cleaned
              and {"openai-token", "github-token"} <= kinds
              and bool({"password-literal", "assigned-secret"} & kinds))
        return ok, f"redacted={len(findings)} types={sorted(kinds)}"
    ctx.check("secret scanner redacts credential-shaped values without echoing them", c_secret_redaction)

    # -- retriever fusion on synthetic in-memory candidate lists ------------
    def c_fusion():
        from .retriever import Retriever
        from .store.base import Candidate
        r = Retriever(cfg, store, embedder=NullEmbeddingProvider(cfg.embeddings.dim))
        a = [Candidate(chunk_id=1, rank=1, score=9), Candidate(chunk_id=2, rank=2, score=8)]
        b = [Candidate(chunk_id=2, rank=1, score=0.9), Candidate(chunk_id=3, rank=2, score=0.8)]
        fused = r._rrf([a, b])
        # chunk 2 appears in both lists -> must outrank singletons 1 and 3
        return (fused[2] > fused[1] and fused[2] > fused[3] and len(fused) == 3,
                f"rrf id2={fused[2]:.4f} > id1={fused[1]:.4f},id3={fused[3]:.4f}")
    ctx.check("retriever RRF fusion ranks doubly-listed chunk highest", c_fusion)

    def c_rank_order():
        from .ranking import reciprocal_rank_order
        lexical = list(range(1, 25))
        semantic = [19, 30, 31]
        order = reciprocal_rank_order((lexical, semantic), rrf_k=60)
        return order[0] == 19, f"top={order[:4]} (id 19 is supported by both rankers)"
    ctx.check("shared RRF ordering lets vector evidence promote a lower lexical fact", c_rank_order)

    # -- BM25 search returns hits (real DB) ---------------------------------
    real_q = "memory recall hook index retriever postgres embedding"
    if store is None:
        ctx.skip("BM25 search returns hits", "store unavailable")
    elif n_chunks == 0:
        ctx.skip("BM25 search returns hits", empty_skip)
    else:
        def c_bm25():
            hits = store.search_bm25(real_q, cfg.recall.bm25_k)
            return len(hits) > 0, f"{len(hits)} bm25 hits"
        ctx.check("BM25 search returns hits (populated DB)", c_bm25)

        def c_read_not_serialized():
            import threading
            held = threading.Event()
            release = threading.Event()

            def hold_writer_lock():
                with store._lock:
                    held.set()
                    release.wait(10)

            t = threading.Thread(target=hold_writer_lock, daemon=True)
            t.start()
            held.wait(2)
            try:
                hits = store.search_bm25(real_q, 4)
                return len(hits) > 0, f"{len(hits)} hits while writer connection lock was held"
            finally:
                release.set()
        if hasattr(store, "_read_conn"):
            ctx.check("SQLite hot-path reads do not serialize behind the writer lock", c_read_not_serialized)
        else:
            ctx.skip("SQLite hot-path reads do not serialize behind the writer lock",
                     "active backend is not SQLite")

    # -- vector search returns hits (real DB; needs embedder) ---------------
    if store is None:
        ctx.skip("vector search returns hits", "store unavailable")
    elif not embed_ok:
        ctx.skip("vector search returns hits", "embedder unavailable")
    elif n_chunks == 0:
        ctx.skip("vector search returns hits", empty_skip)
    else:
        def c_vec():
            qv = embedder.embed_query(real_q)
            hits = store.search_vector(qv, cfg.recall.vector_k)
            return len(hits) > 0, f"{len(hits)} vector hits"
        ctx.check("vector search returns hits (populated DB)", c_vec)

    # -- hybrid fuse returns hits (real DB) ---------------------------------
    if store is None:
        ctx.skip("hybrid search returns hits", "store unavailable")
    elif n_chunks == 0:
        ctx.skip("hybrid search returns hits", empty_skip)
    else:
        def c_hybrid():
            from .retriever import Retriever
            r = Retriever(cfg, store)
            res = r.search(real_q, tier="hot")
            ok_results = all(hasattr(x, "chunk") and hasattr(x, "score") for x in res)
            return len(res) > 0 and ok_results, f"{len(res)} hybrid results"
        ctx.check("hybrid fuse returns hits with .chunk/.score (populated DB)", c_hybrid)

    # -- degrade-to-keyword: NullEmbeddingProvider still retrieves ----------
    if store is None:
        ctx.skip("degrade-to-keyword path", "store unavailable")
    elif n_chunks == 0:
        ctx.skip("degrade-to-keyword path", empty_skip)
    else:
        def c_degrade():
            from .retriever import Retriever
            from .providers.reranker import NoopReranker
            r = Retriever(cfg, store, embedder=NullEmbeddingProvider(cfg.embeddings.dim),
                          reranker=NoopReranker())
            res = r.search(real_q, tier="hot", do_rerank=False)
            return len(res) > 0, f"{len(res)} keyword-only results (no embedder)"
        ctx.check("degrade-to-keyword path retrieves with null embedder", c_degrade)

    # -- recall envelope formatting + char cap (<=10000, <=max_chars) -------
    def c_envelope_cap():
        from .recall_format import format_recall
        from .store.base import Fact
        results = _fake_results(cfg, 3)
        facts = [Fact(id=1, path="p", project="claude-memory", name="n", title="Char Cap",
                      description="how the cap works", type="feedback", tags=[],
                      origin_session_id=None, body="body")]
        text = format_recall(results, facts, cfg)
        ok = (len(text) <= cfg.recall.max_chars and len(text) <= 10000
              and '<recalled-memory trust="data-only">' in text
              and "<curated-notes" in text)
        return ok, f"len={len(text)} <= max_chars={cfg.recall.max_chars}"
    ctx.check("recall envelope formats + respects char cap (<=10000)", c_envelope_cap)

    # -- recall envelope hard char cap under a huge synthetic payload -------
    def c_envelope_overflow():
        from .recall_format import format_recall
        from .store.base import Chunk
        from .retriever import Result
        big = [Result(chunk=Chunk(id=i, source_id=1, kind="text", role="assistant",
                                  session_id=f"sess{i}", project="claude-memory",
                                  cwd=None, ts=datetime.now(timezone.utc),
                                  content="x" * 5000, ordinal=i, meta={}),
                      score=0.5, fused=0.5) for i in range(50)]
        text = format_recall(big, [], cfg)
        return len(text) <= cfg.recall.max_chars <= 10000, f"len={len(text)} (cap enforced under overflow)"
    ctx.check("recall envelope hard-caps a huge synthetic payload", c_envelope_overflow)

    # -- unify map formatting emits <memory-map> ----------------------------
    def c_unify_format():
        from .recall_format import format_unify
        from .store.base import Fact
        tmap = {"website-dokku": [
                    Fact(id=1, path="a", project="website-dokku", name="a", title="Analytics Revamp",
                         description="", type="project", tags=[], origin_session_id=None, body=""),
                    Fact(id=2, path="b", project="website-dokku", name="b", title="Explore first",
                         description="", type="feedback", tags=[], origin_session_id=None, body="")]}
        text = format_unify(tmap, cfg)
        return ("<memory-map" in text and "</memory-map>" in text
                and "## website-dokku" in text and "Analytics Revamp" in text and len(text) <= 9500,
                f"len={len(text)}")
    ctx.check("unify formats a <memory-map> of titles", c_unify_format)

    # -- recall hook subprocess: in-scope -> valid additionalContext JSON ---
    # cwd must derive from config: a hardcoded path is out-of-scope the moment a user sets
    # different workspace_roots, and the hook then correctly emits nothing.
    in_scope_cwd = cfg.scope.workspace_roots[0]

    def c_recall_hook():
        event = {"prompt": real_q, "cwd": in_scope_cwd, "session_id": "selftest-live"}
        out = _run_hook("recall.py", event)
        if not out.strip():
            return False, "hook emitted nothing for an in-scope, term-rich prompt"
        env = json.loads(out)
        hso = env.get("hookSpecificOutput", {})
        return (hso.get("hookEventName") == "UserPromptSubmit"
                and isinstance(hso.get("additionalContext"), str)
                and len(hso["additionalContext"]) > 0
                and len(hso["additionalContext"]) <= 10000,
                f"additionalContext len={len(hso.get('additionalContext',''))}")
    if n_chunks == 0 and n_facts == 0:
        ctx.skip("recall hook (subprocess) emits valid additionalContext envelope", empty_skip)
    else:
        ctx.check("recall hook (subprocess) emits valid additionalContext envelope", c_recall_hook)

    # -- trivial prompt -> no output ----------------------------------------
    def c_recall_trivial():
        event = {"prompt": "hi the a", "cwd": in_scope_cwd, "session_id": "selftest-live"}
        out = _run_hook("recall.py", event)
        return out.strip() == "", "no output for sub-min_terms prompt"
    ctx.check("recall hook: trivial prompt -> no output", c_recall_trivial)

    # -- out-of-scope cwd -> no output --------------------------------------
    def c_recall_oos():
        event = {"prompt": real_q, "cwd": "C:/Windows/Temp", "session_id": "selftest-live"}
        out = _run_hook("recall.py", event)
        return out.strip() == "", "no output for out-of-scope cwd"
    ctx.check("recall hook: out-of-scope cwd -> no output", c_recall_oos)

    # -- unify hook subprocess: emits a <memory-map> ------------------------
    def c_unify_hook():
        event = {"cwd": in_scope_cwd, "session_id": "selftest-live", "source": "startup"}
        out = _run_hook("unify.py", event)
        if not out.strip():
            return False, "unify emitted nothing (expected a memory-map from populated facts)"
        env = json.loads(out)
        ac = env.get("hookSpecificOutput", {}).get("additionalContext", "")
        return ("<memory-map" in ac and env["hookSpecificOutput"]["hookEventName"] == "SessionStart",
                f"additionalContext len={len(ac)}")
    if n_facts == 0:
        ctx.skip("unify hook (subprocess) emits a <memory-map>", empty_skip)
    else:
        ctx.check("unify hook (subprocess) emits a <memory-map>", c_unify_hook)

    # -- kill switch: DISABLED -> both hooks emit nothing -------------------
    def c_killswitch():
        from .paths import DISABLED_SENTINEL, killed, set_killed
        was_killed = killed()
        try:
            set_killed(True)
            r_out = _run_hook("recall.py",
                              {"prompt": real_q, "cwd": "C:/code/claude-memory",
                               "session_id": "selftest-live"})
            u_out = _run_hook("unify.py",
                              {"cwd": "C:/code/claude-memory", "session_id": "selftest-live",
                               "source": "startup"})
        finally:
            # restore prior state precisely (don't leave a stray DISABLED behind)
            set_killed(was_killed)
        both_silent = (r_out.strip() == "" and u_out.strip() == "")
        restored = killed() == was_killed
        return both_silent and restored, f"recall_silent={r_out.strip()==''} unify_silent={u_out.strip()==''} restored={restored}"
    ctx.check("kill switch: DISABLED silences recall + unify, then restored", c_killswitch)

    # -- install-hooks writes valid settings.json (on a COPY) + idempotent --
    def c_install_hooks():
        from . import hooks_install
        orig_settings = hooks_install.SETTINGS
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d) / "settings.json"
            # seed a pre-existing unrelated setting + a foreign hook to prove preservation
            tmp.write_text(json.dumps({
                "model": "opus",
                "hooks": {"UserPromptSubmit": [
                    {"matcher": "", "hooks": [{"type": "command", "command": "other-tool.py"}]}]},
            }), encoding="utf-8")
            try:
                hooks_install.SETTINGS = tmp
                hooks_install.install()
                hooks_install.install()  # idempotent: second run must not duplicate
                data = json.loads(tmp.read_text(encoding="utf-8"))
            finally:
                hooks_install.SETTINGS = orig_settings  # never leave it pointed at the temp file
        h = data.get("hooks", {})
        ups = h.get("UserPromptSubmit", [])
        # Detect our entries with the production matcher — a hardcoded "claude-memory/hooks/"
        # here counted zero entries whenever the repo directory had another name, calling a
        # correct install non-idempotent.
        ours = [e for e in ups if hooks_install._is_ours(e)]
        foreign_kept = any("other-tool.py" in x.get("command", "")
                           for e in ups for x in e.get("hooks", []))
        events_ok = all(k in h for k in ("UserPromptSubmit", "SessionStart", "SessionEnd", "PreCompact"))
        idempotent = len(ours) == 1  # exactly one of our recall entries despite two installs
        preserved = data.get("model") == "opus" and foreign_kept
        return (events_ok and idempotent and preserved,
                f"events_ok={events_ok} idempotent={idempotent} foreign_kept={foreign_kept}")
    ctx.check("install-hooks: valid + idempotent + preserves existing (on a copy)", c_install_hooks)

    # -- uninstall-hooks removes only our entries (on the same copy) --------
    def c_uninstall_hooks():
        from . import hooks_install
        orig_settings = hooks_install.SETTINGS
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d) / "settings.json"
            tmp.write_text(json.dumps({"model": "opus", "hooks": {"UserPromptSubmit": [
                {"matcher": "", "hooks": [{"type": "command", "command": "other-tool.py"}]}]}}),
                encoding="utf-8")
            try:
                hooks_install.SETTINGS = tmp
                hooks_install.install()
                hooks_install.uninstall()
                data = json.loads(tmp.read_text(encoding="utf-8"))
            finally:
                hooks_install.SETTINGS = orig_settings
        h = data.get("hooks", {})
        # Same production matcher as above: the hardcoded substring made this check pass
        # VACUOUSLY in a differently-named clone — it could not see the leftovers it guards
        # against, because it was looking for the wrong path shape.
        ours_left = any(hooks_install._is_ours(e) for e in h.get("UserPromptSubmit", []))
        foreign_kept = any("other-tool.py" in x.get("command", "")
                           for e in h.get("UserPromptSubmit", []) for x in e.get("hooks", []))
        return (not ours_left and foreign_kept and data.get("model") == "opus",
                f"ours_removed={not ours_left} foreign_kept={foreign_kept}")
    ctx.check("uninstall-hooks removes only our entries (on a copy)", c_uninstall_hooks)

    # -- Codex hook installer preserves foreign hooks and is idempotent -----
    def c_codex_hooks():
        from . import codex_hooks_install
        original = codex_hooks_install.HOOKS_FILE
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d) / "hooks.json"
            tmp.write_text(json.dumps({"hooks": {"UserPromptSubmit": [
                {"hooks": [{"type": "command", "command": "foreign-memory.py"}]}]}}),
                encoding="utf-8")
            try:
                codex_hooks_install.HOOKS_FILE = tmp
                codex_hooks_install.install()
                codex_hooks_install.install()
                data = json.loads(tmp.read_text(encoding="utf-8"))
            finally:
                codex_hooks_install.HOOKS_FILE = original
        hooks = data["hooks"]
        ours = [e for e in hooks["UserPromptSubmit"] if codex_hooks_install._is_ours(e)]
        foreign = any("foreign-memory.py" in (h.get("command") or "")
                      for e in hooks["UserPromptSubmit"] for h in e.get("hooks", []))
        complete = all(e in hooks for e in ("UserPromptSubmit", "SessionStart",
                                             "SessionEnd", "PreCompact"))
        context_limit = ours[0]["hooks"][0].get("additionalContextLimit", 0) if ours else 0
        return complete and len(ours) == 1 and foreign and context_limit >= cfg.recall.max_chars, (
            f"events_ok={complete} idempotent={len(ours) == 1} foreign_kept={foreign} "
            f"context_limit={context_limit}")
    ctx.check("install-codex-hooks is valid, idempotent, and preserves foreign hooks", c_codex_hooks)

    def c_verified_backup():
        import sqlite3
        from dataclasses import replace
        from .backup import create_backup, verify_snapshot
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            db = root / "source.db"
            conn = sqlite3.connect(db)
            conn.executescript("""
                CREATE TABLE sources(id INTEGER PRIMARY KEY);
                CREATE TABLE chunks(id INTEGER PRIMARY KEY);
                CREATE TABLE facts(id INTEGER PRIMARY KEY);
                INSERT INTO sources VALUES (1);
                INSERT INTO chunks VALUES (1);
                INSERT INTO facts VALUES (1);
            """)
            conn.commit(); conn.close()
            projects = root / "projects"; memory = projects / "C--code-test" / "memory"
            memory.mkdir(parents=True)
            (memory / "note.md").write_text("durable note\n", encoding="utf-8")
            test_cfg = replace(
                cfg, data_dir=root / "data",
                scope=replace(cfg.scope, claude_projects_dir=str(projects)),
                store=replace(cfg.store, backend="sqlite",
                              sqlite=replace(cfg.store.sqlite, path=str(db))),
            )
            made = create_backup(test_cfg, retention=2)
            checked = verify_snapshot(Path(made["snapshot"]))
            due = create_backup(test_cfg, if_due=True, retention=2)
        ok = (made.get("created") and checked.get("ok") and checked.get("notes") == 1
              and due.get("created") is False and checked["counts"]["chunks"] == 1)
        return ok, f"created={made.get('created')} verified={checked.get('ok')} due_skipped={not due.get('created')}"
    ctx.check("SQLite and curated-note backup is atomic, checksummed, verified, and debounced",
              c_verified_backup)

    # -- real query end-to-end via Retriever returns >0 distinct hits -------
    if store is None:
        ctx.skip("real query end-to-end returns hits", "store unavailable")
    elif n_chunks == 0:
        ctx.skip("real query end-to-end returns hits", empty_skip)
    else:
        def c_e2e():
            from .retriever import Retriever
            r = Retriever(cfg, store)
            res = r.search(real_q, tier="hot", exclude_session="no-such-session")
            facts = r.search_facts("memory")
            return len(res) > 0, f"{len(res)} chunk results, {len(facts)} fact results"
        ctx.check("real query end-to-end (Retriever) returns >0 hits", c_e2e)

    # -- WEDGE GUARDS -------------------------------------------------------
    # On 2026-07-29 the warm server was found wedged since 2026-07-22: the store lock was held
    # forever, /api/stats never answered (proved: 300s, no response), 73 threads were parked, and
    # every recall had been silently degrading to keyword-only for seven days. Nothing noticed,
    # because the watchdog probed with socket.create_connection() and a wedged server still
    # accepts TCP in 88ms. Each check below SEEDS THAT FAILURE and asserts the guard fires; if a
    # guard is ever weakened into something that cannot observe a wedge, this suite goes red.

    def c_lock_timeout():
        """A held store lock must raise StoreBusy, not block forever."""
        import threading
        import time as _t
        from .store.sqlite_store import LOCK_TIMEOUT_S, SqliteStore, StoreBusy
        s = SqliteStore.__new__(SqliteStore)
        s._lock = threading.RLock()
        holder_in = threading.Event()
        release = threading.Event()

        def hold():
            with s._lock:
                holder_in.set()
                release.wait(30)
        t = threading.Thread(target=hold, daemon=True)
        t.start()
        holder_in.wait(5)
        import claudemem.store.sqlite_store as mod
        prev = mod.LOCK_TIMEOUT_S
        mod.LOCK_TIMEOUT_S = 0.5  # keep the suite fast; the mechanism is identical
        t0 = _t.time()
        try:
            with s._locked():
                release.set()
                return False, "acquired a lock held by another thread (guard cannot fire)"
        except StoreBusy:
            dt = _t.time() - t0
            return (dt < 5), f"StoreBusy raised after {dt:.2f}s instead of blocking forever"
        finally:
            mod.LOCK_TIMEOUT_S = prev
            release.set()
    ctx.check("wedge guard: held store lock raises StoreBusy (does not hang)", c_lock_timeout)

    def c_timed_out_work_keeps_slot():
        """A timed-out worker must retain admission until the real thread exits."""
        import asyncio as _a
        import threading as _th
        from concurrent.futures import ThreadPoolExecutor
        from .dashboard import api as _api

        previous = (_api.HOOK_DEADLINE_S, _api._hook_slots, _api._hook_pool)
        release = _th.Event()
        test_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="selftest-memory-hook")
        _api.HOOK_DEADLINE_S = 0.05
        _api._hook_slots = _th.BoundedSemaphore(1)
        _api._hook_pool = test_pool

        def slow():
            release.wait(5)
            return {"value": "late"}

        async def exercise():
            first = await _api._bounded("recall", slow, {"value": "fallback"})
            second = await _api._bounded("recall", lambda: {"value": "wrongly admitted"},
                                         {"value": "fallback"})
            return first, second

        loop = _a.new_event_loop()
        try:
            first, second = loop.run_until_complete(exercise())
            held = first.get("timeout") and second.get("shed")
            return held, f"first_timeout={bool(first.get('timeout'))} second_shed={bool(second.get('shed'))}"
        finally:
            release.set()
            loop.run_until_complete(_a.sleep(0.1))
            loop.close()
            test_pool.shutdown(wait=True)
            _api.HOOK_DEADLINE_S, _api._hook_slots, _api._hook_pool = previous
    ctx.check("admission guard: timed-out vector work retains its slot until exit", c_timed_out_work_keeps_slot)

    def c_healthz_fires():
        """/healthz must return 503 when the store cannot answer - not 200, and not hang."""
        import asyncio as _a
        from fastapi.responses import JSONResponse
        from .dashboard import api as _api

        import time as _t

        class _WedgedStore:
            def counts(self):
                _t.sleep(6)  # never answers within the probe deadline

        class _WedgedState:
            store = _WedgedStore()

        prev = _api.get_state
        prev_deadline, prev_wedge = _api.PROBE_DEADLINE_S, _api.WEDGE_AFTER_S
        prev_ok = _api._last_store_ok["ts"]
        _api.get_state = lambda: _WedgedState()
        _api.PROBE_DEADLINE_S = 1.0
        _api.WEDGE_AFTER_S = 1.0
        _api._last_store_ok["ts"] = _t.time() - 600  # sustained: 10 min with no successful op
        # Time the COROUTINE, not asyncio.run(): run() ends with shutdown_default_executor(),
        # which joins the orphaned probe thread and would measure the wedge itself rather than
        # our response latency. The orphan is expected - to_thread cannot be cancelled, which is
        # exactly why _probe_sem exists so a second probe never spawns another one.
        loop = _a.new_event_loop()
        try:
            t0 = _t.time()
            res = loop.run_until_complete(_api.healthz())
            dt = _t.time() - t0
            code = getattr(res, "status_code", 200)
            ok = (code == 503) and dt < 4
            return ok, f"status={code} in {dt:.2f}s (want 503 in <4s)"
        finally:
            loop.close()
            _api.get_state = prev
            _api.PROBE_DEADLINE_S, _api.WEDGE_AFTER_S = prev_deadline, prev_wedge
            _api._last_store_ok["ts"] = prev_ok
    ctx.check("wedge guard: /healthz returns 503 on a SUSTAINED wedge", c_healthz_fires)

    def c_healthz_tolerates_busy():
        """A momentarily slow store must NOT be reported as wedged.

        Regression lock on a self-inflicted outage (2026-07-29): the first version of /healthz
        503'd on any probe slower than 2s. A probe that merely queued behind an in-flight recall
        does that routinely, so the supervisor restarted a healthy server every ~3 minutes and
        each restart dropped the models, making the next probe slower still. An over-firing guard
        is not a stricter guard - it is an outage with a health check attached.
        """
        import time as _t
        import asyncio as _a
        from .dashboard import api as _api

        class _SlowStore:
            def counts(self):
                _t.sleep(2)  # slower than the deadline, but the store IS alive

        class _SlowState:
            store = _SlowStore()
        prev = _api.get_state
        prev_deadline, prev_wedge = _api.PROBE_DEADLINE_S, _api.WEDGE_AFTER_S
        prev_ok = _api._last_store_ok["ts"]
        _api.get_state = lambda: _SlowState()
        _api.PROBE_DEADLINE_S = 0.5      # force the slow path
        _api.WEDGE_AFTER_S = 45.0        # but it answered recently, so it is not a wedge
        _api._last_store_ok["ts"] = _t.time()
        loop = _a.new_event_loop()
        try:
            res = loop.run_until_complete(_api.healthz())
            code = getattr(res, "status_code", 200)
            store = res.get("store") if isinstance(res, dict) else "?"
            return code == 200, f"status={code} store={store!r} (want 200: busy != wedged)"
        finally:
            loop.close()
            _api.get_state = prev
            _api.PROBE_DEADLINE_S, _api.WEDGE_AFTER_S = prev_deadline, prev_wedge
            _api._last_store_ok["ts"] = prev_ok
    ctx.check("wedge guard: /healthz tolerates a transiently SLOW store", c_healthz_tolerates_busy)

    def c_healthz_quiet_when_healthy():
        """...and must NOT fire on a healthy store, or it gets disabled within a week."""
        import asyncio as _a
        from .dashboard import api as _api

        class _OkStore:
            def counts(self):
                return {"chunks": 1}

        class _OkState:
            store = _OkStore()
        prev = _api.get_state
        _api.get_state = lambda: _OkState()
        try:
            res = _a.run(_api.healthz())
            code = getattr(res, "status_code", 200)
            return code == 200, f"status={code} on a healthy store (want 200)"
        finally:
            _api.get_state = prev
    ctx.check("wedge guard: /healthz stays quiet on a healthy store", c_healthz_quiet_when_healthy)

    def c_probe_is_not_port_only():
        """The session-start watchdog must reject a listening-but-wedged server.

        Regression lock on the exact 7-day outage: a socket-connect probe passes against any
        bound port. We stand up a server that ACCEPTS connections and answers 503, and require
        the watchdog to treat it as unhealthy.
        """
        import http.server
        import socket
        import threading

        class H(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(503)
                self.end_headers()
                self.wfile.write(b'{"ok":false,"store":"wedged"}')

            def log_message(self, *a):
                pass
        srv = http.server.HTTPServer(("127.0.0.1", 0), H)
        port = srv.server_address[1]
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        try:
            # the OLD probe would pass here - prove that, so the test is not vacuous
            with socket.create_connection(("127.0.0.1", port), timeout=1):
                old_probe_says_healthy = True
            import urllib.request
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/healthz", timeout=2) as r:
                    new_probe_says_healthy = (r.status == 200)
            except Exception:
                new_probe_says_healthy = False
            ok = old_probe_says_healthy and not new_probe_says_healthy
            return ok, ("old port-probe=healthy (the bug), new probe=unhealthy (the fix)"
                        if ok else
                        f"old={old_probe_says_healthy} new={new_probe_says_healthy}")
        finally:
            srv.shutdown()
    ctx.check("wedge guard: watchdog rejects a listening-but-503 server", c_probe_is_not_port_only)

    # ----------------------------------------------------------------------- #
    if verbose:
        print("=" * 60)
        print(f"  PASS={ctx.passed}  FAIL={ctx.failed}  SKIP={ctx.skipped}  "
              f"(total {ctx.passed + ctx.failed + ctx.skipped})")
        print("  RESULT:", "OK" if ctx.failed == 0 else "FAILURES PRESENT")
    return ctx.failed == 0


# --------------------------------------------------------------------------- #
# Hook subprocess helper: run a hook with the *current* interpreter, feeding a
# JSON event on stdin, returning stdout (the optional context line).
# --------------------------------------------------------------------------- #

def _run_hook(script: str, event: dict, timeout: float = 30.0) -> str:
    hook_path = ROOT / "hooks" / script
    env = _hook_env()
    try:
        proc = subprocess.run(
            [sys.executable, str(hook_path)],
            input=json.dumps(event).encode("utf-8"),
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            cwd=str(ROOT), env=env, timeout=timeout,
        )
        return proc.stdout.decode("utf-8", "replace")
    except subprocess.TimeoutExpired:
        return ""


def _hook_env() -> dict:
    import os
    env = dict(os.environ)
    env["PYTHONUTF8"] = "1"
    # Ensure the package is importable from the hook subprocess (it adds ROOT to sys.path
    # itself, but be explicit so the test is robust regardless of how it's launched).
    existing = env.get("PYTHONPATH", "")
    root = str(ROOT)
    env["PYTHONPATH"] = root if not existing else (root + os.pathsep + existing)
    return env


if __name__ == "__main__":
    sys.exit(0 if run_selftest(verbose=True) else 1)
