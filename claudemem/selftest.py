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


def _synthetic_note(tmpdir: Path) -> Path:
    """Write a synthetic curated note (with frontmatter) to a throwaway dir for parse testing."""
    p = tmpdir / "synthetic_note.md"
    p.write_text(
        "---\n"
        "name: synthetic-char-cap\n"
        "description: How the recall envelope respects the character cap.\n"
        "metadata:\n"
        "  type: feedback\n"
        "  originSessionId: sess-xyz\n"
        "tags: [recall, budget]\n"
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

    # -- curated note parse: frontmatter + nested type + wikilinks ----------
    def c_note_parse():
        from .facts import load_note
        with tempfile.TemporaryDirectory() as d:
            p = _synthetic_note(Path(d))
            nd = load_note(p, "synthetic-project")
        return (nd is not None and nd.type == "feedback" and nd.origin_session_id == "sess-xyz"
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

    # -- BM25 search returns hits (real DB) ---------------------------------
    real_q = "memory recall hook index retriever postgres embedding"
    if store is None:
        ctx.skip("BM25 search returns hits", "store unavailable")
    else:
        def c_bm25():
            hits = store.search_bm25(real_q, cfg.recall.bm25_k)
            return len(hits) > 0, f"{len(hits)} bm25 hits"
        ctx.check("BM25 search returns hits (populated DB)", c_bm25)

    # -- vector search returns hits (real DB; needs embedder) ---------------
    if store is None:
        ctx.skip("vector search returns hits", "store unavailable")
    elif not embed_ok:
        ctx.skip("vector search returns hits", "embedder unavailable")
    else:
        def c_vec():
            qv = embedder.embed_query(real_q)
            hits = store.search_vector(qv, cfg.recall.vector_k)
            return len(hits) > 0, f"{len(hits)} vector hits"
        ctx.check("vector search returns hits (populated DB)", c_vec)

    # -- hybrid fuse returns hits (real DB) ---------------------------------
    if store is None:
        ctx.skip("hybrid search returns hits", "store unavailable")
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
    def c_recall_hook():
        event = {"prompt": real_q, "cwd": "C:/code/claude-memory", "session_id": "selftest-live"}
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
    ctx.check("recall hook (subprocess) emits valid additionalContext envelope", c_recall_hook)

    # -- trivial prompt -> no output ----------------------------------------
    def c_recall_trivial():
        event = {"prompt": "hi the a", "cwd": "C:/code/claude-memory", "session_id": "selftest-live"}
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
        event = {"cwd": "C:/code/claude-memory", "session_id": "selftest-live", "source": "startup"}
        out = _run_hook("unify.py", event)
        if not out.strip():
            return False, "unify emitted nothing (expected a memory-map from populated facts)"
        env = json.loads(out)
        ac = env.get("hookSpecificOutput", {}).get("additionalContext", "")
        return ("<memory-map" in ac and env["hookSpecificOutput"]["hookEventName"] == "SessionStart",
                f"additionalContext len={len(ac)}")
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
        ours = [e for e in ups if any("claude-memory/hooks/" in
                                      x.get("command", "").replace("\\", "/")
                                      for x in e.get("hooks", []))]
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
        ours_left = any("claude-memory/hooks/" in x.get("command", "").replace("\\", "/")
                        for e in h.get("UserPromptSubmit", []) for x in e.get("hooks", []))
        foreign_kept = any("other-tool.py" in x.get("command", "")
                           for e in h.get("UserPromptSubmit", []) for x in e.get("hooks", []))
        return (not ours_left and foreign_kept and data.get("model") == "opus",
                f"ours_removed={not ours_left} foreign_kept={foreign_kept}")
    ctx.check("uninstall-hooks removes only our entries (on a copy)", c_uninstall_hooks)

    # -- real query end-to-end via Retriever returns >0 distinct hits -------
    if store is None:
        ctx.skip("real query end-to-end returns hits", "store unavailable")
    else:
        def c_e2e():
            from .retriever import Retriever
            r = Retriever(cfg, store)
            res = r.search(real_q, tier="hot", exclude_session="no-such-session")
            facts = r.search_facts("memory")
            return len(res) > 0, f"{len(res)} chunk results, {len(facts)} fact results"
        ctx.check("real query end-to-end (Retriever) returns >0 hits", c_e2e)

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
