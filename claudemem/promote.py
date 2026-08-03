"""Auto-promotion mining: scan indexed transcript chunks for recurring lessons / gotchas /
decisions that are NOT yet captured as curated facts, cluster the near-duplicates, draft
concise candidate notes, and stash them as promotion candidates for human review.

Deterministic + fully offline: no external LLM. We pull lesson-bearing chunks with the
store's BM25 (works on both backends), extract the salient sentence with cue-phrase
heuristics, cluster by keyword (Jaccard) overlap, score by support (cluster size) and
novelty vs. existing facts, and draft a title + short body + a guessed type. Acceptance
(actually writing a note) happens in the dashboard — we never auto-write notes here.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field

from .config import Config, load_config
from .facts import load_notes
from .log import get_logger
from .store.base import Chunk, Store
from .store.factory import get_store
from .text import collapse_ws, extract_terms

log = get_logger(__name__)

# Cue phrases that mark a correction / lesson / gotcha / decision. Lowercased; matched as
# word-ish substrings. Order roughly by strength (used only for sentence selection ranking).
CUE_PHRASES: tuple[str, ...] = (
    "from now on", "make sure", "be sure to", "remember to", "remember that",
    "the issue was", "the problem was", "root cause", "turns out", "the fix was",
    "the bug was", "should always", "should never", "you must", "always", "never",
    "don't", "do not", "avoid", "instead of", "make sure to", "note that",
    "important:", "gotcha", "lesson", "in the future", "going forward",
    "next time", "the key is", "the trick is", "needs to", "has to",
)

# A precompiled alternation for fast cue detection (longest-first so multi-word cues win).
_CUE_RE = re.compile(
    "|".join(re.escape(p) for p in sorted(CUE_PHRASES, key=len, reverse=True)),
    re.IGNORECASE,
)

# Sentence splitter — coarse but adequate for transcript prose.
_SENT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'`])|\n+")

# Assistant "status update" chatter that matches a cue phrase but is NOT a durable lesson.
# These are progress narration, not feedback/decisions, so we drop them.
_NOISE_RE = re.compile(
    r"^[\s\"'(\[]*(let me|let'?s (?:do|run|add|make|update|look|check)|now (?:i|let|update|make|add|run|the)|"
    r"i'?ll|i'?m going to|i need to|i want to|i should|here'?s|memory saved|refresh|done[.! ]|fixed[.! ]|"
    r"this (?:turn|session|pass|run|sprint) |this is a comprehensive|the workflow (?:completed|confirms)|"
    r"which turns out|all (?:checks|green|tests)|next[,:]|step \d|\d+\.\s)"
    r"|[,;:]\s+(?:let me|let'?s do|i'?ll|i'?m going to|i need to)\b"      # narration after a clause
    r"|\s[—–-]\s*(?:let me|let'?s do|i'?ll|i'?m going to|i need to)\b",   # …or after a dash
    re.IGNORECASE,
)
# A sentence that's mostly code / paths / URLs / markup is poor note material.
_CODE_RE = re.compile(r"https?://|`[^`]+`|[A-Za-z0-9_]+\.(?:js|ts|py|html|css|md|json)\b|[{}<>]")

# Phrase -> heuristic note type. Corrective/imperative feedback -> 'feedback'; decisions /
# project facts -> 'project'; everything else -> 'reference'.
_FEEDBACK_CUES = (
    "from now on", "make sure", "be sure to", "remember to", "you must",
    "should always", "should never", "always", "never", "don't", "do not",
    "avoid", "in the future", "going forward", "next time", "remember that",
)
_PROJECT_CUES = (
    "the issue was", "the problem was", "root cause", "the fix was", "the bug was",
    "turns out", "decided", "we use", "we should use", "the key is", "the trick is",
)

def _cue_strength(cue: str) -> int:
    """Higher = a stronger/more durable rule cue (imperatives beat weak observations)."""
    try:
        return len(CUE_PHRASES) - CUE_PHRASES.index(cue.lower())
    except ValueError:
        return 0


# Min meaningful terms a candidate sentence must carry to be worth promoting.
_MIN_TERMS = 4
# How many chunks to pull per cue query, and the overall cap on drafted candidates.
_BM25_K = 60
_CAP = 15
# Score floor: without it, the fixed cap dredges the next-15-worst clusters from the corpus on
# every run, refilling the review queue with dross forever. Above-floor ≈ strong-cue user-voiced
# or multi-session lessons; the long tail of weak assistant observations never drafts.
_MIN_SCORE = 1.0
# Jaccard threshold for treating two lesson sentences as near-duplicates (same cluster).
_DUP_THRESHOLD = 0.45
# Keyword-overlap above this against an existing fact = "already captured" (not novel).
_FACT_OVERLAP_DROP = 0.6


@dataclass
class _Lesson:
    chunk: Chunk
    sentence: str            # the salient lesson sentence (cleaned)
    terms: set[str]          # meaningful term set, for clustering / novelty
    cue: str                 # the strongest cue phrase matched


@dataclass
class _Cluster:
    lessons: list[_Lesson] = field(default_factory=list)
    terms: set[str] = field(default_factory=set)

    def add(self, l: _Lesson) -> None:
        self.lessons.append(l)
        self.terms |= l.terms

    @property
    def support(self) -> int:
        # Support = number of *distinct sessions* the lesson recurs across (falls back to
        # chunk count when sessions are absent), so a single rambling message can't inflate it.
        sessions = {l.chunk.session_id for l in self.lessons if l.chunk.session_id}
        return max(len(sessions), 1) if sessions else len(self.lessons)

    @property
    def cue_strength(self) -> int:
        return max((_cue_strength(l.cue) for l in self.lessons), default=0)

    @property
    def has_user(self) -> bool:
        return any(l.chunk.role == "user" for l in self.lessons)


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    if not inter:
        return 0.0
    return inter / len(a | b)


def _overlap(a: set[str], b: set[str]) -> float:
    """Overlap coefficient: |a∩b| / min(|a|,|b|). More forgiving than Jaccard for the short,
    differently-padded sentences typical of transcripts — better recall when clustering."""
    if not a or not b:
        return 0.0
    inter = len(a & b)
    if not inter:
        return 0.0
    return inter / min(len(a), len(b))


def _is_quality(sent: str, role: str) -> bool:
    """Reject assistant progress-narration, mid-quote split artifacts, and markup-heavy text."""
    if _NOISE_RE.search(sent):
        return False
    # Sentence-splitter artifact: a chopped word-tail like 'ity is WAYYYY...' or 'n AMAZING...'
    # (a short lowercase token glued to the front by the coarse splitter). Users legitimately
    # start lowercase ("you MUST", "don't run"), so only reject a tiny leading fragment + space.
    if re.match(r'^["\']?[a-z]{1,3}\s', sent) or re.match(r'^["\']?[a-z]+["\']?\s+-\s', sent):
        return False
    # A pasted recap of earlier turns (quote-dash-quote chains) isn't a fresh lesson.
    if sent.count('" - "') >= 1 or sent.count('" "') >= 2:
        return False
    # Too much code/markup/path/URL noise for a clean lesson.
    if len(_CODE_RE.findall(sent)) >= 2:
        return False
    return True


_CLAUSE_SEPS = (". ", "; ", " — ", " - ", " · ", " | ", ", ")


def _clause_around(sent: str, cue_start: int, cue_end: int) -> str:
    """Tighten a long sentence to the clause containing the cue (split on . ; — - · | , ),
    so a buried directive like '...use 8888 for everything from now on**' becomes the title."""
    if len(sent) <= 140:
        return sent
    left = max((sent.rfind(d, 0, cue_start) + len(d) - 1 for d in _CLAUSE_SEPS
                if sent.rfind(d, 0, cue_start) != -1), default=-1)
    right_candidates = [sent.find(d, cue_end) for d in _CLAUSE_SEPS]
    right_candidates = [r for r in right_candidates if r != -1]
    right = min(right_candidates) if right_candidates else -1
    start = left + 1 if left != -1 else 0
    end = right if right != -1 else len(sent)
    clause = sent[start:end].strip(" .,;—-·|")
    return clause if len(clause) >= 16 else sent


def _best_sentence(text: str, role: str = "user") -> tuple[str, str] | None:
    """Return (sentence, strongest_cue) for the most lesson-like sentence in `text`, or None."""
    clean = collapse_ws(text)
    if not clean:
        return None
    best: tuple[str, str] | None = None
    best_rank = -1.0
    for raw in _SENT_RE.split(text):
        sent = collapse_ws(raw)
        if not (16 <= len(sent) <= 320):
            continue
        m = _CUE_RE.search(sent)
        if not m:
            continue
        if not _is_quality(sent, role):
            continue
        cue = m.group(0).lower()
        sent = _clause_around(sent, m.start(), m.end())
        # Prefer earlier (stronger) cue phrases; tie-break on a shorter, punchier sentence.
        strength = _cue_strength(cue)
        # User-authored corrections are the gold standard; nudge them ahead on ties.
        role_bonus = 0.5 if role == "user" else 0.0
        rank = (strength + role_bonus) * 1000 - len(sent)
        if rank > best_rank:
            best_rank = rank
            best = (sent, cue)
    return best


def _guess_type(cue: str, terms: set[str]) -> str:
    c = cue.lower()
    if any(c.startswith(p) or c == p for p in _FEEDBACK_CUES):
        return "feedback"
    if any(p in c for p in _PROJECT_CUES):
        return "project"
    return "reference"


def _title_from(sentence: str) -> str:
    """Condense a lesson sentence into a short imperative-ish title (<= ~80 chars)."""
    t = collapse_ws(sentence)
    # Strip leading markdown markers / quotes the splitter may have carried in.
    t = t.lstrip("*_#>-•·\"' \t")
    # Drop a leading conversational filler / cue lead-in for a cleaner title.
    t = re.sub(r"^(ok(ay)?|so|also|and|but|well|hey|please|just|now)[,:\s]+", "", t,
               flags=re.IGNORECASE)
    t = t.strip("*_ ").rstrip(".!?")
    if len(t) > 80:
        t = t[:77].rstrip() + "..."
    if t:
        t = t[0].upper() + t[1:]
    return t or "Recurring lesson"


def _canonical(cluster: _Cluster) -> _Lesson:
    """The representative lesson for a cluster: strongest cue, preferring user voice, then the
    most term-rich (specific) phrasing. Deterministic via the final sentence tie-break."""
    return max(
        cluster.lessons,
        key=lambda l: (_cue_strength(l.cue), l.chunk.role == "user", len(l.terms),
                       l.sentence.lower()),
    )


def _eligible_for_review(cluster: _Cluster) -> bool:
    """Single user corrections are signals; assistant-only lessons must recur."""
    return cluster.has_user or cluster.support >= 2


def _draft_body(cluster: _Cluster) -> str:
    """2-4 sentence body: the canonical lesson + supporting context across sessions."""
    canonical = _canonical(cluster)
    sentences: list[str] = [canonical.sentence.rstrip(".!?") + "."]

    # Add up to two distinct supporting paraphrases from other sessions.
    seen_norms = {canonical.sentence.lower()}
    for l in sorted(cluster.lessons, key=lambda l: len(l.terms), reverse=True):
        if len(sentences) >= 3:
            break
        norm = l.sentence.lower()
        if norm in seen_norms or _jaccard(l.terms, canonical.terms) > 0.85:
            continue
        seen_norms.add(norm)
        sentences.append(l.sentence.rstrip(".!?") + ".")

    n_sessions = cluster.support
    projects = sorted({l.chunk.project for l in cluster.lessons if l.chunk.project})
    if n_sessions > 1:
        proj_txt = f" in {', '.join(projects[:3])}" if projects else ""
        sentences.append(
            f"This came up across {n_sessions} sessions{proj_txt}; "
            f"consider capturing it as a standing rule."
        )
    return " ".join(sentences)[:800]


def _collect_lessons(cfg: Config, store: Store) -> list[_Lesson]:
    """Pull lesson-bearing text chunks via BM25 over the cue vocabulary, dedup by chunk id."""
    seen: dict[int, _Lesson] = {}
    # Query in a few cue groups so one giant OR query doesn't drown weaker cues.
    groups = [
        "always never don't avoid you must make sure be sure remember to from now on",
        "the issue was the problem was root cause the fix was the bug was turns out",
        "should always should never in the future going forward next time note that important",
        "gotcha lesson the key is the trick is instead of decided we use needs to has to",
    ]
    chunk_ids: set[int] = set()
    ranked: list = []
    for q in groups:
        try:
            ranked = store.search_bm25(q, _BM25_K, kinds=("text",))
        except Exception as e:
            log.warning("bm25 cue query failed: %s", e)
            continue
        for cand in ranked:
            chunk_ids.add(cand.chunk_id)

    for ch in store.get_chunks(chunk_ids):
        if ch.kind != "text" or not ch.content:
            continue
        picked = _best_sentence(ch.content, ch.role)
        if not picked:
            continue
        sentence, cue = picked
        terms = set(extract_terms(sentence))
        if len(terms) < _MIN_TERMS:
            continue
        seen[ch.id] = _Lesson(chunk=ch, sentence=sentence, terms=terms, cue=cue)
    return list(seen.values())


def _cluster(lessons: list[_Lesson]) -> list[_Cluster]:
    """Greedy single-pass clustering by keyword (Jaccard) overlap. Deterministic given input
    order, which we fix by sorting on term count then sentence text."""
    ordered = sorted(lessons, key=lambda l: (-len(l.terms), l.sentence.lower()))
    clusters: list[_Cluster] = []
    for l in ordered:
        best: _Cluster | None = None
        best_sim = 0.0
        for cl in clusters:
            sim = _overlap(l.terms, cl.terms)
            if sim > best_sim:
                best_sim = sim
                best = cl
        if best is not None and best_sim >= _DUP_THRESHOLD:
            best.add(l)
        else:
            nc = _Cluster()
            nc.add(l)
            clusters.append(nc)
    return clusters


def _fact_term_sets(cfg: Config, store: Store) -> list[set[str]]:
    """Term sets of existing curated facts (from notes on disk + indexed facts in the store),
    for novelty scoring. Both sources are unioned so we don't re-promote captured lessons."""
    sets: list[set[str]] = []
    try:
        for f in store.list_facts():
            blob = " ".join([f.title or "", f.description or "", f.body or ""])
            ts = set(extract_terms(blob))
            if ts:
                sets.append(ts)
    except Exception as e:
        log.warning("list_facts failed during novelty scoring: %s", e)
    try:
        for nd in load_notes(cfg):
            blob = " ".join([nd.title or "", nd.description or "", nd.body or ""])
            ts = set(extract_terms(blob))
            if ts:
                sets.append(ts)
    except Exception as e:
        log.warning("load_notes failed during novelty scoring: %s", e)
    return sets


def _novelty(terms: set[str], fact_sets: list[set[str]]) -> float:
    """1 - max keyword overlap to any existing fact. 1.0 = totally novel, 0.0 = already captured."""
    if not fact_sets or not terms:
        return 1.0
    return 1.0 - max(_jaccard(terms, fs) for fs in fact_sets)


def mine_candidates(cfg: Config | None = None, store: Store | None = None,
                    *, cap: int = _CAP) -> int:
    """Scan indexed transcript chunks for recurring, not-yet-captured lessons; draft candidate
    notes for the top `cap` clusters and persist them via store.add_promotion_candidate.

    Returns the number of candidates stored. Self-contained: builds config/store if not given
    (so `mem promote` can call it with no args).
    """
    cfg = cfg or load_config()
    store = store or get_store(cfg)

    lessons = _collect_lessons(cfg, store)
    log.info("promote: %d lesson-bearing chunks", len(lessons))
    if not lessons:
        return 0

    clusters = _cluster(lessons)
    fact_sets = _fact_term_sets(cfg, store)
    # Existing candidates (any status) suppress re-drafting: a rejected candidate must not
    # resurrect on the next mining run, and a pending one must not accumulate duplicates.
    # Overlap coefficient of the canonical sentence vs the candidate text — jaccard against the
    # cluster's term-union is far too weak to catch even an identical sentence.
    cand_sets: list[set[str]] = []
    try:
        for c in store.list_promotions():
            ts = set(extract_terms(f"{c.get('title', '')} {c.get('body', '')}"))
            if ts:
                cand_sets.append(ts)
    except Exception as e:
        log.warning("list_promotions failed during candidate dedup: %s", e)

    max_cue = max((_cue_strength(c) for c in CUE_PHRASES), default=1)
    scored: list[tuple[float, _Cluster, float, float, str]] = []  # (score, cluster, support, novelty, type)
    for cl in clusters:
        # One-off assistant prose is overwhelmingly progress narration with a lesson-shaped cue
        # ("reporting the root cause", "capturing this so..."). It may be useful transcript data,
        # but it is not durable-note material unless it recurs independently. A direct user
        # correction remains reviewable after one occurrence.
        if not _eligible_for_review(cl):
            continue
        novelty = _novelty(cl.terms, fact_sets)
        if novelty < (1.0 - _FACT_OVERLAP_DROP):
            continue  # essentially already captured as a fact
        canon_terms = _canonical(cl).terms
        if cand_sets and max((_overlap(canon_terms, cs) for cs in cand_sets), default=0.0) >= 0.7:
            continue  # already drafted (pending/accepted/rejected) on a previous run
        support = float(cl.support)
        # The signal that actually predicts a durable, promotable rule is the *quality* of the
        # cue (imperatives like "always/never/from now on/you must" >> weak observations) and
        # whether the user said it (a direct correction). Recurrence (support) and novelty are
        # multipliers on top — not the primary driver — so long one-off assistant musings don't
        # outrank a crisp user rule. A non-recurring lesson is still eligible (support 1).
        quality = cl.cue_strength / max_cue                    # 0..1
        if cl.has_user:
            quality = min(1.0, quality + 0.35)                 # strong nudge for user corrections
        score = (1.0 + math.log1p(support)) * (0.5 + 0.5 * novelty) * (0.2 + 0.8 * quality)
        ctype = _guess_type(_canonical(cl).cue, cl.terms)
        if ctype == "feedback" and not cl.has_user:
            # a "rule" no user ever voiced is usually assistant self-talk; demote it
            ctype = "reference"
            score *= 0.5
        scored.append((score, cl, support, novelty, ctype))

    scored = [s for s in scored if s[0] >= _MIN_SCORE]
    # Deterministic ordering: score desc, then support desc, then a stable text key.
    scored.sort(key=lambda x: (-x[0], -x[2], _canonical(x[1]).sentence.lower()))

    written = 0
    for score, cl, support, novelty, ctype in scored[:cap]:
        canonical = _canonical(cl)
        title = _title_from(canonical.sentence)
        body = _draft_body(cl)
        sample = cl.lessons[: min(8, len(cl.lessons))]
        support_meta = {
            "method": "keyword-cluster",
            "cluster_size": len(cl.lessons),
            "sessions": sorted({l.chunk.session_id for l in cl.lessons if l.chunk.session_id}),
            "projects": sorted({l.chunk.project for l in cl.lessons if l.chunk.project}),
            "chunk_ids": [l.chunk.id for l in sample],
            "cues": [c for c, _ in Counter(l.cue for l in cl.lessons).most_common(4)],
            "support": support,
            "novelty": round(novelty, 4),
            "examples": [l.sentence[:200] for l in sample[:3]],
        }
        try:
            store.add_promotion_candidate(title=title, body=body, type=ctype,
                                          support=support_meta, score=round(score, 4))
            written += 1
        except Exception as e:
            log.warning("add_promotion_candidate failed for %r: %s", title, e)
    log.info("promote: stored %d candidates", written)
    return written


if __name__ == "__main__":  # pragma: no cover - manual smoke test
    print(mine_candidates(), "candidates")
