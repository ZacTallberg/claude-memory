from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from .models import RecallEvidence, RecallQuery, RecallResult, RecallScope
from .recall import RecallEngine


class EvidenceLabel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    stable_id: str = Field(min_length=1)
    memory_type: str = "event"
    relevance: int = Field(default=3, ge=1, le=3)
    provider: str | None = None
    project_id: str | None = None
    session_id: str | None = None
    role: str | None = None
    authority: str | None = None
    valid_from: datetime | None = None
    valid_to: datetime | None = None
    text_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")

    @property
    def key(self) -> tuple[str, str]:
        return self.memory_type, self.stable_id


class EvidenceGroup(BaseModel):
    model_config = ConfigDict(extra="forbid")

    group_id: str = Field(min_length=1)
    required: bool = True
    items: tuple[EvidenceLabel, ...] = Field(min_length=1)


class ForbiddenEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    stable_id: str = Field(min_length=1)
    memory_type: str = "event"
    reason: str = Field(min_length=1)

    @property
    def key(self) -> tuple[str, str]:
        return self.memory_type, self.stable_id


class EvalCase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    case_id: str = Field(min_length=1)
    split: Literal["dev", "test"]
    category: Literal[
        "user_facts_values",
        "assistant_decisions",
        "cross_session_project",
        "temporal_updates",
        "procedures_gotchas",
        "attribution",
        "abstention_negatives",
        "contamination_forgetting",
    ]
    ability: str = Field(min_length=1)
    query: str = Field(min_length=1)
    query_time: datetime | None = None
    current_project_id: str | None = None
    scope: RecallScope = Field(default_factory=RecallScope)
    answerable: bool = True
    evidence_groups: tuple[EvidenceGroup, ...] = ()
    forbidden_evidence: tuple[ForbiddenEvidence, ...] = ()
    limit: int = Field(default=6, ge=1, le=50)
    max_chars: int = Field(default=8_000, ge=256, le=100_000)
    notes: str | None = None

    @model_validator(mode="after")
    def evidence_matches_answerability(self):
        required = [group for group in self.evidence_groups if group.required]
        if self.answerable and not required:
            raise ValueError("answerable cases require at least one required evidence group")
        if not self.answerable and required:
            raise ValueError("unanswerable cases cannot require evidence")
        return self

    def recall_query(self) -> RecallQuery:
        return RecallQuery(
            query=self.query,
            current_project_id=self.current_project_id,
            as_of=self.query_time,
            scope=self.scope,
            limit=self.limit,
            max_chars=self.max_chars,
        )


class CaseScore(BaseModel):
    case_id: str
    split: str
    category: str
    strict_pass: bool
    required_groups: int
    groups_hit: int
    ndcg_at_k: float | None
    reciprocal_rank: float | None
    abstained: bool
    false_context: bool
    forbidden_hits: tuple[str, ...]
    attribution_fields_checked: int
    attribution_fields_correct: int
    result_refs: tuple[str, ...]
    mode: str
    elapsed_ms: float


class EvalSummary(BaseModel):
    cases: int
    answerable_cases: int
    negative_cases: int
    strict_passes: int
    strict_rate: float
    strict_wilson_lower_95: float
    evidence_group_recall_at_k: float
    mean_ndcg_at_k: float
    mean_reciprocal_rank: float
    false_context_rate: float
    forbidden_hits: int
    attribution_exactness: float
    category_strict_rate: dict[str, float]
    mode_counts: dict[str, int]
    latency_ms: dict[str, float]
    case_scores: tuple[CaseScore, ...]


def load_cases(path: Path) -> list[EvalCase]:
    cases: list[EvalCase] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                case = EvalCase.model_validate_json(line)
            except ValidationError as error:
                detail = json.dumps(
                    error.errors(include_url=False, include_context=False, include_input=False),
                    sort_keys=True,
                )
                raise ValueError(
                    f"invalid evaluation case at line {line_number}: {detail}"
                ) from error
            if case.case_id in seen:
                raise ValueError(f"duplicate evaluation case_id: {case.case_id}")
            seen.add(case.case_id)
            cases.append(case)
    if not cases:
        raise ValueError("evaluation case file is empty")
    return cases


def validate_gold64(cases: list[EvalCase]) -> dict[str, int]:
    if len(cases) != 64:
        raise ValueError(f"Gold-64 requires exactly 64 cases, found {len(cases)}")
    split_counts = Counter(case.split for case in cases)
    if split_counts != {"dev": 24, "test": 40}:
        raise ValueError(f"Gold-64 split must be 24 dev / 40 test, found {dict(split_counts)}")
    category_counts = Counter(case.category for case in cases)
    incorrect = {name: amount for name, amount in category_counts.items() if amount != 8}
    if len(category_counts) != 8 or incorrect:
        raise ValueError(
            "Gold-64 requires eight cases in each category; "
            f"found {dict(sorted(category_counts.items()))}"
        )
    return dict(sorted(category_counts.items()))


def _evidence_key(item: RecallEvidence) -> tuple[str, str]:
    return item.memory_type, item.ref_id


def _dcg(relevances: list[int]) -> float:
    return sum(
        (2**relevance - 1) / math.log2(rank + 2) for rank, relevance in enumerate(relevances)
    )


def _score_case(case: EvalCase, result: RecallResult) -> CaseScore:
    ranked = list(result.evidence[: case.limit])
    ranked_keys = [_evidence_key(item) for item in ranked]
    labels = {item.key: item for group in case.evidence_groups for item in group.items}
    required = [group for group in case.evidence_groups if group.required]
    groups_hit = sum(
        1 for group in required if any(item.key in ranked_keys for item in group.items)
    )
    if case.answerable:
        strict = groups_hit == len(required)
        relevance = [labels[key].relevance if key in labels else 0 for key in ranked_keys]
        ideal = sorted((item.relevance for item in labels.values()), reverse=True)[: case.limit]
        ideal_dcg = _dcg(ideal)
        ndcg = _dcg(relevance) / ideal_dcg if ideal_dcg else 0.0
        first_relevant = next(
            (rank for rank, key in enumerate(ranked_keys, start=1) if key in labels), None
        )
        reciprocal_rank = 1.0 / first_relevant if first_relevant else 0.0
        false_context = False
    else:
        strict = result.abstained and not ranked
        ndcg = None
        reciprocal_rank = None
        false_context = bool(ranked) or not result.abstained

    forbidden = {item.key: item for item in case.forbidden_evidence}
    forbidden_hits = tuple(forbidden[key].stable_id for key in ranked_keys if key in forbidden)
    checked = 0
    correct = 0
    for retrieved in ranked:
        label = labels.get(_evidence_key(retrieved))
        if not label:
            continue
        for field in ("provider", "project_id", "session_id", "role", "authority"):
            expected = getattr(label, field)
            if expected is None:
                continue
            checked += 1
            correct += int(getattr(retrieved, field) == expected)

    return CaseScore(
        case_id=case.case_id,
        split=case.split,
        category=case.category,
        strict_pass=strict and not forbidden_hits,
        required_groups=len(required),
        groups_hit=groups_hit,
        ndcg_at_k=ndcg,
        reciprocal_rank=reciprocal_rank,
        abstained=result.abstained,
        false_context=false_context,
        forbidden_hits=forbidden_hits,
        attribution_fields_checked=checked,
        attribution_fields_correct=correct,
        result_refs=tuple(item.ref_id for item in ranked),
        mode=result.mode,
        elapsed_ms=result.elapsed_ms,
    )


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _wilson_lower(successes: int, total: int, z: float = 1.959963984540054) -> float:
    if total <= 0:
        return 0.0
    proportion = successes / total
    denominator = 1 + z * z / total
    center = proportion + z * z / (2 * total)
    margin = z * math.sqrt(proportion * (1 - proportion) / total + z * z / (4 * total * total))
    return max(0.0, (center - margin) / denominator)


def summarize(scores: list[CaseScore]) -> EvalSummary:
    if not scores:
        raise ValueError("cannot summarize an empty evaluation run")
    answerable = [score for score in scores if score.required_groups]
    negatives = [score for score in scores if not score.required_groups]
    strict = sum(score.strict_pass for score in scores)
    total_groups = sum(score.required_groups for score in answerable)
    groups_hit = sum(score.groups_hit for score in answerable)
    ndcg = [score.ndcg_at_k for score in answerable if score.ndcg_at_k is not None]
    reciprocal = [
        score.reciprocal_rank for score in answerable if score.reciprocal_rank is not None
    ]
    checked = sum(score.attribution_fields_checked for score in scores)
    correct = sum(score.attribution_fields_correct for score in scores)
    categories: dict[str, list[bool]] = defaultdict(list)
    for score in scores:
        categories[score.category].append(score.strict_pass)
    latencies = [score.elapsed_ms for score in scores]
    return EvalSummary(
        cases=len(scores),
        answerable_cases=len(answerable),
        negative_cases=len(negatives),
        strict_passes=strict,
        strict_rate=strict / len(scores),
        strict_wilson_lower_95=_wilson_lower(strict, len(scores)),
        evidence_group_recall_at_k=groups_hit / total_groups if total_groups else 0.0,
        mean_ndcg_at_k=sum(ndcg) / len(ndcg) if ndcg else 0.0,
        mean_reciprocal_rank=sum(reciprocal) / len(reciprocal) if reciprocal else 0.0,
        false_context_rate=(
            sum(score.false_context for score in negatives) / len(negatives) if negatives else 0.0
        ),
        forbidden_hits=sum(len(score.forbidden_hits) for score in scores),
        attribution_exactness=correct / checked if checked else 1.0,
        category_strict_rate={
            category: sum(values) / len(values) for category, values in sorted(categories.items())
        },
        mode_counts=dict(sorted(Counter(score.mode for score in scores).items())),
        latency_ms={
            "p50": _percentile(latencies, 0.50),
            "p95": _percentile(latencies, 0.95),
            "p99": _percentile(latencies, 0.99),
            "max": max(latencies),
        },
        case_scores=tuple(scores),
    )


def run_evaluation(engine: RecallEngine, cases: list[EvalCase]) -> EvalSummary:
    return summarize([_score_case(case, engine.recall(case.recall_query())) for case in cases])


def write_schema(path: Path) -> None:
    path.write_text(
        json.dumps(EvalCase.model_json_schema(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
