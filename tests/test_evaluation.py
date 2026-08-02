from __future__ import annotations

import json

import pytest

from system_memory.evaluation import (
    EvalCase,
    EvidenceGroup,
    EvidenceLabel,
    ForbiddenEvidence,
    _score_case,
    load_cases,
    summarize,
    validate_gold64,
)
from system_memory.models import RecallEvidence, RecallResult


def _result(*refs: str, abstained: bool = False) -> RecallResult:
    return RecallResult(
        request_id="request-1",
        mode="empty" if abstained else "keyword_only",
        evidence=tuple(
            RecallEvidence(
                document_id=f"doc-{ref}",
                memory_type="event",
                ref_id=ref,
                title="title",
                text="body",
                provider="codex",
                project_id="project-a",
                session_id="session-a",
                role="user",
                authority="user_authored",
                occurred_at=None,
                score=1.0,
                reasons=("test",),
            )
            for ref in refs
        ),
        elapsed_ms=10.0,
        generation_id="generation-1",
        abstained=abstained,
        reason="test abstention" if abstained else None,
    )


def _answerable(case_id: str = "case-1") -> EvalCase:
    return EvalCase(
        case_id=case_id,
        split="dev",
        category="cross_session_project",
        ability="two-source synthesis",
        query="What was decided across the two sessions?",
        evidence_groups=(
            EvidenceGroup(
                group_id="first",
                items=(
                    EvidenceLabel(
                        stable_id="event-a",
                        provider="codex",
                        project_id="project-a",
                        session_id="session-a",
                        role="user",
                        authority="user_authored",
                    ),
                ),
            ),
            EvidenceGroup(
                group_id="second",
                items=(EvidenceLabel(stable_id="event-b", relevance=2),),
            ),
        ),
        forbidden_evidence=(ForbiddenEvidence(stable_id="obsolete", reason="superseded"),),
    )


def test_evidence_groups_are_anded_and_alternatives_are_ored():
    case = _answerable()
    partial = _score_case(case, _result("event-a"))
    complete = _score_case(case, _result("event-b", "event-a"))

    assert partial.groups_hit == 1
    assert partial.strict_pass is False
    assert complete.groups_hit == 2
    assert complete.strict_pass is True
    assert complete.attribution_fields_checked == 5
    assert complete.attribution_fields_correct == 5
    assert complete.ndcg_at_k is not None and complete.ndcg_at_k > 0


def test_negatives_forbidden_evidence_and_summary_are_explicit():
    negative = EvalCase(
        case_id="negative",
        split="dev",
        category="abstention_negatives",
        ability="false premise awareness",
        query="What is the invented deployment rule?",
        answerable=False,
    )
    correct_negative = _score_case(negative, _result(abstained=True))
    false_context = _score_case(negative, _result("unrelated"))
    forbidden = _score_case(_answerable(), _result("event-a", "event-b", "obsolete"))
    summary = summarize([correct_negative, false_context, forbidden])

    assert correct_negative.strict_pass is True
    assert false_context.false_context is True
    assert forbidden.strict_pass is False
    assert forbidden.forbidden_hits == ("obsolete",)
    assert summary.false_context_rate == 0.5
    assert summary.forbidden_hits == 1
    assert summary.strict_wilson_lower_95 < summary.strict_rate


def test_loader_rejects_duplicates_and_gold64_requires_real_shape(tmp_path):
    case = _answerable()
    path = tmp_path / "cases.jsonl"
    path.write_text(
        case.model_dump_json() + "\n" + case.model_dump_json() + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate"):
        load_cases(path)
    with pytest.raises(ValueError, match="exactly 64"):
        validate_gold64([case])


def test_eval_case_schema_rejects_answerable_case_without_evidence():
    payload = {
        "case_id": "missing-evidence",
        "split": "dev",
        "category": "user_facts_values",
        "ability": "direct recall",
        "query": "What does the user prefer?",
        "answerable": True,
    }
    with pytest.raises(ValueError, match="required evidence group"):
        EvalCase.model_validate_json(json.dumps(payload))
