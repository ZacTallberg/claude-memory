from __future__ import annotations

from datetime import UTC, datetime

import pytest

from system_memory.models import Authority, ClaimOperation, ClaimProposal, EvidenceRef
from system_memory.store import EvidenceError

from .conftest import make_event


def proposal(event_id: str, **changes) -> ClaimProposal:
    values = {
        "operation": ClaimOperation.ADD,
        "subject": "user",
        "predicate": "preferred accent color",
        "value": "coral",
        "rendering": "The user prefers coral as the accent color.",
        "authority": Authority.USER_DECLARATION,
        "confidence": 1.0,
        "valid_from": datetime(2026, 8, 2, tzinfo=UTC),
        "created_by": "consolidator-v2",
        "evidence": (EvidenceRef(event_id=event_id),),
    }
    values.update(changes)
    return ClaimProposal(**values)


def test_claim_updates_are_temporal_revisions_not_overwrites(store):
    coral_event = store.ingest(make_event(content="I prefer coral as the accent color."))
    first = store.propose_claim(proposal(coral_event.event_id))
    assert first is not None
    store.accept_claim(first, reviewer="user-review")

    teal_event = store.ingest(
        make_event(
            event_key="message-2",
            content="I now prefer teal as the accent color.",
            offset=200,
        )
    )
    with store.database.read() as connection:
        series_id = connection.execute(
            "SELECT series_id FROM claim_revisions WHERE id=?", (first,)
        ).fetchone()[0]
    second = store.propose_claim(
        proposal(
            teal_event.event_id,
            series_id=series_id,
            operation=ClaimOperation.SUPERSEDE,
            value="teal",
            rendering="The user now prefers teal as the accent color.",
            predecessor_revision_id=first,
            valid_from=datetime(2026, 8, 3, tzinfo=UTC),
        )
    )
    assert second is not None
    store.accept_claim(second, reviewer="user-review")

    with store.database.read() as connection:
        rows = connection.execute(
            "SELECT id,state,value_json,valid_to,reviewed_by FROM claim_revisions "
            "WHERE series_id=? ORDER BY revision_no",
            (series_id,),
        ).fetchall()
    assert [row["state"] for row in rows] == ["superseded", "accepted"]
    assert rows[0]["valid_to"] is not None
    assert rows[1]["value_json"] == '"teal"'
    assert rows[1]["reviewed_by"] == "user-review"


def test_missing_evidence_rolls_back_entire_claim(store):
    with pytest.raises(EvidenceError):
        store.propose_claim(proposal("evt_missing"))
    assert store.counts()["claim_revisions"] == 0


def test_noop_is_not_persisted_as_fake_work(store):
    event = store.ingest(make_event())
    assert store.propose_claim(proposal(event.event_id, operation=ClaimOperation.NOOP)) is None
    assert store.counts()["claim_revisions"] == 0
