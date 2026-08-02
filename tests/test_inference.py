from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from system_memory.inference import (
    InferenceScheduler,
    InferenceShed,
    InferenceTimeout,
    WorkKind,
)


def test_timed_out_work_keeps_its_admission_slot_until_model_exit():
    scheduler = InferenceScheduler(capacity=1)
    release = threading.Event()

    def slow():
        release.wait(timeout=5)
        return "finished"

    with pytest.raises(InferenceTimeout):
        scheduler.submit(WorkKind.DOCUMENT_EMBEDDING, slow, timeout=0.01)
    with pytest.raises(InferenceShed):
        scheduler.submit(WorkKind.QUERY_EMBEDDING, lambda: "query", timeout=0.1)
    release.set()
    deadline = time.monotonic() + 2
    while scheduler.queued and time.monotonic() < deadline:
        time.sleep(0.01)
    # Semaphore release occurs just after queue task completion; allow that bounded handoff.
    time.sleep(0.02)
    assert scheduler.submit(WorkKind.QUERY_EMBEDDING, lambda: "query", timeout=1) == "query"
    scheduler.close()


def test_waiting_query_runs_before_waiting_background_batch():
    scheduler = InferenceScheduler(capacity=4)
    release = threading.Event()
    order: list[str] = []

    def first_document():
        release.wait(timeout=5)
        order.append("first-document")
        return 1

    def second_document():
        order.append("second-document")
        return 2

    def query():
        order.append("query")
        return 3

    with ThreadPoolExecutor(max_workers=3) as pool:
        first = pool.submit(
            scheduler.submit,
            WorkKind.DOCUMENT_EMBEDDING,
            first_document,
            timeout=2,
        )
        time.sleep(0.03)
        second = pool.submit(
            scheduler.submit,
            WorkKind.DOCUMENT_EMBEDDING,
            second_document,
            timeout=2,
        )
        requested = pool.submit(
            scheduler.submit,
            WorkKind.QUERY_EMBEDDING,
            query,
            timeout=2,
        )
        time.sleep(0.03)
        release.set()
        assert first.result() == 1
        assert requested.result() == 3
        assert second.result() == 2
    assert order == ["first-document", "query", "second-document"]
    scheduler.close()
