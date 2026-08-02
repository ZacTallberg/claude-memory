from __future__ import annotations

import os
from datetime import UTC, datetime

import pytest

import system_memory.supervisor as supervisor_module
from system_memory.supervisor import (
    OwnedChild,
    OwnershipLost,
    Supervisor,
    SupervisorConfig,
    SupervisorState,
    process_fingerprint,
)


def test_process_fingerprint_is_stable_for_current_process():
    first = process_fingerprint(os.getpid())
    second = process_fingerprint(os.getpid())
    assert first
    assert first == second


def test_force_termination_refuses_pid_reuse(monkeypatch, tmp_path):
    supervisor = Supervisor(SupervisorConfig(root=tmp_path), build_id="test")
    child = OwnedChild(pid=12345, nonce="expected", fingerprint="original", process=None)
    killed = []
    monkeypatch.setattr(supervisor_module, "process_fingerprint", lambda _pid: "replacement")
    monkeypatch.setattr(os, "kill", lambda *args: killed.append(args))

    with pytest.raises(OwnershipLost, match="creation identity changed"):
        supervisor._terminate_owned(child)
    assert killed == []


def test_adoption_requires_pid_nonce_and_creation_fingerprint(monkeypatch, tmp_path):
    supervisor = Supervisor(SupervisorConfig(root=tmp_path), build_id="test")
    state = SupervisorState(
        status="running",
        supervisor_pid=1,
        supervisor_nonce="old-supervisor",
        child_pid=321,
        child_nonce="child-nonce",
        child_fingerprint="fingerprint",
        build_id="test",
        started_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
    )
    monkeypatch.setattr(supervisor, "_live_identity", lambda: (321, "child-nonce", "test"))
    monkeypatch.setattr(supervisor, "_read_state", lambda: state)
    monkeypatch.setattr(supervisor_module, "process_fingerprint", lambda _pid: "fingerprint")
    monkeypatch.setattr(supervisor, "_write_state", lambda *args, **kwargs: None)
    monkeypatch.setattr(supervisor, "_log", lambda *args, **kwargs: None)

    adopted = supervisor._adopt_or_start()

    assert adopted.pid == 321
    assert adopted.nonce == "child-nonce"
    assert adopted.process is None
