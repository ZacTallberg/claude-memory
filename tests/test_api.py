from __future__ import annotations

from fastapi.testclient import TestClient

from system_memory.api import create_app
from system_memory.auth import CredentialStore
from system_memory.models import Authority, Role
from system_memory.settings import Settings

from .conftest import make_event


def client_and_tokens(store, tmp_path, *, shutdown_callback=None):
    credentials = CredentialStore(store.database)
    _, admin = credentials.create(actor_id="admin", label="admin", scopes={"*"})
    _, worker = credentials.create(
        actor_id="codex-main",
        label="codex",
        scopes={"read", "recall", "ingest:self"},
    )
    settings = Settings(root=tmp_path, port=7788, request_body_limit=2_000)
    app = create_app(
        settings,
        store=store,
        instance_nonce="test-instance",
        shutdown_callback=shutdown_callback,
    )
    return TestClient(app), admin, worker


def bearer(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def test_liveness_is_public_but_private_routes_require_credentials(store, tmp_path):
    client, _, _ = client_and_tokens(store, tmp_path)
    live = client.get("/livez")
    assert live.status_code == 200
    assert live.json()["nonce"] == "test-instance"
    assert client.get("/v1/stats").status_code == 401


def test_ingest_identity_is_bound_to_credential(store, tmp_path):
    client, admin, worker = client_and_tokens(store, tmp_path)
    own_event = make_event().model_copy(
        update={"role": Role.ASSISTANT, "authority": Authority.ASSISTANT_SYNTHESIS}
    )
    own = client.post("/v1/events", headers=bearer(worker), json=own_event.model_dump(mode="json"))
    assert own.status_code == 200

    impersonated = own_event.model_copy(
        update={"provider_event_id": "impostor", "agent_id": "claude-main"}
    )
    denied = client.post(
        "/v1/events",
        headers=bearer(worker),
        json=impersonated.model_dump(mode="json"),
    )
    assert denied.status_code == 403

    allowed = client.post(
        "/v1/events",
        headers=bearer(admin),
        json=impersonated.model_dump(mode="json"),
    )
    assert allowed.status_code == 200


def test_provider_credential_cannot_assert_user_authority_without_hook_scope(store, tmp_path):
    client, _, worker = client_and_tokens(store, tmp_path)
    user_event = make_event(event_key="forged-user-authority")
    denied = client.post(
        "/v1/events", headers=bearer(worker), json=user_event.model_dump(mode="json")
    )
    assert denied.status_code == 403

    credentials = CredentialStore(store.database)
    _, hook = credentials.create(
        actor_id="codex-main",
        label="trusted prompt hook",
        scopes={"ingest:user-authored"},
    )
    allowed = client.post(
        "/v1/events", headers=bearer(hook), json=user_event.model_dump(mode="json")
    )
    assert allowed.status_code == 200


def test_origin_host_and_body_boundaries(store, tmp_path):
    client, admin, _ = client_and_tokens(store, tmp_path)
    payload = make_event().model_dump(mode="json")
    bad_origin = client.post(
        "/v1/events",
        headers={**bearer(admin), "Origin": "https://evil.example"},
        json=payload,
    )
    assert bad_origin.status_code == 403

    bad_host = client.get("/livez", headers={"Host": "evil.example"})
    assert bad_host.status_code == 400

    oversized = client.post(
        "/v1/events",
        headers={**bearer(admin), "Content-Type": "application/json"},
        content="x" * 2_001,
    )
    assert oversized.status_code == 413


def test_ready_and_recall_report_actual_available_mode(store, tmp_path):
    client, admin, _ = client_and_tokens(store, tmp_path)
    event = store.ingest(make_event(content="A global system memory remembers this phrase."))
    generation = store.create_search_generation(corpus_sha256="f" * 64, chunker_version="event-v1")
    store.index_event(event.event_id, generation)
    store.activate_generation(generation)

    ready = client.get("/readyz")
    assert ready.status_code == 200
    assert ready.json()["ok"] is True
    assert ready.json()["available_modes"] == ["keyword_only"]

    recalled = client.post(
        "/v1/recall",
        headers=bearer(admin),
        json={"query": "global system memory"},
    )
    assert recalled.status_code == 200
    assert recalled.json()["mode"] == "keyword_only"


def test_authenticated_shutdown_is_deferred_and_admin_only(store, tmp_path):
    called = []
    client, admin, worker = client_and_tokens(
        store, tmp_path, shutdown_callback=lambda: called.append(True)
    )

    denied = client.post("/v1/admin/shutdown", headers=bearer(worker))
    allowed = client.post("/v1/admin/shutdown", headers=bearer(admin))

    assert denied.status_code == 403
    assert allowed.status_code == 200
    assert called == [True]
