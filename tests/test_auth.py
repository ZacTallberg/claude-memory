from __future__ import annotations

from datetime import UTC, datetime, timedelta

from system_memory.auth import CredentialStore


def test_credentials_bind_actor_scopes_expiry_and_revocation(store):
    credentials = CredentialStore(store.database)
    credential, token = credentials.create(
        actor_id="codex-main",
        label="Codex main adapter",
        scopes={"recall", "ingest:self"},
    )
    authenticated = credentials.authenticate(token)
    assert authenticated is not None
    assert authenticated.actor_id == "codex-main"
    assert authenticated.permits("recall")
    assert not authenticated.permits("ingest:any")
    assert credentials.revoke(credential.credential_id)
    assert credentials.authenticate(token) is None


def test_expired_credential_never_authenticates(store):
    credentials = CredentialStore(store.database)
    _, token = credentials.create(
        actor_id="expired-worker",
        label="expired",
        scopes={"recall"},
        expires_at=datetime.now(UTC) - timedelta(seconds=1),
    )
    assert credentials.authenticate(token) is None


def test_admin_wildcard_permits_every_scope(store):
    credentials = CredentialStore(store.database)
    _, token = credentials.create(actor_id="admin", label="admin", scopes={"*"})
    authenticated = credentials.authenticate(token)
    assert authenticated is not None
    assert authenticated.permits("arbitrary:future-scope")
