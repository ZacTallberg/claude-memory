from __future__ import annotations

import hashlib
import hmac
import json
import secrets
from dataclasses import dataclass
from datetime import datetime

from .clock import utc_iso, utc_now
from .database import Database
from .ids import stable_id


@dataclass(frozen=True)
class Credential:
    credential_id: str
    actor_id: str
    label: str
    scopes: frozenset[str]
    expires_at: str | None

    def permits(self, scope: str) -> bool:
        return "*" in self.scopes or scope in self.scopes


class CredentialStore:
    def __init__(self, database: Database) -> None:
        self.database = database

    @staticmethod
    def token_hash(token: str) -> str:
        return hashlib.sha256(token.encode("utf-8")).hexdigest()

    def create(
        self,
        *,
        actor_id: str,
        label: str,
        scopes: set[str] | frozenset[str],
        expires_at: datetime | None = None,
        token: str | None = None,
    ) -> tuple[Credential, str]:
        if not actor_id.strip() or not label.strip() or not scopes:
            raise ValueError("actor_id, label, and at least one scope are required")
        secret = token or secrets.token_urlsafe(48)
        digest = self.token_hash(secret)
        credential_id = stable_id("cred", actor_id, label, digest)
        now = utc_iso()
        expires = utc_iso(expires_at) if expires_at else None
        with self.database.write() as connection:
            connection.execute(
                """INSERT INTO credentials(
                       id,actor_id,label,token_sha256,scopes_json,created_at,expires_at,metadata
                   ) VALUES (?,?,?,?,?,?,?,?)""",
                (
                    credential_id,
                    actor_id,
                    label,
                    digest,
                    json.dumps(sorted(scopes)),
                    now,
                    expires,
                    json.dumps({}, sort_keys=True),
                ),
            )
        return (
            Credential(credential_id, actor_id, label, frozenset(scopes), expires),
            secret,
        )

    def authenticate(self, token: str) -> Credential | None:
        if not token:
            return None
        supplied = self.token_hash(token)
        with self.database.read() as connection:
            row = connection.execute(
                """SELECT id,actor_id,label,token_sha256,scopes_json,expires_at
                   FROM credentials WHERE token_sha256=? AND revoked_at IS NULL""",
                (supplied,),
            ).fetchone()
            if not row or not hmac.compare_digest(row["token_sha256"], supplied):
                return None
            expires_at = row["expires_at"]
            if expires_at:
                parsed = datetime.fromisoformat(expires_at.replace("Z", "+00:00"))
                if parsed <= utc_now():
                    return None
        return Credential(
            credential_id=row["id"],
            actor_id=row["actor_id"],
            label=row["label"],
            scopes=frozenset(json.loads(row["scopes_json"])),
            expires_at=expires_at,
        )

    def revoke(self, credential_id: str) -> bool:
        with self.database.write() as connection:
            changed = connection.execute(
                "UPDATE credentials SET revoked_at=? WHERE id=? AND revoked_at IS NULL",
                (utc_iso(), credential_id),
            ).rowcount
        return changed == 1
