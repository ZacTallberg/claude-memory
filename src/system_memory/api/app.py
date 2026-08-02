from __future__ import annotations

import os
import secrets
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel

from ..archive import CanonicalArchive
from ..auth import Credential, CredentialStore
from ..database import Database
from ..models import IngestEvent, IngestResult, RecallQuery, RecallResult
from ..recall import RecallEngine
from ..settings import Settings
from ..store import MemoryStore
from .middleware import RequestBoundaryMiddleware


class StatsResponse(BaseModel):
    ok: bool
    schema_version: int
    counts: dict[str, int]
    active_generation_id: str | None


class AppContext:
    def __init__(self, settings: Settings, store: MemoryStore, nonce: str) -> None:
        self.settings = settings
        self.store = store
        self.credentials = CredentialStore(store.database)
        self.recall = RecallEngine(
            store,
            lexical_candidates=settings.lexical_limit,
            abstention_min_score=settings.abstention_min_score,
        )
        self.nonce = nonce


_bearer = HTTPBearer(auto_error=False)


def create_app(
    settings: Settings | None = None,
    *,
    store: MemoryStore | None = None,
    instance_nonce: str | None = None,
) -> FastAPI:
    cfg = settings or Settings()
    memory = store or MemoryStore(
        Database(cfg.resolved_database_path, busy_timeout_ms=cfg.busy_timeout_ms),
        CanonicalArchive(cfg.resolved_archive_path),
    )
    version = memory.initialize()
    if version < 1:
        raise RuntimeError("database migration did not initialize canonical memory")
    context = AppContext(cfg, memory, instance_nonce or secrets.token_urlsafe(24))

    app = FastAPI(
        title="System Memory",
        version="0.1.0",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )
    app.state.context = context
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=["127.0.0.1", "localhost", "testserver"],
    )
    app.add_middleware(
        RequestBoundaryMiddleware,
        body_limit=cfg.request_body_limit,
        allowed_origins=frozenset(
            {
                f"http://127.0.0.1:{cfg.port}",
                f"http://localhost:{cfg.port}",
            }
        ),
    )

    async def credential(
        request: Request,
        supplied: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)],
    ) -> Credential:
        if not supplied or supplied.scheme.casefold() != "bearer":
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="bearer credential required",
            )
        authenticated = request.app.state.context.credentials.authenticate(supplied.credentials)
        if not authenticated:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="credential is invalid, expired, or revoked",
            )
        return authenticated

    def require_scope(actor: Credential, scope: str) -> None:
        if not actor.permits(scope):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN, detail="scope is not allowed"
            )

    authenticated_actor = Depends(credential)

    @app.middleware("http")
    async def no_store(request: Request, call_next):
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        return response

    @app.get("/livez")
    def livez() -> dict[str, object]:
        return {"ok": True, "pid": os.getpid(), "nonce": context.nonce, "build": "0.1.0"}

    @app.get("/readyz")
    def readyz() -> dict[str, object]:
        health = context.store.database.health()
        generation = context.store.active_generation_id()
        return {
            "ok": bool(health["ok"] and generation),
            "database": health["quick_check"],
            "schema_version": health["schema_version"],
            "active_generation_id": generation,
            "available_modes": ["keyword_only"] if generation else [],
        }

    @app.get("/healthz")
    def healthz(actor: Credential = authenticated_actor) -> dict[str, object]:
        require_scope(actor, "read")
        return {
            **context.store.database.health(),
            "active_generation_id": context.store.active_generation_id(),
            "counts": context.store.counts(),
        }

    @app.get("/v1/stats", response_model=StatsResponse)
    def stats(actor: Credential = authenticated_actor) -> StatsResponse:
        require_scope(actor, "read")
        health = context.store.database.health()
        return StatsResponse(
            ok=bool(health["ok"]),
            schema_version=int(health["schema_version"]),
            counts=context.store.counts(),
            active_generation_id=context.store.active_generation_id(),
        )

    @app.post("/v1/events", response_model=IngestResult)
    def ingest(
        incoming: IngestEvent,
        actor: Credential = authenticated_actor,
    ) -> IngestResult:
        if actor.permits("ingest:any") or (
            actor.permits("ingest:self") and actor.actor_id == incoming.agent_id
        ):
            pass
        else:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="credential cannot ingest for the asserted agent identity",
            )
        return context.store.ingest(incoming)

    @app.post("/v1/recall", response_model=RecallResult)
    def recall(
        query: RecallQuery,
        actor: Credential = authenticated_actor,
    ) -> RecallResult:
        require_scope(actor, "recall")
        return context.recall.recall(query)

    return app
