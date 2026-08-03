from __future__ import annotations

import os
import secrets
import signal
from collections.abc import Callable
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Request, status
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel

from ..archive import CanonicalArchive
from ..auth import Credential, CredentialStore
from ..database import Database
from ..embeddings import EmbeddingProvider, FastEmbedProvider
from ..indexing import LiveEmbeddingWorker
from ..inference import InferenceScheduler
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
    def __init__(
        self,
        settings: Settings,
        store: MemoryStore,
        nonce: str,
        shutdown_callback: Callable[[], None],
        embedding_provider: EmbeddingProvider | None = None,
        inference_scheduler: InferenceScheduler | None = None,
    ) -> None:
        self.settings = settings
        self.store = store
        self.credentials = CredentialStore(store.database)
        self.embedding_generation_id = store.active_generation_id()
        self.embedding_manifest = (
            store.embedding_manifest(self.embedding_generation_id)
            if self.embedding_generation_id
            else None
        )
        self.embedding_error: str | None = None
        self.embedding_provider = embedding_provider
        self.inference_scheduler = inference_scheduler
        self._owns_scheduler = False
        self.live_embedding_worker: LiveEmbeddingWorker | None = None
        if self.embedding_manifest:
            try:
                if self.embedding_provider is None:
                    settings.resolved_embedding_cache_path.mkdir(parents=True, exist_ok=True)
                    self.embedding_provider = FastEmbedProvider(
                        self.embedding_manifest,
                        cache_dir=settings.resolved_embedding_cache_path,
                        threads=settings.embedding_threads,
                        local_files_only=True,
                    )
                if self.embedding_provider.manifest != self.embedding_manifest:
                    raise ValueError("loaded embedding provider does not match active generation")
                if self.inference_scheduler is None:
                    self.inference_scheduler = InferenceScheduler(
                        capacity=settings.inference_capacity
                    )
                    self._owns_scheduler = True
                self.live_embedding_worker = LiveEmbeddingWorker(
                    store,
                    self.embedding_provider,
                    self.inference_scheduler,
                    poll_seconds=settings.live_embedding_poll_seconds,
                )
            except (ImportError, OSError, RuntimeError, ValueError) as error:
                self.embedding_error = f"{type(error).__name__}: {error}"
                self.embedding_provider = None
                if self._owns_scheduler and self.inference_scheduler:
                    self.inference_scheduler.close()
                self.inference_scheduler = None
                self._owns_scheduler = False
        self.recall = RecallEngine(
            store,
            lexical_candidates=settings.lexical_limit,
            abstention_min_score=settings.abstention_min_score,
            vector_min_similarity=settings.vector_min_similarity,
            embedder=self.embedding_provider,
            scheduler=self.inference_scheduler,
            query_timeout_seconds=settings.query_inference_timeout_seconds,
        )
        self.nonce = nonce
        self.shutdown_callback = shutdown_callback

    def start(self) -> None:
        if self.live_embedding_worker:
            self.live_embedding_worker.start()

    def close(self) -> None:
        if self.live_embedding_worker:
            self.live_embedding_worker.close()
        if self._owns_scheduler and self.inference_scheduler:
            self.inference_scheduler.close()

    def embedding_health(self) -> dict[str, object]:
        generation = self.embedding_generation_id
        status = self.store.embedding_status(generation) if generation else None
        ready = bool(
            self.embedding_manifest
            and self.embedding_provider
            and self.inference_scheduler
            and status
            and status["documents"] > 0
            and status["documents"] == status["vectors"]
        )
        reason = self.embedding_error
        if not self.embedding_manifest:
            reason = "active generation has no dense index"
        return {
            "ready": ready,
            "reason": reason,
            "generation_id": generation,
            "manifest": self.embedding_manifest.model_dump(mode="json")
            if self.embedding_manifest
            else None,
            "coverage": status,
            "scheduler_queue": self.inference_scheduler.queued
            if self.inference_scheduler
            else None,
            "live_worker": self.live_embedding_worker.status()
            if self.live_embedding_worker
            else None,
        }

    def available_modes(self, usable: bool) -> list[str]:
        if not usable:
            return []
        modes = ["keyword_only"]
        if self.embedding_health()["ready"]:
            modes.insert(0, "hybrid")
        return modes


_bearer = HTTPBearer(auto_error=False)


def create_app(
    settings: Settings | None = None,
    *,
    store: MemoryStore | None = None,
    instance_nonce: str | None = None,
    shutdown_callback: Callable[[], None] | None = None,
    embedding_provider: EmbeddingProvider | None = None,
    inference_scheduler: InferenceScheduler | None = None,
) -> FastAPI:
    cfg = settings or Settings()
    memory = store or MemoryStore(
        Database(cfg.resolved_database_path, busy_timeout_ms=cfg.busy_timeout_ms),
        CanonicalArchive(cfg.resolved_archive_path),
    )
    version = memory.initialize()
    if version < 1:
        raise RuntimeError("database migration did not initialize canonical memory")

    def default_shutdown() -> None:
        os.kill(os.getpid(), signal.SIGINT)

    context = AppContext(
        cfg,
        memory,
        instance_nonce
        or os.environ.get("SYSTEM_MEMORY_INSTANCE_NONCE")
        or secrets.token_urlsafe(24),
        shutdown_callback or default_shutdown,
        embedding_provider,
        inference_scheduler,
    )

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        context.start()
        try:
            yield
        finally:
            context.close()

    app = FastAPI(
        title="System Memory",
        version="0.1.0",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
        lifespan=lifespan,
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
        return {
            "ok": True,
            "pid": os.getpid(),
            "nonce": context.nonce,
            "build": os.environ.get("SYSTEM_MEMORY_BUILD_ID", "0.1.0"),
        }

    @app.get("/readyz")
    def readyz() -> dict[str, object]:
        health = context.store.database.health()
        generation = context.store.active_generation_id()
        live_documents = context.store.live_document_count()
        usable = bool(generation or live_documents)
        return {
            "ok": bool(health["ok"] and usable),
            "database": health["quick_check"],
            "schema_version": health["schema_version"],
            "active_generation_id": generation,
            "available_modes": context.available_modes(usable),
            "live_documents": live_documents,
            "embedding": context.embedding_health(),
        }

    @app.get("/healthz")
    def healthz(actor: Credential = authenticated_actor) -> dict[str, object]:
        require_scope(actor, "read")
        return {
            **context.store.database.health(),
            "active_generation_id": context.store.active_generation_id(),
            "counts": context.store.counts(),
            "embedding": context.embedding_health(),
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
        own_identity = actor.actor_id == incoming.agent_id
        user_authority = incoming.role.value == "user" or incoming.authority.value in {
            "user_authored",
            "user_declaration",
            "user_behavior",
        }
        explicit_decision = incoming.authority.value == "explicit_decision"
        allowed = actor.permits("ingest:any")
        if own_identity and user_authority:
            allowed = allowed or actor.permits("ingest:user-authored")
        elif own_identity and explicit_decision:
            allowed = allowed or actor.permits("ingest:decision")
        elif own_identity:
            allowed = allowed or actor.permits("ingest:self")
        if not allowed:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="credential cannot ingest the asserted identity or authority",
            )
        result = context.store.ingest(incoming)
        if context.live_embedding_worker:
            context.live_embedding_worker.wake()
        return result

    @app.post("/v1/recall", response_model=RecallResult)
    def recall(
        query: RecallQuery,
        actor: Credential = authenticated_actor,
    ) -> RecallResult:
        require_scope(actor, "recall")
        return context.recall.recall(query)

    @app.post("/v1/admin/shutdown")
    def shutdown(
        background: BackgroundTasks,
        actor: Credential = authenticated_actor,
    ) -> dict[str, bool]:
        require_scope(actor, "admin:shutdown")
        background.add_task(context.shutdown_callback)
        return {"ok": True}

    return app
