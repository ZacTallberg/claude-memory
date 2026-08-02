from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from starlette.datastructures import Headers
from starlette.responses import JSONResponse

ASGIApp = Callable[
    [
        dict[str, Any],
        Callable[[], Awaitable[dict[str, Any]]],
        Callable[[dict[str, Any]], Awaitable[None]],
    ],
    Awaitable[None],
]


class RequestBoundaryMiddleware:
    """Enforce body and browser-origin boundaries before application parsing."""

    def __init__(self, app: ASGIApp, *, body_limit: int, allowed_origins: frozenset[str]) -> None:
        self.app = app
        self.body_limit = body_limit
        self.allowed_origins = allowed_origins

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        headers = Headers(scope=scope)
        length = headers.get("content-length")
        if length:
            try:
                too_large = int(length) > self.body_limit
            except ValueError:
                too_large = True
            if too_large:
                response = JSONResponse({"detail": "request body too large"}, status_code=413)
                await response(scope, receive, send)
                return
        origin = headers.get("origin")
        if (
            origin
            and scope.get("method") not in {"GET", "HEAD", "OPTIONS"}
            and origin.rstrip("/") not in self.allowed_origins
        ):
            response = JSONResponse({"detail": "origin is not allowed"}, status_code=403)
            await response(scope, receive, send)
            return
        await self.app(scope, receive, send)
