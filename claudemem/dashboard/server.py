"""FastAPI app factory + runner. One persistent process: warm models + store, serving the
hub UI and the hook recall/unify endpoints."""
from __future__ import annotations

import threading
import webbrowser
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.trustedhost import TrustedHostMiddleware

from claudemem.local_security import is_loopback_host, is_safe_origin
from claudemem.log import get_logger

from .api import live_index_loop, router
from .state import get_state

log = get_logger(__name__)
_STATIC = Path(__file__).parent / "static"


def create_app() -> FastAPI:
    app = FastAPI(title="shared agent memory", docs_url="/api/docs")
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=["127.0.0.1", "localhost", "[::1]", "testserver"],
    )

    @app.middleware("http")
    async def local_only(request: Request, call_next):
        peer = request.client.host if request.client else None
        # ``testclient`` is Starlette's in-process test transport, never a network peer.
        if peer != "testclient" and not is_loopback_host(peer):
            return JSONResponse({"error": "local access only"}, status_code=403)
        if not is_safe_origin(request.headers.get("origin")):
            return JSONResponse({"error": "cross-origin access refused"}, status_code=403)
        return await call_next(request)

    app.include_router(router)
    if _STATIC.exists():
        app.mount("/static", StaticFiles(directory=str(_STATIC)), name="static")

    @app.get("/")
    def home():
        idx = _STATIC / "index.html"
        if idx.exists():
            return FileResponse(str(idx))
        return HTMLResponse("<h1>shared agent memory</h1><p>UI assets not built yet. "
                            "API is live at <a href='/api/docs'>/api/docs</a>.</p>")

    @app.on_event("startup")
    def _warm():
        threading.Thread(target=get_state, daemon=True).start()
        threading.Thread(target=live_index_loop, daemon=True,
                         name="memory-live-index-scheduler").start()

    return app


app = create_app()


def run(host: str = "127.0.0.1", port: int = 7777, open_browser: bool = True) -> None:
    import uvicorn
    if not is_loopback_host(host):
        raise SystemExit(
            f"refusing to expose the unauthenticated local memory service on {host!r}; "
            "use 127.0.0.1, ::1, or localhost"
        )
    if open_browser:
        threading.Timer(1.8, lambda: webbrowser.open(f"http://{host}:{port}")).start()
    log.info("serving shared agent memory on http://%s:%d", host, port)
    uvicorn.run(app, host=host, port=port, log_level="warning")
