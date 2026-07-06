"""FastAPI app factory + runner. One persistent process: warm models + store, serving the
hub UI and the hook recall/unify endpoints."""
from __future__ import annotations

import threading
import webbrowser
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from claudemem.log import get_logger

from .api import router
from .state import get_state

log = get_logger(__name__)
_STATIC = Path(__file__).parent / "static"


def create_app() -> FastAPI:
    app = FastAPI(title="claude-memory hub", docs_url="/api/docs")
    app.include_router(router)
    if _STATIC.exists():
        app.mount("/static", StaticFiles(directory=str(_STATIC)), name="static")

    @app.get("/")
    def home():
        idx = _STATIC / "index.html"
        if idx.exists():
            return FileResponse(str(idx))
        return HTMLResponse("<h1>claude-memory hub</h1><p>UI assets not built yet. "
                            "API is live at <a href='/api/docs'>/api/docs</a>.</p>")

    @app.on_event("startup")
    def _warm():
        threading.Thread(target=get_state, daemon=True).start()

    return app


app = create_app()


def run(host: str = "127.0.0.1", port: int = 7777, open_browser: bool = True) -> None:
    import uvicorn
    if open_browser:
        threading.Timer(1.8, lambda: webbrowser.open(f"http://{host}:{port}")).start()
    log.info("serving claude-memory hub on http://%s:%d", host, port)
    uvicorn.run(app, host=host, port=port, log_level="warning")
