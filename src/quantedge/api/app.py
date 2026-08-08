"""FastAPI App initialization."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from quantedge.api.routes import router

STATIC_DIR = Path(__file__).resolve().parents[3] / "static"


def create_app() -> FastAPI:
    """Create and configure FastAPI instance."""
    app_inst = FastAPI(
        title="QuantEdge AI Market Gateway",
        description=(
            "Live market data, deterministic analytics, scanning, memory post-mortem, "
            "and LLM signal review."
        ),
        version="0.1.0",
    )

    app_inst.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app_inst.include_router(router)

    if STATIC_DIR.exists():
        app_inst.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

        @app_inst.get("/", response_class=FileResponse)
        def read_root() -> FileResponse:
            return FileResponse(str(STATIC_DIR / "index.html"))

    return app_inst


app = create_app()
