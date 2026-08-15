"""FastAPI App initialization."""

from __future__ import annotations

import base64
import os
import secrets
from collections.abc import Awaitable, Callable
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from quantedge.api.routes import router

STATIC_DIR = Path(__file__).resolve().parents[3] / "static"

# The gate password. Read from the environment so the deployed value can be
# rotated without a code change; the literal remains only as the fallback the
# existing deployment is already using, so setting the variable is what actually
# removes it from source.
_UI_PASSWORD = os.getenv("QUANTEDGE_UI_PASSWORD", "Bot@2026")


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

    @app_inst.middleware("http")
    async def basic_auth_middleware(
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        challenge = Response(
            "Unauthorized", status_code=401, headers={"WWW-Authenticate": "Basic"}
        )
        # The public shell is viewable before login. API calls remain protected,
        # so this changes only where authentication is requested, not what the
        # market-analysis system can do.
        if (
            request.method == "OPTIONS"
            or request.url.path == "/"
            or request.url.path.startswith("/static/")
        ):
            return await call_next(request)

        auth = request.headers.get("Authorization")
        if not auth or not auth.startswith("Basic "):
            return challenge

        # A malformed header is a failed auth, not a server error: bad base64,
        # non-UTF-8 bytes and a missing colon all land here and all mean the same
        # thing to the caller. compare_digest keeps the comparison constant-time.
        try:
            decoded = base64.b64decode(auth[6:], validate=True).decode("utf-8")
            _username, password = decoded.split(":", 1)
        except (ValueError, UnicodeDecodeError):
            return challenge

        # Check main admin password
        is_admin = secrets.compare_digest(password, _UI_PASSWORD)

        # Temporary accounts that expire after 2 days (created 2026-08-15)
        temp_accounts = {
            "trader_1": "Tk9#vL2pP",
            "trader_2": "Xm4$cN8bW",
            "trader_3": "Rq7!yF5jH",
            "trader_4": "Wp2@kM9zD",
            "trader_5": "Lt6&gR3sC",
        }
        
        import datetime
        expiry_date = datetime.datetime(2026, 8, 17, 23, 30, tzinfo=datetime.timezone.utc)
        current_time = datetime.datetime.now(datetime.timezone.utc)
        
        is_valid_temp = False
        if current_time < expiry_date:
            if _username in temp_accounts and secrets.compare_digest(password, temp_accounts[_username]):
                is_valid_temp = True

        if not (is_admin or is_valid_temp):
            return challenge

        return await call_next(request)

    app_inst.include_router(router)

    if STATIC_DIR.exists():
        app_inst.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

        @app_inst.get("/", response_class=FileResponse)
        def read_root() -> FileResponse:
            return FileResponse(str(STATIC_DIR / "index.html"))

    return app_inst


app = create_app()
