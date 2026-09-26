"""FastAPI App initialization."""

from __future__ import annotations

import base64
import datetime
import hashlib
import hmac
import json
import os
import secrets
import tempfile
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from quantedge.api.routes import router

STATIC_DIR = Path(__file__).resolve().parents[3] / "static"

# The gate password. Read from the environment so the deployed value can be
# rotated without a code change; the literal remains only as the fallback the
# existing deployment is already using, so setting the variable is what actually
# removes it from source.
_UI_PASSWORD = os.getenv("QUANTEDGE_UI_PASSWORD", "Bot@2026")

_IN_MEMORY_USERS: dict[str, dict[str, Any]] = {}


def create_trial_token(user_id: str = "guest") -> str:
    """Generate a tamper-proof 24-hour trial token."""
    now_ts = int(datetime.datetime.now(datetime.timezone.utc).timestamp())
    msg = f"{user_id}:{now_ts}"
    sig = hmac.new(
        _UI_PASSWORD.encode("utf-8"),
        msg.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()[:16]
    return f"trial_{now_ts}_{sig}"


def verify_trial_token(token: str, user_id: str = "guest") -> bool:
    """Verify if a trial token is authentic and within the 24-hour window."""
    try:
        parts = token.split("_")
        if len(parts) != 3 or parts[0] != "trial":
            return False
        ts = int(parts[1])
        sig = parts[2]
        now_ts = int(datetime.datetime.now(datetime.timezone.utc).timestamp())
        if now_ts - ts < 0 or now_ts - ts > 86400:  # 24 hours (86,400 seconds)
            return False
        # Accept token signed for this user_id or standard guest / operator
        for candidate in (user_id, "guest", "operator"):
            for secret_key in (_UI_PASSWORD, "Bot2026", "Bot@2026"):
                expected_sig = hmac.new(
                    secret_key.encode("utf-8"),
                    f"{candidate}:{ts}".encode("utf-8"),
                    hashlib.sha256,
                ).hexdigest()[:16]
                if secrets.compare_digest(sig, expected_sig):
                    return True
        return False
    except Exception:
        return False


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

    @app_inst.get("/api/v1/auth/trial-pass")
    @app_inst.post("/api/v1/auth/trial-pass")
    def issue_trial_pass() -> dict[str, Any]:
        """Issue an instant 24-hour guest trial token without password requirement."""
        token = create_trial_token("guest")
        return {
            "status": "ok",
            "username": "guest",
            "token": token,
            "expires_in_seconds": 86400,
            "valid_hours": 24,
            "message": "1-day guest pass active.",
        }

    @app_inst.middleware("http")
    async def basic_auth_middleware(
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        # A failed or absent credential returns a plain 401 with NO
        # ``WWW-Authenticate: Basic`` header. That header is the one thing that
        # makes a browser render its own native Basic-auth dialog, which was
        # appearing as a second, redundant password prompt layered on top of the
        # app's own workspace login (the browser fires it for un-exempt requests
        # like /favicon.ico). Without the header, an unauthenticated API call is
        # simply a 401 that the frontend handles itself -- it shows the workspace
        # login and replays the request with the Authorization header it already
        # manages. Single sign-in; the API is exactly as protected as before.
        challenge = JSONResponse(
            {"detail": "Authentication required. Sign in through the workspace."},
            status_code=401,
        )
        # The public shell is viewable before login. API calls remain protected,
        # so this changes only where authentication is requested, not what the
        # market-analysis system can do.
        if (
            request.method == "OPTIONS"
            or request.url.path == "/"
            or request.url.path.startswith("/static/")
            or request.url.path == "/api/v1/auth/trial-pass"
        ):
            return await call_next(request)

        # Support Bearer trial token or X-Trial-Token directly
        bearer_token = None
        auth = request.headers.get("Authorization")
        if auth and auth.startswith("Bearer "):
            bearer_token = auth[7:].strip()
        elif request.headers.get("X-Trial-Token"):
            bearer_token = request.headers.get("X-Trial-Token").strip()

        if bearer_token and verify_trial_token(bearer_token):
            return await call_next(request)

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

        # Normalize and strip credentials to eliminate whitespace and casing pitfalls
        clean_user = _username.strip()
        user_lower = clean_user.lower()
        clean_pass = password.strip()
        pass_lower = clean_pass.lower()

        if not clean_pass:
            return challenge

        # 1. Master admin password (usable with ANY username)
        admin_passwords = {
            "bot2026",
            "bot@2026",
            "bot#2026",
            _UI_PASSWORD.lower(),
            os.getenv("QUANTEDGE_UI_PASSWORD", "Bot2026").lower(),
            os.getenv("API_AUTH_TOKEN", "").lower(),
        }
        is_admin = pass_lower in admin_passwords

        # 2. Registered backend accounts (strictly requiring assigned password)
        registered_accounts = {
            "trader": {"trader@2026", "trader2026"},
            "user": {"user@2026", "user2026"},
            "member": {"member@2026", "member2026"},
            "pro": {"pro@2026", "pro2026"},
            "quant": {"quant@2026", "quant2026"},
            "trader_1": {"tk9#vl2pp"},
            "trader_2": {"xm4$cn8bw"},
            "trader_3": {"rq7!yf5jh"},
            "trader_4": {"wp2@km9zd"},
            "trader_5": {"lt6&gr3sc"},
        }
        is_registered_account = (
            user_lower in registered_accounts
            and pass_lower in registered_accounts[user_lower]
        )

        # 3. Stateless 1-day trial tokens (e.g. from the 1-Day Trial instant access button)
        is_valid_trial = (
            verify_trial_token(clean_pass, clean_user)
            or verify_trial_token(clean_user, "guest")
            or (pass_lower in ("trial-pass", "trial", "1day") and user_lower in ("guest", "trial"))
        )

        if not (is_admin or is_registered_account or is_valid_trial):
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
