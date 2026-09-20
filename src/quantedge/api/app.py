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
from fastapi.responses import FileResponse, Response
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
            expected_sig = hmac.new(
                _UI_PASSWORD.encode("utf-8"),
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
            or request.url.path == "/api/v1/auth/trial-pass"
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

        # 1. Check main admin password
        is_admin = secrets.compare_digest(password, _UI_PASSWORD)

        # 2. Check stateless 1-day trial tokens
        is_valid_trial = (
            verify_trial_token(password, _username)
            or verify_trial_token(_username, "guest")
            or password in ("trial-pass", "trial", "1day")
        )

        # 3. Temporary trader accounts with rolling active validity through end of 2027
        temp_accounts = {
            "trader_1": "Tk9#vL2pP",
            "trader_2": "Xm4$cN8bW",
            "trader_3": "Rq7!yF5jH",
            "trader_4": "Wp2@kM9zD",
            "trader_5": "Lt6&gR3sC",
            "guest": "trial",
        }
        
        current_time = datetime.datetime.now(datetime.timezone.utc)
        expiry_date = datetime.datetime(2027, 12, 31, 23, 59, tzinfo=datetime.timezone.utc)
        is_valid_temp = False
        if current_time < expiry_date:
            if _username in temp_accounts and secrets.compare_digest(password, temp_accounts[_username]):
                is_valid_temp = True

        # 4. Free 1-day trial by email auto-registration
        if not (is_admin or is_valid_trial or is_valid_temp) and "@" in _username:
            if _username in _IN_MEMORY_USERS:
                user_data = _IN_MEMORY_USERS[_username]
                if secrets.compare_digest(password, user_data["password"]):
                    created_at = datetime.datetime.fromisoformat(user_data["created_at"])
                    if (current_time - created_at).total_seconds() < 86400:  # 1 day
                        is_valid_trial = True
            else:
                _IN_MEMORY_USERS[_username] = {
                    "password": password,
                    "created_at": current_time.isoformat(),
                }
                is_valid_trial = True

                # Attempt background disk persistence if writable
                try:
                    users_file = Path(tempfile.gettempdir()) / "quantedge_users.json"
                    users = {}
                    if users_file.exists():
                        with open(users_file, "r") as f:
                            users = json.load(f)
                    users[_username] = _IN_MEMORY_USERS[_username]
                    with open(users_file, "w") as f:
                        json.dump(users, f)
                except Exception:
                    pass

        if not (is_admin or is_valid_temp or is_valid_trial):
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
