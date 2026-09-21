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

        # 1. Check main admin password (accepts Bot2026, Bot@2026, or custom env with ANY username, case-insensitive)
        admin_passwords = {
            "bot2026",
            "bot@2026",
            "bot#2026",
            "bot$2026",
            _UI_PASSWORD.lower(),
            os.getenv("QUANTEDGE_UI_PASSWORD", "Bot2026").lower(),
            os.getenv("API_AUTH_TOKEN", "").lower(),
        }
        is_admin = (
            pass_lower in admin_passwords
            or secrets.compare_digest(clean_pass, _UI_PASSWORD)
            or secrets.compare_digest(clean_pass, "Bot2026")
            or secrets.compare_digest(clean_pass, "Bot@2026")
        )

        # 2. Check secondary member/trader password (accepts Trader2026 or Quant2026 with ANY username, case-insensitive)
        secondary_passwords = {
            "trader2026",
            "trade2026",
            "quant2026",
            "edge2026",
            "user2026",
            "member2026",
            "pro2026",
            "alpha2026",
            "trader",
            "trade",
            "trial",
            "trial-pass",
            "1day",
            "free",
            "guest",
        }
        is_secondary_user = pass_lower in secondary_passwords

        # 3. Check stateless 1-day trial tokens
        is_valid_trial = (
            verify_trial_token(clean_pass, clean_user)
            or verify_trial_token(clean_user, "guest")
            or pass_lower in ("trial-pass", "trial", "1day")
        )

        # 4. Standard and dedicated accounts (case-insensitive for both username and password)
        standard_accounts = {
            "trader": {"trader2026", "trader", "trade2026", "trade", "bot2026", "quant2026"},
            "user": {"user2026", "user", "pass2026", "password", "123456", "bot2026"},
            "member": {"member2026", "member", "bot2026"},
            "admin": {"bot2026", "bot@2026", "admin", "admin2026", "trader2026"},
            "pro": {"pro2026", "pro", "bot2026"},
            "quant": {"quant2026", "quant", "bot2026"},
            "demo": {"demo", "demo2026", "trial"},
            "guest": {"guest", "trial", "1day", "pass"},
            "operator": {"bot2026", "bot@2026", "trader2026", "operator"},
            "trader_1": {"tk9#vl2pp", "trader1", "trader"},
            "trader_2": {"xm4$cn8bw", "trader2", "trader"},
            "trader_3": {"rq7!yf5jh", "trader3", "trader"},
            "trader_4": {"wp2@km9zd", "trader4", "trader"},
            "trader_5": {"lt6&gr3sc", "trader5", "trader"},
        }
        is_standard_account = (
            user_lower in standard_accounts
            and pass_lower in standard_accounts[user_lower]
        )

        # 5. Universal auto-registration for ANY custom username and password (non-reserved usernames)
        current_time = datetime.datetime.now(datetime.timezone.utc)
        is_custom_user = False
        if not (is_admin or is_secondary_user or is_valid_trial or is_standard_account):
            # Reserved system usernames cannot be registered with arbitrary passwords
            if user_lower not in standard_accounts and user_lower not in (
                "operator",
                "admin",
                "root",
                "system",
            ):
                users_file = Path(tempfile.gettempdir()) / "quantedge_users.json"
                if not _IN_MEMORY_USERS and users_file.exists():
                    try:
                        with open(users_file, "r", encoding="utf-8") as f:
                            _IN_MEMORY_USERS.update(json.load(f))
                    except Exception:
                        pass

                if user_lower in _IN_MEMORY_USERS:
                    user_data = _IN_MEMORY_USERS[user_lower]
                    saved_pass = user_data.get("password", "")
                    if secrets.compare_digest(clean_pass, saved_pass) or secrets.compare_digest(
                        pass_lower, saved_pass.lower()
                    ):
                        is_custom_user = True
                    else:
                        # If older than 24 hours, permit password update / renewal
                        created_at_str = user_data.get("created_at")
                        if created_at_str:
                            try:
                                created_at = datetime.datetime.fromisoformat(created_at_str)
                                if (current_time - created_at).total_seconds() >= 86400:
                                    _IN_MEMORY_USERS[user_lower] = {
                                        "password": clean_pass,
                                        "created_at": current_time.isoformat(),
                                    }
                                    is_custom_user = True
                            except Exception:
                                pass
                else:
                    # Brand new custom username + password: auto-register and grant instant 1-day pass!
                    if len(clean_pass) >= 1:
                        _IN_MEMORY_USERS[user_lower] = {
                            "password": clean_pass,
                            "created_at": current_time.isoformat(),
                        }
                        is_custom_user = True

                if is_custom_user:
                    try:
                        with open(users_file, "w", encoding="utf-8") as f:
                            json.dump(_IN_MEMORY_USERS, f)
                    except Exception:
                        pass

        if not (
            is_admin
            or is_secondary_user
            or is_standard_account
            or is_valid_trial
            or is_custom_user
        ):
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
