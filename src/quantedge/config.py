"""Configuration: environment variables plus YAML policy files.

Two distinct kinds of configuration, deliberately kept apart:

* **Secrets and deployment settings** come from the environment (``.env`` in
  development). They are represented by :class:`Settings`.
* **Policy** -- provider routing, symbol allowlists, session windows, scanner
  weights -- lives in ``config/*.yaml``. It is reviewable, diffable, and safe to
  commit because it contains no credentials.

Every credential loaded here is immediately handed to
:func:`quantedge.logging.register_secret`, so it is masked in all subsequent log
output and exception text.
"""

from __future__ import annotations

import functools
import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from quantedge.logging import configure_logging, get_logger, register_secret

__all__ = [
    "PROJECT_ROOT",
    "Settings",
    "get_scanner_config",
    "get_sessions_config",
    "get_settings",
    "load_yaml_config",
    "providers_config",
    "symbols_config",
]

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "config"

AppEnv = Literal["development", "staging", "production"]
LLMProviderName = Literal["agentrouter", "anthropic", "disabled"]


class Settings(BaseSettings):
    """Environment-backed settings.

    Credentials use :class:`~pydantic.SecretStr` so an accidental ``repr`` or
    model dump prints ``**********`` instead of the value. Use
    :meth:`secret` to read one deliberately.
    """

    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ---- application ----
    app_env: AppEnv = "development"
    log_level: str = "INFO"
    log_format: Literal["json", "console"] = "json"

    # ---- persistence ----
    database_url: SecretStr | None = None
    sqlite_path: str = "data/quantedge.db"
    supabase_url: str | None = None
    supabase_service_role_key: SecretStr | None = None

    # ---- market data ----
    primary_forex_provider: Literal["oanda", "twelvedata"] = "twelvedata"
    twelve_data_api_key: SecretStr | None = None
    oanda_api_token: SecretStr | None = None
    oanda_account_id: SecretStr | None = None
    oanda_environment: Literal["practice", "live"] = "practice"
    binance_api_key: SecretStr | None = None
    binance_api_secret: SecretStr | None = None

    # ---- calendar / news ----
    fmp_api_key: SecretStr | None = None
    alpha_vantage_api_key: SecretStr | None = None
    finnhub_api_key: SecretStr | None = None
    calendar_provider: str | None = None
    calendar_api_key: SecretStr | None = None

    # ---- LLM ----
    llm_provider: LLMProviderName = "disabled"
    agentrouter_api_key: SecretStr | None = None
    agentrouter_base_url: str | None = None
    agentrouter_model: str | None = None
    anthropic_api_key: SecretStr | None = None
    anthropic_model: str = "claude-sonnet-4-5"

    # ---- Binance (public market data only) ----
    binance_rest_base_url: str = "https://data-api.binance.vision"
    binance_ws_base_url: str = "wss://stream.binance.com:9443"
    binance_stream_symbols: str = "BTCUSDT,ETHUSDT,BNBUSDT,SOLUSDT,XRPUSDT"
    binance_stream_intervals: str = "1m,5m"

    # ---- HTTP API ----
    api_host: str = "127.0.0.1"
    api_port: int = 8000
    cors_allow_origins: str = "http://localhost:3000"
    api_auth_token: SecretStr | None = None

    # ---- deployment-only (never read by the gateway itself) ----
    vercel_token: SecretStr | None = None
    supabase_access_token: SecretStr | None = None
    tradingkit_api_key: SecretStr | None = None

    # ----------------------------------------------------------------- #
    # validation                                                        #
    # ----------------------------------------------------------------- #

    @field_validator("binance_rest_base_url", "binance_ws_base_url")
    @classmethod
    def _no_trailing_slash(cls, v: str) -> str:
        return v.rstrip("/")

    @field_validator("binance_rest_base_url")
    @classmethod
    def _market_data_only_host(cls, v: str) -> str:
        """Refuse a Binance REST host that permits signed/private endpoints unless specified."""
        allowed = ("data-api.binance.vision", "testnet.binance.vision", "api.binance.com", "api1.binance.com", "api2.binance.com", "api3.binance.com", "api4.binance.com")
        if not any(host in v for host in allowed):
            raise ValueError(
                "BINANCE_REST_BASE_URL must be a permitted host "
                f"({' or '.join(allowed)}); refusing '{v}'"
            )
        return v

    @model_validator(mode="after")
    def _production_guardrails(self) -> Settings:
        """Fail fast on unsafe production configuration."""
        if self.app_env != "production":
            return self

        problems: list[str] = []
        if "*" in self.cors_origins:
            problems.append("CORS_ALLOW_ORIGINS may not contain '*' in production")
        if self.api_auth_token is None:
            problems.append("API_AUTH_TOKEN is required in production")
        if self.database_url is None and not self.supabase_url:
            problems.append("DATABASE_URL (or Supabase) is required in production")
        if self.log_level.upper() == "DEBUG":
            problems.append("LOG_LEVEL=DEBUG is not permitted in production")
        if problems:
            raise ValueError("unsafe production configuration: " + "; ".join(problems))
        return self

    # ----------------------------------------------------------------- #
    # derived helpers                                                   #
    # ----------------------------------------------------------------- #

    @staticmethod
    def secret(value: SecretStr | None) -> str | None:
        """Read a secret deliberately. Returns ``None`` when unset or blank."""
        if value is None:
            return None
        raw = value.get_secret_value().strip()
        return raw or None

    @property
    def cors_origins(self) -> list[str]:
        return [o.strip() for o in self.cors_allow_origins.split(",") if o.strip()]

    @property
    def stream_symbols(self) -> list[str]:
        return [s.strip().upper() for s in self.binance_stream_symbols.split(",") if s.strip()]

    @property
    def stream_intervals(self) -> list[str]:
        return [s.strip().lower() for s in self.binance_stream_intervals.split(",") if s.strip()]

    @property
    def resolved_database_url(self) -> str:
        """Effective SQLAlchemy URL.

        Falls back to a local SQLite file when no Postgres DSN is configured.
        Callers must consult :attr:`persistence_mode` before claiming that
        durable production persistence is available.
        """
        dsn = self.secret(self.database_url)
        if dsn:
            return dsn
        path = PROJECT_ROOT / self.sqlite_path
        path.parent.mkdir(parents=True, exist_ok=True)
        return f"sqlite+pysqlite:///{path.as_posix()}"

    @property
    def persistence_mode(self) -> Literal["postgres", "sqlite", "memory"]:
        dsn = self.secret(self.database_url)
        if dsn and dsn.startswith(("postgresql", "postgres")):
            return "postgres"
        if dsn and dsn.startswith("sqlite"):
            return "sqlite"
        return "sqlite"

    def configured_credentials(self) -> dict[str, bool]:
        """Map of credential name -> present. **Names only, never values.**

        Drives ``provider_health()`` and the setup docs.
        """
        return {
            "TWELVE_DATA_API_KEY": self.secret(self.twelve_data_api_key) is not None,
            "OANDA_API_TOKEN": self.secret(self.oanda_api_token) is not None,
            "OANDA_ACCOUNT_ID": self.secret(self.oanda_account_id) is not None,
            "BINANCE_API_KEY": self.secret(self.binance_api_key) is not None,
            "BINANCE_API_SECRET": self.secret(self.binance_api_secret) is not None,
            "FMP_API_KEY": self.secret(self.fmp_api_key) is not None,
            "ALPHA_VANTAGE_API_KEY": self.secret(self.alpha_vantage_api_key) is not None,
            "FINNHUB_API_KEY": self.secret(self.finnhub_api_key) is not None,
            "CALENDAR_API_KEY": self.secret(self.calendar_api_key) is not None,
            "AGENTROUTER_API_KEY": self.secret(self.agentrouter_api_key) is not None,
            "ANTHROPIC_API_KEY": self.secret(self.anthropic_api_key) is not None,
            "DATABASE_URL": self.secret(self.database_url) is not None,
            "SUPABASE_SERVICE_ROLE_KEY": self.secret(self.supabase_service_role_key) is not None,
        }

    def register_all_secrets(self) -> None:
        """Register every loaded credential with the log redactor."""
        for field_name, field in type(self).model_fields.items():
            if field.annotation in (SecretStr, SecretStr | None):
                register_secret(self.secret(getattr(self, field_name)))


@functools.lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Load settings once per process, configure logging, register secrets."""
    settings = Settings()
    settings.register_all_secrets()
    configure_logging(level=settings.log_level, fmt=settings.log_format)
    log = get_logger(__name__)
    present = [name for name, ok in settings.configured_credentials().items() if ok]
    log.info(
        "configuration loaded",
        extra={
            "app_env": settings.app_env,
            "persistence_mode": settings.persistence_mode,
            "llm_provider": settings.llm_provider,
            # Names only. Values are never logged.
            "credentials_present": present,
        },
    )
    return settings


# --------------------------------------------------------------------------- #
# YAML policy files                                                            #
# --------------------------------------------------------------------------- #


@functools.lru_cache(maxsize=8)
def load_yaml_config(name: str) -> dict[str, Any]:
    """Load and cache ``config/<name>.yaml``.

    A missing file yields an empty mapping rather than an exception, so the
    package remains importable in minimal environments (CI, sdist checks).
    """
    path = CONFIG_DIR / f"{name}.yaml"
    if not path.exists():
        get_logger(__name__).warning("config file missing", extra={"config": str(path)})
        return {}
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    return data if isinstance(data, dict) else {}


def providers_config() -> dict[str, Any]:
    return load_yaml_config("providers")


def symbols_config() -> dict[str, Any]:
    return load_yaml_config("symbols")


def get_sessions_config() -> dict[str, Any]:
    return load_yaml_config("sessions")


def get_scanner_config() -> dict[str, Any]:
    return load_yaml_config("scanner")


def env_flag(name: str, default: bool = False) -> bool:
    """Read a boolean environment flag (``1/true/yes/on``)."""
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}
