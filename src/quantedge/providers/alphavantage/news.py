"""Alpha Vantage news adapter.

Scope discipline
----------------
This key was supplied labelled as a calendar credential. Probing it
(``scripts/probe_credentials.py``) established that it is an **Alpha Vantage**
key, and Alpha Vantage publishes **no economic-calendar endpoint at all**. So
this adapter registers as a *news* provider only; ``config/providers.yaml`` sets
``economic_calendar: false`` for it so the event-risk service never routes a
calendar request here and never receives news dressed up as a calendar.

What is verified to work on this key: ``NEWS_SENTIMENT`` returns a live feed.

Sentiment handling
------------------
Alpha Vantage ships its own ``overall_sentiment_score``. That number is *their*
model's output, not ours and not a calibrated probability of anything. It is
carried through as a labelled provider annotation and is never renamed to a
confidence, a probability, or an edge.

Free-tier limits: **5 requests/minute, 25/day**. That is small enough that news
must be cached, not polled.
"""

from __future__ import annotations

import time
from datetime import UTC, datetime
from typing import Any

from quantedge.config import get_settings, providers_config
from quantedge.contracts import AssetClass, HealthStatus, NewsItem, ProviderHealth
from quantedge.errors import (
    ProviderBadResponseError,
    ProviderDisabledError,
    ProviderRateLimitError,
    QuantEdgeError,
)
from quantedge.logging import get_logger
from quantedge.providers.base import NewsProvider
from quantedge.providers.http import HttpClientConfig, ResilientHttpClient
from quantedge.symbols import asset_class_for, normalize_symbol

__all__ = ["AlphaVantageNewsProvider"]

log = get_logger(__name__)

PROVIDER = "alphavantage"
_CREDENTIAL_ENV = "ALPHA_VANTAGE_API_KEY"

# Alpha Vantage answers "quota exhausted" and "premium endpoint" with HTTP 200
# and a prose field. Status codes carry no reliability signal here.
_GATE_FIELDS = ("Information", "Note", "Error Message")


class AlphaVantageNewsProvider(NewsProvider):
    """Market and symbol news from Alpha Vantage's NEWS_SENTIMENT feed."""

    name = PROVIDER

    def __init__(self, base_url: str | None = None) -> None:
        settings = get_settings()
        self._api_key = settings.secret(settings.alpha_vantage_api_key)
        self._base_url = (base_url or "https://www.alphavantage.co").rstrip("/")

        cfg = providers_config()
        defaults = cfg.get("defaults", {})
        provider_cfg = cfg.get("providers", {}).get(PROVIDER, {})
        rate = provider_cfg.get("rate_limit", {})

        self._config_enabled = bool(provider_cfg.get("enabled", True))
        self._capabilities = provider_cfg.get(
            "capabilities",
            {"economic_calendar": False, "market_news": True, "symbol_news": True},
        )
        self._client = ResilientHttpClient(
            provider=PROVIDER,
            base_url=self._base_url,
            config=HttpClientConfig(
                timeout_seconds=float(defaults.get("timeout_seconds", 10.0)),
                connect_timeout_seconds=float(defaults.get("connect_timeout_seconds", 5.0)),
                max_retries=int(defaults.get("max_retries", 3)),
                backoff_base_seconds=float(defaults.get("backoff_base_seconds", 0.5)),
                backoff_max_seconds=float(defaults.get("backoff_max_seconds", 8.0)),
                requests_per_minute=rate.get("requests_per_minute", 5),
                requests_per_day=rate.get("requests_per_day", 25),
                min_interval_seconds=float(rate.get("min_interval_seconds", 1.2)),
            ),
        )

    # ------------------------------------------------------------------ #
    # identity / health                                                  #
    # ------------------------------------------------------------------ #

    @property
    def enabled(self) -> bool:
        return self._config_enabled and self.credentials_present

    @property
    def credentials_present(self) -> bool:
        return bool(self._api_key)

    def capabilities(self) -> dict[str, bool]:
        return dict(self._capabilities)

    def missing_env(self) -> list[str]:
        return [] if self._api_key else [_CREDENTIAL_ENV]

    def _require_credentials(self) -> str:
        if not self._api_key:
            raise ProviderDisabledError(PROVIDER, [_CREDENTIAL_ENV])
        return self._api_key

    async def health_check(self) -> ProviderHealth:
        """Probe the news feed. Never raises; costs one of 25 daily calls."""
        if not self._config_enabled:
            return self._health(HealthStatus.DISABLED, "disabled in config/providers.yaml")
        if not self.credentials_present:
            return self._health(
                HealthStatus.DISABLED, f"credential not configured: {_CREDENTIAL_ENV}"
            )

        started = time.perf_counter()
        try:
            payload = await self._query(
                {"function": "NEWS_SENTIMENT", "topics": "financial_markets", "limit": "1"},
                dedupe=False,
            )
            latency_ms = round((time.perf_counter() - started) * 1000.0, 2)
            count = len(payload.get("feed") or [])
            return self._health(
                HealthStatus.OK,
                f"news feed reachable ({count} item(s) in probe)",
                latency_ms=latency_ms,
            )
        except ProviderRateLimitError as exc:
            # A quota wall is a real limitation, not a broken provider.
            return self._health(HealthStatus.DEGRADED, exc.message)
        except QuantEdgeError as exc:
            return self._health(HealthStatus.ERROR, exc.message)
        except Exception as exc:  # noqa: BLE001 - health must never propagate
            return self._health(HealthStatus.ERROR, f"unexpected error: {type(exc).__name__}")

    def _health(
        self,
        status: HealthStatus,
        message: str,
        *,
        latency_ms: float | None = None,
    ) -> ProviderHealth:
        return ProviderHealth(
            provider=PROVIDER,
            kind=self.kind,
            status=status,
            enabled=self._config_enabled,
            credentials_present=self.credentials_present,
            latency_ms=latency_ms,
            capabilities=self.capabilities(),
            message=message,
            limitations=[
                "NO economic calendar: Alpha Vantage publishes no such endpoint",
                "free plan: 5 requests/minute, 25/day",
                "provider sentiment scores are Alpha Vantage's own model output, "
                "not calibrated probabilities",
            ],
            missing_env=self.missing_env(),
            circuit_state=self._client.circuit_state,  # type: ignore[arg-type]
        )

    # ------------------------------------------------------------------ #
    # transport                                                          #
    # ------------------------------------------------------------------ #

    async def _query(self, params: dict[str, str], *, dedupe: bool = True) -> dict[str, Any]:
        """One ``/query`` call, with gate messages turned into real errors."""
        params = {**params, "apikey": self._require_credentials()}
        payload = await self._client.get_json("/query", params=params, dedupe=dedupe)
        if not isinstance(payload, dict):
            raise ProviderBadResponseError(
                PROVIDER, f"expected a JSON object, got {type(payload).__name__}"
            )

        for field in _GATE_FIELDS:
            text = payload.get(field)
            if not text:
                continue
            lowered = str(text).lower()
            if "rate limit" in lowered or "requests per day" in lowered or "frequency" in lowered:
                raise ProviderRateLimitError(PROVIDER, f"quota exhausted: {text}")
            if "premium" in lowered or "subscribe" in lowered:
                raise ProviderBadResponseError(PROVIDER, f"endpoint is premium-gated: {text}")
            raise ProviderBadResponseError(PROVIDER, f"provider message: {text}")
        return payload

    # ------------------------------------------------------------------ #
    # news                                                               #
    # ------------------------------------------------------------------ #

    async def get_market_news(self, limit: int = 20) -> list[NewsItem]:
        """Broad market news, newest first."""
        limit = max(1, min(limit, 1000))
        payload = await self._query(
            {
                "function": "NEWS_SENTIMENT",
                "topics": "financial_markets,economy_macro,finance",
                "limit": str(limit),
                "sort": "LATEST",
            }
        )
        return self._parse_feed(payload, limit=limit)

    async def get_symbol_news(self, symbol: str, limit: int = 20) -> list[NewsItem]:
        """News mentioning one instrument, newest first.

        Alpha Vantage's ``tickers`` filter covers equities and crypto (as
        ``CRYPTO:BTC``) but not FX pairs. For forex the ticker filter is dropped
        and the macro feed is returned instead -- correctly labelled, because
        claiming pair-specific coverage the provider cannot deliver would be a
        fabrication.
        """
        limit = max(1, min(limit, 1000))
        canonical = normalize_symbol(symbol)
        ticker = self._to_ticker(canonical)

        params = {
            "function": "NEWS_SENTIMENT",
            "limit": str(limit),
            "sort": "LATEST",
        }
        if ticker is not None:
            params["tickers"] = ticker
        else:
            params["topics"] = "economy_macro,economy_monetary,financial_markets"

        payload = await self._query(params)
        items = self._parse_feed(payload, limit=limit)
        if ticker is None:
            # Do not imply the provider matched this symbol when it did not.
            return [item.model_copy(update={"symbols": []}) for item in items]
        return items

    @staticmethod
    def _to_ticker(canonical: str) -> str | None:
        """Alpha Vantage ticker for a canonical symbol, or ``None`` if unsupported."""
        try:
            asset_class = asset_class_for(canonical)
        except QuantEdgeError:
            return None

        if asset_class is AssetClass.STOCK:
            return canonical
        if asset_class is AssetClass.CRYPTO:
            for quote in ("USDT", "USDC", "USD", "BTC", "ETH"):
                if canonical.endswith(quote) and len(canonical) > len(quote):
                    return f"CRYPTO:{canonical[: -len(quote)]}"
            return f"CRYPTO:{canonical}"
        # Forex, indices and commodities have no usable ticker filter here.
        return None

    def _parse_feed(self, payload: dict[str, Any], *, limit: int) -> list[NewsItem]:
        feed = payload.get("feed")
        if not isinstance(feed, list):
            raise ProviderBadResponseError(PROVIDER, "NEWS_SENTIMENT payload missing 'feed'")

        items: list[NewsItem] = []
        skipped = 0
        for raw in feed:
            if not isinstance(raw, dict):
                skipped += 1
                continue
            published = self._parse_time(raw.get("time_published"))
            title = str(raw.get("title") or "").strip()
            if published is None or not title:
                # An item with no timestamp cannot be aged, and an item with no
                # headline carries nothing. Dropping beats guessing.
                skipped += 1
                continue
            items.append(
                NewsItem(
                    provider=PROVIDER,
                    headline=title,
                    summary=(str(raw.get("summary")).strip() or None)
                    if raw.get("summary")
                    else None,
                    url=str(raw.get("url")) if raw.get("url") else None,
                    source=str(raw.get("source")) if raw.get("source") else None,
                    symbols=self._extract_symbols(raw),
                    published_at_utc=published,
                )
            )

        if skipped:
            log.warning("dropped unusable news items", extra={"count": skipped})

        items.sort(key=lambda i: i.published_at_utc, reverse=True)
        return items[:limit]

    @staticmethod
    def _extract_symbols(raw: dict[str, Any]) -> list[str]:
        """Canonical symbols the provider itself attached to the article."""
        out: list[str] = []
        for entry in raw.get("ticker_sentiment") or []:
            if not isinstance(entry, dict):
                continue
            ticker = str(entry.get("ticker") or "").strip().upper()
            if not ticker:
                continue
            # "CRYPTO:BTC" -> "BTC"; a bare equity ticker passes through.
            out.append(ticker.split(":", 1)[1] if ":" in ticker else ticker)
        return out

    @staticmethod
    def _parse_time(value: Any) -> datetime | None:
        """Parse Alpha Vantage's ``YYYYMMDDTHHMMSS`` stamp as UTC.

        Returns ``None`` rather than substituting "now": an article stamped with
        the retrieval time would read as breaking news forever.
        """
        if not value:
            return None
        text = str(value).strip()
        for fmt in ("%Y%m%dT%H%M%S", "%Y%m%dT%H%M", "%Y%m%d"):
            try:
                return datetime.strptime(text, fmt).replace(tzinfo=UTC)
            except ValueError:
                continue
        return None

    def rate_limit_snapshot(self) -> dict[str, Any]:
        return self._client.rate_limit_snapshot()

    async def aclose(self) -> None:
        await self._client.aclose()
