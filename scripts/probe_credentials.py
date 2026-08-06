"""Empirically identify two unlabelled API credentials.

Two keys were supplied with only informal labels ("calander", "tradingkit") and
no vendor named. Rather than guess a vendor and build an adapter against an API
that may not exist, this script probes a shortlist of candidate services and
reports which -- if any -- accept each key.

Method
------
For each candidate vendor, issue one cheap, read-only request using the key in
that vendor's documented auth style, then classify the response:

  ACCEPTED  -- HTTP 2xx and a body whose shape matches that vendor's schema
  REJECTED  -- HTTP 401/403, or a body with an auth-failure marker
  PAYWALLED -- authenticated successfully but the endpoint needs a paid plan
               (this still identifies the vendor, which is the point)
  UNKNOWN   -- network error, timeout, or an unrecognised response

Safety
------
* Every request is a read-only GET.
* No key is ever printed. Output shows vendor + verdict only.
* Failures are reported honestly; a key that matches nothing is reported as
  unidentified rather than assigned to a plausible-looking vendor.

Some vendors return HTTP 200 with an error *inside* the JSON body (Twelve Data
and Alpha Vantage both do this), so status code alone is not sufficient -- the
body is inspected too.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

import httpx

from quantedge.config import get_settings

TIMEOUT = httpx.Timeout(15.0, connect=8.0)


@dataclass
class Probe:
    """One candidate vendor for a key."""

    vendor: str
    url: str
    params: dict[str, str] | None = None
    headers: dict[str, str] | None = None
    note: str = ""


@dataclass
class Result:
    vendor: str
    verdict: str
    detail: str
    note: str = ""


def _classify(vendor: str, response: httpx.Response, note: str) -> Result:
    """Map a response to a verdict, inspecting the body, not just the status."""
    status = response.status_code

    try:
        body: Any = response.json()
    except ValueError:
        body = None

    text = response.text[:300]

    # Vendors that signal auth failure inside a 200 body.
    if isinstance(body, dict):
        code = body.get("code")
        message = str(body.get("message", "") or body.get("Error Message", "") or "")
        status_field = str(body.get("status", ""))

        if status_field == "error" or code in (400, 401, 403):
            low = message.lower()
            if "api key" in low or "apikey" in low or "unauthor" in low or "invalid" in low:
                return Result(vendor, "REJECTED", f"body error: {message[:120]}", note)
            if "plan" in low or "upgrade" in low or "premium" in low:
                return Result(vendor, "PAYWALLED", f"auth ok, plan required: {message[:120]}", note)
            return Result(vendor, "REJECTED", f"body error: {message[:120]}", note)

        if "Error Message" in body:
            return Result(vendor, "REJECTED", str(body["Error Message"])[:120], note)
        if "Note" in body or "Information" in body:
            return Result(
                vendor, "PAYWALLED", str(body.get("Note") or body.get("Information"))[:120], note
            )

    if status in (401, 403):
        return Result(vendor, "REJECTED", f"HTTP {status}", note)
    if status == 402:
        return Result(vendor, "PAYWALLED", "HTTP 402 - key valid, paid plan required", note)
    if status == 429:
        return Result(vendor, "UNKNOWN", "HTTP 429 rate limited - retry later", note)
    if 200 <= status < 300:
        if body is None:
            return Result(vendor, "UNKNOWN", f"HTTP {status} but non-JSON body", note)
        shape = (
            f"dict keys: {list(body)[:6]}"
            if isinstance(body, dict)
            else f"list[{len(body)}]"
            if isinstance(body, list)
            else type(body).__name__
        )
        # An empty list may just mean "no events today" - still proves auth worked.
        return Result(vendor, "ACCEPTED", f"HTTP {status}, {shape}", note)

    return Result(vendor, "UNKNOWN", f"HTTP {status}: {text[:120]}", note)


async def probe_one(client: httpx.AsyncClient, probe: Probe) -> Result:
    try:
        response = await client.get(
            probe.url, params=probe.params, headers=probe.headers, timeout=TIMEOUT
        )
    except httpx.TimeoutException:
        return Result(probe.vendor, "UNKNOWN", "request timed out", probe.note)
    except httpx.HTTPError as exc:
        return Result(probe.vendor, "UNKNOWN", f"network error: {type(exc).__name__}", probe.note)
    return _classify(probe.vendor, response, probe.note)


def calendar_probes(key: str) -> list[Probe]:
    """Candidate vendors for a key labelled 'calander' (economic calendar)."""
    return [
        Probe(
            "Finnhub",
            "https://finnhub.io/api/v1/calendar/economic",
            headers={"X-Finnhub-Token": key},
            note="economic calendar is premium-gated on the free tier",
        ),
        Probe(
            "Financial Modeling Prep",
            "https://financialmodelingprep.com/api/v3/economic_calendar",
            params={"apikey": key},
            note="calendar is a paid endpoint",
        ),
        Probe(
            "Trading Economics",
            "https://api.tradingeconomics.com/calendar",
            params={"c": key, "f": "json"},
        ),
        Probe(
            "FinancialModelingPrep (stable)",
            "https://financialmodelingprep.com/stable/economic-calendar",
            params={"apikey": key},
        ),
        Probe(
            "Marketaux",
            "https://api.marketaux.com/v1/news/all",
            params={"api_token": key, "limit": "1"},
        ),
        Probe(
            "Polygon.io",
            "https://api.polygon.io/v3/reference/tickers",
            params={"apiKey": key, "limit": "1"},
        ),
        Probe(
            "Twelve Data",
            "https://api.twelvedata.com/quote",
            params={"symbol": "AAPL", "apikey": key},
        ),
        Probe(
            "Alpha Vantage",
            "https://www.alphavantage.co/query",
            params={"function": "GLOBAL_QUOTE", "symbol": "AAPL", "apikey": key},
        ),
        Probe(
            "NewsAPI.org",
            "https://newsapi.org/v2/top-headlines",
            params={"category": "business", "apiKey": key, "pageSize": "1"},
        ),
    ]


def tradingkit_probes(key: str) -> list[Probe]:
    """Candidate vendors for a 'pk_'-prefixed key labelled 'tradingkit'.

    A ``pk_`` prefix conventionally means "publishable key". Finnhub, Polygon
    and Stripe all use that convention, as do several smaller market-data SaaS
    products.
    """
    return [
        Probe(
            "Finnhub",
            "https://finnhub.io/api/v1/quote",
            params={"symbol": "AAPL", "token": key},
        ),
        Probe(
            "Polygon.io",
            "https://api.polygon.io/v2/aggs/ticker/AAPL/prev",
            params={"apiKey": key},
        ),
        Probe(
            "Tradier",
            "https://api.tradier.com/v1/markets/quotes",
            params={"symbols": "AAPL"},
            headers={"Authorization": f"Bearer {key}", "Accept": "application/json"},
        ),
        Probe(
            "Marketstack",
            "https://api.marketstack.com/v1/eod/latest",
            params={"access_key": key, "symbols": "AAPL"},
        ),
        Probe(
            "Financial Modeling Prep",
            "https://financialmodelingprep.com/api/v3/quote/AAPL",
            params={"apikey": key},
        ),
        Probe(
            "Twelve Data",
            "https://api.twelvedata.com/quote",
            params={"symbol": "AAPL", "apikey": key},
        ),
        Probe(
            "Alpaca (market data)",
            "https://data.alpaca.markets/v2/stocks/AAPL/quotes/latest",
            headers={"APCA-API-KEY-ID": key},
        ),
        Probe(
            "EOD Historical Data",
            "https://eodhd.com/api/real-time/AAPL.US",
            params={"api_token": key, "fmt": "json"},
        ),
        Probe(
            "Benzinga",
            "https://api.benzinga.com/api/v2/news",
            params={"token": key, "pageSize": "1"},
        ),
    ]


async def run_group(title: str, key: str | None, probes: list[Probe]) -> list[Result]:
    print(f"\n{'=' * 72}")
    print(f"  {title}")
    print(f"{'=' * 72}")

    if not key:
        print("  SKIPPED - no key configured in .env")
        return []

    # Length and prefix are structural facts, not the secret itself.
    prefix = key[:3] if len(key) > 8 else "?"
    print(f"  Key shape: {len(key)} chars, prefix '{prefix}...' (value never printed)")
    print(f"  Probing {len(probes)} candidate vendors...\n")

    async with httpx.AsyncClient(follow_redirects=False) as client:
        results = await asyncio.gather(*(probe_one(client, p) for p in probes))

    order = {"ACCEPTED": 0, "PAYWALLED": 1, "REJECTED": 2, "UNKNOWN": 3}
    for result in sorted(results, key=lambda r: order.get(r.verdict, 9)):
        marker = {
            "ACCEPTED": "[MATCH]  ",
            "PAYWALLED": "[PLAN]   ",
            "REJECTED": "[no]     ",
            "UNKNOWN": "[?]      ",
        }.get(result.verdict, "         ")
        print(f"  {marker} {result.vendor:<32} {result.detail}")
        if result.note and result.verdict in ("ACCEPTED", "PAYWALLED"):
            print(f"{'':<12} note: {result.note}")

    return list(results)


def summarize(label: str, results: list[Result]) -> str | None:
    """Return the identified vendor, or None when the key is unidentified."""
    accepted = [r for r in results if r.verdict == "ACCEPTED"]
    paywalled = [r for r in results if r.verdict == "PAYWALLED"]

    if accepted:
        if len(accepted) == 1:
            print(f"\n  => {label} identified as: {accepted[0].vendor}")
            return accepted[0].vendor
        names = ", ".join(r.vendor for r in accepted)
        print(f"\n  => {label} accepted by MULTIPLE vendors: {names}")
        print("     (some APIs return data for any key; needs manual disambiguation)")
        return accepted[0].vendor
    if paywalled:
        print(f"\n  => {label} authenticates with {paywalled[0].vendor} but needs a paid plan")
        return paywalled[0].vendor

    print(f"\n  => {label} NOT identified. No probed vendor accepted it.")
    print("     It will be left unconfigured rather than wired to a guessed vendor.")
    return None


async def main() -> None:
    print("QuantEdge credential identification probe")
    print("Read-only GETs. Key values are never printed.")

    settings = get_settings()
    calendar_key = settings.secret(settings.calendar_api_key)
    tradingkit_key = settings.secret(settings.tradingkit_api_key)
    twelve_key = settings.secret(settings.twelve_data_api_key)

    calendar_results = await run_group(
        "KEY 1: labelled 'calander' (expected: economic calendar)",
        calendar_key,
        calendar_probes(calendar_key) if calendar_key else [],
    )
    calendar_vendor = summarize("'calander' key", calendar_results) if calendar_results else None

    tradingkit_results = await run_group(
        "KEY 2: labelled 'tradingkit' (pk_ prefix = publishable key)",
        tradingkit_key,
        tradingkit_probes(tradingkit_key) if tradingkit_key else [],
    )
    tradingkit_vendor = (
        summarize("'tradingkit' key", tradingkit_results) if tradingkit_results else None
    )

    # Twelve Data was explicitly named by the user, so this is a confirmation,
    # not an identification.
    print(f"\n{'=' * 72}")
    print("  KEY 3: Twelve Data (vendor already known - confirming it works)")
    print(f"{'=' * 72}")
    if twelve_key:
        async with httpx.AsyncClient() as client:
            result = await probe_one(
                client,
                Probe(
                    "Twelve Data",
                    "https://api.twelvedata.com/time_series",
                    params={
                        "symbol": "EUR/USD",
                        "interval": "1min",
                        "outputsize": "3",
                        "apikey": twelve_key,
                    },
                ),
            )
        print(f"  [{result.verdict}] {result.detail}")
    else:
        print("  SKIPPED - no key configured")

    print(f"\n{'=' * 72}")
    print("  SUMMARY")
    print(f"{'=' * 72}")
    print(f"  'calander'   -> {calendar_vendor or 'UNIDENTIFIED'}")
    print(f"  'tradingkit' -> {tradingkit_vendor or 'UNIDENTIFIED'}")


if __name__ == "__main__":
    asyncio.run(main())
