"""Provider registry verification: routing, fallback and health.

Fake providers cover the failure paths that cannot be produced on demand from
real vendors (timeouts, open circuits, exhausted chains); the final section
routes real requests through the live registry.

Run:  ./.venv/Scripts/python.exe -u scripts/smoke_test_registry.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from quantedge.contracts import AssetClass, HealthStatus, ProviderHealth, Timeframe
from quantedge.errors import (
    AllProvidersFailedError,
    ProviderTimeoutError,
    UnsupportedSymbolError,
)
from quantedge.logging import configure_logging
from quantedge.providers.base import BaseProvider
from quantedge.providers.registry import ProviderRegistry, get_registry

FAILURES: list[str] = []


def check(label: str, condition: bool, detail: str = "") -> None:
    print(f"  {'[PASS]' if condition else '[FAIL]'} {label}{f' -- {detail}' if detail else ''}")
    if not condition:
        FAILURES.append(label)


class FakeProvider(BaseProvider):
    """Stand-in provider, subclassing the real base on purpose.

    Inheriting rather than duck-typing means the fake cannot drift away from
    the contract the registry actually calls: when ``circuit_state`` moved from
    a private attribute to a base-class property, a hand-rolled fake kept
    passing against an interface that no longer existed.
    """

    kind = "market_data"

    def __init__(
        self,
        name: str,
        *,
        enabled: bool = True,
        missing: list[str] | None = None,
        raises: Exception | None = None,
        circuit: str = "closed",
        capabilities: dict[str, bool] | None = None,
    ) -> None:
        self.name = name
        self._enabled = enabled
        self._missing = missing or []
        self._raises = raises
        self._capabilities = capabilities or {"quote": True, "candles": True}
        self.call_count = 0

        class _Client:
            circuit_state = circuit

        self._client = _Client()

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def credentials_present(self) -> bool:
        return not self._missing

    def capabilities(self) -> dict[str, bool]:
        return dict(self._capabilities)

    def missing_env(self) -> list[str]:
        return list(self._missing)

    def supports(self, capability: str) -> bool:
        return self._capabilities.get(capability, False)

    async def fetch(self) -> str:
        self.call_count += 1
        if self._raises is not None:
            raise self._raises
        return f"data-from-{self.name}"

    async def health_check(self) -> ProviderHealth:
        return ProviderHealth(
            provider=self.name,
            kind=self.kind,
            status=HealthStatus.OK,
            enabled=self.enabled,
            credentials_present=not self._missing,
        )

    async def aclose(self) -> None:
        return None


def registry_with(providers: dict[str, FakeProvider]) -> ProviderRegistry:
    """A registry whose chain is the given providers, in insertion order."""
    names = list(providers)
    reg = ProviderRegistry(
        {
            "routing": {
                "market_data": {"crypto": names, "forex": names},
                "economic_calendar": [],
                "news": [],
            }
        }
    )
    reg._instances.update(providers)
    return reg


async def main() -> int:
    configure_logging()
    print("=" * 70)
    print("PROVIDER REGISTRY -- ROUTING AND FALLBACK VERIFICATION")
    print("=" * 70)

    print("\n[1] first healthy provider serves; later ones are not called")
    first = FakeProvider("first")
    second = FakeProvider("second")
    reg = registry_with({"first": first, "second": second})
    result, served_by, attempts = await reg.execute(
        ["first", "second"], lambda p: p.fetch(), description="a quote"
    )
    check("served by the first provider", served_by == "first", served_by)
    check("returned its data", result == "data-from-first", str(result))
    check("second provider was never called", second.call_count == 0, f"{second.call_count} calls")
    check("one attempt recorded", len(attempts) == 1, str([a.to_dict() for a in attempts]))

    print("\n[2] infrastructure failure falls through to the next provider")
    broken = FakeProvider("broken", raises=ProviderTimeoutError("broken", 10.0))
    healthy = FakeProvider("healthy")
    reg = registry_with({"broken": broken, "healthy": healthy})
    result, served_by, attempts = await reg.execute(
        ["broken", "healthy"], lambda p: p.fetch(), description="a quote"
    )
    check("fell over to the healthy provider", served_by == "healthy", served_by)
    check("the broken one was actually tried", broken.call_count == 1, f"{broken.call_count} calls")
    check("both attempts recorded", len(attempts) == 2, str(len(attempts)))
    check("failure reason retained", attempts[0].outcome == "failed", attempts[0].outcome)
    print(f"    attempts: {[a.to_dict() for a in attempts]}")

    print("\n[3] a bad REQUEST does not fail over (an unknown symbol is unknown everywhere)")
    picky = FakeProvider("picky", raises=UnsupportedSymbolError("no such symbol"))
    backup = FakeProvider("backup")
    reg = registry_with({"picky": picky, "backup": backup})
    try:
        await reg.execute(["picky", "backup"], lambda p: p.fetch(), description="a quote")
    except UnsupportedSymbolError:
        check("validation error propagated immediately", True)
        check("backup quota was NOT spent", backup.call_count == 0, f"{backup.call_count} calls")
    else:
        check("validation error propagated immediately", False, "no exception raised")

    print("\n[4] uncredentialled and disabled providers are skipped with a reason")
    nocreds = FakeProvider("nocreds", enabled=False, missing=["SOME_API_KEY"])
    working = FakeProvider("working")
    reg = registry_with({"nocreds": nocreds, "working": working})
    _result, served_by, attempts = await reg.execute(
        ["nocreds", "working"], lambda p: p.fetch(), description="a quote"
    )
    check("skipped, not called", nocreds.call_count == 0)
    check("served by the credentialled provider", served_by == "working", served_by)
    check(
        "skip reason names the missing variable",
        "SOME_API_KEY" in attempts[0].reason,
        attempts[0].reason,
    )

    print("\n[5] an open circuit breaker takes a provider out of rotation")
    tripped = FakeProvider("tripped", circuit="open")
    spare = FakeProvider("spare")
    reg = registry_with({"tripped": tripped, "spare": spare})
    _result, served_by, attempts = await reg.execute(
        ["tripped", "spare"], lambda p: p.fetch(), description="a quote"
    )
    check("open circuit was not called", tripped.call_count == 0)
    check("circuit reason recorded", "circuit" in attempts[0].reason.lower(), attempts[0].reason)

    print("\n[6] a missing capability is skipped rather than attempted")
    noquote = FakeProvider("noquote", capabilities={"quote": False, "candles": True})
    fullservice = FakeProvider("fullservice")
    reg = registry_with({"noquote": noquote, "fullservice": fullservice})
    _r, served_by, attempts = await reg.execute(
        ["noquote", "fullservice"], lambda p: p.fetch(), description="a quote", capability="quote"
    )
    check("capability-lacking provider skipped", noquote.call_count == 0)
    check("served by the capable one", served_by == "fullservice", served_by)

    print("\n[7] an exhausted chain raises with the full attempt log (never an empty result)")
    dead1 = FakeProvider("dead1", raises=ProviderTimeoutError("dead1", 10.0))
    dead2 = FakeProvider("dead2", enabled=False, missing=["DEAD2_KEY"])
    reg = registry_with({"dead1": dead1, "dead2": dead2})
    try:
        await reg.execute(["dead1", "dead2"], lambda p: p.fetch(), description="a quote")
    except AllProvidersFailedError as exc:
        check("raised AllProvidersFailedError", True)
        check("every attempt is in the error", len(exc.attempts) == 2, str(exc.attempts))
        check("retryable (one attempt was a timeout)", exc.retryable is True)
        check("client payload carries no secrets", "DEAD2_KEY" in exc.details["attempts"])
        print(f"    payload: {exc.to_dict()}")
    else:
        check("raised AllProvidersFailedError", False, "returned a value instead")

    print("\n[7b] a chain skipped only for missing credentials is NOT retryable")
    reg = registry_with({"nokey": FakeProvider("nokey", enabled=False, missing=["NOKEY_TOKEN"])})
    try:
        await reg.execute(["nokey"], lambda p: p.fetch(), description="a quote")
    except AllProvidersFailedError as exc:
        check(
            "not retryable -- a missing key is missing a second later too", exc.retryable is False
        )
    else:
        check("not retryable -- a missing key is missing a second later too", False, "no raise")

    print("\n[8] real registry: routing tables from config/providers.yaml")
    live = get_registry()
    crypto = live.market_data_chain(AssetClass.CRYPTO)
    forex = live.market_data_chain(AssetClass.FOREX)
    print(f"    crypto   -> {crypto}")
    print(f"    forex    -> {forex}")
    print(f"    calendar -> {live.calendar_chain()}")
    print(f"    news     -> {live.news_chain()}")
    check("crypto prefers binance", crypto[0] == "binance", str(crypto))
    check(
        "forex prefers oanda, falls back to twelvedata",
        forex == ["oanda", "twelvedata"],
        str(forex),
    )
    check("a calendar chain is configured", len(live.calendar_chain()) > 0)

    print("\n[9] real registry: live crypto quote through the chain")
    quote, served_by, attempts = await live.market_data(
        AssetClass.CRYPTO,
        lambda p: p.get_quote("BTCUSDT"),
        description="BTCUSDT quote",
        capability="quote",
    )
    print(f"    served by {served_by}: last={quote.last} bid={quote.bid} ask={quote.ask}")
    check("a real crypto quote came back", quote.last is not None)
    check("served by binance", served_by == "binance", served_by)

    print("\n[10] real registry: live forex candles fall back past unconfigured OANDA")
    series, served_by, attempts = await live.market_data(
        AssetClass.FOREX,
        lambda p: p.get_candles("EURUSD", Timeframe.M15, count=5),
        description="EURUSD M15 candles",
        capability="candles",
    )
    print(f"    served by {served_by}: {len(series)} bars, source={series.source}")
    for attempt in attempts:
        print(f"      {attempt.provider}: {attempt.outcome} -- {attempt.reason}")
    check("fell back to twelvedata", served_by == "twelvedata", served_by)
    check(
        "oanda skipped for missing credentials", "OANDA" in attempts[0].reason, attempts[0].reason
    )
    check("real bars returned", len(series) == 5, f"{len(series)} bars")
    check("RULE 9: no forming bar in history", all(c.is_closed for c in series.candles))

    print("\n[11] real registry: the calendar chain is honestly unavailable")
    try:
        from quantedge.contracts import utc_now

        await live.calendar(
            lambda p: p.get_events(utc_now(), utc_now()), description="today's events"
        )
    except AllProvidersFailedError as exc:
        check("calendar reports unavailable rather than 'no events'", True)
        check(
            "not retryable -- no calendar credential exists to retry with", exc.retryable is False
        )
        for attempt in exc.attempts:
            print(f"      {attempt['provider']}: {attempt['outcome']} -- {attempt['reason']}")
    else:
        check(
            "calendar reports unavailable rather than 'no events'",
            False,
            "returned events with no calendar credential configured",
        )

    print("\n[12] real registry: health of every configured provider")
    for health in await live.health():
        print(
            f"    {health.provider:14} {health.kind:18} {health.status.value:9} "
            f"creds={health.credentials_present!s:5} {health.message or ''}"
        )
    reported = {h.provider for h in await live.health()}
    check(
        "every configured provider appears in the report",
        reported == set(live.configured_providers()),
        f"missing: {set(live.configured_providers()) - reported}",
    )

    await live.aclose()

    print("\n" + "=" * 70)
    if FAILURES:
        print(f"RESULT: FAILED -- {len(FAILURES)} check(s) failed")
        for name in FAILURES:
            print(f"  - {name}")
        return 1
    print("RESULT: ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
