"""Persistence verification -- both backends, against a real temporary database.

Nothing here is mocked. Section 1 onward runs against a SQLite file in a temp
directory, created through the same :func:`create_all` the application uses, and
the file is deleted at the end. Section 8 runs the identical assertions against
:class:`MemoryRepository`, because the two backends are only interchangeable if
they actually agree -- a divergence would surface as a bug in whichever one was
configured second.

The properties that matter most are the ones a passing round-trip does not prove:

* **[3] Idempotence.** Re-writing a closed bar must not duplicate it and must not
  overwrite it. First observation wins, so a provider restating a bar is visible
  as a restatement rather than silently replacing history.
* **[5] Append-only.** ``settled_signals`` and ``audit_logs`` must refuse UPDATE
  and DELETE. This is the performance record; editing it is editing the score.
* **[6] Quota durability.** The counter has to live in the database, not in a
  limiter object. Alpha Vantage allows 25 requests a day, and a per-process
  counter resets on every CLI invocation.
* **[7] Rule 9.** A forming candle must be refused, not filtered. Filtering lets
  a caller believe a 200-bar write landed when 199 did, and the missing bar is
  always the most recent one.

Run:  ./.venv/Scripts/python.exe -u scripts/verify_persistence.py
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from quantedge.contracts import (
    AssetClass,
    Candle,
    DataQualityReport,
    HealthStatus,
    ProviderHealth,
    QualityStatus,
    Quote,
    SettledSignal,
    SettlementOutcome,
    SignalDirection,
    SymbolInfo,
    Timeframe,
)
from quantedge.errors import PersistenceError

FAILURES: list[str] = []


def check(label: str, condition: bool, detail: str = "") -> None:
    print(f"  {'[PASS]' if condition else '[FAIL]'} {label}{f' -- {detail}' if detail else ''}")
    if not condition:
        FAILURES.append(label)


START = datetime(2026, 1, 1, tzinfo=UTC)


def bar(index: int, close: float, *, is_closed: bool = True) -> Candle:
    price = Decimal(str(close))
    return Candle(
        provider="fixture",
        symbol="BTCUSDT",
        asset_class=AssetClass.CRYPTO,
        timeframe=Timeframe.M5,
        open_time_utc=START + timedelta(minutes=5 * index),
        close_time_utc=START + timedelta(minutes=5 * (index + 1)),
        open=price,
        high=price + Decimal("5"),
        low=price - Decimal("5"),
        close=price,
        volume=Decimal("10.5"),
        is_closed=is_closed,
    )


def section_schema(engine: object) -> None:
    from quantedge.repositories.database import backend_info, create_all

    print("\n[1] Schema creation")
    names = create_all(engine)  # type: ignore[arg-type]
    check("15 tables created", len(names) == 15, f"got {len(names)}")
    for required in ("candles", "settled_signals", "audit_logs", "provider_quota"):
        check(f"table '{required}' exists", required in names)

    check("create_all is idempotent", create_all(engine) == names)  # type: ignore[arg-type]

    info = backend_info(engine)  # type: ignore[arg-type]
    check("backend reports sqlite", info["backend"] == "sqlite", info["backend"])
    check("backend reports durable", info["durable"] is True)


def section_dsn_redaction() -> None:
    from quantedge.repositories.database import redact_dsn

    print("\n[2] DSN redaction -- a DSN is a secret")
    raw = "postgresql+psycopg://qe_user:sup3r-s3cret@db.example.com:5432/quantedge"
    safe = redact_dsn(raw)
    check("password is gone", "sup3r-s3cret" not in safe, safe)
    check("username is gone", "qe_user" not in safe)
    check("host is kept", "db.example.com" in safe)
    check("database is kept", "quantedge" in safe)
    check("keyless DSN is untouched", redact_dsn("sqlite:///./local.db") == "sqlite:///./local.db")


def section_candles(repo: object) -> None:
    print("\n[3] Candles -- round-trip, idempotence, ordering")
    bars = [bar(i, 50000.0 + i * 10) for i in range(20)]
    written = repo.save_candles(bars)  # type: ignore[attr-defined]
    check("20 bars written", written == 20, f"got {written}")

    again = repo.save_candles(bars)  # type: ignore[attr-defined]
    check("re-writing the same 20 inserts nothing", again == 0, f"got {again}")
    check("count is still 20", repo.candle_count("BTCUSDT", Timeframe.M5) == 20)  # type: ignore[attr-defined]

    # First observation wins: a restated bar must not overwrite the original.
    restated = bar(5, 99999.0)
    repo.save_candles([restated])  # type: ignore[attr-defined]
    loaded = repo.load_candles("BTCUSDT", Timeframe.M5, limit=100)  # type: ignore[attr-defined]
    kept = next(c for c in loaded if c.open_time_utc == restated.open_time_utc)
    check("restatement did not overwrite", kept.close == Decimal("50050"), str(kept.close))

    check(
        "load returns oldest-first",
        [c.open_time_utc for c in loaded] == sorted(c.open_time_utc for c in loaded),
    )
    check("Decimal survives the round-trip", loaded[0].close == Decimal("50000"))
    check("volume survives the round-trip", loaded[0].volume == Decimal("10.5"))
    check("timestamps come back tz-aware", loaded[0].open_time_utc.tzinfo is not None)

    tail = repo.load_candles("BTCUSDT", Timeframe.M5, limit=5)  # type: ignore[attr-defined]
    check("limit returns the newest 5", len(tail) == 5 and tail[-1] == loaded[-1])

    latest = repo.latest_candle("BTCUSDT", Timeframe.M5)  # type: ignore[attr-defined]
    check("latest_candle is the newest bar", latest is not None and latest == loaded[-1])

    since = repo.load_candles(  # type: ignore[attr-defined]
        "BTCUSDT", Timeframe.M5, limit=100, since=START + timedelta(minutes=50)
    )
    check("since filters older bars", len(since) == 10, f"got {len(since)}")
    check("unknown symbol returns empty", repo.load_candles("NOPEUSDT", Timeframe.M5) == [])  # type: ignore[attr-defined]


def section_quotes_symbols(repo: object) -> None:
    print("\n[4] Quotes, symbols, health")
    quote = Quote(
        provider="fixture",
        symbol="EURUSD",
        asset_class=AssetClass.FOREX,
        bid=Decimal("1.085"),
        ask=Decimal("1.0852"),
        provider_time_utc=START,
    )
    repo.save_quote(quote)  # type: ignore[attr-defined]
    back = repo.latest_quote("EURUSD")  # type: ignore[attr-defined]
    check("quote round-trips", back is not None and back.bid == Decimal("1.085"))
    check("ask round-trips", back is not None and back.ask == Decimal("1.0852"))
    check("missing quote returns None", repo.latest_quote("GBPJPY") is None)  # type: ignore[attr-defined]

    # Built rather than model_copy'd: model_copy skips validation, so a fixture
    # that crossed the market would pass here and only fail on the read path.
    newer = Quote(
        provider="fixture",
        symbol="EURUSD",
        asset_class=AssetClass.FOREX,
        bid=Decimal("1.09"),
        ask=Decimal("1.0902"),
        provider_time_utc=START + timedelta(minutes=1),
    )
    repo.save_quote(newer)  # type: ignore[attr-defined]
    latest = repo.latest_quote("EURUSD")  # type: ignore[attr-defined]
    check(
        "latest_quote prefers the newer tick", latest is not None and latest.bid == Decimal("1.09")
    )

    written = repo.upsert_symbols(  # type: ignore[attr-defined]
        [
            SymbolInfo(
                provider="fixture",
                symbol="BTCUSDT",
                provider_symbol="BTCUSDT",
                asset_class=AssetClass.CRYPTO,
                is_tradable=True,
            ),
            SymbolInfo(
                provider="fixture",
                symbol="DELISTED",
                provider_symbol="DELISTED",
                asset_class=AssetClass.CRYPTO,
                is_tradable=False,
            ),
        ]
    )
    check("2 symbols upserted", written == 2, f"got {written}")
    check("tradable symbol is allowed", repo.is_symbol_allowed("BTCUSDT") is True)  # type: ignore[attr-defined]
    check("untradable symbol is refused", repo.is_symbol_allowed("DELISTED") is False)  # type: ignore[attr-defined]
    check("unknown symbol is refused", repo.is_symbol_allowed("MADEUP") is False)  # type: ignore[attr-defined]

    # Scanning is an operator decision, not a consequence of a provider listing
    # the instrument. A fresh upsert must leave it off.
    check("upsert does not enable scanning", repo.scannable_symbols() == [])  # type: ignore[attr-defined]
    repo.set_symbol_scanning("BTCUSDT", enabled=True)  # type: ignore[attr-defined]
    check("operator can enable scanning", repo.scannable_symbols() == ["BTCUSDT"])  # type: ignore[attr-defined]

    # And a provider refresh must not re-enable something a human switched off,
    # nor switch off something a human enabled.
    repo.upsert_symbols(  # type: ignore[attr-defined]
        [
            SymbolInfo(
                provider="fixture",
                symbol="BTCUSDT",
                provider_symbol="BTCUSDT",
                asset_class=AssetClass.CRYPTO,
                is_tradable=True,
                price_precision=2,
            )
        ]
    )
    check("refresh preserves the operator's choice", repo.scannable_symbols() == ["BTCUSDT"])  # type: ignore[attr-defined]

    repo.set_symbol_scanning("DELISTED", enabled=True)  # type: ignore[attr-defined]
    check("untradable stays out of the scan list", repo.scannable_symbols() == ["BTCUSDT"])  # type: ignore[attr-defined]
    repo.set_symbol_scanning("BTCUSDT", enabled=False)  # type: ignore[attr-defined]
    check("operator can disable scanning", repo.scannable_symbols() == [])  # type: ignore[attr-defined]
    repo.set_symbol_scanning("BTCUSDT", enabled=True)  # type: ignore[attr-defined]

    repo.record_health(  # type: ignore[attr-defined]
        ProviderHealth(
            provider="fixture",
            kind="market_data",
            status=HealthStatus.OK,
            enabled=True,
            credentials_present=False,
            latency_ms=12.5,
        )
    )
    history = repo.recent_health("fixture")  # type: ignore[attr-defined]
    check("health recorded", len(history) == 1 and history[0]["status"] == "ok")
    check(
        "health row carries no credential value",
        all("api_key" not in str(v).lower() for v in history[0].values()),
    )


def section_append_only(repo: object) -> None:
    print("\n[5] Append-only: settled_signals and audit_logs refuse mutation")
    settled = SettledSignal(
        signal_id="sig-verify-1",
        symbol="BTCUSDT",
        horizon="15m",
        direction=SignalDirection.UP,
        reference_price=Decimal("50000"),
        settlement_price=Decimal("50120"),
        outcome=SettlementOutcome.WIN,
        expiry_utc=START + timedelta(minutes=15),
        settlement_provider="fixture",
        notes=["verified by scripts/verify_persistence.py"],
    )
    repo.settle_signal(settled)  # type: ignore[attr-defined]
    stored = repo.settled_signals(symbol="BTCUSDT")  # type: ignore[attr-defined]
    check("settlement round-trips", len(stored) == 1 and stored[0].signal_id == "sig-verify-1")
    check("outcome round-trips", stored[0].outcome is SettlementOutcome.WIN)
    check("direction round-trips", stored[0].direction is SignalDirection.UP)
    check("notes round-trip", stored[0].notes == ["verified by scripts/verify_persistence.py"])

    try:
        repo.settle_signal(settled)
        check("re-settling the same ID is refused", False, "no error raised")
    except PersistenceError as exc:
        check("re-settling the same ID is refused", True, type(exc).__name__)

    # A settlement missing its scoring fields cannot be scored, and inventing a
    # price to fill the gap would put a number in the performance history that no
    # provider ever quoted (Rule 2).
    try:
        repo.settle_signal(  # type: ignore[attr-defined]
            SettledSignal(
                signal_id="sig-verify-incomplete",
                symbol="BTCUSDT",
                horizon="15m",
                outcome=SettlementOutcome.VOID,
            )
        )
        check("incomplete settlement is refused", False, "no error raised")
    except PersistenceError as exc:
        check("incomplete settlement is refused", True, "names the missing fields")
        check("error names direction", "direction" in str(exc), str(exc)[:60])

    repo.log_event(  # type: ignore[attr-defined]
        "verification.started",
        actor="verify_persistence",
        symbol="BTCUSDT",
        message="append-only check",
        details={"section": 5},
    )
    events = repo.recent_events()  # type: ignore[attr-defined]
    check("audit event appended", any(e["event_type"] == "verification.started" for e in events))
    filtered = repo.recent_events(event_type="verification.started")  # type: ignore[attr-defined]
    check("audit filter works", len(filtered) == 1)
    check("audit filter excludes others", repo.recent_events(event_type="nope") == [])  # type: ignore[attr-defined]


def section_orm_guards(factory: object) -> None:
    """The ORM-level half of the append-only guarantee. SQL backend only.

    Section 5 proves the repository refuses a duplicate settlement. This proves
    that an UPDATE or DELETE issued through the ORM -- the way a future feature
    would most plausibly do it by accident -- is refused too.
    """
    import quantedge.repositories.models as m
    from quantedge.repositories.database import session_scope

    print("\n[6] ORM guards: UPDATE and DELETE on immutable tables")
    for label, table in (("settled_signals", m.SettledSignalRow), ("audit_logs", m.AuditLog)):
        for verb in ("update", "delete"):
            try:
                with session_scope(factory) as session:  # type: ignore[arg-type]
                    row = session.query(table).first()
                    if row is None:
                        check(f"{label} has a row to test {verb}", False, "table empty")
                        continue
                    if verb == "update":
                        row.symbol = "TAMPERED"
                    else:
                        session.delete(row)
                check(f"{label} refuses {verb.upper()}", False, "no error raised")
            except PersistenceError:
                check(f"{label} refuses {verb.upper()}", True)

    # And the data is genuinely unchanged afterwards -- a refused write that
    # still mutated the row would be worse than no guard at all.
    with session_scope(factory) as session:  # type: ignore[arg-type]
        symbols = [r.symbol for r in session.query(m.SettledSignalRow).all()]
    check("no row was tampered with", "TAMPERED" not in symbols, str(symbols))


def section_quota(repo: object) -> None:
    """The bug this table exists to fix.

    Alpha Vantage allows 25 requests a day. A limiter holding that count in
    memory resets to zero on every CLI invocation, so twenty invocations spend
    the day's budget while each one believes it has spent nothing. The counter
    has to survive the process, which means it has to be in the database.
    """
    print("\n[7] Provider quota -- the counter outlives the caller")
    first = repo.consume_quota("alphavantage", limit=25)  # type: ignore[attr-defined]
    check("first request allowed", first["allowed"] is True)
    check("counter reads 1", first["requests_made"] == 1, str(first["requests_made"]))

    second = repo.consume_quota("alphavantage", limit=25)  # type: ignore[attr-defined]
    check("counter incremented, not reset", second["requests_made"] == 2)

    state = repo.quota_state("alphavantage")  # type: ignore[attr-defined]
    check("state reports 23 remaining", state is not None and state["remaining"] == 23)
    check("state reports the cap", state is not None and state["requests_allowed"] == 25)

    # Exhaust a small budget and confirm the refusal, since an off-by-one here
    # spends a real vendor's daily allowance.
    for _ in range(3):
        repo.consume_quota("tinyprovider", limit=3)  # type: ignore[attr-defined]
    exhausted = repo.consume_quota("tinyprovider", limit=3)  # type: ignore[attr-defined]
    check("request 4 of 3 is refused", exhausted["allowed"] is False)
    check("refusal does not inflate the count", exhausted["requests_made"] == 3)

    unlimited = repo.consume_quota("nolimit")  # type: ignore[attr-defined]
    check("absent limit means unmetered", unlimited["allowed"] is True)
    check("unmetered reports no cap", unlimited["requests_allowed"] is None)
    check("unknown provider has no state", repo.quota_state("neverseen") is None)  # type: ignore[attr-defined]

    # A different day is a different window, so the counter starts over.
    tomorrow = repo.consume_quota(  # type: ignore[attr-defined]
        "alphavantage", limit=25, now=datetime.now(UTC) + timedelta(days=1)
    )
    check("a new day opens a new window", tomorrow["requests_made"] == 1)


def section_rule_9(repo: object) -> None:
    print("\n[8] Rule 9 -- a forming candle is refused, not filtered")
    forming = bar(99, 51000.0, is_closed=False)
    before = repo.candle_count("BTCUSDT", Timeframe.M5)  # type: ignore[attr-defined]
    try:
        repo.save_candles([forming])  # type: ignore[attr-defined]
        check("forming candle alone is refused", False, "no error raised")
    except PersistenceError as exc:
        check("forming candle alone is refused", True, type(exc).__name__)
        check("error mentions Rule 9", "Rule 9" in str(exc), str(exc)[:70])

    # The mixed batch is the case that matters: filtering would report success
    # while dropping the newest bar, which is the one the next calculation needs.
    try:
        repo.save_candles([bar(100, 51100.0), forming])  # type: ignore[attr-defined]
        check("mixed batch is refused whole", False, "no error raised")
    except PersistenceError:
        check("mixed batch is refused whole", True)

    after = repo.candle_count("BTCUSDT", Timeframe.M5)  # type: ignore[attr-defined]
    check("no partial write landed", after == before, f"{before} -> {after}")


def section_analysis_outputs(repo: object) -> None:
    print("\n[9] Analysis outputs and retention")
    repo.save_quality_report(  # type: ignore[attr-defined]
        DataQualityReport(
            status=QualityStatus.DEGRADED,
            quality_score=0.62,
            freshness_ms=4200,
            provider="fixture",
            symbol="BTCUSDT",
            timeframe=Timeframe.M5,
            candles_checked=200,
            closed_candles=199,
            warnings=["one gap detected"],
            checks_run=["gaps", "freshness", "monotonic_time"],
        )
    )
    check("quality report accepted", True)

    blocking = DataQualityReport(
        status=QualityStatus.FAIL,
        quality_score=0.1,
        freshness_ms=900_000,
        provider="fixture",
        blocking_reasons=["stale beyond tolerance"],
    )
    repo.save_quality_report(blocking)  # type: ignore[attr-defined]
    check("is_blocking derives from FAIL", blocking.is_blocking is True)
    check("a report with no timeframe is storable", True)

    removed = repo.prune_candles_before(START + timedelta(minutes=25))  # type: ignore[attr-defined]
    check("retention removed the 5 oldest", removed == 5, f"got {removed}")
    remaining = repo.load_candles("BTCUSDT", Timeframe.M5, limit=500)  # type: ignore[attr-defined]
    check("15 bars remain", len(remaining) == 15, f"got {len(remaining)}")
    check(
        "oldest survivor is at the cutoff",
        remaining[0].open_time_utc >= START + timedelta(minutes=25),
    )


def section_memory_parity() -> None:
    """The same assertions against the in-memory backend.

    The two backends are only interchangeable if they agree. Running the shared
    sections against both is what turns "the signatures match" into "the
    behaviour matches".
    """
    from quantedge.repositories.memory import MemoryRepository

    print("\n[10] In-memory backend parity")
    repo = MemoryRepository()
    check("memory backend reports no durability", repo.persistence_available is False)

    section_candles(repo)
    section_quotes_symbols(repo)
    section_append_only(repo)
    section_quota(repo)
    section_rule_9(repo)
    section_analysis_outputs(repo)


def section_factory() -> None:
    print("\n[11] Backend factory and honest durability reporting")
    from quantedge.repositories import describe_persistence, get_repository, reset_repository

    reset_repository()
    repo = get_repository(force_memory=True)
    info = describe_persistence()
    check("forced memory backend selected", type(repo).__name__ == "MemoryRepository")
    check("durable is False for memory", info["durable"] is False)
    check("persistence_available agrees", info["persistence_available"] is False)
    check("the warning is present", "warning" in info, info.get("warning", ""))
    check("no DSN in the memory report", "dsn" not in info)

    reset_repository()
    sql_repo = get_repository()
    sql_info = describe_persistence()
    check("default backend is SQL", type(sql_repo).__name__ == "SqlRepository")
    check("durable is True for SQL", sql_info["durable"] is True)
    check("report carries a redacted DSN", "***" in sql_info["dsn"] or "@" not in sql_info["dsn"])
    reset_repository()


def section_production_refuses_fallback() -> None:
    """Production must fail loudly rather than degrade into forgetting.

    This is the one branch where the fallback is wrong. A production deployment
    silently running on in-memory storage would accumulate a settled-signal
    history that disappears at the next restart -- and a performance record that
    resets is worse than no performance record, because it looks like one.
    """
    from quantedge.config import get_settings
    from quantedge.repositories import get_repository, reset_repository
    from quantedge.repositories.database import reset_engine

    print("\n[12] Production refuses the in-memory fallback")
    saved_env = os.environ.get("APP_ENV")
    saved_url = os.environ.get("DATABASE_URL")
    try:
        # A DSN pointing at a driver that is not installed: unreachable in a way
        # no retry fixes, which is exactly the case that must not degrade.
        os.environ["APP_ENV"] = "production"
        os.environ["DATABASE_URL"] = "postgresql+psycopg://u:p@127.0.0.1:1/nope"
        get_settings.cache_clear()
        reset_engine()
        reset_repository()

        try:
            get_repository()
            check("production refuses to fall back", False, "returned a repository")
        except PersistenceError as exc:
            check("production refuses to fall back", True, type(exc).__name__)
            check("the refusal explains why", "durable" in str(exc), str(exc)[:70])
            check("the refusal leaks no DSN", "u:p@" not in str(exc))

        # The same failure in development degrades instead, loudly.
        os.environ["APP_ENV"] = "development"
        get_settings.cache_clear()
        reset_engine()
        reset_repository()
        repo = get_repository()
        check("development falls back instead", type(repo).__name__ == "MemoryRepository")
        check("and reports itself as not durable", repo.persistence_available is False)
    finally:
        reset_repository()
        reset_engine()
        for key, value in (("APP_ENV", saved_env), ("DATABASE_URL", saved_url)):
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        get_settings.cache_clear()


def main() -> int:
    print("=" * 70)
    print("PERSISTENCE VERIFICATION -- real SQLite file + in-memory parity")
    print("=" * 70)

    tmp = Path(tempfile.mkdtemp(prefix="quantedge-verify-"))
    db_path = tmp / "verify.db"
    os.environ["DATABASE_URL"] = f"sqlite+pysqlite:///{db_path.as_posix()}"

    from quantedge.config import get_settings
    from quantedge.repositories.database import get_session_factory, make_engine, reset_engine

    get_settings.cache_clear()
    reset_engine()

    engine = make_engine()
    try:
        section_schema(engine)
        section_dsn_redaction()

        from quantedge.repositories.sql import SqlRepository

        factory = get_session_factory()
        repo = SqlRepository(factory)
        section_candles(repo)
        section_quotes_symbols(repo)
        section_append_only(repo)
        section_orm_guards(factory)
        section_quota(repo)
        section_rule_9(repo)
        section_analysis_outputs(repo)

        check("the database file exists on disk", db_path.exists(), str(db_path.name))

        section_memory_parity()
        section_factory()
        section_production_refuses_fallback()
    finally:
        reset_engine()
        engine.dispose()
        shutil.rmtree(tmp, ignore_errors=True)
        os.environ.pop("DATABASE_URL", None)
        get_settings.cache_clear()

    print("\n" + "=" * 70)
    if FAILURES:
        print(f"RESULT: {len(FAILURES)} CHECK(S) FAILED")
        for name in FAILURES:
            print(f"  - {name}")
        return 1
    print("RESULT: ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
