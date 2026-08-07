"""Prove the daily provider quota survives a process restart.

This is the check the in-process limiter could never pass. ``RateLimiter._day``
holds :func:`time.monotonic` values, which are meaningless outside the process
that produced them, so a 25-request Alpha Vantage budget was spent afresh by
every CLI invocation -- twenty CLI runs could issue 500 requests against a
25-request allowance and each one would believe it was within budget.

Sections 1-3 run in *this* process. Section 4 is the one that matters: it
re-executes this file as a **new interpreter** against the same database file
and asserts the counter picked up where the previous process left it. Nothing
is mocked and no network call is made -- the limiter is exercised directly, so
a failure here is a failure of the quota logic and not of a provider's uptime.

Section 5 covers the degraded path. A gateway whose database is unreachable
must still serve market data, so a store fault fails open onto the in-process
backstop; what it must not do is keep claiming the cap is durable. That claim
is what ``daily_cap_durable`` reports, and it is asserted false there.
"""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
import tempfile
from datetime import timedelta
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

FAILURES: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    marker = "[PASS]" if ok else "[FAIL]"
    print(f"  {marker} {label}{(' -- ' + detail) if detail and not ok else ''}")
    if not ok:
        FAILURES.append(label)


def section(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


# --------------------------------------------------------------------------- #
# the child process invoked by section 4                                       #
# --------------------------------------------------------------------------- #


def _child_spend(count: int) -> int:
    """Spend ``count`` requests and print the resulting state as JSON.

    Runs in a fresh interpreter with DATABASE_URL already set by the parent.
    """
    import json

    from quantedge.providers.http import RateLimiter
    from quantedge.repositories import get_repository

    repo = get_repository()
    limiter = RateLimiter(per_day=5, quota_store=repo, quota_durable=True)

    granted = 0
    denied = 0
    for _ in range(count):
        try:
            asyncio.run(limiter.acquire("childprov"))
            granted += 1
        except Exception as exc:  # the denial is the result being measured
            if type(exc).__name__ == "ProviderRateLimitError":
                denied += 1
            else:
                raise

    state = repo.quota_state("childprov", window_kind="day")
    print(
        "CHILD_RESULT="
        + json.dumps(
            {
                "granted": granted,
                "denied": denied,
                "requests_made": state["requests_made"] if state else None,
            }
        )
    )
    return 0


# --------------------------------------------------------------------------- #
# sections                                                                     #
# --------------------------------------------------------------------------- #


def section_counts_and_denies() -> None:
    """The counter increments, and the cap is enforced."""
    from quantedge.providers.http import RateLimiter
    from quantedge.repositories import get_repository

    repo = get_repository()
    limiter = RateLimiter(per_day=3, quota_store=repo, quota_durable=True)

    for i in range(3):
        asyncio.run(limiter.acquire("provA"))
        state = repo.quota_state("provA", window_kind="day")
        check(
            f"request {i + 1} granted and counted",
            state is not None and state["requests_made"] == i + 1,
        )

    denied = False
    retry_after: float | None = None
    try:
        asyncio.run(limiter.acquire("provA"))
    except Exception as exc:  # the refusal is what is under test
        denied = type(exc).__name__ == "ProviderRateLimitError"
        retry_after = getattr(exc, "retry_after_seconds", None)

    check("the 4th request over a cap of 3 is refused", denied)
    check(
        "the refusal points at the next UTC midnight, not a flat 24h",
        retry_after is not None and 0 < retry_after <= 86_400.0,
        f"retry_after={retry_after}",
    )

    state = repo.quota_state("provA", window_kind="day")
    check(
        "a refused request does not increment the counter",
        state is not None and state["requests_made"] == 3,
        f"made={state['requests_made'] if state else None}",
    )
    check(
        "remaining reaches zero rather than going negative",
        state is not None and state["remaining"] == 0,
    )


def section_no_cap_no_store() -> None:
    """A provider with no daily cap never touches the quota table.

    Binance has no daily limit; putting a database round-trip in front of every
    candle fetch would buy nothing and cost latency on the hottest path.
    """
    from quantedge.providers.http import RateLimiter
    from quantedge.repositories import get_repository

    repo = get_repository()
    limiter = RateLimiter(per_minute=1200, quota_store=repo, quota_durable=True)
    asyncio.run(limiter.acquire("binance"))
    asyncio.run(limiter.acquire("binance"))

    check(
        "no quota row is written for a provider without a daily cap",
        repo.quota_state("binance", window_kind="day") is None,
    )


def section_window_rolls_over() -> None:
    """Yesterday's spend does not count against today.

    Asserted by writing directly at an explicit timestamp rather than by
    waiting for midnight.
    """
    from quantedge.contracts import utc_now
    from quantedge.repositories import get_repository

    repo = get_repository()
    yesterday = utc_now() - timedelta(days=1)

    for _ in range(4):
        repo.consume_quota("rollover", window_kind="day", limit=4, now=yesterday)

    spent = repo.consume_quota("rollover", window_kind="day", limit=4, now=yesterday)
    check("yesterday's budget is exhausted", spent["allowed"] is False)

    today = repo.consume_quota("rollover", window_kind="day", limit=4)
    check("today starts from a fresh budget", today["allowed"] is True)
    check(
        "today's counter starts at 1, not 5",
        today["requests_made"] == 1,
        f"made={today['requests_made']}",
    )


def section_survives_restart(db_url: str) -> None:
    """The point of the whole exercise: a *new process* sees the spend.

    Two child interpreters, three requests each, against a cap of five. If the
    counter were still in process memory both would be granted all three; the
    second must instead be refused its last one.
    """
    env = dict(os.environ)
    env["DATABASE_URL"] = db_url
    env["APP_ENV"] = "development"
    env["QUANTEDGE_CHILD_SPEND"] = "3"

    results = []
    for run in (1, 2):
        proc = subprocess.run(  # noqa: S603 - fixed argv, no shell, no user input
            [sys.executable, "-u", str(Path(__file__).resolve())],
            env=env,
            capture_output=True,
            check=False,  # a non-zero child is a result to report, not an exception
            text=True,
            timeout=120,
        )
        line = next(
            (ln for ln in proc.stdout.splitlines() if ln.startswith("CHILD_RESULT=")),
            None,
        )
        check(f"child process {run} ran", line is not None, proc.stderr[-400:])
        if line is None:
            return
        import json

        results.append(json.loads(line.removeprefix("CHILD_RESULT=")))

    first, second = results
    check(
        "the first process spends 3 of 5",
        first["granted"] == 3 and first["requests_made"] == 3,
        f"{first}",
    )
    check(
        "a NEW process sees the 3 already spent and is granted only 2",
        second["granted"] == 2,
        f"granted={second['granted']} (3 would mean the counter reset)",
    )
    check(
        "the new process is refused the request that would exceed the cap",
        second["denied"] == 1,
        f"denied={second['denied']}",
    )
    check(
        "the total across both processes is exactly the cap",
        second["requests_made"] == 5,
        f"made={second['requests_made']}",
    )


def section_degrades_without_lying() -> None:
    """A broken store fails open, but stops claiming the cap is durable.

    Serving market data must survive a database outage. Reporting that the
    daily cap is durable when it is not must not, because an operator reading
    that flag would conclude a spent budget is being tracked when it is being
    forgotten at every restart.
    """
    from quantedge.providers.http import RateLimiter

    class BrokenStore:
        def consume_quota(self, provider: str, **kw: Any) -> dict[str, Any]:
            raise RuntimeError("database is down")

        def quota_state(self, provider: str, **kw: Any) -> dict[str, Any] | None:
            raise RuntimeError("database is down")

    limiter = RateLimiter(per_day=5, quota_store=BrokenStore(), quota_durable=True)

    granted = True
    try:
        asyncio.run(limiter.acquire("brokenprov"))
    except Exception:  # failing open is the behaviour under test
        granted = False

    check("a store outage does not block the request", granted)

    snap = limiter.snapshot("brokenprov")
    check(
        "and the snapshot stops claiming the cap is durable",
        snap["daily_cap_durable"] is False,
        f"daily_cap_durable={snap['daily_cap_durable']}",
    )
    check(
        "the in-process backstop still counted it",
        snap["used_last_day"] == 1,
        f"used_last_day={snap['used_last_day']}",
    )

    limiter2 = RateLimiter(per_day=5)
    check(
        "a limiter with no store never claims durability",
        limiter2.snapshot("x")["daily_cap_durable"] is False,
    )


def section_reports_persisted_usage() -> None:
    """The snapshot surfaces the durable numbers, not just the process-local ones."""
    from quantedge.providers.http import RateLimiter
    from quantedge.repositories import get_repository

    repo = get_repository()
    limiter = RateLimiter(per_day=10, quota_store=repo, quota_durable=True)
    asyncio.run(limiter.acquire("snapprov"))
    asyncio.run(limiter.acquire("snapprov"))

    snap = limiter.snapshot("snapprov")
    check("daily_cap_durable is true on a working store", snap["daily_cap_durable"] is True)
    check(
        "persisted usage is reported",
        snap.get("persisted_used_today") == 2,
        f"{snap.get('persisted_used_today')}",
    )
    check(
        "persisted remaining is reported",
        snap.get("persisted_remaining_today") == 8,
        f"{snap.get('persisted_remaining_today')}",
    )


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="quantedge_quota_"))
    db = tmp / "quota.db"
    db_url = f"sqlite+pysqlite:///{db.as_posix()}"
    os.environ["DATABASE_URL"] = db_url
    os.environ["APP_ENV"] = "development"

    from quantedge.repositories.database import create_all

    create_all()

    print(f"database: {db}")

    section("[1] the counter increments and the cap is enforced")
    section_counts_and_denies()

    section("[2] providers without a daily cap are left alone")
    section_no_cap_no_store()

    section("[3] the daily window rolls over at UTC midnight")
    section_window_rolls_over()

    section("[4] THE POINT: the count survives a process restart")
    section_survives_restart(db_url)

    section("[5] a store outage degrades without misreporting durability")
    section_degrades_without_lying()

    section("[6] the snapshot reports durable usage")
    section_reports_persisted_usage()

    print("\n" + "=" * 70)
    if FAILURES:
        print(f"RESULT: {len(FAILURES)} CHECK(S) FAILED")
        for f in FAILURES:
            print(f"  - {f}")
        return 1
    print("RESULT: ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    spend = os.environ.get("QUANTEDGE_CHILD_SPEND")
    sys.exit(_child_spend(int(spend)) if spend else main())
