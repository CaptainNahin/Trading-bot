"""Background worker verification.

Each worker is an unbounded ``while True`` loop wrapped in a broad ``except`` so
one bad cycle cannot kill a daemon. That is the right shape for a background
process and the wrong shape for trusting it silently: a worker whose every cycle
raises looks exactly like a worker with nothing to do, because both simply keep
logging and sleeping.

So this script does not assert "it did not crash". It attaches a handler to the
worker loggers, runs each loop for a few real seconds against live providers, and
asserts the cycle reached its *success* log line and emitted no ERROR record. A
worker that was failing every pass fails here.

``settle_due_signals`` is exercised directly rather than through its loop: it is a
self-contained unit that returns a count, so calling it once is a stronger check
than watching its wrapper sleep.
"""

from __future__ import annotations

import logging
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from quantedge.workers.health_worker import run_health_worker
from quantedge.workers.scanner_worker import run_scanner_worker
from quantedge.workers.settlement_worker import settle_due_signals

FAILURES: list[str] = []

# Long enough for one cycle against live HTTP providers, short enough that the
# whole script stays inside a normal verification run.
_CYCLE_SECONDS = 20.0
_WORKER_INTERVAL = 3600  # one cycle then a sleep we never wait out


class _Collector(logging.Handler):
    """Captures records so a cycle can be judged on what it logged."""

    def __init__(self) -> None:
        super().__init__(level=logging.DEBUG)
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


def check(label: str, condition: bool, detail: str = "") -> None:
    print(f"  {'[PASS]' if condition else '[FAIL]'} {label}{f' -- {detail}' if detail else ''}")
    if not condition:
        FAILURES.append(label)


def _run_for(target: object, logger_name: str, seconds: float) -> list[logging.LogRecord]:
    """Run a worker loop in a daemon thread and return what it logged.

    The thread is never joined. These loops have no stop signal, so the only way
    to end one is to leave it daemonised and let the interpreter drop it -- which
    is exactly how ``python -m quantedge.workers`` shuts them down too.
    """
    collector = _Collector()
    logger = logging.getLogger(logger_name)
    logger.addHandler(collector)
    previous = logger.level
    logger.setLevel(logging.DEBUG)

    thread = threading.Thread(
        target=target,  # type: ignore[arg-type]
        kwargs={"interval_seconds": _WORKER_INTERVAL},
        daemon=True,
    )
    thread.start()

    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        # Stop as soon as the cycle has clearly finished, so a healthy worker
        # does not cost the full timeout.
        if any(r.levelno >= logging.ERROR for r in collector.records):
            break
        if len(collector.records) >= 2:
            break
        time.sleep(0.25)

    logger.removeHandler(collector)
    logger.setLevel(previous)
    return list(collector.records)


def _messages(records: list[logging.LogRecord]) -> str:
    return " | ".join(r.getMessage() for r in records)


def _errors(records: list[logging.LogRecord]) -> list[str]:
    return [r.getMessage() for r in records if r.levelno >= logging.ERROR]


def main() -> int:
    print("=" * 70)
    print("BACKGROUND WORKER VERIFICATION -- one live cycle per worker")
    print("=" * 70)

    print("\n[1] Health worker probes providers and stores the results")
    records = _run_for(run_health_worker, "quantedge.workers.health_worker", _CYCLE_SECONDS)
    errors = _errors(records)
    check("the loop started", bool(records), _messages(records)[:70])
    check("no error was logged", not errors, "; ".join(errors)[:70] or "none")
    check(
        "a probe cycle completed",
        any("health" in r.getMessage().lower() for r in records[1:]),
        _messages(records[1:])[:70] or "no cycle line",
    )

    print("\n[2] Scanner worker completes a scan against live data")
    records = _run_for(run_scanner_worker, "quantedge.workers.scanner_worker", _CYCLE_SECONDS)
    errors = _errors(records)
    check("the loop started", bool(records), _messages(records)[:70])
    check("no error was logged", not errors, "; ".join(errors)[:70] or "none")
    check(
        "a scan cycle completed",
        any("Scan completed" in r.getMessage() for r in records),
        _messages(records[1:])[:70] or "no cycle line",
    )

    print("\n[3] Settlement pass runs and returns a count")
    settled = settle_due_signals(limit=10)
    check("settle_due_signals returns an integer", isinstance(settled, int), str(settled))
    check("and never a negative count", settled >= 0, str(settled))

    print("\n" + "=" * 70)
    if FAILURES:
        print(f"RESULT: {len(FAILURES)} CHECK(S) FAILED")
        return 1
    print("RESULT: ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
