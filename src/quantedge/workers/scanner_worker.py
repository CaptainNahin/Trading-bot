"""Background scanner worker."""

from __future__ import annotations

import time

from quantedge.logging import get_logger
from quantedge.providers.registry import get_registry
from quantedge.services import scanner

logger = get_logger(__name__)


def run_scanner_worker(interval_seconds: int = 60) -> None:
    """Periodically execute scanning pipeline."""
    logger.info("Starting scanner worker with interval=%ds", interval_seconds)
    registry = get_registry()
    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]

    while True:
        try:
            res = scanner.run_scan(symbols, horizon="swing", registry=registry)
            logger.info(
                "Scan completed: %d candidates found out of %d scanned",
                len(res.candidates),
                res.scanned,
            )
        except Exception as exc:  # noqa: BLE001 - a daemon must outlive one bad cycle
            logger.error("Scanner worker loop error: %s", exc)
        time.sleep(interval_seconds)


if __name__ == "__main__":
    run_scanner_worker()
