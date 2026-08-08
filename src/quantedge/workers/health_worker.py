"""Background health monitoring worker.

Probes every configured provider on an interval and records each result.

The counts logged are of rows actually written, not of providers probed. An
earlier revision swallowed persistence errors with a bare ``pass`` and then
logged "recorded health checks for N providers" regardless, so a worker whose
every write was failing looked identical in the log to one working perfectly.
"""

from __future__ import annotations

import time

from quantedge.errors import QuantEdgeError
from quantedge.logging import get_logger
from quantedge.providers.registry import get_registry
from quantedge.repositories import get_repository

logger = get_logger(__name__)


def run_health_worker(interval_seconds: int = 60) -> None:
    """Periodically check provider health and record each result."""
    logger.info("starting health worker", extra={"interval_seconds": interval_seconds})
    registry = get_registry()
    repo = get_repository()

    while True:
        try:
            health_list = registry.health_check_all()
        except QuantEdgeError as exc:
            logger.error("health probe failed", extra={"code": exc.code, "error": exc.message})
            time.sleep(interval_seconds)
            continue

        stored = 0
        failed = 0
        for health in health_list:
            try:
                repo.record_health(health)
                stored += 1
            except Exception as exc:  # noqa: BLE001 - one bad row must not stop the loop
                failed += 1
                logger.warning(
                    "could not record provider health",
                    extra={"provider": health.provider, "error": type(exc).__name__},
                )

        logger.info(
            "health probe complete",
            extra={"probed": len(health_list), "stored": stored, "failed": failed},
        )
        time.sleep(interval_seconds)


if __name__ == "__main__":
    run_health_worker()
