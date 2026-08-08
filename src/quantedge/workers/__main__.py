"""CLI entry point for QuantEdge background workers.

Runs all workers (scanner, settlement, health) concurrently in a
single process using their respective thread loops.

Usage:
    python -m quantedge.workers
    quantedge-worker           # via pyproject.toml [project.scripts]
"""

from __future__ import annotations

import os
import signal
import sys
import threading
import time

from quantedge.logging import get_logger

log = get_logger(__name__)


def main() -> None:
    """Start all background workers and block until interrupted."""
    from quantedge.workers.health_worker import run_health_worker
    from quantedge.workers.scanner_worker import run_scanner_worker
    from quantedge.workers.settlement_worker import run_settlement_worker

    shutdown = threading.Event()

    def _on_signal(signum: int, _frame: object) -> None:
        log.info("received signal %s, shutting down workers", signum)
        shutdown.set()

    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)

    workers = [
        threading.Thread(target=run_health_worker, name="health-worker", daemon=True),
        threading.Thread(target=run_scanner_worker, name="scanner-worker", daemon=True),
        threading.Thread(target=run_settlement_worker, name="settlement-worker", daemon=True),
    ]

    log.info("starting %d background workers", len(workers))
    for w in workers:
        w.start()

    try:
        while not shutdown.is_set():
            time.sleep(1)
    except KeyboardInterrupt:
        pass

    log.info("workers stopped")


if __name__ == "__main__":
    src = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
    src = os.path.abspath(src)
    if src not in sys.path:
        sys.path.insert(0, src)

    main()
