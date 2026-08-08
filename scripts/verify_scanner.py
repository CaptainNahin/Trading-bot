"""Scanner pipeline verification script."""

from __future__ import annotations

import sys
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from quantedge.contracts import AssetClass, Candle, Timeframe
from quantedge.services.scanner import run_scan

FAILURES: list[str] = []


def check(label: str, condition: bool, detail: str = "") -> None:
    print(f"  {'[PASS]' if condition else '[FAIL]'} {label}{f' -- {detail}' if detail else ''}")
    if not condition:
        FAILURES.append(label)


START = datetime(2026, 1, 1, tzinfo=UTC)


def mock_candles(symbol: str, tf: Timeframe, limit: int) -> list[Candle]:
    base = 100.0 if symbol == "BTCUSDT" else 10.0
    drift = 0.1 if symbol == "BTCUSDT" else -0.1
    res: list[Candle] = []
    for i in range(limit):
        price = Decimal(str(round(base + drift * i, 4)))
        res.append(
            Candle(
                provider="mock",
                symbol=symbol,
                asset_class=AssetClass.CRYPTO,
                timeframe=tf,
                open_time_utc=START + timedelta(minutes=i),
                close_time_utc=START + timedelta(minutes=i + 1),
                open=price,
                high=price + Decimal("0.5"),
                low=price - Decimal("0.5"),
                close=price,
                volume=Decimal("1000"),
                is_closed=True,
            )
        )
    return res


def main() -> int:
    print("=" * 70)
    print("SCANNER VERIFICATION -- 12-step pipeline checks")
    print("=" * 70)

    symbols = ["BTCUSDT", "ETHUSDT"]
    scan_res = run_scan(symbols, horizon="swing", candle_fetcher=mock_candles)

    check("Scanned symbol count matches", scan_res.scanned == 2, f"scanned={scan_res.scanned}")
    check(
        "Candidates or rejections produced",
        len(scan_res.candidates) + len(scan_res.rejections) == 2,
    )

    if scan_res.candidates:
        cand = scan_res.candidates[0]
        check("Candidate carries valid heuristic_score", 0.0 <= cand.heuristic_score <= 1.0)
        check("Candidate carries valid provider", cand.provider == "binance")

    if scan_res.rejections:
        rej = scan_res.rejections[0]
        check("Rejection carries reason code", bool(rej.reason_code))

    print("\n" + "=" * 70)
    if FAILURES:
        print(f"RESULT: {len(FAILURES)} CHECK(S) FAILED")
        return 1
    print("RESULT: ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
