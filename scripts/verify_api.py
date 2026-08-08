"""FastAPI routes & OpenAPI schema verification script."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fastapi.testclient import TestClient

from quantedge.api import app

FAILURES: list[str] = []


def check(label: str, condition: bool, detail: str = "") -> None:
    print(f"  {'[PASS]' if condition else '[FAIL]'} {label}{f' -- {detail}' if detail else ''}")
    if not condition:
        FAILURES.append(label)


def main() -> int:
    print("=" * 70)
    print("FASTAPI VERIFICATION -- routes & OpenAPI schema checks")
    print("=" * 70)

    client = TestClient(app)

    # OpenAPI schema test
    schema_res = client.get("/openapi.json")
    check("OpenAPI schema returns 200 OK", schema_res.status_code == 200)

    # Health endpoint test
    health_res = client.get("/api/v1/health")
    check("GET /api/v1/health returns 200 OK", health_res.status_code == 200)

    # Performance summary endpoint test
    perf_res = client.get("/api/v1/performance")
    check("GET /api/v1/performance returns 200 OK", perf_res.status_code == 200)

    print("\n" + "=" * 70)
    if FAILURES:
        print(f"RESULT: {len(FAILURES)} CHECK(S) FAILED")
        return 1
    print("RESULT: ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
