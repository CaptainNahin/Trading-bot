"""MCP Server tools discovery & execution verification script."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from quantedge.mcp_server import server

FAILURES: list[str] = []

# Capabilities the brief prohibits this surface from carrying. A model driving
# the server must not be able to reach a shell, the filesystem, raw SQL or a
# broker through it, so the names are asserted absent rather than merely never
# written -- a tool added later would otherwise slip in unremarked.
_FORBIDDEN_SUBSTRINGS = (
    "shell",
    "exec",
    "command",
    "subprocess",
    "read_file",
    "write_file",
    "filesystem",
    "sql",
    "query_db",
    "order",
    "buy",
    "sell",
    "withdraw",
    "wallet",
    "account",
    "login",
    "scrape",
)


def check(label: str, condition: bool, detail: str = "") -> None:
    print(f"  {'[PASS]' if condition else '[FAIL]'} {label}{f' -- {detail}' if detail else ''}")
    if not condition:
        FAILURES.append(label)


def main() -> int:
    print("=" * 70)
    print("MCP SERVER VERIFICATION -- tool discovery and direct invocation checks")
    print("=" * 70)

    print("\n[1] Registered surface")
    tools = asyncio.run(server.mcp_server.list_tools())
    names = sorted(t.name for t in tools)
    check("tools are registered", len(names) > 0, f"{len(names)} found")
    check(
        "the count matches the number documented in the module docstring",
        f"Exposes {len(names)} deterministic" in (server.__doc__ or ""),
        f"{len(names)} registered",
    )
    check("every tool name is unique", len(set(names)) == len(names))
    undocumented = [t.name for t in tools if not (t.description or "").strip()]
    check("every tool carries a description", not undocumented, ", ".join(undocumented) or "all")

    print("\n[2] Prohibited capabilities are absent")
    # `get_order_book` is public depth data, not order placement; it is the one
    # legitimate name containing a forbidden substring, so it is matched exactly.
    allowed = {"get_order_book"}
    breaches = [
        n
        for n in names
        if n not in allowed and any(bad in n.lower() for bad in _FORBIDDEN_SUBSTRINGS)
    ]
    check("no shell, filesystem, SQL or broker-execution tool is exposed", not breaches,
          ", ".join(breaches) or "none")

    print("\n[3] Direct invocation")
    q_str = server.get_live_quote("BTCUSDT")
    check("get_live_quote returns JSON payload", q_str.startswith("{"))

    h_str = server.get_provider_health()
    check("get_provider_health returns JSON array", h_str.startswith("["))

    s_str = server.get_system_status()
    check("get_system_status returns OK status", "QuantEdge AI" in s_str)

    print("\n" + "=" * 70)
    if FAILURES:
        print(f"RESULT: {len(FAILURES)} CHECK(S) FAILED")
        return 1
    print("RESULT: ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
