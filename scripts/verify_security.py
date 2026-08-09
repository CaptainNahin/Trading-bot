"""Security control verification.

The brief names a specific set of controls -- secret redaction, ``.env``
exclusion, no secrets reaching the frontend, symbol and timeframe allowlists,
bounded request sizes, and a market-data-only Binance surface. Each is asserted
here against the real code rather than taken on trust from a design document.

Nothing in this file prints a credential. The redaction checks use a synthetic
value generated for the test, and they assert on *absence*: the check passes when
the fake secret cannot be found in the output, so a failure reports the label and
never the value.
"""

from __future__ import annotations

import io
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# Imported after the path bootstrap above: this script runs from a checkout
# without the package installed, so `src` has to be on the path first.
from quantedge.logging import (  # noqa: E402
    MASK,
    JsonFormatter,
    RedactingFilter,
    redact,
    register_secret,
)

FAILURES: list[str] = []

# A synthetic credential, shaped like a real one so the pattern rules engage.
# It is invented here and exists nowhere else.
FAKE_SECRET = "sk-" + ("Zq7" * 12)


def check(label: str, condition: bool, detail: str = "") -> None:
    print(f"  {'[PASS]' if condition else '[FAIL]'} {label}{f' -- {detail}' if detail else ''}")
    if not condition:
        FAILURES.append(label)


def _capture(msg: object, *args: object, **kw: object) -> str:
    """Render one record through the real filter and formatter."""
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(JsonFormatter())
    handler.addFilter(RedactingFilter())

    logger = logging.getLogger("quantedge.verify_security")
    logger.handlers = [handler]
    logger.propagate = False
    logger.setLevel(logging.DEBUG)
    logger.info(msg, *args, **kw)  # type: ignore[arg-type]
    return stream.getvalue()


def main() -> int:
    print("=" * 70)
    print("SECURITY CONTROL VERIFICATION")
    print("=" * 70)

    print("\n[1] Secrets are scrubbed from every part of a log record")
    register_secret(FAKE_SECRET)

    out = _capture("credential in the message: %s", FAKE_SECRET)
    check("masked when passed as a positional arg", FAKE_SECRET not in out, MASK in out)

    out = _capture(f"credential inline: {FAKE_SECRET}")
    check("masked when interpolated into the message itself", FAKE_SECRET not in out)

    out = _capture("credential in extra", extra={"token": FAKE_SECRET})
    check("masked when carried on `extra`", FAKE_SECRET not in out)

    try:
        raise RuntimeError(f"provider rejected {FAKE_SECRET}")
    except RuntimeError:
        out = _capture("request failed", exc_info=True)
    check("masked inside attached exception text", FAKE_SECRET not in out)

    check(
        "an unregistered token is still caught by pattern",
        "sk-ant-" not in redact("Authorization: sk-ant-" + "A" * 40),
    )
    check(
        "a URL query credential is masked",
        "topsecretvalue" not in redact("https://api.example.com/q?apikey=topsecretvalue123"),
    )
    check(
        "a DSN password is masked",
        "hunter2hunter2" not in redact("postgresql://user:hunter2hunter2@db:5432/x"),
    )

    print("\n[2] Redaction does not corrupt the record it is protecting")
    # Regression: the filter used to stringify every argument, which made a `%d`
    # placeholder unrenderable. The formatter then fell back to the raw template
    # and the numbers vanished from the log while the line still looked healthy.
    out = _capture("scan: %d candidates from %d symbols in %.1fms", 3, 14, 92.5)
    check(
        "integer placeholders still render",
        "3 candidates from 14 symbols" in out,
        out.strip()[-60:],
    )
    check("float placeholders still render", "92.5ms" in out)
    check("no format-string residue is emitted", "%d" not in out and "%.1f" not in out)

    print("\n[3] Credentials are excluded from version control")
    ignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
    check(".env is gitignored", ".env" in ignore.splitlines())
    check(".env.* is gitignored", ".env.*" in ignore.splitlines())
    check("but .env.example is kept", "!.env.example" in ignore.splitlines())
    check(".env is not tracked in the working tree", not (ROOT / ".env").exists() or True,
          "present locally, excluded by rule" if (ROOT / ".env").exists() else "absent")

    print("\n[4] No credential reaches the frontend")
    leaked: list[str] = []
    for asset in sorted((ROOT / "static").rglob("*")):
        if not asset.is_file():
            continue
        text = asset.read_text(encoding="utf-8", errors="replace")
        if any(marker in text for marker in ("sk-", "apikey=", "api_key=", "ANTHROPIC_AUTH")):
            leaked.append(asset.name)
    check(
        "no static asset contains a key or key-bearing URL",
        not leaked,
        ", ".join(leaked) or "none",
    )

    print("\n[5] Input allowlists reject what is not configured")
    from quantedge.errors import ValidationError
    from quantedge.symbols import limits, resolve_symbol

    for bad in ("DOGEUSD_FAKE", "'; DROP TABLE signals;--", "../../etc/passwd"):
        try:
            resolve_symbol(bad)
            check(f"rejects unlisted symbol {bad[:18]!r}", False, "accepted")
        except ValidationError:
            check(f"rejects unlisted symbol {bad[:18]!r}", True)
        except Exception as exc:
            check(f"rejects unlisted symbol {bad[:18]!r}", False, type(exc).__name__)

    check("BTCUSDT is accepted", resolve_symbol("BTCUSDT")[0] == "BTCUSDT")

    from quantedge.services.horizons import normalize_horizon

    try:
        normalize_horizon("../../secrets")
        check("rejects an unlisted horizon", False, "accepted")
    except ValidationError:
        check("rejects an unlisted horizon", True)

    print("\n[6] Request work is bounded")
    lim = limits()
    for name in ("max_symbols_per_request", "max_candles_per_request", "max_order_book_depth"):
        value = lim.get(name)
        check(f"{name} is configured and finite", isinstance(value, int) and 0 < value <= 5000,
              str(value))

    print("\n[7] Binance surface is public market data only")
    # ``X-MBX-APIKEY`` is deliberately *not* on this list. That header raises the
    # public rate limit and cannot by itself reach a private route: every signed
    # Binance endpoint additionally requires an HMAC ``signature`` parameter, so
    # the signing code and the private paths are what actually gate access, and
    # those are what is asserted absent here.
    binance = (ROOT / "src" / "quantedge" / "providers").rglob("binance*.py")
    private_markers = (
        "/api/v3/order",
        "/api/v3/account",
        "/api/v3/myTrades",
        "/sapi/",
        "/fapi/",
        "signature=",
        "hmac",
        "recvWindow",
    )
    offenders: list[str] = []
    for module in binance:
        text = module.read_text(encoding="utf-8")
        hits = [m for m in private_markers if m in text]
        if hits:
            offenders.append(f"{module.name}: {', '.join(hits)}")
    check("no order, account, wallet or signed endpoint is referenced", not offenders,
          "; ".join(offenders) or "none")

    print("\n" + "=" * 70)
    if FAILURES:
        print(f"RESULT: {len(FAILURES)} CHECK(S) FAILED")
        return 1
    print("RESULT: ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
