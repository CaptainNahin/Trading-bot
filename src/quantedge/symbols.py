"""Symbol allowlisting and normalization.

Two jobs:

1. **Security.** Only symbols present in ``config/symbols.yaml`` may be
   requested. This bounds the work any caller can trigger and prevents a
   crafted symbol from reaching a provider URL.
2. **Correctness.** Each canonical symbol maps to exactly one asset class and,
   for forex, to an explicit currency pair used by event-risk mapping.

Canonical form is uppercase with separators stripped: ``eur/usd`` -> ``EURUSD``.
Provider-specific spellings (``EUR/USD``, ``EUR_USD``) are produced by each
adapter, never stored internally.
"""

from __future__ import annotations

import functools

from quantedge.config import symbols_config
from quantedge.contracts.enums import AssetClass
from quantedge.errors import UnsupportedSymbolError, ValidationError

__all__ = [
    "asset_class_for",
    "currencies_for_symbol",
    "is_supported",
    "limits",
    "normalize_symbol",
    "resolve_symbol",
    "supported_symbols",
]

_SECTIONS: tuple[tuple[str, AssetClass], ...] = (
    ("crypto", AssetClass.CRYPTO),
    ("forex", AssetClass.FOREX),
    ("commodity", AssetClass.COMMODITY),
    ("index", AssetClass.INDEX),
    ("stock", AssetClass.STOCK),
)


def normalize_symbol(symbol: str) -> str:
    """Canonicalize a user-supplied symbol.

    Uppercases and strips the separators declared in ``config/symbols.yaml``.
    Does **not** check the allowlist -- use :func:`resolve_symbol` for that.
    """
    if not isinstance(symbol, str) or not symbol.strip():
        raise ValidationError("symbol must be a non-empty string")
    cfg = symbols_config().get("normalization", {})
    out = symbol.strip()
    if cfg.get("uppercase", True):
        out = out.upper()
    for char in cfg.get("strip_characters", ["/", "-", "_", ":", " "]):
        out = out.replace(char, "")
    if not out:
        raise ValidationError(f"symbol '{symbol}' normalized to an empty string")
    if len(out) > 20:
        raise ValidationError(f"symbol '{out[:20]}...' exceeds the 20-character limit")
    if not out.isalnum():
        raise ValidationError(f"symbol '{out}' contains non-alphanumeric characters")
    return out


@functools.lru_cache(maxsize=1)
def _symbol_index() -> dict[str, AssetClass]:
    """Build ``{canonical symbol: asset class}`` from the config, once."""
    index: dict[str, AssetClass] = {}
    cfg = symbols_config()
    for section, asset_class in _SECTIONS:
        for raw in cfg.get(section, {}).get("symbols", []) or []:
            index[str(raw).strip().upper()] = asset_class
    return index


def supported_symbols(asset_class: AssetClass | str | None = None) -> list[str]:
    """All allowlisted symbols, optionally filtered by asset class."""
    index = _symbol_index()
    if asset_class is None:
        return sorted(index)
    target = AssetClass(str(asset_class).lower())
    return sorted(s for s, ac in index.items() if ac is target)


_KNOWN_COMMODITIES = {
    "XAUUSD", "GOLD", "XAGUSD", "SILVER", "WTICOUSD", "WTI", "USOIL", "BRENT", "UKOIL", "NATGAS", "COPPER", "PLATINUM", "PALLADIUM"
}
_KNOWN_INDICES = {
    "SPX", "NDX", "DJI", "US30", "NAS100", "US500", "DAX", "FTSE", "NIFTY", "VIX", "DXY"
}
_KNOWN_CRYPTO_SUFFIXES = ("USDT", "BUSD", "USDC", "BTC", "ETH")
_KNOWN_CRYPTO_TOKENS = {
    "BTC", "ETH", "SOL", "BNB", "XRP", "ADA", "DOGE", "AVAX", "DOT", "LINK", "MATIC", "POL", "LTC", "NEAR", "SUI", "PEPE", "SHIB", "TRX", "TON", "UNI", "APT"
}
_ISO_CURRENCIES = {
    "USD", "EUR", "GBP", "JPY", "CHF", "CAD", "AUD", "NZD", "ARS", "TRY", "BRL", "MXN",
    "INR", "ZAR", "SGD", "HKD", "NOK", "SEK", "DKK", "PLN", "CZK", "HUF", "ILS", "THB",
    "IDR", "MYR", "PHP", "KRW", "CNY", "CNH", "RUB", "CLP", "COP", "PEN", "TWD", "AED", "SAR"
}


def is_supported(symbol: str) -> bool:
    """True for any valid symbol that normalizes cleanly."""
    try:
        normalize_symbol(symbol)
        return True
    except ValidationError:
        return False


def resolve_symbol(symbol: str) -> tuple[str, AssetClass]:
    """Normalize and resolve symbol and its AssetClass.

    Checks config index first. If not listed in config, dynamically infers
    the asset class (Forex, Crypto, Commodity, Index, Stock) so that any
    valid global market (e.g. USDARS, USDBRL, NVDA) is fully supported.
    """
    canonical = normalize_symbol(symbol)
    index = _symbol_index()
    if canonical in index:
        return canonical, index[canonical]

    # Dynamic classification for universal markets
    if canonical in _KNOWN_COMMODITIES:
        return canonical, AssetClass.COMMODITY
    if canonical in _KNOWN_INDICES:
        return canonical, AssetClass.INDEX
    if any(canonical.endswith(suf) for suf in _KNOWN_CRYPTO_SUFFIXES) or canonical in _KNOWN_CRYPTO_TOKENS:
        return canonical, AssetClass.CRYPTO
    if len(canonical) == 6 and (canonical[:3] in _ISO_CURRENCIES or canonical[3:] in _ISO_CURRENCIES):
        return canonical, AssetClass.FOREX

    # Default to STOCK / EQUITY for standard tickers (e.g. AAPL, NVDA, TSLA)
    if len(canonical) <= 6:
        return canonical, AssetClass.STOCK

    return canonical, AssetClass.FOREX


def asset_class_for(symbol: str) -> AssetClass:
    return resolve_symbol(symbol)[1]


def currencies_for_symbol(symbol: str) -> list[str]:
    """Currencies whose economic events are relevant to ``symbol``.

    Forex pairs use the explicit ``currency_map``. Crypto falls back to the
    broad risk currencies configured for the crypto section (USD by default),
    because macro USD releases move crypto too. Returns ``[]`` when no mapping
    is configured -- callers must then report ``UNKNOWN`` event risk rather than
    guessing.
    """
    canonical, asset_class = resolve_symbol(symbol)
    cfg = symbols_config()

    mapped = cfg.get("currency_map", {}).get(canonical)
    if mapped:
        return [str(c).upper() for c in mapped]

    if asset_class is AssetClass.CRYPTO:
        broad = cfg.get("crypto", {}).get("quote_currency_risk", ["USD"])
        return [str(c).upper() for c in broad]

    if asset_class is AssetClass.FOREX and len(canonical) == 6:
        # A 6-character forex code decomposes unambiguously.
        return [canonical[:3], canonical[3:]]

    if asset_class in (AssetClass.STOCK, AssetClass.INDEX):
        return ["USD"]

    return []


@functools.lru_cache(maxsize=1)
def limits() -> dict[str, int]:
    """Hard input limits from ``config/symbols.yaml``."""
    defaults = {
        "max_symbols_per_request": 25,
        "max_candles_per_request": 1000,
        "max_order_book_depth": 100,
        "max_recent_trades": 500,
        "max_stream_symbols": 20,
        "max_stream_intervals": 4,
    }
    configured = symbols_config().get("limits", {}) or {}
    return {k: int(configured.get(k, v)) for k, v in defaults.items()}


def enforce_limit(value: int, limit_name: str, label: str) -> int:
    """Clamp-check an integer against a configured limit."""
    max_value = limits()[limit_name]
    if value < 1:
        raise ValidationError(f"{label} must be >= 1; got {value}")
    if value > max_value:
        raise ValidationError(
            f"{label} {value} exceeds the maximum of {max_value}",
            details={"limit": max_value},
        )
    return value


def default_stream_symbols() -> list[str]:
    cfg = symbols_config()
    raw = cfg.get("default_stream_symbols") or ["BTCUSDT", "ETHUSDT"]
    return [str(s).upper() for s in raw]
