"""Automated test suite for symbol extraction, provider routing, and chat signal flows."""

from fastapi.testclient import TestClient

from quantedge.api.app import app
from quantedge.services.chat import Intent, _extract_symbol, handle_message, parse_intent


def test_symbol_extraction_slash_pairs():
    """Verify that forex and crypto pairs with slashes or hyphens resolve correctly."""
    assert _extract_symbol("Give me a signal USD/JPY for 5 min") == "USDJPY"
    assert _extract_symbol("give me EUR/USD 15m") == "EURUSD"
    assert _extract_symbol("check GBP/USD 5 min") == "GBPUSD"
    assert _extract_symbol("USD-CAD analysis") == "USDCAD"
    assert _extract_symbol("BTC/USDT 5m") == "BTCUSDT"
    assert _extract_symbol("ETH/USDT 15m") == "ETHUSDT"
    assert _extract_symbol("XAU/USD gold setup") == "XAUUSD"


def test_symbol_extraction_space_separated():
    """Verify two-token currency pairs like 'USD JPY'."""
    assert _extract_symbol("signal for USD JPY 5 min") == "USDJPY"
    assert _extract_symbol("EUR USD trade") == "EURUSD"
    assert _extract_symbol("BTC USDT 15m") == "BTCUSDT"


def test_symbol_extraction_aliases():
    """Verify common shorthands and aliases."""
    assert _extract_symbol("BTC 15m") == "BTCUSDT"
    assert _extract_symbol("ETH 5m") == "ETHUSDT"
    assert _extract_symbol("YEN 5m") == "USDJPY"
    assert _extract_symbol("GOLD 15m") == "XAUUSD"
    assert _extract_symbol("SOL 15m") == "SOLUSDT"


def test_parse_intent_signal_usdjpy():
    """Verify parse_intent captures symbol, minutes, and Intent.SIGNAL."""
    parsed = parse_intent("Give me a signal USD/JPY for 5 min")
    assert parsed.intent == Intent.SIGNAL
    assert parsed.symbol == "USDJPY"
    assert parsed.minutes == 5


def test_parse_intent_eurusd():
    parsed = parse_intent("EUR/USD 15m")
    assert parsed.intent == Intent.SIGNAL
    assert parsed.symbol == "EURUSD"
    assert parsed.minutes == 15


def test_handle_message_usdjpy():
    """Verify handle_message returns a grounded reply for USD/JPY without crashing."""
    reply = handle_message("Give me a signal USD/JPY for 5 min")
    assert reply.intent == Intent.SIGNAL
    assert isinstance(reply.text, str)
    assert len(reply.text) > 10
    # Must never produce a raw CancelledError
    assert "CancelledError" not in reply.text


def test_api_chat_signal_usdjpy():
    """Verify /api/v1/bot/chat endpoint returns 200 with grounded reply."""
    client = TestClient(app)
    # Obtain a valid trial pass token
    auth_resp = client.get("/api/v1/auth/trial-pass")
    assert auth_resp.status_code == 200
    token = auth_resp.json()["token"]

    response = client.post(
        "/api/v1/bot/chat",
        headers={"Authorization": f"Bearer {token}"},
        json={"message": "Give me a signal USD/JPY for 5 min", "session_id": "test_session"},
    )
    assert response.status_code == 200
    data = response.json()
    assert "text" in data
    assert "CancelledError" not in data["text"]
    assert data.get("intent") == "SIGNAL"
