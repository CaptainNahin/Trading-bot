"""Chat conversation path: a real brain reply passes through; a brain timeout
falls back to an honest, question-aware message -- never the old command menu."""
from __future__ import annotations

import quantedge.providers.llm as llm_pkg
from quantedge.errors import ProviderTimeoutError
from quantedge.services import chat as chat_svc


class _ReplyProvider:
    provider_name = "seekai"

    def __init__(self, reply=None, exc=None):
        self._reply = reply
        self._exc = exc

    def generate_chat_reply(self, message, conversation_history=None, system_prompt=None):
        if self._exc is not None:
            raise self._exc
        return self._reply


def _run(monkeypatch, provider):
    monkeypatch.setattr(llm_pkg, "default_llm_provider", lambda: provider)
    return chat_svc._handle_conversation(
        "Well, I think you are giving only down on EURGBP, so what do you think?",
        {"conversation_history": []},
    )


def test_real_brain_reply_passes_through(monkeypatch):
    reply = _run(monkeypatch, _ReplyProvider(reply="Fair pushback -- EURGBP has leaned bearish, but I'd watch for a 4H CHoCH."))
    assert reply.text.startswith("Fair pushback")
    assert (reply.data or {}).get("fallback") is None
    assert (reply.data or {}).get("provider") == "seekai"


def test_brain_timeout_gives_honest_question_aware_fallback(monkeypatch):
    reply = _run(monkeypatch, _ReplyProvider(exc=ProviderTimeoutError("seekai", 54.0)))
    data = reply.data or {}
    assert data.get("fallback") is True
    assert "ProviderTimeoutError" in (data.get("fallback_reason") or "")
    # It must acknowledge the question and invite a retry, NOT dump the old menu.
    assert "ask me again" in reply.text.lower()
    assert "Available Actions" not in reply.text
    assert "I received your question" not in reply.text


def test_empty_brain_reply_falls_back(monkeypatch):
    reply = _run(monkeypatch, _ReplyProvider(reply="   "))
    data = reply.data or {}
    assert data.get("fallback") is True
    assert data.get("fallback_reason") == "AI brain returned an empty reply"


def test_meta_identity_question_routes_to_conversation():
    """The image-10 misroute: an identity/meta question that merely CONTAINS the
    word 'signal' must be a conversation, never a spurious BTCUSDT signal."""
    q = "Which model are you using? Who are you giving me this sandbox? How is the signal coming?"
    assert chat_svc.parse_intent(q).intent == chat_svc.Intent.CONVERSATION


def test_meta_questions_stay_conversation():
    for q in (
        "which llm is this",
        "what model are you",
        "are you using deepseek",
        "how accurate is this bot",
        "why did you only give down",
        "you are giving only down, what do you think?",
    ):
        assert chat_svc.parse_intent(q).intent == chat_svc.Intent.CONVERSATION, q


def test_real_signal_commands_still_route_to_signal():
    """The guard must not swallow genuine imperatives to produce a signal."""
    for q in (
        "Give me a signal USD/JPY for 5 min",
        "scan BTCUSDT 15m",
        "generate a setup for ETHUSDT",
        "send me a trade on gold",
    ):
        assert chat_svc.parse_intent(q).intent == chat_svc.Intent.SIGNAL, q
