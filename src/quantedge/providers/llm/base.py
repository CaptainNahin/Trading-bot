"""Abstract Base Class for LLM providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from quantedge.contracts import LLMSignalResponse, ProviderHealth, SignalContext


class BaseLLMProvider(ABC):
    """Base interface for all LLM providers in QuantEdge.

    Integrations must ensure strict adherence to the output schema.
    """

    def __init__(self, provider_name: str, model_name: str):
        self.provider_name = provider_name
        self.model_name = model_name

    @abstractmethod
    def health(self) -> ProviderHealth:
        """Check provider health and return status."""
        ...

    @abstractmethod
    def evaluate_signal_context(self, context: SignalContext) -> LLMSignalResponse:
        """Evaluate a SignalContext and produce a structured LLMSignalResponse."""
        ...

    def generate_chat_reply(
        self,
        message: str,
        conversation_history: list[dict[str, str]] | None = None,
        system_prompt: str | None = None,
    ) -> str:
        """Generate a natural conversational response."""
        raise NotImplementedError(f"{self.provider_name} does not implement generate_chat_reply")

    def decide_trade(
        self,
        evidence: dict[str, Any],
        *,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        """Decide a trade from pre-verified evidence (optional decision authority).

        Providers that can act as the decision brain return
        ``{"decision", "conviction", "reason", "invalidation", "brain"}``.
        Providers that only review need not implement this; the orchestrator
        checks for the capability and falls back to the deterministic gate.
        """
        raise NotImplementedError(f"{self.provider_name} does not implement decide_trade")

