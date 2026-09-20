"""Abstract Base Class for LLM providers."""

from __future__ import annotations

from abc import ABC, abstractmethod

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

