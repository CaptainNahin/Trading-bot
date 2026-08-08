"""LLM reviewer adapters and the selection of one from configuration."""

from __future__ import annotations

from quantedge.config import get_settings
from quantedge.logging import get_logger
from quantedge.providers.llm.agentrouter import AgentRouterLLMProvider
from quantedge.providers.llm.anthropic import AnthropicLLMProvider
from quantedge.providers.llm.base import BaseLLMProvider

__all__ = [
    "AgentRouterLLMProvider",
    "AnthropicLLMProvider",
    "BaseLLMProvider",
    "default_llm_provider",
]

log = get_logger(__name__)


def default_llm_provider() -> BaseLLMProvider | None:
    """The reviewer named by ``LLM_PROVIDER``, or ``None`` when unavailable.

    ``None`` is a supported state, not a failure: the deterministic scanner,
    risk levels and memory all work without a model, and the caller reports the
    candidate on its own rather than blocking on a review it cannot obtain.
    Returning ``None`` rather than raising is what keeps that path open.

    A provider named in configuration but missing its credential also yields
    ``None``, logged once so the reason is visible in the service log instead of
    surfacing later as an unexplained absence of review.
    """
    settings = get_settings()
    name = settings.llm_provider

    if name == "disabled":
        return None

    provider: BaseLLMProvider
    if name == "agentrouter":
        provider = AgentRouterLLMProvider()
    elif name == "anthropic":
        provider = AnthropicLLMProvider()
    else:  # pragma: no cover - Settings restricts the literal
        log.warning("unknown LLM_PROVIDER; reviews disabled", extra={"configured": name})
        return None

    if not getattr(provider, "credentials_present", False):
        log.info(
            "LLM_PROVIDER is set but its credential is missing; reviews disabled",
            extra={"provider": name},
        )
        return None
    return provider
