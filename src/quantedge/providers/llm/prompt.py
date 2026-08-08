"""The prompt handed to a reviewing model, and the parser for what comes back.

Division of labour
------------------
Every number in the prompt was computed deterministically in
:mod:`quantedge.services.indicators`, :mod:`~quantedge.services.structure` and
:mod:`~quantedge.services.scoring`. The model's job is to *review* that evidence
-- weigh contradictions, name what would invalidate the setup, say when the case
is too thin -- not to calculate anything. Rule 10 is explicit that indicator
values must never come from a model's intuition, so the system prompt forbids
arithmetic outright and the response schema has no field a computed value could
be smuggled into.

Why the response is JSON with a closed schema
---------------------------------------------
:class:`~quantedge.contracts.LLMSignalResponse` sets ``extra="forbid"``. A model
that invents ``"win_rate": 0.87`` fails validation rather than having the number
accepted and displayed. The prompt therefore lists the permitted keys exactly,
and the parser hands the raw text to that contract instead of reading fields out
by hand.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from quantedge.errors import ProviderBadResponseError

if TYPE_CHECKING:
    from quantedge.contracts import SignalContext

__all__ = ["SYSTEM_PROMPT", "build_user_prompt", "extract_json_object"]

SYSTEM_PROMPT = """You are the review stage of a quantitative trading pipeline.

Every indicator, structure level and score in the payload was already computed
deterministically from closed candles. You must NOT recalculate, estimate or
adjust any number, and you must NOT introduce a number that is not in the
payload. Arithmetic is not your task; judgement is.

Your job:
1. Decide whether the deterministic evidence genuinely supports the candidate
   direction, or whether the contradictions outweigh it.
2. Name the concrete conditions that would invalidate the setup.
3. List what is not known. If required information is missing, say so.

Choose exactly one status:
- "SIGNAL"            the evidence supports the candidate direction
- "NO_TRADE"          the evidence is present but too weak or too conflicted
- "INSUFFICIENT_DATA" required data is missing, stale or failed quality checks

Hard rules:
- Never state or imply a win rate, accuracy figure, or probability of profit.
- Never output a confidence number. The calibrated_probability field must be
  null: no calibration model has been fitted, so any value would be invented.
- Prefer NO_TRADE over a marginal SIGNAL. Declining is a valid answer.
- If data quality is DEGRADED or FAIL, that must appear in your reasoning.

Reply with a single JSON object and nothing else -- no prose, no code fence.
Permitted keys, all optional except status and generated_at_utc:
  status, asset, direction, generated_at_utc, entry_window_start_utc,
  entry_window_end_utc, expiry_utc, horizon, reference_price, regime,
  calibrated_probability, heuristic_score, supporting_evidence,
  contradictory_evidence, invalidation_conditions, missing_information
direction is "UP", "DOWN" or null. The three *_evidence/conditions/information
fields are arrays of short strings. Timestamps are ISO 8601 with a UTC offset.
Any key not on this list will cause the response to be rejected."""


def build_user_prompt(context: SignalContext) -> str:
    """Serialise the verified context as the model's only source of fact.

    The payload is the contract's own JSON dump rather than a prose summary, so
    what the model reviews is exactly what the pipeline computed and persisted --
    a hand-written summary could omit a contradiction or round a level.
    """
    payload = context.model_dump(mode="json", exclude_none=True)
    return (
        "Review this verified market context and reply with the JSON object.\n\n"
        f"```json\n{json.dumps(payload, indent=2, sort_keys=True)}\n```\n\n"
        f"Candidate direction to assess: {context.candidate_direction or 'none established'}\n"
        f"Data quality status: {context.quality.status.value} "
        f"(score {context.quality.quality_score})\n"
        f"Known gaps: {'; '.join(context.missing_information) or 'none recorded'}"
    )


def extract_json_object(text: str, *, provider: str) -> dict[str, Any]:
    """Parse the one JSON object out of a model's reply.

    Tolerates a `````json`` fence and surrounding chatter, because models add
    both despite instructions. Does not tolerate ambiguity beyond that: if no
    object can be isolated the call fails with the offending sample rather than
    returning a partial dict that would validate into a signal.
    """
    stripped = text.strip()
    if stripped.startswith("```"):
        # Drop the opening fence (with or without a language tag) and the closer.
        stripped = stripped.split("\n", 1)[-1] if "\n" in stripped else ""
        if stripped.rstrip().endswith("```"):
            stripped = stripped.rstrip()[:-3]
        stripped = stripped.strip()

    if not stripped.startswith("{"):
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start == -1 or end <= start:
            raise ProviderBadResponseError(
                provider,
                "model reply contained no JSON object",
                sample=text[:200],
            )
        stripped = stripped[start : end + 1]

    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError as exc:
        raise ProviderBadResponseError(
            provider,
            f"model reply was not valid JSON: {exc.msg}",
            sample=stripped[:200],
        ) from exc

    if not isinstance(parsed, dict):
        raise ProviderBadResponseError(
            provider,
            f"model reply was a {type(parsed).__name__}, not a JSON object",
            sample=stripped[:200],
        )
    return parsed
