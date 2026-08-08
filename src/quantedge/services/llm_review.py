"""LLM Response Validation & Verification Engine.

Enforces strict compliance of LLM outputs against system rules:
- Fails validation if extra fields are present (`extra="forbid"`).
- Refuses upgrading any candidate built on failing data quality (`QualityStatus.FAIL`).
- Forces `calibrated_probability = None` when no calibration model is active.
"""

from __future__ import annotations

import json
from typing import Any

from pydantic import ValidationError

from quantedge.contracts import (
    LLMSignalResponse,
    QualityStatus,
    SignalContext,
    SignalStatus,
)
from quantedge.errors import QuantEdgeError


class LLMValidationError(QuantEdgeError):
    """Raised when an LLM response violates contract rules."""

    def __init__(self, message: str, *, violations: list[str] | None = None):
        super().__init__(message, code="LLM_VALIDATION_ERROR")
        self.violations = violations or []


def validate_llm_response(
    raw_response: str | dict[str, Any] | LLMSignalResponse,
    context: SignalContext,
) -> LLMSignalResponse:
    """Validate and normalize an LLM response against context rules.

    Parameters
    ----------
    raw_response:
        Raw JSON string, dictionary, or LLMSignalResponse instance.
    context:
        The SignalContext payload that was passed to the LLM.

    Returns
    -------
    LLMSignalResponse
        A fully validated, compliant response contract.
    """
    violations: list[str] = []

    # 1. Parse JSON / Dict into Pydantic model
    if isinstance(raw_response, LLMSignalResponse):
        response = raw_response
    elif isinstance(raw_response, dict):
        try:
            response = LLMSignalResponse.model_validate(raw_response)
        except ValidationError as exc:
            raise LLMValidationError(
                f"LLM response failed schema validation: {exc}",
                violations=[str(e) for e in exc.errors()],
            ) from exc
    elif isinstance(raw_response, str):
        try:
            parsed = json.loads(raw_response)
            response = LLMSignalResponse.model_validate(parsed)
        except (json.JSONDecodeError, ValidationError) as exc:
            raise LLMValidationError(
                f"Failed to parse LLM JSON response: {exc}",
                violations=[str(exc)],
            ) from exc
    else:
        raise LLMValidationError(f"Invalid response type: {type(raw_response)}")

    # 2. Hard Gate: Data Quality FAIL must force NO_TRADE / INSUFFICIENT_DATA
    if context.quality.status == QualityStatus.FAIL and response.status not in (
        SignalStatus.NO_TRADE,
        SignalStatus.INSUFFICIENT_DATA,
    ):
        violations.append(
            f"Cannot upgrade signal status to '{response.status.value}' "
            "when data quality status is FAIL"
        )
        response = response.model_copy(
            update={
                "status": SignalStatus.NO_TRADE,
                "missing_information": [
                    *response.missing_information,
                    "Data quality check failed",
                ],
            }
        )

    # 3. Rule 6: Forced calibrated_probability = None when no calibration model is active
    if not context.calibration_model_available and response.calibrated_probability is not None:
        violations.append("calibrated_probability must be null when no calibration model is active")
        response = response.model_copy(update={"calibrated_probability": None})

    return response
