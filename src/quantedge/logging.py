"""Structured logging with mandatory secret redaction.

This module is imported by :mod:`quantedge.errors`, so it must not import any
other project module -- keep it dependency-free.

Why redaction lives at the logging layer
----------------------------------------
Provider SDKs and ``httpx`` happily include full request URLs in exception
messages, and market APIs commonly authenticate via query string
(``?apikey=...``). A single unhandled traceback is therefore enough to write a
live credential into a log file. Rather than trusting every call site to be
careful, we redact centrally:

* :func:`register_secret` records every value that must never appear in output.
  :func:`quantedge.config.get_settings` registers all loaded credentials at
  startup.
* :func:`redact` also applies pattern-based scrubbing, so a secret that was
  never registered (a token pasted into a URL by a third-party library, for
  instance) is still masked.
* :class:`RedactingFilter` is attached to the root logger, covering messages,
  formatting arguments, and exception text.

Redaction is best-effort defence in depth, not a licence to log secrets.
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
from datetime import UTC, datetime
from typing import Any

__all__ = [
    "RedactingFilter",
    "configure_logging",
    "get_logger",
    "redact",
    "register_secret",
    "registered_secret_count",
]

MASK = "***REDACTED***"

# Minimum length before a registered value is treated as a secret. Guards
# against a short/blank config value turning into a catastrophic global replace.
_MIN_SECRET_LENGTH = 8

_SECRETS: set[str] = set()

# Pattern-based fallbacks for values we were never told about.
_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    # key=value in query strings and connection strings
    (
        re.compile(
            r"(?i)\b(api[-_]?key|apikey|token|access[-_]?token|secret|password|passwd"
            r"|pwd|auth|authorization|bearer|service[-_]?role[-_]?key|signature)"
            r"(\s*[=:]\s*|%3D)([\"']?)([A-Za-z0-9_\-./+=]{6,})\3"
        ),
        r"\1\2\3" + MASK + r"\3",
    ),
    # Vendor-prefixed tokens
    (re.compile(r"\bsk-ant-[A-Za-z0-9_\-]{10,}"), MASK),
    (re.compile(r"\bsk-[A-Za-z0-9_\-]{20,}"), MASK),
    (re.compile(r"\bsbp_[A-Za-z0-9]{20,}"), MASK),
    (re.compile(r"\bvck_[A-Za-z0-9]{20,}"), MASK),
    (re.compile(r"\bpk_[A-Za-z0-9]{20,}"), MASK),
    (re.compile(r"\bghp_[A-Za-z0-9]{20,}"), MASK),
    # JWTs (Supabase service-role keys are JWTs)
    (re.compile(r"\beyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}"), MASK),
    # Credentials embedded in a DSN: scheme://user:password@host
    (re.compile(r"(?i)\b([a-z][a-z0-9+.\-]*://[^:/\s]+:)([^@/\s]+)(@)"), r"\1" + MASK + r"\3"),
)


def register_secret(value: str | None) -> None:
    """Record ``value`` so it is masked everywhere.

    Blank, ``None`` and implausibly short values are ignored -- replacing a
    2-character string globally would corrupt every log line.
    """
    if not value:
        return
    stripped = value.strip()
    if len(stripped) < _MIN_SECRET_LENGTH:
        return
    _SECRETS.add(stripped)


def registered_secret_count() -> int:
    """Number of registered secrets. Used by tests; never logs the values."""
    return len(_SECRETS)


def redact(text: Any) -> str:
    """Return ``text`` with every known or suspected secret masked."""
    if text is None:
        return ""
    out = text if isinstance(text, str) else str(text)

    # Exact registered values first: longest to shortest, so an overlapping
    # shorter secret cannot partially mask a longer one.
    for secret in sorted(_SECRETS, key=len, reverse=True):
        if secret in out:
            out = out.replace(secret, MASK)

    for pattern, replacement in _PATTERNS:
        out = pattern.sub(replacement, out)

    return out


def _redact_arg(value: Any) -> Any:
    """Scrub one log argument without destroying its type.

    Redacting unconditionally returned a ``str`` for every argument, which broke
    every printf-style call using a numeric placeholder: ``"%d" % ("3",)`` raises,
    the record can no longer render, and :class:`JsonFormatter` falls back to the
    raw template -- so ``"Scan completed: %d candidates"`` reached the operator
    with the count silently missing. The line looked fine; the number was gone.

    Redaction strength is unchanged. When scrubbing alters the text, the scrubbed
    string is returned exactly as before. When it does not, the original object is
    passed through and the formatter renders the same characters from it. Numbers
    are exempt outright: they have no string form a secret can hide in.
    """
    if value is None or isinstance(value, bool | int | float | complex):
        return value
    scrubbed = redact(value)
    return scrubbed if scrubbed != str(value) else value


class RedactingFilter(logging.Filter):
    """Scrub secrets from the message, its args, and any attached exception."""

    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.msg, str):
            record.msg = redact(record.msg)

        if record.args:
            if isinstance(record.args, dict):
                record.args = {k: _redact_arg(v) for k, v in record.args.items()}
            elif isinstance(record.args, tuple):
                record.args = tuple(_redact_arg(a) for a in record.args)

        # Exception text is the highest-risk path: render it now, scrub it, and
        # drop the original exc_info so the formatter cannot re-expand it.
        if record.exc_info and record.exc_info[0] is not None:
            record.exc_text = redact(
                logging.Formatter().formatException(record.exc_info)  # type: ignore[arg-type]
            )
            record.exc_info = None

        if record.exc_text:
            record.exc_text = redact(record.exc_text)

        return True


_RESERVED = frozenset(
    {
        "args",
        "asctime",
        "created",
        "exc_info",
        "exc_text",
        "filename",
        "funcName",
        "levelname",
        "levelno",
        "lineno",
        "module",
        "msecs",
        "message",
        "msg",
        "name",
        "pathname",
        "process",
        "processName",
        "relativeCreated",
        "stack_info",
        "thread",
        "threadName",
        "taskName",
    }
)


class JsonFormatter(logging.Formatter):
    """Emit one JSON object per record (machine-ingestible logs)."""

    def format(self, record: logging.LogRecord) -> str:
        try:
            msg = record.getMessage()
        except Exception:
            msg = str(record.msg)
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": msg,
        }
        for key, value in record.__dict__.items():
            if key not in _RESERVED and not key.startswith("_"):
                try:
                    json.dumps(value)
                    payload[key] = _redact_arg(value)
                except (TypeError, ValueError):
                    payload[key] = redact(str(value))
        if record.exc_text:
            payload["exception"] = record.exc_text
        return json.dumps(payload, default=str)


class ConsoleFormatter(logging.Formatter):
    """Compact human-readable format for local development."""

    def __init__(self) -> None:
        super().__init__(
            fmt="%(asctime)s %(levelname)-7s %(name)-34s %(message)s",
            datefmt="%H:%M:%S",
        )


_configured = False


def configure_logging(level: str = "INFO", fmt: str = "json", *, force: bool = False) -> None:
    """Install the root handler, formatter and redaction filter.

    Idempotent unless ``force`` is set.

    Notes
    -----
    Logs go to **stderr**. This is mandatory, not cosmetic: the MCP server
    speaks JSON-RPC over *stdout*, so a single stray stdout log line would
    corrupt the protocol stream and break every tool call.
    """
    global _configured  # noqa: PLW0603 - process-wide logging setup, by design
    if _configured and not force:
        return

    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)

    handler = logging.StreamHandler(stream=sys.stderr)
    handler.setFormatter(JsonFormatter() if fmt.lower() == "json" else ConsoleFormatter())
    handler.addFilter(RedactingFilter())

    root.addHandler(handler)
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Third-party loggers are chatty at DEBUG and can echo request URLs.
    for noisy in ("httpx", "httpcore", "websockets", "asyncio", "sqlalchemy.engine"):
        logging.getLogger(noisy).setLevel(max(root.level, logging.WARNING))

    _configured = True


def get_logger(name: str) -> logging.LoggerAdapter[logging.Logger] | logging.Logger:
    """Return a logger, auto-configuring from the environment on first use.

    Reads ``LOG_LEVEL``/``LOG_FORMAT`` directly rather than importing
    :mod:`quantedge.config`, which would create an import cycle.
    """
    if not _configured:
        configure_logging(
            level=os.getenv("LOG_LEVEL", "INFO"),
            fmt=os.getenv("LOG_FORMAT", "json"),
        )
    return logging.getLogger(name)
