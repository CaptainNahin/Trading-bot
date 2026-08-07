"""Persistence layer: the repository, its backends, and the factory that picks one.

:func:`get_repository` resolves the backend from configuration -- Postgres or
SQLite through :class:`~quantedge.repositories.sql.SqlRepository`, or
:class:`~quantedge.repositories.memory.MemoryRepository` when no database can be
reached. Callers depend on the method surface, which is identical across both, so
the choice of backend is not visible in business logic.

The fallback is deliberate but never silent. When a database was configured and
could not be opened, the failure is logged at ERROR and the returned repository
reports ``persistence_available is False``. Nothing in this system may report a
successful write as durable without consulting that flag: a scan whose results
vanished on restart and a scan whose results were saved look identical from the
inside, and only this flag distinguishes them.

In production the fallback is refused outright rather than degraded into. A
production deployment that silently ran on in-memory storage would accumulate a
performance record that disappears at the next restart, and the settled-signal
history is the one thing in this system that must not be reconstructable-by-luck.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from quantedge.config import get_settings
from quantedge.errors import PersistenceError
from quantedge.logging import get_logger

if TYPE_CHECKING:
    from quantedge.repositories.memory import MemoryRepository
    from quantedge.repositories.sql import SqlRepository

__all__ = [
    "Repository",
    "describe_persistence",
    "get_repository",
    "reset_repository",
]

log = get_logger(__name__)


@runtime_checkable
class Repository(Protocol):
    """The method surface both backends implement.

    Declared as a Protocol rather than a base class so neither backend inherits
    behaviour it does not want, and so a test double satisfies the type without
    importing SQLAlchemy.
    """

    def save_candles(self, candles: Any) -> int: ...
    def load_candles(self, symbol: str, timeframe: Any, **kwargs: Any) -> list[Any]: ...
    def save_quote(self, quote: Any) -> None: ...
    def latest_quote(self, symbol: str) -> Any: ...
    def is_symbol_allowed(self, symbol: str, **kwargs: Any) -> bool: ...
    def consume_quota(self, provider: str, **kwargs: Any) -> dict[str, Any]: ...
    def log_event(self, event_type: str, **kwargs: Any) -> None: ...


_repository: SqlRepository | MemoryRepository | None = None
_mode: str = "uninitialised"


def get_repository(*, force_memory: bool = False) -> SqlRepository | MemoryRepository:
    """The process-wide repository, created on first use.

    Parameters
    ----------
    force_memory:
        Skip the database entirely. For tests that must not touch a file.

    Raises
    ------
    PersistenceError
        When the database is unreachable and ``APP_ENV`` is production. Falling
        back there would mean running the live system on storage that forgets.
    """
    global _repository, _mode  # noqa: PLW0603 - one repository per process, by design
    if _repository is not None:
        return _repository

    settings = get_settings()

    if force_memory:
        from quantedge.repositories.memory import MemoryRepository

        _repository = MemoryRepository()
        _mode = "memory"
        return _repository

    try:
        from quantedge.repositories.database import create_all, get_engine
        from quantedge.repositories.sql import SqlRepository

        engine = get_engine()
        # SQLite gets its schema created here; Postgres is migrated by Alembic
        # and create_all would drift from the migration history.
        if engine.url.get_backend_name() == "sqlite":
            create_all(engine)
        _repository = SqlRepository()
        _mode = settings.persistence_mode
        log.info("repository ready", extra={"mode": _mode, "durable": True})
        return _repository
    except Exception as exc:
        if settings.app_env == "production":
            raise PersistenceError(
                "database is unreachable and in-memory fallback is refused in production; "
                "settled signals and audit logs must be durable",
                details={"persistence_mode": settings.persistence_mode},
            ) from exc

        from quantedge.repositories.memory import MemoryRepository

        log.error(
            "database unavailable; falling back to in-memory storage that does not persist",
            extra={"error": type(exc).__name__, "durable": False},
        )
        _repository = MemoryRepository()
        _mode = "memory"
        return _repository


def reset_repository() -> None:
    """Drop the cached repository. For tests and reconfiguration."""
    global _repository, _mode  # noqa: PLW0603 - clears the process-wide cache above
    _repository = None
    _mode = "uninitialised"


def describe_persistence() -> dict[str, Any]:
    """What the active backend is and whether it survives a restart.

    ``durable`` is the field callers must consult before reporting that anything
    was saved. It is false for the in-memory backend, and a caller that reports
    "stored 300 candles" without checking it has told the user something untrue.
    """
    repo = get_repository()
    durable = not isinstance(repo, _memory_class())
    info: dict[str, Any] = {
        "mode": _mode,
        "durable": durable,
        "persistence_available": durable,
        "backend_class": type(repo).__name__,
    }
    if durable:
        from quantedge.repositories.database import backend_info

        info.update(backend_info())
    else:
        info["warning"] = "in-memory storage: nothing survives a process restart"
    return info


def _memory_class() -> type:
    from quantedge.repositories.memory import MemoryRepository

    return MemoryRepository
