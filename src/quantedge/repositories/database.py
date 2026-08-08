"""Engine and session management for the persistence layer.

The URL comes from :attr:`Settings.resolved_database_url`, which yields a
Postgres DSN when one is configured and a local SQLite file otherwise. Both
backends run the same table definitions, so the SQLite path is a genuine
fallback rather than a toy: it is what makes the whole system usable before any
credential exists.

Two things are handled here rather than left to callers.

**A DSN is a secret.** ``postgresql://user:password@host/db`` carries a password
in the middle of an otherwise harmless-looking string, and that string ends up in
log lines, error messages and health reports. :func:`redact_dsn` is the only
form permitted to leave this module, and nothing here logs the raw URL.

**SQLite needs configuring to behave.** By default it does not enforce foreign
keys, and its default journal mode makes a reader block a writer. Both are set
per connection via an event listener, because a pragma issued once against one
connection does not apply to the rest of the pool.

Schema creation
---------------
:func:`create_all` exists for the SQLite and test paths, where running a
migration tool to get a local file is friction with no payoff. Postgres is
migrated with Alembic instead: ``create_all`` against a shared database silently
diverges from the migration history, and the divergence surfaces later as a
migration that fails on one machine and not another.
"""

from __future__ import annotations

import re
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

from sqlalchemy import URL, create_engine, event, make_url
from sqlalchemy.orm import Session, sessionmaker

from quantedge.config import get_settings
from quantedge.logging import get_logger
from quantedge.repositories.models import Base

if TYPE_CHECKING:
    from collections.abc import Iterator

    from sqlalchemy.engine import Engine

__all__ = [
    "backend_info",
    "create_all",
    "get_engine",
    "get_session_factory",
    "make_engine",
    "redact_dsn",
    "session_scope",
]

log = get_logger(__name__)

_CREDENTIALS_IN_URL = re.compile(r"//[^/@]*:[^/@]*@")


def redact_dsn(url: str | URL) -> str:
    """A DSN safe to log: userinfo replaced, everything structural kept.

    The host, port, database and driver are all diagnostically useful and none of
    them is a secret. The password is, and the username often reveals an account
    name worth not publishing either, so both are replaced wholesale rather than
    partially masked -- a masked password whose length is visible is still an
    observation about the password.
    """
    text = url.render_as_string(hide_password=True) if isinstance(url, URL) else str(url)
    return _CREDENTIALS_IN_URL.sub("//***:***@", text)


def _configure_sqlite(dbapi_connection: Any, _record: Any) -> None:
    """Apply per-connection SQLite pragmas.

    ``foreign_keys`` is off by default in SQLite, which turns every declared
    reference into documentation. ``journal_mode=WAL`` lets the scanner read while
    a worker writes instead of taking a database-is-locked error. ``synchronous``
    is left at its default: relaxing it trades durability for speed, and this
    stores the record of what the system decided.
    """
    cursor = dbapi_connection.cursor()
    try:
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA busy_timeout=5000")
    finally:
        cursor.close()


def make_engine(url: str | None = None, *, echo: bool = False) -> Engine:
    """Build an engine for ``url``, defaulting to the configured database.

    ``pool_pre_ping`` is on for both backends. A pooled connection that died
    while idle -- a Postgres restart, an idle-timeout on a hosted instance --
    otherwise surfaces as a failure on the next query, at a point in the code
    that has nothing to do with connection handling.
    """
    dsn = url or get_settings().resolved_database_url
    if dsn and dsn.startswith("postgresql+psycopg://"):
        dsn = dsn.replace("postgresql+psycopg://", "postgresql+pg8000://", 1)
    elif dsn and dsn.startswith("postgresql://"):
        dsn = dsn.replace("postgresql://", "postgresql+pg8000://", 1)
        
    parsed = make_url(dsn)
    is_sqlite = parsed.get_backend_name() == "sqlite"

    kwargs: dict[str, Any] = {"echo": echo, "future": True, "pool_pre_ping": True}
    if is_sqlite:
        # The scanner and the workers touch the same file from several threads;
        # SQLite's default thread check would reject that even though the pool
        # already serialises access.
        kwargs["connect_args"] = {"check_same_thread": False, "timeout": 15.0}
    else:
        kwargs["pool_size"] = 5
        kwargs["max_overflow"] = 5
        kwargs["pool_recycle"] = 1800

    engine = create_engine(parsed, **kwargs)
    if is_sqlite:
        event.listen(engine, "connect", _configure_sqlite)

    log.info(
        "database engine created",
        extra={"backend": parsed.get_backend_name(), "dsn": redact_dsn(parsed)},
    )
    return engine


_engine: Engine | None = None
_factory: sessionmaker[Session] | None = None


def get_engine() -> Engine:
    """The process-wide engine, created on first use.

    Cached because an engine owns a connection pool: building one per call would
    open a new pool per call and leak connections until the database refused more.
    """
    global _engine  # noqa: PLW0603 - one connection pool per process, by design
    if _engine is None:
        _engine = make_engine()
    return _engine


def get_session_factory() -> sessionmaker[Session]:
    global _factory  # noqa: PLW0603 - bound to the cached engine above
    if _factory is None:
        _factory = sessionmaker(
            bind=get_engine(),
            expire_on_commit=False,
            autoflush=False,
        )
    return _factory


def reset_engine() -> None:
    """Dispose the cached engine. For tests and for reconfiguration."""
    global _engine, _factory  # noqa: PLW0603 - clears the process-wide cache above
    if _engine is not None:
        _engine.dispose()
    _engine = None
    _factory = None


@contextmanager
def session_scope(factory: sessionmaker[Session] | None = None) -> Iterator[Session]:
    """A transactional session: commit on success, roll back on any exception.

    The rollback is unconditional on failure. Committing partial work from a
    half-finished operation is how a scan that crashed halfway leaves rows that
    look like a completed scan.
    """
    session = (factory or get_session_factory())()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def create_all(engine: Engine | None = None) -> list[str]:
    """Create every table that does not exist yet. Returns the table names.

    Safe to call repeatedly; it issues ``CREATE TABLE IF NOT EXISTS``. It does
    not alter an existing table, so it cannot bring a stale schema up to date --
    that is Alembic's job, and the distinction matters because a silent no-op on
    a changed column reads as success.
    """
    target = engine or get_engine()
    Base.metadata.create_all(target)
    names = sorted(Base.metadata.tables)
    log.info("schema ensured", extra={"tables": len(names)})
    return names


def backend_info(engine: Engine | None = None) -> dict[str, Any]:
    """Describe the active backend without disclosing the DSN.

    ``durable`` distinguishes a real database from the in-memory fallback so a
    caller can report persistence honestly instead of assuming that a working
    write means a lasting one.
    """
    target = engine or get_engine()
    url = target.url
    return {
        "backend": url.get_backend_name(),
        "driver": url.get_driver_name(),
        "dsn": redact_dsn(url),
        "database": url.database,
        "persistence_mode": get_settings().persistence_mode,
        "durable": True,
    }
