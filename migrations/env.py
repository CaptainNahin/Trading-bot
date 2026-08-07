"""Alembic environment: the DSN comes from Settings, never from alembic.ini.

``alembic.ini`` is committed, so a DSN written there would put a database
password into version control. The URL is read from
:attr:`Settings.resolved_database_url` instead -- the same value the application
uses, sourced from ``.env``, which is gitignored. One source, never checked in.

Nothing here logs the raw URL. :func:`redact_dsn` is the only form permitted to
reach a log line, because a migration that fails prints its context, and that is
exactly the moment a DSN would otherwise end up in a terminal scrollback or a CI
log that outlives the credential.

Why Alembic exists here at all
------------------------------
SQLite gets its schema from ``create_all`` (see
:mod:`quantedge.repositories.database`); Postgres is migrated. Running
``create_all`` against a shared database silently diverges from the migration
history, and the divergence surfaces later as a migration that fails on one
machine and not another.
"""

from __future__ import annotations

import sys
from logging.config import fileConfig
from pathlib import Path

from alembic import context
from sqlalchemy import engine_from_config, pool

# The project is a src-layout package and alembic runs from the repo root, so
# src/ is not on the path yet.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from quantedge.config import get_settings
from quantedge.repositories.database import redact_dsn
from quantedge.repositories.models import Base, UtcDateTime

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Autogenerate compares this against the live database. Importing Base is what
# makes every table in models.py visible to it; a table whose module is never
# imported looks to autogenerate like a table the migration should drop.
target_metadata = Base.metadata


def _url() -> str:
    """The DSN to migrate, from settings unless alembic.ini overrides it.

    The ini value is honoured when non-empty so a one-off migration against
    another database stays possible without editing code, but it is empty in the
    committed file and the environment is the normal path.
    """
    from_ini = config.get_main_option("sqlalchemy.url")
    if from_ini:
        return from_ini
    return get_settings().resolved_database_url


def _render_item(type_: str, obj: object, autogen_context: object) -> str | bool:
    """Render custom column types as their plain SQLAlchemy equivalent.

    ``UtcDateTime`` is a ``TypeDecorator`` whose behaviour is entirely
    Python-side -- it rejects naive datetimes on bind and re-attaches UTC on
    result. The DDL it emits is exactly ``DateTime(timezone=True)``, so that is
    what the migration should say.

    The point is that a migration must not import application code. Migrations
    are permanent and models are not: rendering ``quantedge...UtcDateTime`` into
    a revision file means renaming or deleting that class years later breaks a
    migration that already ran everywhere, and ``alembic upgrade head`` on a
    fresh database fails on history nobody is allowed to edit.

    ``env.py`` itself is exempt from that rule and has to import the models --
    autogenerate has nothing to compare against otherwise. It is the revision
    files, which are permanent, that must stay free of application imports.
    """
    if type_ == "type" and isinstance(obj, UtcDateTime):
        return "sa.DateTime(timezone=True)"
    return False


def run_migrations_offline() -> None:
    """Emit SQL to stdout instead of running it.

    This is how a migration gets reviewed before it touches a production
    database, and how a DBA who owns the schema applies it themselves.
    """
    url = _url()
    print(f"-- target: {redact_dsn(url)}", file=sys.stderr)
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
        render_item=_render_item,
        render_as_batch=url.startswith("sqlite"),
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Apply migrations against a live connection."""
    url = _url()
    section = config.get_section(config.config_ini_section, {}) or {}
    section["sqlalchemy.url"] = url

    connectable = engine_from_config(
        section,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    try:
        with connectable.connect() as connection:
            context.configure(
                connection=connection,
                target_metadata=target_metadata,
                # Autogenerate ignores type changes by default, which means a
                # column widened from Numeric(18,8) to Numeric(24,12) produces an
                # empty migration and a silent truncation later.
                compare_type=True,
                render_item=_render_item,
                # SQLite cannot ALTER a column; batch mode rewrites the table
                # instead. Harmless on Postgres, so it is keyed off the dialect.
                render_as_batch=connection.dialect.name == "sqlite",
            )
            with context.begin_transaction():
                context.run_migrations()
    finally:
        connectable.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
