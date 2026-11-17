"""Programmatic Alembic migration entry point shared by API, worker and tests."""

from __future__ import annotations

from pathlib import Path

from alembic import command
from alembic.config import Config
from sqlalchemy import Engine, inspect


def alembic_config(engine: Engine | None = None) -> Config:
    from shop_core.config import PROJECT_ROOT

    config = Config(str(PROJECT_ROOT / "alembic.ini"))
    config.set_main_option("script_location", str(PROJECT_ROOT / "migrations"))
    if engine is not None:
        config.attributes["connection"] = engine.connect()
    return config


def upgrade_database(engine: Engine) -> None:
    """Apply all migrations.

    An existing pre-migration database created by an older development build is
    not silently stamped: callers get a clear failure instead of a half-applied
    schema. Fresh installs upgrade normally.
    """
    existing = set(inspect(engine).get_table_names())
    config = alembic_config(engine)
    connection = config.attributes.get("connection")
    try:
        if existing and "alembic_version" not in existing:
            raise RuntimeError(
                "Existing database has no migration history. Back it up and run the documented "
                "one-time migration adoption step before starting Shop."
            )
        command.upgrade(config, "head")
    finally:
        if connection is not None:
            connection.close()


def database_head(engine: Engine) -> str | None:
    """Return the current Alembic revision without mutating the database."""
    from sqlalchemy import text

    try:
        with engine.connect() as connection:
            row = connection.execute(text("SELECT version_num FROM alembic_version LIMIT 1")).first()
            return str(row[0]) if row else None
    except Exception:
        return None


def migration_script_dir() -> Path:
    from shop_core.config import PROJECT_ROOT

    return PROJECT_ROOT / "migrations"
