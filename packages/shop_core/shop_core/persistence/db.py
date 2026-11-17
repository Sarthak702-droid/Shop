"""Database engine, pragmas and transaction helpers."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from sqlalchemy import Engine, create_engine, event
from sqlalchemy.orm import Session, sessionmaker

from shop_core.config import Settings
from shop_core.persistence.models import Base


def create_db_engine(database_url: str | None = None, storage_root: Path | None = None) -> Engine:
    if database_url is None:
        if storage_root is None:
            raise ValueError("storage_root or database_url is required")
        database_url = f"sqlite:///{storage_root / 'shop.db'}"
    engine = create_engine(
        database_url,
        future=True,
        connect_args={"timeout": 15, "check_same_thread": False},
    )

    @event.listens_for(engine, "connect")
    def _set_sqlite_pragmas(dbapi_connection, _record) -> None:  # noqa: ANN001
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.execute("PRAGMA busy_timeout=15000")
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA synchronous=NORMAL")
        cursor.close()

    return engine


class Database:
    def __init__(self, settings: Settings) -> None:
        settings.storage.root.mkdir(parents=True, exist_ok=True)
        self.settings = settings
        self.engine = create_db_engine(settings.database_url)
        self.session_factory = sessionmaker(bind=self.engine, expire_on_commit=False, future=True)

    def create_all(self) -> None:
        Base.metadata.create_all(self.engine)

    @contextmanager
    def session(self) -> Iterator[Session]:
        session = self.session_factory()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def dispose(self) -> None:
        self.engine.dispose()
