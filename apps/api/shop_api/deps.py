"""Dependency wiring."""

from __future__ import annotations

from collections.abc import Iterator

from fastapi import Request
from sqlalchemy.orm import Session

from shop_api.service import AppContext


def get_context(request: Request) -> AppContext:
    return request.app.state.context


def get_session(request: Request) -> Iterator[Session]:
    context = get_context(request)
    with context.db.session() as session:
        yield session
