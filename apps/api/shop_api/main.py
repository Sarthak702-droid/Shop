"""Uvicorn entry point.

    .venv/bin/uvicorn shop_api.main:app --host 127.0.0.1 --port 8000
"""

from __future__ import annotations

from shop_api.app import create_app

app = create_app()
