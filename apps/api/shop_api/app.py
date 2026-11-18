"""FastAPI application factory for Shop's local review service."""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import APIRouter, Depends, FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException

from shop_api.errors import install_error_handlers
from shop_api.middleware import RequestContextMiddleware, RequestSizeMiddleware
from shop_api.security import require_api_auth
from shop_api.service import APP_VERSION, AppContext
from shop_core.config import PROJECT_ROOT, Settings, load_settings
from shop_core.persistence.db import Database
from shop_core.persistence.migrate import upgrade_database

logger = logging.getLogger("shop.api")


def _cleanup_partial_uploads(root: Path) -> None:
    cutoff = time.time() - 60 * 60
    try:
        for path in root.glob(".*.part"):
            try:
                if path.stat().st_mtime < cutoff:
                    path.unlink(missing_ok=True)
            except OSError:
                continue
    except OSError:
        return


def create_app(
    settings: Settings | None = None,
    *,
    auto_migrate: bool | None = None,
) -> FastAPI:
    settings = settings or load_settings()
    settings.ensure_directories()
    db = Database(settings)
    should_migrate = settings.storage.auto_migrate if auto_migrate is None else auto_migrate
    if should_migrate:
        upgrade_database(db.engine)
    _cleanup_partial_uploads(settings.uploads_dir)
    context = AppContext(settings=settings, db=db)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        logger.info("shop api starting version=%s bind=%s", APP_VERSION, settings.security.bind_host)
        try:
            yield
        finally:
            db.dispose()

    app = FastAPI(
        title="Shop local review API",
        version=APP_VERSION,
        description=(
            "Local-first retail activity review. Events are descriptive observations for human "
            "review, never a determination of theft or guilt."
        ),
        lifespan=lifespan,
        docs_url="/api/docs",
        redoc_url=None,
        openapi_url="/api/openapi.json",
    )
    app.state.context = context
    install_error_handlers(app)
    app.add_middleware(RequestSizeMiddleware)
    app.add_middleware(RequestContextMiddleware)

    from shop_api.routes import admin, events, jobs, research, sources, system, uploads

    public = APIRouter(prefix="/api/v1")
    public.include_router(system.public_router)
    app.include_router(public)

    protected = APIRouter(prefix="/api/v1", dependencies=[Depends(require_api_auth)])
    protected.include_router(system.router)
    protected.include_router(uploads.router)
    protected.include_router(sources.router)
    protected.include_router(jobs.router)
    protected.include_router(events.router)
    protected.include_router(admin.router)
    protected.include_router(research.router)
    app.include_router(protected)

    dist = PROJECT_ROOT / "apps" / "web" / "dist"
    assets = dist / "assets"
    if assets.is_dir():
        app.mount("/assets", StaticFiles(directory=assets), name="assets")

    @app.get("/", include_in_schema=False)
    def index():
        index_file = dist / "index.html"
        if not index_file.is_file():
            return JSONResponse(
                {
                    "service": "shop-api",
                    "version": APP_VERSION,
                    "detail": (
                        "API is running. The review interface build is absent; run "
                        "`npm install && npm run build` in apps/web."
                    ),
                    "docs": "/api/docs",
                }
            )
        return FileResponse(
            index_file,
            media_type="text/html",
            headers={
                "Content-Security-Policy": (
                    "default-src 'self'; img-src 'self' data: blob:; media-src 'self' blob:; "
                    "style-src 'self'; script-src 'self'; font-src 'self' data:; "
                    "connect-src 'self'; frame-ancestors 'none'; base-uri 'self'"
                ),
                "Cache-Control": "no-cache",
            },
        )

    @app.get("/{path:path}", include_in_schema=False)
    def spa_fallback(path: str):
        if path.startswith("api/"):
            raise StarletteHTTPException(status_code=404)
        candidate = (dist / path).resolve()
        try:
            candidate.relative_to(dist.resolve())
        except ValueError:
            raise StarletteHTTPException(status_code=404)
        if candidate.is_file() and path:
            return FileResponse(candidate)
        index_file = dist / "index.html"
        if not index_file.is_file():
            raise StarletteHTTPException(status_code=404)
        return FileResponse(
            index_file,
            media_type="text/html",
            headers={
                "Content-Security-Policy": (
                    "default-src 'self'; img-src 'self' data: blob:; media-src 'self' blob:; "
                    "style-src 'self'; script-src 'self'; font-src 'self' data:; "
                    "connect-src 'self'; frame-ancestors 'none'; base-uri 'self'"
                ),
                "Cache-Control": "no-cache",
            },
        )

    return app
