"""Request identity, access logging and defensive response headers."""

from __future__ import annotations

import logging
import re
import time
import uuid
from collections.abc import Awaitable, Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from shop_api.errors import envelope

logger = logging.getLogger("shop.api.access")
REQUEST_ID_PATTERN = re.compile(r"^[A-Za-z0-9._-]{1,100}$")


class RequestContextMiddleware(BaseHTTPMiddleware):
    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        supplied = request.headers.get("x-request-id", "")
        rid = supplied if REQUEST_ID_PATTERN.fullmatch(supplied) else str(uuid.uuid4())
        request.state.request_id = rid
        started = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            logger.exception(
                "request_id=%s method=%s path=%s status=500", rid, request.method, request.url.path
            )
            raise
        elapsed_ms = round((time.perf_counter() - started) * 1000.0, 2)
        response.headers["X-Request-ID"] = rid
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Cross-Origin-Resource-Policy"] = "same-origin"
        response.headers["Cache-Control"] = response.headers.get(
            "Cache-Control", "no-store" if request.url.path.startswith("/api/") else "no-cache"
        )
        response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
        logger.info(
            "request_id=%s method=%s path=%s status=%s elapsed_ms=%s",
            rid,
            request.method,
            request.url.path,
            response.status_code,
            elapsed_ms,
        )
        return response


class RequestSizeMiddleware(BaseHTTPMiddleware):
    """Reject oversized non-upload JSON before a route reads it.

    Upload bodies are streamed and counted byte-by-byte in the upload route so
    an untrusted Content-Length cannot bypass the configured limit.
    """

    def __init__(self, app, *, max_json_bytes: int = 1_048_576, upload_path: str = "/api/v1/uploads") -> None:  # noqa: ANN001
        super().__init__(app)
        self.max_json_bytes = max_json_bytes
        self.upload_path = upload_path

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        if (
            request.method in {"POST", "PUT", "PATCH"}
            and not request.url.path.rstrip("/").endswith(self.upload_path)
            and request.url.path.startswith("/api/")
        ):
            raw_length = request.headers.get("content-length")
            if raw_length:
                try:
                    length = int(raw_length)
                except ValueError:
                    length = self.max_json_bytes + 1
                if length > self.max_json_bytes:
                    return _too_large(request)
        return await call_next(request)


def _too_large(request: Request) -> Response:
    from fastapi.responses import JSONResponse

    return JSONResponse(
        status_code=413,
        content=envelope(
            request,
            code="REQUEST_TOO_LARGE",
            message="The request body exceeds the configured limit.",
            retryable=False,
        ),
    )
