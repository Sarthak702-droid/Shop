"""Safe API error types and response envelope."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

logger = logging.getLogger("shop.api")


class ApiError(Exception):
    def __init__(
        self,
        status_code: int,
        code: str,
        message: str,
        *,
        retryable: bool = False,
        details: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message
        self.retryable = retryable
        self.details = details or {}
        self.headers = headers or {}


def request_id(request: Request) -> str:
    return str(getattr(request.state, "request_id", "unavailable"))


def envelope(
    request: Request,
    *,
    code: str,
    message: str,
    retryable: bool,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "error": {
            "code": code,
            "message": message,
            "retryable": retryable,
            "details": details or {},
        },
        "request_id": request_id(request),
    }


def install_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(ApiError)
    async def _api_error(request: Request, exc: ApiError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content=envelope(
                request,
                code=exc.code,
                message=exc.message,
                retryable=exc.retryable,
                details=exc.details,
            ),
            headers=exc.headers,
        )

    @app.exception_handler(RequestValidationError)
    async def _validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        fields = []
        for error in exc.errors():
            fields.append(
                {
                    "location": ".".join(str(part) for part in error.get("loc", ())),
                    "message": error.get("msg", "Invalid value."),
                    "type": error.get("type", "value_error"),
                }
            )
        return JSONResponse(
            status_code=422,
            content=envelope(
                request,
                code="VALIDATION_ERROR",
                message="The request did not match the required schema.",
                retryable=False,
                details={"fields": fields[:20]},
            ),
        )

    @app.exception_handler(StarletteHTTPException)
    async def _http_error(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        code = {404: "NOT_FOUND", 405: "METHOD_NOT_ALLOWED", 401: "UNAUTHENTICATED"}.get(
            exc.status_code, "HTTP_ERROR"
        )
        message = {
            404: "The requested resource is not available.",
            405: "The HTTP method is not allowed for this resource.",
            401: "Authentication is required.",
        }.get(exc.status_code, "The request could not be completed.")
        return JSONResponse(
            status_code=exc.status_code,
            content=envelope(
                request,
                code=code,
                message=message,
                retryable=exc.status_code >= 500,
            ),
            headers=exc.headers,
        )

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("unhandled api exception request_id=%s", request_id(request))
        return JSONResponse(
            status_code=500,
            content=envelope(
                request,
                code="INTERNAL_ERROR",
                message="An internal error occurred. No raw exception data is exposed.",
                retryable=False,
            ),
        )
