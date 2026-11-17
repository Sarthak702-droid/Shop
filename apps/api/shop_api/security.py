"""Local bearer authentication and reviewer identity resolution."""

from __future__ import annotations

import hmac

from fastapi import Depends, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from shop_api.errors import ApiError

_bearer = HTTPBearer(auto_error=False, scheme_name="ShopLocalBearer")


def require_api_auth(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
) -> None:
    settings = request.app.state.context.settings
    expected = settings.api_token
    if expected is None:
        return
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise ApiError(
            401,
            "UNAUTHENTICATED",
            "A local bearer token is required for this operation.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    if not hmac.compare_digest(credentials.credentials, expected):
        raise ApiError(
            401,
            "INVALID_TOKEN",
            "The supplied local bearer token is not valid.",
            headers={"WWW-Authenticate": "Bearer"},
        )


def resolve_operator(request: Request) -> str:
    """Choose the reviewer identity from an authenticated local header.

    The token proves local API access; it does not identify a person. The
    reviewer header is checked against the explicit local allowlist. Unknown
    identities are rejected so review audit records always refer to an approved
    actor.
    """
    settings = request.app.state.context.settings
    requested = request.headers.get("x-shop-reviewer", "").strip()
    allowed = settings.security.reviewers
    if requested:
        if requested not in allowed:
            raise ApiError(
                403,
                "REVIEWER_NOT_APPROVED",
                "The requested reviewer identity is not in the local allowlist.",
            )
        return requested
    return allowed[0]
