"""Health, capability, model and metrics endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

from shop_api.deps import get_context, get_session
from shop_api.errors import request_id
from shop_api.service import (
    APP_VERSION,
    AppContext,
    capabilities_payload,
    metrics_payload,
    model_capability_payload,
    readiness_payload,
    worker_heartbeat,
)
from shop_core.persistence.repository import iso
from datetime import UTC, datetime

public_router = APIRouter(prefix="/health", tags=["health"])
router = APIRouter(tags=["system"])


@public_router.get("/live")
def live(request: Request) -> dict:
    return {
        "status": "ok",
        "service": "shop-api",
        "version": APP_VERSION,
        "checked_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "request_id": request_id(request),
    }


@public_router.get("/ready")
def ready(request: Request, context: AppContext = Depends(get_context)):
    is_ready, payload = readiness_payload(context)
    payload["request_id"] = request_id(request)
    return JSONResponse(status_code=200 if is_ready else 503, content=payload)


@router.get("/capabilities")
def capabilities(context: AppContext = Depends(get_context)) -> dict:
    return capabilities_payload(context)


@router.get("/models")
def models(context: AppContext = Depends(get_context)) -> dict:
    return {
        "bundle": context.model_bundle(),
        **model_capability_payload(context),
        "labels": {
            "commercial_status": "manifest declaration, not a measured accuracy claim",
            "evaluation": "unmeasured unless details.measured_on_this_host is true",
        },
    }


@router.get("/metrics")
def metrics(
    session: Session = Depends(get_session), context: AppContext = Depends(get_context)
) -> dict:
    return metrics_payload(session, context)


@router.get("/config")
def config(context: AppContext = Depends(get_context)) -> dict:
    from shop_core.config import redacted_config

    payload = redacted_config(context.settings)
    payload["worker"] = worker_heartbeat(context.settings)
    return payload
