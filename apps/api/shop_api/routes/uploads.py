"""Bounded upload ingestion, probing and deletion."""

from __future__ import annotations

import hashlib
import os
import uuid
from pathlib import Path
from uuid import UUID

from fastapi import APIRouter, Depends, File, Request, UploadFile
from fastapi.responses import JSONResponse
from sqlalchemy import select
from sqlalchemy.orm import Session

from shop_api.deps import get_context, get_session
from shop_api.errors import ApiError, request_id
from shop_api.security import resolve_operator
from shop_api.service import AppContext, disk_status, upload_api
from shop_core.persistence.models import AuditRow, JobRow, UploadRow, utcnow
from shop_core.persistence.repository import iso, new_id
from shop_core.sources.media import SourceFault, probe_video

router = APIRouter(tags=["uploads"])
CHUNK = 1024 * 1024
ALLOWED_EXTENSIONS = {".mp4", ".mov", ".webm", ".mkv", ".avi", ".m4v"}


def _safe_filename(value: str | None) -> str:
    name = Path(value or "recording").name.replace("\x00", "").strip()
    if not name or name in {".", ".."}:
        return "recording"
    return name[:255]


@router.post("/uploads", status_code=201)
async def create_upload(
    request: Request,
    file: UploadFile = File(...),
    session: Session = Depends(get_session),
    context: AppContext = Depends(get_context),
) -> JSONResponse:
    settings = context.settings
    actor = resolve_operator(request)
    disk = disk_status(settings)
    if disk["below_minimum"]:
        raise ApiError(
            503,
            "STORAGE_LOW",
            "Upload is disabled because free storage is below the configured reserve.",
            retryable=True,
        )
    raw_length = request.headers.get("content-length")
    if raw_length:
        try:
            if int(raw_length) > settings.storage.max_upload_bytes + 1_048_576:
                raise ApiError(
                    413, "UPLOAD_TOO_LARGE", "The upload exceeds the configured byte limit."
                )
        except ValueError:
            raise ApiError(400, "INVALID_CONTENT_LENGTH", "Content-Length is not a valid integer.")

    filename = _safe_filename(file.filename)
    extension = Path(filename).suffix.lower()
    if extension not in ALLOWED_EXTENSIONS:
        raise ApiError(
            415,
            "UNSUPPORTED_EXTENSION",
            "Upload a supported video container such as MP4, MOV, WebM, MKV or AVI.",
            details={"accepted_extensions": sorted(ALLOWED_EXTENSIONS)},
        )
    upload_id = new_id()
    temp_path = settings.uploads_dir / f".{upload_id}.part"
    final_path = settings.uploads_dir / f"{upload_id}{extension}"
    digest = hashlib.sha256()
    size = 0
    handle = None
    try:
        handle = temp_path.open("xb")
        while True:
            chunk = await file.read(CHUNK)
            if not chunk:
                break
            size += len(chunk)
            if size > settings.storage.max_upload_bytes:
                raise ApiError(
                    413,
                    "UPLOAD_TOO_LARGE",
                    "The upload exceeds the configured byte limit.",
                    details={"max_bytes": settings.storage.max_upload_bytes},
                )
            if size > 8 * 1024 * 1024 and size % (64 * 1024 * 1024) < CHUNK:
                current = disk_status(settings)
                if current["below_minimum"]:
                    raise ApiError(
                        503,
                        "STORAGE_LOW",
                        "Upload was stopped because the free-space reserve was reached.",
                        retryable=True,
                    )
            digest.update(chunk)
            handle.write(chunk)
        handle.flush()
        os.fsync(handle.fileno())
        handle.close()
        handle = None
        if size <= 0:
            raise ApiError(422, "UPLOAD_EMPTY", "The uploaded file is empty.")
        try:
            probe = probe_video(temp_path)
        except SourceFault as error:
            status = 415 if error.code == "MEDIA_UNSUPPORTED" else 422
            raise ApiError(status, error.code, error.message) from error
        if probe.get("frame_count") is not None and not probe.get("first_frame_decoded"):
            raise ApiError(422, "MEDIA_DECODE_FAILED", "No decodable first frame was found.")
        # The temporary suffix is an implementation detail. The extension is
        # only a container label; the decoder observation remains authoritative.
        probe["container"] = extension.lstrip(".")
        duration = probe.get("duration_ms")
        if duration is not None and duration > settings.storage.max_upload_duration_ms:
            raise ApiError(
                422,
                "MEDIA_DURATION_EXCEEDED",
                "The video duration exceeds the configured upload limit.",
                details={
                    "duration_ms": duration,
                    "max_duration_ms": settings.storage.max_upload_duration_ms,
                },
            )
        width = int(probe.get("width") or 0)
        height = int(probe.get("height") or 0)
        if width > settings.storage.max_upload_width or height > settings.storage.max_upload_height:
            raise ApiError(
                422,
                "MEDIA_DIMENSIONS_EXCEEDED",
                "The video dimensions exceed the configured upload limit.",
                details={
                    "width": width,
                    "height": height,
                    "max_width": settings.storage.max_upload_width,
                    "max_height": settings.storage.max_upload_height,
                },
            )
        os.replace(temp_path, final_path)
        row = UploadRow(
            id=upload_id,
            owner=actor,
            original_filename=filename,
            stored_path=final_path.relative_to(settings.storage.root).as_posix(),
            sha256=digest.hexdigest(),
            bytes=size,
            content_type=file.content_type,
            status="ready",
            probe=probe,
        )
        session.add(row)
        session.add(
            AuditRow(
                id=new_id(),
                actor=actor,
                action="upload_validated",
                object_type="upload",
                object_id=upload_id,
                request_id=request_id(request),
                metadata_json={
                    "bytes": size,
                    "sha256": row.sha256,
                    "codec": probe.get("codec"),
                    "duration_ms": probe.get("duration_ms"),
                },
            )
        )
        session.commit()
    except ApiError:
        temp_path.unlink(missing_ok=True)
        raise
    except Exception:
        temp_path.unlink(missing_ok=True)
        final_path.unlink(missing_ok=True)
        raise
    finally:
        if handle is not None:
            handle.close()
        await file.close()
    return JSONResponse(
        status_code=201,
        content={"upload": upload_api(row), "request_id": request_id(request)},
    )


@router.get("/uploads")
def list_uploads(
    session: Session = Depends(get_session), context: AppContext = Depends(get_context)
) -> dict:
    rows = session.execute(
        select(UploadRow)
        .where(UploadRow.deleted_at.is_(None))
        .order_by(UploadRow.created_at.desc(), UploadRow.id.desc())
        .limit(100)
    ).scalars().all()
    return {"items": [upload_api(row) for row in rows], "next_cursor": None}


@router.get("/uploads/{upload_id}")
def get_upload(upload_id: UUID, session: Session = Depends(get_session)) -> dict:
    row = session.get(UploadRow, str(upload_id))
    if row is None or row.deleted_at is not None:
        raise ApiError(404, "UPLOAD_NOT_FOUND", "The upload is not available.")
    return {"upload": upload_api(row)}


@router.delete("/uploads/{upload_id}", status_code=202)
def delete_upload(
    upload_id: UUID,
    request: Request,
    session: Session = Depends(get_session),
    context: AppContext = Depends(get_context),
) -> dict:
    row = session.get(UploadRow, str(upload_id))
    if row is None:
        raise ApiError(404, "UPLOAD_NOT_FOUND", "The upload is not available.")
    actor = resolve_operator(request)
    if row.deleted_at is not None:
        return {"upload_id": row.id, "status": "deleted", "request_id": request_id(request)}
    active = session.execute(
        select(JobRow).where(
            JobRow.upload_id == row.id,
            JobRow.state.in_(("queued", "running", "cancelling")),
        )
    ).scalars().first()
    if active is not None:
        raise ApiError(
            409,
            "UPLOAD_ACTIVE_HOLD",
            "This upload has an active analysis job. Cancel or finish the job before deletion.",
            details={"job_id": active.id},
        )
    path = (context.settings.storage.root / row.stored_path).resolve()
    try:
        path.relative_to(context.settings.storage.root.resolve())
    except ValueError:
        raise ApiError(409, "UPLOAD_PATH_REFUSED", "The stored upload path is not inside managed storage.")
    row.deleted_at = utcnow()
    row.status = "deleted"
    path.unlink(missing_ok=True)
    session.add(
        AuditRow(
            id=new_id(),
            actor=actor,
            action="upload_deleted",
            object_type="upload",
            object_id=row.id,
            request_id=request_id(request),
            metadata_json={"deleted": True},
        )
    )
    return {
        "upload_id": row.id,
        "status": "deleted",
        "deleted_at": iso(row.deleted_at),
        "request_id": request_id(request),
    }
