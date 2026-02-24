# Copyright (c) 2026 Harsh Dwivedi
# Licensed under the Harsh Non-Commercial Attribution License (HNCAL) v1.0
# Commercial use requires written permission. See LICENSE for details.

"""
PieceWise — GET /assets/{job_id}/{file_path}
Streams output files (overlays, step cards, solution.json) from
per-job namespaced storage. Handles nested paths like step_cards/step_0001.jpg.
"""

from __future__ import annotations

import mimetypes
from pathlib import Path

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import FileResponse

from app.api.middleware.error_handler import JobNotFoundError
from app.dependencies import JobStoreDep
from app.utils.logger import get_logger
from app.utils.storage import job_dir, outputs_dir

router = APIRouter(tags=["assets"])
log = get_logger(__name__)

# Only these extensions are servable — prevents path traversal to cache/uploads
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".json", ".webp"}


def _safe_resolve(job_id: str, file_path: str) -> Path:
    """
    Resolve and validate the requested file path.
    - Ensures the path is inside the job's outputs directory
    - Rejects path traversal attempts (../ etc.)
    - Rejects disallowed file extensions
    Raises HTTPException on any violation.
    """
    out_dir = outputs_dir(job_id).resolve()
    # Normalise and resolve without requiring file to exist yet
    requested = (out_dir / file_path).resolve()

    # Must be inside outputs directory
    try:
        requested.relative_to(out_dir)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Path traversal not allowed.",
        )

    # Extension allowlist
    if requested.suffix.lower() not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"File type '{requested.suffix}' not servable.",
        )

    return requested


@router.get(
    "/assets/{job_id}/{file_path:path}",
    summary="Retrieve a job output asset",
    description=(
        "Stream an output file for a completed job. "
        "file_path can include subdirectories, e.g. 'step_cards/step_0001.jpg'. "
        "Only files inside the job's outputs/ directory are accessible."
    ),
)
async def get_asset(
    job_id: str,
    file_path: str,
    store: JobStoreDep,
) -> FileResponse:
    # Verify job exists
    job = store.get_job(job_id)
    if job is None:
        raise JobNotFoundError(job_id)

    resolved = _safe_resolve(job_id, file_path)

    if not resolved.exists():
        log.warning("asset_not_found", job_id=job_id, file_path=file_path)
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Asset '{file_path}' not yet available for job {job_id}.",
        )

    media_type, _ = mimetypes.guess_type(str(resolved))
    media_type = media_type or "application/octet-stream"

    log.debug("asset_served", job_id=job_id, file_path=file_path)
    return FileResponse(path=str(resolved), media_type=media_type)