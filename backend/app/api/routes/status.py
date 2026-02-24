# Copyright (c) 2026 Harsh Dwivedi
# Licensed under the Harsh Non-Commercial Attribution License (HNCAL) v1.0
# Commercial use requires written permission. See LICENSE for details.

"""
PieceWise — GET /status/{job_id}
Returns current job status, stage, and progress for frontend polling.
"""

from __future__ import annotations

from fastapi import APIRouter

from app.api.middleware.error_handler import JobNotFoundError
from app.dependencies import JobStoreDep
from app.utils.logger import get_logger

router = APIRouter(tags=["status"])
log = get_logger(__name__)


@router.get(
    "/status/{job_id}",
    summary="Poll job progress",
    description=(
        "Returns current status, pipeline stage, and progress percentage (0–100). "
        "Poll every 1–2 seconds while status is 'pending' or 'running'. "
        "When status is 'done', result.solution_manifest_url contains the full solution."
    ),
)
async def get_status(job_id: str, store: JobStoreDep) -> dict:
    job = store.get_job(job_id)
    if job is None:
        raise JobNotFoundError(job_id)

    log.debug("status_polled", job_id=job_id, status=job.status.value, progress=job.progress)
    return job.to_status_response()