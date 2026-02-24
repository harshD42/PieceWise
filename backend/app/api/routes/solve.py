# Copyright (c) 2026 Harsh Dwivedi
# Licensed under the Harsh Non-Commercial Attribution License (HNCAL) v1.0
# Commercial use requires written permission. See LICENSE for details.

"""
PieceWise — POST /solve + PATCH /solve/{job_id}/correct
Accepts uploaded image pairs, creates a job, and enqueues
the pipeline as a FastAPI background task.
"""

from __future__ import annotations

from fastapi import APIRouter, BackgroundTasks, HTTPException, UploadFile, status

from app.api.middleware.error_handler import ImageValidationError, JobNotFoundError
from app.config import get_settings
from app.core.pipeline import rerun_from_correction, run_pipeline
from app.dependencies import JobStoreDep
from app.models.job import CorrectionRequest, CorrectionResponse, JobStatus, SolveResponse
from app.utils.image_utils import is_valid_image_bytes
from app.utils.logger import get_logger
from app.utils.storage import (
    init_job_dirs,
    pieces_upload_path,
    reference_upload_path,
)

router = APIRouter(tags=["solve"])
log = get_logger(__name__)

ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp"}


def _validate_and_save_upload(upload: UploadFile, dest_path) -> None:
    """
    Read, validate, and save an uploaded image file.
    Raises ImageValidationError on format/size/content failures.
    """
    settings = get_settings()

    if upload.content_type not in ALLOWED_CONTENT_TYPES:
        raise ImageValidationError(
            f"Unsupported file type '{upload.content_type}'. "
            f"Allowed: {', '.join(ALLOWED_CONTENT_TYPES)}"
        )

    data = upload.file.read()

    if len(data) > settings.upload_max_bytes:
        raise ImageValidationError(
            f"File '{upload.filename}' exceeds maximum size "
            f"of {settings.upload_max_mb} MB."
        )

    if not is_valid_image_bytes(data):
        raise ImageValidationError(
            f"File '{upload.filename}' could not be decoded as a valid image. "
            "Ensure it is a non-corrupted JPEG, PNG, or WebP file."
        )

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    dest_path.write_bytes(data)
    log.debug("upload_saved", path=str(dest_path), size_bytes=len(data))


@router.post(
    "/solve",
    response_model=SolveResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Submit a puzzle solve job",
    description=(
        "Upload a reference image and a scattered pieces image. "
        "Returns a job_id for polling progress via GET /status/{job_id}."
    ),
)
async def submit_solve(
    reference_image: UploadFile,
    pieces_image: UploadFile,
    background_tasks: BackgroundTasks,
    store: JobStoreDep,
) -> SolveResponse:
    # Create job first so we have a job_id for storage paths
    job = store.create_job()
    job_id = job.job_id

    log.info("solve_request_received", job_id=job_id)

    # Initialise per-job directory structure
    init_job_dirs(job_id)

    # Validate and persist uploads
    _validate_and_save_upload(reference_image, reference_upload_path(job_id))
    _validate_and_save_upload(pieces_image, pieces_upload_path(job_id))

    log.info("uploads_saved", job_id=job_id)

    # Enqueue pipeline as background task (non-blocking)
    background_tasks.add_task(run_pipeline, job_id, store)

    log.info("pipeline_enqueued", job_id=job_id)

    return SolveResponse(job_id=job_id)


@router.patch(
    "/solve/{job_id}/correct",
    response_model=CorrectionResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Apply a human-in-the-loop piece correction",
    description=(
        "Correct the grid position of a flagged piece. "
        "Re-runs adjacency refinement, sequencing, and rendering. "
        "Segmentation and matching are NOT re-run."
    ),
)
async def correct_piece(
    job_id: str,
    correction: CorrectionRequest,
    background_tasks: BackgroundTasks,
    store: JobStoreDep,
) -> CorrectionResponse:
    job = store.get_job(job_id)
    if job is None:
        raise JobNotFoundError(job_id)

    if job.status != JobStatus.DONE:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                f"Corrections can only be applied to completed jobs. "
                f"Current status: {job.status.value}"
            ),
        )

    log.info(
        "correction_request",
        job_id=job_id,
        piece_id=correction.piece_id,
        corrected_pos=correction.corrected_grid_pos,
    )

    background_tasks.add_task(
        rerun_from_correction,
        job_id,
        correction.piece_id,
        correction.corrected_grid_pos,
        store,
    )

    return CorrectionResponse(
        job_id=job_id,
        piece_id=correction.piece_id,
        corrected_grid_pos=correction.corrected_grid_pos,
    )