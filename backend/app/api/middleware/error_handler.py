# Copyright (c) 2026 Harsh Dwivedi
# Licensed under the Harsh Non-Commercial Attribution License (HNCAL) v1.0
# Commercial use requires written permission. See LICENSE for details.

"""
PieceWise — Global Error Handler
Converts all unhandled exceptions into structured JSON error responses.
Registered on the FastAPI app in main.py.
"""

from __future__ import annotations

import traceback

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse

from app.utils.logger import get_logger

log = get_logger(__name__)


class ImageValidationError(ValueError):
    """Raised when an uploaded image fails format or size validation."""


class JobNotFoundError(KeyError):
    """Raised when a job_id does not exist in the store."""


class PipelineError(RuntimeError):
    """Raised when a pipeline stage fails in a recoverable way."""


def _error_body(code: str, message: str, detail: str | None = None) -> dict:
    body = {"error": {"code": code, "message": message}}
    if detail:
        body["error"]["detail"] = detail
    return body


def register_error_handlers(app: FastAPI) -> None:
    """
    Register all global exception handlers on the FastAPI application.
    Call this in main.py after creating the app instance.
    """

    @app.exception_handler(ImageValidationError)
    async def image_validation_handler(
        req: Request, exc: ImageValidationError
    ) -> JSONResponse:
        log.warning("image_validation_error", path=str(req.url), error=str(exc))
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content=_error_body(
                code="IMAGE_VALIDATION_ERROR",
                message=str(exc),
            ),
        )

    @app.exception_handler(JobNotFoundError)
    async def job_not_found_handler(
        req: Request, exc: JobNotFoundError
    ) -> JSONResponse:
        log.warning("job_not_found", path=str(req.url), error=str(exc))
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content=_error_body(
                code="JOB_NOT_FOUND",
                message=f"Job not found: {exc}",
            ),
        )

    @app.exception_handler(PipelineError)
    async def pipeline_error_handler(
        req: Request, exc: PipelineError
    ) -> JSONResponse:
        log.error("pipeline_error", path=str(req.url), error=str(exc))
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=_error_body(
                code="PIPELINE_ERROR",
                message=str(exc),
            ),
        )

    @app.exception_handler(NotImplementedError)
    async def not_implemented_handler(
        req: Request, exc: NotImplementedError
    ) -> JSONResponse:
        log.warning("not_implemented", path=str(req.url), error=str(exc))
        return JSONResponse(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            content=_error_body(
                code="NOT_IMPLEMENTED",
                message="This pipeline stage is not yet available.",
                detail=str(exc),
            ),
        )

    @app.exception_handler(Exception)
    async def generic_handler(req: Request, exc: Exception) -> JSONResponse:
        tb = traceback.format_exc()
        log.error(
            "unhandled_exception",
            path=str(req.url),
            error=str(exc),
            exc_type=type(exc).__name__,
            traceback=tb,
        )
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=_error_body(
                code="INTERNAL_SERVER_ERROR",
                message="An unexpected error occurred.",
            ),
        )