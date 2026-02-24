# Copyright (c) 2026 Harsh Dwivedi
# Licensed under the Harsh Non-Commercial Attribution License (HNCAL) v1.0
# Commercial use requires written permission. See LICENSE for details.

"""
PieceWise — Job State Models
Tracks lifecycle of a solve request from submission through completion.
Used by the abstract JobStore and all API status endpoints.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


class JobStage(str, Enum):
    """Pipeline stage labels — used for progress UI display."""
    QUEUED = "queued"
    PREPROCESSING = "preprocessing"
    SEGMENTATION = "segmentation"
    FEATURE_EXTRACTION = "feature_extraction"
    MATCHING = "matching"
    ADJACENCY_REFINEMENT = "adjacency_refinement"
    SEQUENCING = "sequencing"
    RENDERING = "rendering"
    DONE = "done"
    FAILED = "failed"


# Progress percentage at the START of each stage
STAGE_PROGRESS: dict[JobStage, int] = {
    JobStage.QUEUED: 0,
    JobStage.PREPROCESSING: 5,
    JobStage.SEGMENTATION: 10,
    JobStage.FEATURE_EXTRACTION: 35,
    JobStage.MATCHING: 55,
    JobStage.ADJACENCY_REFINEMENT: 70,
    JobStage.SEQUENCING: 78,
    JobStage.RENDERING: 85,
    JobStage.DONE: 100,
    JobStage.FAILED: 0,
}


class OutputBundle(BaseModel):
    """URLs to all generated output assets for a completed job."""
    overlay_reference_url: str
    overlay_pieces_url: str
    solution_manifest_url: str
    step_card_urls: list[str] = Field(default_factory=list)
    total_pieces: int = 0
    flagged_count: int = 0
    mean_confidence: float = 0.0


class Job(BaseModel):
    """Full job state record stored in JobStore."""
    job_id: str
    status: JobStatus = JobStatus.PENDING
    stage: JobStage = JobStage.QUEUED
    progress: int = Field(0, ge=0, le=100)
    error: Optional[str] = None
    result: Optional[OutputBundle] = None
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    def to_status_response(self) -> dict:
        """Serialise to the shape returned by GET /status/{job_id}."""
        resp = {
            "job_id": self.job_id,
            "status": self.status.value,
            "stage": self.stage.value,
            "progress": self.progress,
            "error": self.error,
        }
        if self.result:
            resp["result"] = self.result.model_dump()
        return resp


# ─── API Request/Response Schemas ────────────────────────────────────────────

class SolveResponse(BaseModel):
    """Response body for POST /solve."""
    job_id: str
    status: JobStatus = JobStatus.PENDING
    message: str = "Puzzle solve job created. Poll /status/{job_id} for progress."


class CorrectionRequest(BaseModel):
    """Request body for PATCH /solve/{job_id}/correct."""
    piece_id: int = Field(..., description="ID of the piece to correct")
    corrected_grid_pos: tuple[int, int] = Field(
        ..., description="User-selected correct (row, col) grid position"
    )


class CorrectionResponse(BaseModel):
    """Response body for PATCH /solve/{job_id}/correct."""
    job_id: str
    piece_id: int
    corrected_grid_pos: tuple[int, int]
    message: str = "Correction applied. Sequencing and rendering re-running."