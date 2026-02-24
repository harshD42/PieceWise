# Copyright (c) 2026 Harsh Dwivedi
# Licensed under the Harsh Non-Commercial Attribution License (HNCAL) v1.0
# Commercial use requires written permission. See LICENSE for details.

"""
PieceWise — Output / Manifest Models
Defines the structure of solution.json — the complete output
artifact consumed by the frontend solution viewer.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field

from app.models.piece import AssemblyStep, CandidateMatch, PieceType


class StepManifestEntry(BaseModel):
    """
    One entry in solution.json steps array.
    Extends AssemblyStep with asset URLs for the frontend.
    """
    step_num: int
    piece_id: int
    grid_pos: tuple[int, int]
    rotation_deg: int
    piece_type: PieceType
    composite_confidence: float
    adjacency_score: float
    curvature_complement_score: float
    flagged: bool

    # Asset URLs served by GET /assets/{job_id}/...
    piece_crop_url: str
    step_card_url: str

    # Only populated for flagged pieces — drives human-in-the-loop UI
    top3_candidates: list[CandidateMatch] = Field(default_factory=list)


class SolutionManifest(BaseModel):
    """
    Complete solution.json structure.
    Written to disk by the manifest builder and served as a static asset.
    Consumed by the frontend to render the full solution viewer.
    """
    job_id: str
    grid_shape: tuple[int, int] = Field(..., description="(n_rows, n_cols)")
    total_pieces: int
    flagged_count: int
    mean_confidence: float
    min_confidence: float
    max_confidence: float

    steps: list[StepManifestEntry] = Field(default_factory=list)

    asset_urls: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Named asset URLs: overlay_reference, overlay_pieces, "
            "solution_manifest, step_cards array"
        ),
    )

    # Piece type summary — used by frontend for progress indicators
    corner_count: int = 0
    edge_count: int = 0
    interior_count: int = 0


class PieceCropManifestEntry(BaseModel):
    """
    Metadata for a single piece crop image stored on disk.
    Used internally by the rendering module.
    """
    piece_id: int
    crop_url: str
    bbox: tuple[int, int, int, int]
    rotation_deg: int
    piece_type: PieceType = PieceType.UNKNOWN
    flagged: bool = False