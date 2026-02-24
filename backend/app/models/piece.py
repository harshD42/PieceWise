# Copyright (c) 2026 Harsh Dwivedi
# Licensed under the Harsh Non-Commercial Attribution License (HNCAL) v1.0
# Commercial use requires written permission. See LICENSE for details.

"""
PieceWise — Piece Data Models
Pydantic models representing a puzzle piece through each stage
of the pipeline: raw crop → embedding → match → assembly step.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field


class PieceType(str, Enum):
    CORNER = "corner"
    EDGE = "edge"
    INTERIOR = "interior"
    UNKNOWN = "unknown"


class CurvatureProfile(BaseModel):
    """
    Encoded curvature for one side of a puzzle piece.
    Sampled to a fixed-length vector for fast comparison.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    side_index: int = Field(..., ge=0, le=3, description="0=top 1=right 2=bottom 3=left")
    curvature_vector: Any = Field(..., description="np.ndarray shape (32,) normalised")
    is_flat: bool = Field(..., description="True if this side is a border/straight edge")
    # Positive peak → tab protrusion; negative peak → blank indentation
    peak_value: float = Field(0.0, description="Signed peak curvature value")


class PieceCrop(BaseModel):
    """
    A single segmented puzzle piece as produced by the segmentation module.
    Carries both the raw image data and all classical CV descriptors needed
    by the feature extraction and matching stages.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    piece_id: int = Field(..., description="Sequential integer ID assigned at segmentation")
    # BGR numpy array (H×W×3)
    image: Any = Field(..., description="np.ndarray BGR crop of the piece")
    # Binary mask, same spatial dimensions as image crop
    alpha_mask: Any = Field(..., description="np.ndarray uint8 binary mask (0 or 255)")
    # (x, y, w, h) in the normalised pieces image coordinate space
    bbox: tuple[int, int, int, int] = Field(..., description="(x, y, w, h) bounding box")
    # OpenCV contour array (N, 1, 2) int32
    contour: Any = Field(..., description="np.ndarray OpenCV contour")

    # ── Shape descriptors ──
    area_px: float = Field(..., description="Contour pixel area")
    solidity: float = Field(..., description="contour_area / convex_hull_area")
    compactness: float = Field(..., description="4π·area / perimeter²")

    # ── Curvature encoding (4 sides) ──
    curvature_profiles: list[CurvatureProfile] = Field(
        default_factory=list,
        description="One CurvatureProfile per side (top/right/bottom/left)",
    )
    flat_side_count: int = Field(
        0, ge=0, le=4,
        description="Number of flat (border) sides detected on this piece",
    )

    # ── PCA orientation ──
    # Degrees by which the piece was rotated to align its principal axis
    # before DINOv2 embedding. Added back when reporting final rotation.
    pca_correction_deg: float = Field(
        0.0,
        description="PCA orientation correction applied before 4-rotation embedding",
    )


class PieceEmbedding(BaseModel):
    """
    DINOv2 spatial token grids for a single piece across 4 rotations.
    Token grids are stored as numpy arrays; actual torch tensors live
    in the GPU embedding store and are referenced by piece_id.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    piece_id: int
    # Maps rotation_deg → spatial token grid as numpy (H_tok × W_tok × D)
    # D = 128 if PCA enabled, 768 otherwise
    token_grids: dict[int, Any] = Field(
        default_factory=dict,
        description="rotation_deg → np.ndarray token grid",
    )
    # Maps rotation_deg → CLS vector as numpy (D,)
    cls_vectors: dict[int, Any] = Field(
        default_factory=dict,
        description="rotation_deg → np.ndarray CLS embedding",
    )


class CandidateMatch(BaseModel):
    """A single (grid_pos, rotation) candidate for a piece, with scores."""
    grid_pos: tuple[int, int] = Field(..., description="(row, col) in puzzle grid")
    rotation_deg: int = Field(..., description="Rotation in degrees (0/90/180/270)")
    spatial_score: float = Field(..., ge=0.0, le=1.0)
    flat_side_score: float = Field(..., ge=0.0, le=1.0)
    composite_score: float = Field(..., ge=0.0, le=1.0)


class PieceMatch(BaseModel):
    """
    Final assignment of a piece to a grid cell, produced by the matching engine
    after Hungarian resolution and adjacency refinement.
    """
    piece_id: int
    grid_pos: tuple[int, int] = Field(..., description="(row, col) in puzzle grid")
    rotation_deg: int = Field(..., description="Final rotation (0/90/180/270)")

    # ── Scores ──
    spatial_score: float = Field(0.0, ge=0.0, le=1.0)
    flat_side_score: float = Field(0.0, ge=0.0, le=1.0)
    composite_confidence: float = Field(0.0, ge=0.0, le=1.0)
    # Added by adjacency refiner (post-Hungarian)
    adjacency_score: float = Field(0.0, ge=0.0, le=1.0)
    curvature_complement_score: float = Field(0.0, ge=0.0, le=1.0)

    flagged: bool = Field(False, description="True if confidence below threshold")
    # Top-3 alternatives for flagged pieces (shown in human-in-the-loop UI)
    top3_candidates: list[CandidateMatch] = Field(default_factory=list)


class AssemblyStep(BaseModel):
    """
    A single step in the BFS-ordered assembly sequence.
    One step = one piece to physically place.
    """
    step_num: int = Field(..., ge=1)
    piece_id: int
    grid_pos: tuple[int, int]
    rotation_deg: int
    piece_type: PieceType = PieceType.UNKNOWN
    composite_confidence: float = Field(0.0, ge=0.0, le=1.0)
    adjacency_score: float = Field(0.0, ge=0.0, le=1.0)
    curvature_complement_score: float = Field(0.0, ge=0.0, le=1.0)
    flagged: bool = False