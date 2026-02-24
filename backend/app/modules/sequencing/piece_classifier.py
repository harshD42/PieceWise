# Copyright (c) 2026 Harsh Dwivedi
# Licensed under the Harsh Non-Commercial Attribution License (HNCAL) v1.0
# Commercial use requires written permission. See LICENSE for details.

"""
PieceWise — Piece Classifier
Classifies each matched piece as CORNER, EDGE, or INTERIOR based on
its assigned grid position, cross-validated with its flat_side_count
from Phase 3 curvature analysis.

Primary source of truth: grid position (reliable, derived from matching)
Secondary validation: flat_side_count (noisy, derived from contour analysis)

Cross-validation logic:
  - Corner cell expects flat_side_count == 2
  - Edge cell expects flat_side_count == 1
  - Interior cell expects flat_side_count == 0
  If mismatch: log a warning, trust grid position over curvature.
  Curvature analysis can miscount due to image noise, piece orientation,
  or segmentation artifacts at 1000-piece resolution.

This classification is used by the BFS assembler for ordering priority
and by the step generator for display in the solution viewer.
"""

from __future__ import annotations

from app.models.piece import AssemblyStep, PieceCrop, PieceMatch, PieceType
from app.utils.geometry_utils import (
    expected_flat_sides,
    is_corner_pos,
    is_edge_pos,
)
from app.utils.logger import get_logger

log = get_logger(__name__)


def classify_piece(
    match: PieceMatch,
    grid_shape: tuple[int, int],
) -> PieceType:
    """
    Classify a single piece based on its assigned grid position.

    Args:
        match:       PieceMatch with grid_pos set
        grid_shape:  (n_rows, n_cols)

    Returns:
        PieceType enum value
    """
    n_rows, n_cols = grid_shape
    row, col = match.grid_pos

    if is_corner_pos(row, col, n_rows, n_cols):
        return PieceType.CORNER
    if is_edge_pos(row, col, n_rows, n_cols):
        return PieceType.EDGE
    return PieceType.INTERIOR


def classify_and_validate(
    matches: list[PieceMatch],
    pieces: list[PieceCrop],
    grid_shape: tuple[int, int],
) -> dict[int, PieceType]:
    """
    Classify all pieces and cross-validate with flat_side_count.

    Args:
        matches:    List of PieceMatch with grid positions assigned
        pieces:     List of PieceCrop with flat_side_count from Phase 3
        grid_shape: (n_rows, n_cols)

    Returns:
        Dict mapping piece_id → PieceType
    """
    piece_map = {p.piece_id: p for p in pieces}
    classifications: dict[int, PieceType] = {}

    corner_count = edge_count = interior_count = mismatch_count = 0

    for match in matches:
        piece_type = classify_piece(match, grid_shape)
        piece = piece_map.get(match.piece_id)

        # Cross-validate with flat_side_count if curvature profiles exist
        if piece is not None and len(piece.curvature_profiles) > 0:
            expected = expected_flat_sides(*match.grid_pos, *grid_shape)
            actual = piece.flat_side_count

            if expected != actual:
                mismatch_count += 1
                log.debug(
                    "flat_side_mismatch",
                    piece_id=match.piece_id,
                    grid_pos=match.grid_pos,
                    piece_type=piece_type.value,
                    expected_flat_sides=expected,
                    actual_flat_sides=actual,
                )
                # Trust grid position — do NOT change classification

        classifications[match.piece_id] = piece_type

        if piece_type == PieceType.CORNER:
            corner_count += 1
        elif piece_type == PieceType.EDGE:
            edge_count += 1
        else:
            interior_count += 1

    log.info(
        "piece_classification_complete",
        corners=corner_count,
        edges=edge_count,
        interior=interior_count,
        mismatches=mismatch_count,
        total=len(matches),
    )

    return classifications