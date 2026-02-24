# Copyright (c) 2026 Harsh Dwivedi
# Licensed under the Harsh Non-Commercial Attribution License (HNCAL) v1.0
# Commercial use requires written permission. See LICENSE for details.

"""
PieceWise — Flat-Side Constraint Scorer (Pre-Hungarian)

This scorer enforces the border constraint:
  - Corner cells (2 border sides) → piece must have flat_side_count == 2
  - Edge cells   (1 border side)  → piece must have flat_side_count == 1
  - Interior cells (0 border sides) → piece must have flat_side_count == 0

CRITICAL DESIGN NOTE (from architecture review):
  This is the ONLY curvature signal used BEFORE Hungarian assignment.
  It uses ONLY the piece's own flat_side_count — NO neighbor information.
  Full tab/blank complement scoring is deferred to the adjacency refiner
  (Phase 6) where neighbor assignments are fixed and non-circular.

  Violating this would introduce a feedback loop:
    curvature score → influences cost matrix → influences assignment →
    determines neighbors → was used to compute curvature score → circular.

Score formula:
  score = 1.0                  if flat_side_count matches exactly
  score = 1.0 - 0.5 * |diff|  for each unit of mismatch (floor at 0.0)

This gives a graceful penalty rather than hard rejection, because
contour analysis can miscount flat sides (noisy edges, touching pieces).
"""

from __future__ import annotations

from app.models.piece import PieceCrop
from app.utils.geometry_utils import expected_flat_sides
from app.utils.logger import get_logger

log = get_logger(__name__)

# Penalty per unit of flat_side mismatch
_MISMATCH_PENALTY = 0.5


def flat_side_score(
    piece: PieceCrop,
    grid_pos: tuple[int, int],
    grid_shape: tuple[int, int],
) -> float:
    """
    Compute the flat-side constraint score for placing piece at grid_pos.

    Args:
        piece:       PieceCrop with flat_side_count populated by Phase 3
        grid_pos:    (row, col) candidate grid position
        grid_shape:  (n_rows, n_cols) of the puzzle grid

    Returns:
        Score in [0.0, 1.0] — 1.0 = perfect match, 0.0 = maximum mismatch.
    """
    n_rows, n_cols = grid_shape
    row, col = grid_pos

    expected = expected_flat_sides(row, col, n_rows, n_cols)
    actual = piece.flat_side_count
    diff = abs(expected - actual)

    score = max(0.0, 1.0 - _MISMATCH_PENALTY * diff)

    return score


def score_all_candidates(
    pieces: list[PieceCrop],
    candidates: dict[int, dict[int, list[tuple[tuple[int, int], float, float]]]],
    grid_shape: tuple[int, int],
) -> dict[int, dict[int, list[tuple[tuple[int, int], float, float, float]]]]:
    """
    Augment candidate lists with flat-side scores.

    Args:
        pieces:     All PieceCrop objects (used to look up flat_side_count)
        candidates: Output of candidate_selector.select_candidates()
                    piece_id → rot_deg → [(grid_pos, coarse, fine), ...]
        grid_shape: (n_rows, n_cols)

    Returns:
        Same structure with flat_side_score appended:
        piece_id → rot_deg → [(grid_pos, coarse, fine, flat_side), ...]
    """
    piece_map = {p.piece_id: p for p in pieces}

    scored: dict[int, dict[int, list[tuple[tuple[int, int], float, float, float]]]] = {}

    for piece_id, rot_dict in candidates.items():
        piece = piece_map.get(piece_id)
        if piece is None:
            continue

        scored[piece_id] = {}
        for rot_deg, candidate_list in rot_dict.items():
            augmented = []
            for grid_pos, coarse, fine in candidate_list:
                fs = flat_side_score(piece, grid_pos, grid_shape)
                augmented.append((grid_pos, coarse, fine, fs))
            scored[piece_id][rot_deg] = augmented

    return scored