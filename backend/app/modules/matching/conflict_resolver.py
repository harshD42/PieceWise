# Copyright (c) 2026 Harsh Dwivedi
# Licensed under the Harsh Non-Commercial Attribution License (HNCAL) v1.0
# Commercial use requires written permission. See LICENSE for details.

"""
PieceWise — Hungarian Conflict Resolver
Assigns each piece to exactly one grid cell using the Hungarian algorithm
(scipy.optimize.linear_sum_assignment) for optimal global assignment.

Without this, greedy assignment collapses: multiple pieces fight over
the same high-confidence cell and weaker cells go unassigned.
Hungarian guarantees a 1:1 bijection that minimises total assignment cost.

Cost formula (pre-Hungarian composite):
  composite_score = 0.7 × spatial_score + 0.3 × flat_side_score
  cost = 1.0 - composite_score

The spatial score is the primary signal (70%) — it encodes visual similarity.
The flat-side score is a hard structural constraint (30%) — it encodes
puzzle geometry and cannot be derived from image content alone.

Grid cells with no candidate match get cost = 1.0 (maximum cost).
This allows Hungarian to handle puzzles where piece_count < total_cells
gracefully — unmatched cells are simply assigned the "worst" pieces.

After Hungarian, the winning rotation is the rotation with the best
composite score among the 4 rotations tested per piece.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

from app.models.piece import CandidateMatch, PieceCrop, PieceMatch
from app.utils.logger import get_logger

log = get_logger(__name__)

# Composite score weights — must sum to 1.0
_W_SPATIAL = 0.70
_W_FLAT_SIDE = 0.30


def _composite_score(spatial: float, flat_side: float) -> float:
    """Compute weighted composite score."""
    return _W_SPATIAL * spatial + _W_FLAT_SIDE * flat_side


def build_cost_matrix(
    pieces: list[PieceCrop],
    scored_candidates: dict[int, dict[int, list[tuple[tuple[int, int], float, float, float]]]],
    grid_shape: tuple[int, int],
) -> tuple[np.ndarray, list[int], list[tuple[int, int]]]:
    """
    Build the cost matrix for Hungarian assignment.

    Args:
        pieces:            All PieceCrop objects
        scored_candidates: piece_id → rot → [(grid_pos, coarse, fine, flat_side), ...]
        grid_shape:        (n_rows, n_cols)

    Returns:
        (cost_matrix, piece_ids, cell_positions)
        cost_matrix:    (n_pieces, n_cells) float32 — lower = better match
        piece_ids:      Ordered list of piece IDs (row index mapping)
        cell_positions: Ordered list of grid_pos (col index mapping)
    """
    n_rows, n_cols = grid_shape
    n_cells = n_rows * n_cols

    # Build ordered index lists
    piece_ids = [p.piece_id for p in pieces]
    cell_positions = [(r, c) for r in range(n_rows) for c in range(n_cols)]
    cell_to_col = {pos: i for i, pos in enumerate(cell_positions)}

    n_pieces = len(piece_ids)
    # Initialise with maximum cost — cells with no candidate get cost=1.0
    cost_matrix = np.ones((n_pieces, n_cells), dtype=np.float32)

    for row_idx, pid in enumerate(piece_ids):
        rot_dict = scored_candidates.get(pid, {})

        for rot_deg, candidate_list in rot_dict.items():
            for grid_pos, coarse, fine, flat_s in candidate_list:
                col_idx = cell_to_col.get(grid_pos)
                if col_idx is None:
                    continue

                comp = _composite_score(fine, flat_s)
                cost = 1.0 - comp

                # Keep the best (lowest cost) rotation for this cell
                if cost < cost_matrix[row_idx, col_idx]:
                    cost_matrix[row_idx, col_idx] = cost

    return cost_matrix, piece_ids, cell_positions


def resolve(
    pieces: list[PieceCrop],
    scored_candidates: dict[int, dict[int, list[tuple[tuple[int, int], float, float, float]]]],
    grid_shape: tuple[int, int],
) -> list[PieceMatch]:
    """
    Run Hungarian algorithm and produce final PieceMatch assignments.

    For each assigned (piece, cell) pair:
      - Find the best rotation from the candidate list
      - Compute final scores
      - Store top-3 candidates for human-in-the-loop UI (used in Phase 8)

    Args:
        pieces:            All PieceCrop objects from segmentation
        scored_candidates: Scored candidates from flat_side_scorer
        grid_shape:        (n_rows, n_cols)

    Returns:
        List of PieceMatch objects — one per piece, with grid_pos assigned.
    """
    cost_matrix, piece_ids, cell_positions = build_cost_matrix(
        pieces, scored_candidates, grid_shape
    )

    log.info(
        "hungarian_start",
        n_pieces=len(piece_ids),
        n_cells=len(cell_positions),
        cost_matrix_shape=cost_matrix.shape,
    )

    # Run Hungarian algorithm
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    log.info("hungarian_complete", assigned_pairs=len(row_ind))

    # Build PieceMatch objects from assignments
    matches: list[PieceMatch] = []
    piece_id_to_row = {pid: i for i, pid in enumerate(piece_ids)}

    for row, col in zip(row_ind, col_ind):
        pid = piece_ids[row]
        grid_pos = cell_positions[col]
        assignment_cost = float(cost_matrix[row, col])
        composite = 1.0 - assignment_cost

        # Find the best rotation for this (piece, cell) assignment
        best_rot, best_spatial, best_flat = _find_best_rotation(
            pid, grid_pos, scored_candidates
        )

        # Gather top-3 candidates across all rotations for this piece
        top3 = _gather_top3(pid, scored_candidates)

        match = PieceMatch(
            piece_id=pid,
            grid_pos=grid_pos,
            rotation_deg=best_rot,
            spatial_score=best_spatial,
            flat_side_score=best_flat,
            composite_confidence=composite,
            top3_candidates=top3,
        )
        matches.append(match)

    log.info(
        "conflict_resolution_complete",
        total_matches=len(matches),
    )

    return matches


def _find_best_rotation(
    piece_id: int,
    grid_pos: tuple[int, int],
    scored_candidates: dict,
) -> tuple[int, float, float]:
    """
    For an assigned (piece, cell), find the rotation with the best
    composite score for that specific cell.

    Returns: (best_rotation_deg, spatial_score, flat_side_score)
    """
    best_rot = 0
    best_comp = -1.0
    best_spatial = 0.0
    best_flat = 0.0

    rot_dict = scored_candidates.get(piece_id, {})
    for rot_deg, candidate_list in rot_dict.items():
        for gpos, coarse, fine, flat_s in candidate_list:
            if gpos == grid_pos:
                comp = _composite_score(fine, flat_s)
                if comp > best_comp:
                    best_comp = comp
                    best_rot = rot_deg
                    best_spatial = fine
                    best_flat = flat_s
                break

    return best_rot, best_spatial, best_flat


def _gather_top3(
    piece_id: int,
    scored_candidates: dict,
) -> list[CandidateMatch]:
    """
    Collect the top-3 candidate cells across all rotations for a piece.
    Used for the human-in-the-loop UI on flagged pieces.
    """
    all_candidates: list[tuple[float, tuple[int, int], int, float, float]] = []

    rot_dict = scored_candidates.get(piece_id, {})
    for rot_deg, candidate_list in rot_dict.items():
        for grid_pos, coarse, fine, flat_s in candidate_list:
            comp = _composite_score(fine, flat_s)
            all_candidates.append((comp, grid_pos, rot_deg, fine, flat_s))

    # Sort by composite score descending
    all_candidates.sort(key=lambda x: x[0], reverse=True)

    # Deduplicate by grid_pos (keep best rotation per cell)
    seen_cells: set[tuple[int, int]] = set()
    top3: list[CandidateMatch] = []

    for comp, grid_pos, rot_deg, fine, flat_s in all_candidates:
        if grid_pos in seen_cells:
            continue
        seen_cells.add(grid_pos)
        top3.append(CandidateMatch(
            grid_pos=grid_pos,
            rotation_deg=rot_deg,
            spatial_score=fine,
            flat_side_score=flat_s,
            composite_score=comp,
        ))
        if len(top3) == 3:
            break

    return top3