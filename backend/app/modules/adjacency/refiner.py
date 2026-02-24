# Copyright (c) 2026 Harsh Dwivedi
# Licensed under the Harsh Non-Commercial Attribution License (HNCAL) v1.0
# Commercial use requires written permission. See LICENSE for details.

"""
PieceWise — Adjacency Refiner Orchestrator
Wires the 4 adjacency refinement stages in sequence:

  Stage 1: Extract all neighbor pairs (post-Hungarian, fixed neighbors)
  Stage 2: Score all pairs via edge color histogram comparison
  Stage 3: Score all pairs via curvature complement analysis
  Stage 4: Run local swap engine to improve low-scoring assignments

All operations are POST-Hungarian and therefore non-circular.
Called from pipeline._refine_adjacency().
"""

from __future__ import annotations

from app.models.piece import PieceCrop, PieceMatch
from app.modules.adjacency.neighbor_extractor import extract_neighbor_pairs
from app.modules.adjacency.swap_engine import run_swap_engine
from app.utils.logger import get_logger

log = get_logger(__name__)


def refine_adjacency(
    matches: list[PieceMatch],
    pieces: list[PieceCrop],
    grid_shape: tuple[int, int],
) -> list[PieceMatch]:
    """
    Run the complete adjacency refinement pipeline.

    Args:
        matches:    Post-Hungarian PieceMatch list from Phase 5
        pieces:     All PieceCrop objects (carry image + curvature profiles)
        grid_shape: (n_rows, n_cols)

    Returns:
        Refined PieceMatch list with adjacency_score and
        curvature_complement_score populated on each match.
    """
    log.info(
        "adjacency_refinement_start",
        n_pieces=len(matches),
        grid_shape=grid_shape,
    )

    # Stages 1–4 are all orchestrated inside run_swap_engine
    # which internally calls neighbor_extractor, edge_histogram,
    # curvature_complement, and then the swap loop
    refined = run_swap_engine(matches, pieces, grid_shape)

    # Log summary — guard against empty input
    if refined:
        adj_scores = [m.adjacency_score for m in refined]
        curv_scores = [m.curvature_complement_score for m in refined]

        import numpy as np
        log.info(
            "adjacency_refinement_complete",
            mean_adjacency=round(float(np.mean(adj_scores)), 3),
            mean_curvature_complement=round(float(np.mean(curv_scores)), 3),
            total_pieces=len(refined),
        )
    else:
        log.info("adjacency_refinement_complete", total_pieces=0)

    return refined