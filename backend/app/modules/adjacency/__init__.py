# Copyright (c) 2026 Harsh Dwivedi
# Licensed under the Harsh Non-Commercial Attribution License (HNCAL) v1.0
# Commercial use requires written permission. See LICENSE for details.

"""
PieceWise — Adjacency Refiner Module
Public API for the post-Hungarian adjacency refinement stage.
"""

from app.modules.adjacency.curvature_complement import (
    curvature_complement_score,
    score_all_pairs_curvature,
)
from app.modules.adjacency.edge_histogram import (
    edge_histogram_score,
    score_all_pairs_histogram,
)
from app.modules.adjacency.neighbor_extractor import (
    NeighborPair,
    build_piece_neighbor_map,
    extract_neighbor_pairs,
)
from app.modules.adjacency.refiner import refine_adjacency
from app.modules.adjacency.swap_engine import run_swap_engine

__all__ = [
    # Neighbor extraction
    "NeighborPair",
    "extract_neighbor_pairs",
    "build_piece_neighbor_map",
    # Edge histogram
    "edge_histogram_score",
    "score_all_pairs_histogram",
    # Curvature complement
    "curvature_complement_score",
    "score_all_pairs_curvature",
    # Swap engine
    "run_swap_engine",
    # Orchestrator
    "refine_adjacency",
]