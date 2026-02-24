# Copyright (c) 2026 Harsh Dwivedi
# Licensed under the Harsh Non-Commercial Attribution License (HNCAL) v1.0
# Commercial use requires written permission. See LICENSE for details.

"""
PieceWise — Matching Engine Module
Public API for the coarse-to-fine matching pipeline.
"""

from app.modules.matching.candidate_selector import select_candidates
from app.modules.matching.confidence_scorer import (
    compute_confidence_stats,
    flag_low_confidence,
)
from app.modules.matching.conflict_resolver import (
    build_cost_matrix,
    resolve,
)
from app.modules.matching.flat_side_scorer import (
    flat_side_score,
    score_all_candidates,
)
from app.modules.matching.matcher import match_pieces
from app.modules.matching.similarity import (
    batch_spatial_similarity,
    spatial_similarity,
)

__all__ = [
    # Similarity
    "spatial_similarity",
    "batch_spatial_similarity",
    # Candidate selection
    "select_candidates",
    # Flat-side scoring
    "flat_side_score",
    "score_all_candidates",
    # Hungarian resolver
    "build_cost_matrix",
    "resolve",
    # Confidence
    "flag_low_confidence",
    "compute_confidence_stats",
    # Orchestrator
    "match_pieces",
]