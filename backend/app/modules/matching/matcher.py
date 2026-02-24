# Copyright (c) 2026 Harsh Dwivedi
# Licensed under the Harsh Non-Commercial Attribution License (HNCAL) v1.0
# Commercial use requires written permission. See LICENSE for details.

"""
PieceWise — Matching Engine Orchestrator
Wires all matching sub-modules into the complete 5-stage pipeline:

  Stage 1 (COARSE): CLS cosine similarity → top-K shortlist
  Stage 2 (FINE):   Spatial torch.mm correlation on shortlist
  Stage 3:          Flat-side constraint scoring (pre-Hungarian, no neighbors)
  Stage 4:          Hungarian optimal 1:1 assignment
  Stage 5:          Confidence scoring + low-confidence flagging

Called from pipeline._match_pieces().
"""

from __future__ import annotations

from app.models.piece import PieceCrop, PieceMatch
from app.modules.feature_extraction.dino_loader import get_device
from app.modules.feature_extraction.embedding_store import EmbeddingStore
from app.modules.matching.candidate_selector import select_candidates
from app.modules.matching.confidence_scorer import (
    compute_confidence_stats,
    flag_low_confidence,
)
from app.modules.matching.conflict_resolver import resolve
from app.modules.matching.flat_side_scorer import score_all_candidates
from app.utils.logger import get_logger

log = get_logger(__name__)


def match_pieces(
    emb_store: EmbeddingStore,
    pieces: list[PieceCrop],
    grid_shape: tuple[int, int],
) -> list[PieceMatch]:
    """
    Run the complete coarse-to-fine matching pipeline.

    Args:
        emb_store:  Populated EmbeddingStore (reference + piece embeddings)
        pieces:     All PieceCrop objects (carry flat_side_count for scoring)
        grid_shape: (n_rows, n_cols) puzzle grid dimensions

    Returns:
        List of PieceMatch — one per piece, fully scored and flagged.
    """
    device = get_device()

    log.info(
        "matching_pipeline_start",
        n_pieces=len(pieces),
        grid_shape=grid_shape,
        device=device,
    )

    # Stage 1+2: Coarse CLS filter → Fine spatial correlation
    candidates = select_candidates(emb_store, device=device)

    # Stage 3: Flat-side constraint scoring (pre-Hungarian, no neighbors)
    scored = score_all_candidates(pieces, candidates, grid_shape)

    # Stage 4: Hungarian optimal 1:1 assignment
    matches = resolve(pieces, scored, grid_shape)

    # Stage 5: Confidence flagging
    matches = flag_low_confidence(matches)

    # Log summary stats
    stats = compute_confidence_stats(matches)

    log.info(
        "matching_pipeline_complete",
        total_matches=len(matches),
        mean_confidence=round(stats["mean"], 3),
        flagged=stats["flagged_count"],
    )

    return matches