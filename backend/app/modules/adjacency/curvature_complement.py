# Copyright (c) 2026 Harsh Dwivedi
# Licensed under the Harsh Non-Commercial Attribution License (HNCAL) v1.0
# Commercial use requires written permission. See LICENSE for details.

"""
PieceWise — Curvature Complement Scorer (POST-Hungarian)

This is the FULL tab/blank curvature scoring stage.
It is called POST-Hungarian where all neighbor assignments are FIXED.
This is architecturally required to avoid the circular dependency
identified in the algorithm review:
  - Pre-Hungarian: only flat-side count used (no neighbor info)
  - Post-Hungarian: full profile comparison between fixed neighbors

For each adjacent pair, compare the curvature profile of the touching
sides between the two pieces.

Complementarity principle:
  A tab protrusion (positive curvature peak) should be adjacent to
  a blank indentation (negative curvature peak) in its neighbor.
  A flat side should be adjacent to a flat side (border pieces).

Score formula:
  1. Retrieve curvature vectors for the touching sides of both pieces
  2. Negate one vector (complementary profiles should anti-correlate)
  3. Pearson correlation of (side_a, -side_b) → score in [-1, 1]
  4. Rescale to [0, 1]: score = (correlation + 1) / 2

Special cases:
  - Both sides flat → score = 1.0 (correct border alignment)
  - Missing curvature profiles → neutral score = 0.5
"""

from __future__ import annotations

import numpy as np

from app.models.piece import CurvatureProfile, PieceCrop
from app.modules.adjacency.neighbor_extractor import NeighborPair
from app.utils.logger import get_logger

log = get_logger(__name__)


def _pearson_correlation(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute Pearson correlation between two equal-length vectors.
    Returns value in [-1, 1]. Returns 0.0 if either vector is constant.
    """
    if len(a) != len(b) or len(a) < 2:
        return 0.0

    a_centered = a - a.mean()
    b_centered = b - b.mean()

    std_a = np.std(a)
    std_b = np.std(b)

    if std_a < 1e-8 or std_b < 1e-8:
        # Both flat (constant) — check if both are near-zero curvature
        if std_a < 1e-8 and std_b < 1e-8:
            return 1.0  # Both flat → perfect complement for border sides
        return 0.0

    return float(np.dot(a_centered, b_centered) / (len(a) * std_a * std_b))


def _get_curvature_profile(
    piece: PieceCrop, side_index: int
) -> CurvatureProfile | None:
    """Return the curvature profile for a specific side of a piece."""
    for profile in piece.curvature_profiles:
        if profile.side_index == side_index:
            return profile
    return None


def curvature_complement_score(
    piece_a: PieceCrop,
    piece_b: PieceCrop,
    pair: NeighborPair,
) -> float:
    """
    Compute curvature complement score for a pair of adjacent pieces.

    A high score means the touching sides are geometrically compatible
    (tab fits blank, or both flat at a border).

    Args:
        piece_a: PieceCrop with curvature_profiles populated (Phase 3)
        piece_b: PieceCrop with curvature_profiles populated (Phase 3)
        pair:    NeighborPair with side_a and side_b indices

    Returns:
        Score in [0, 1] — higher = better geometric compatibility.
        Returns 0.5 (neutral) when profiles are missing.
    """
    profile_a = _get_curvature_profile(piece_a, pair.side_a)
    profile_b = _get_curvature_profile(piece_b, pair.side_b)

    # Missing profiles → neutral
    if profile_a is None or profile_b is None:
        return 0.5

    # Both flat sides → perfect border alignment
    if profile_a.is_flat and profile_b.is_flat:
        return 1.0

    # One flat, one not → weak mismatch (border vs interior side)
    if profile_a.is_flat != profile_b.is_flat:
        return 0.25

    # Both non-flat: compute complementarity
    vec_a = np.array(profile_a.curvature_vector, dtype=np.float32)
    vec_b = np.array(profile_b.curvature_vector, dtype=np.float32)

    if len(vec_a) == 0 or len(vec_b) == 0:
        return 0.5

    # Resample to same length if needed
    if len(vec_a) != len(vec_b):
        x_old = np.linspace(0, 1, len(vec_b))
        x_new = np.linspace(0, 1, len(vec_a))
        vec_b = np.interp(x_new, x_old, vec_b)

    # Negate one side — complementary profiles anti-correlate
    # Tab (positive peak) should pair with blank (negative peak)
    correlation = _pearson_correlation(vec_a, -vec_b)

    # Rescale from [-1, 1] to [0, 1]
    score = (correlation + 1.0) / 2.0
    return float(np.clip(score, 0.0, 1.0))


def score_all_pairs_curvature(
    pairs: list[NeighborPair],
    piece_map: dict[int, PieceCrop],
) -> dict[tuple[int, int], float]:
    """
    Compute curvature complement scores for all neighbor pairs.

    Returns:
        Dict mapping (min_piece_id, max_piece_id) → curvature score.
    """
    scores: dict[tuple[int, int], float] = {}

    for pair in pairs:
        pa = piece_map.get(pair.piece_id_a)
        pb = piece_map.get(pair.piece_id_b)

        if pa is None or pb is None:
            continue

        score = curvature_complement_score(pa, pb, pair)
        key = (min(pair.piece_id_a, pair.piece_id_b),
               max(pair.piece_id_a, pair.piece_id_b))
        scores[key] = score

    return scores