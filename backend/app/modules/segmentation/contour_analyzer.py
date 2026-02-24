# Copyright (c) 2026 Harsh Dwivedi
# Licensed under the Harsh Non-Commercial Attribution License (HNCAL) v1.0
# Commercial use requires written permission. See LICENSE for details.

"""
PieceWise — Contour Analyzer
Enriches PieceCrop objects with curvature profiles and flat_side_count.

Pipeline:
  1. Find 4 dominant corner points on the piece contour
     (Douglas-Peucker simplification + corner detection)
  2. Segment the contour into 4 sides between corner points
  3. For each side: compute discrete curvature, sample to 32-point vector
  4. Classify each side: flat (border) / tab (protrusion) / blank (indentation)
  5. Count flat sides → flat_side_count (0=interior, 1=edge, 2=corner piece)

Pre-Hungarian use: flat_side_count is computed here with NO neighbor dependency.
Full tab/blank complement scoring happens in the adjacency refiner (Phase 6).
"""

from __future__ import annotations

import cv2
import numpy as np

from app.models.piece import CurvatureProfile, PieceCrop
from app.utils.logger import get_logger

log = get_logger(__name__)

# Number of points to sample per side for the curvature vector
_CURVATURE_SAMPLES = 32

# Flat side detection threshold
# A side is "flat" if |mean_curvature| < this value
_FLAT_CURVATURE_THRESHOLD = 0.15

# Douglas-Peucker epsilon as fraction of contour perimeter
# Controls how aggressively the contour is simplified to find corners
_DP_EPSILON_FRACTION = 0.02

# Expected number of sides for a puzzle piece
_EXPECTED_SIDES = 4


def _find_corner_points(
    contour: np.ndarray,
    n_sides: int = _EXPECTED_SIDES,
) -> np.ndarray:
    """
    Find dominant corner points on a puzzle piece contour.

    Strategy:
      1. Simplify contour with Douglas-Peucker
      2. If simplified contour has exactly n_sides points, use those
      3. Otherwise: find the n_sides points with highest corner angle
         using the simplified contour as a guide

    Args:
        contour: OpenCV contour array (N, 1, 2) int32
        n_sides: Expected number of corners (4 for a puzzle piece)

    Returns:
        Array of n_sides corner points (n_sides, 2) int32
    """
    perimeter = cv2.arcLength(contour, closed=True)
    epsilon = _DP_EPSILON_FRACTION * perimeter

    # Simplify contour
    approx = cv2.approxPolyDP(contour, epsilon, closed=True)
    pts = approx.reshape(-1, 2)

    if len(pts) == n_sides:
        return pts

    # If too many or too few points: find n_sides maximally-separated points
    # by computing the corner angle at each contour point and picking peaks
    contour_pts = contour.reshape(-1, 2)
    n = len(contour_pts)

    if n < n_sides:
        # Not enough points — return evenly spaced
        indices = np.linspace(0, n - 1, n_sides, dtype=int)
        return contour_pts[indices]

    # Compute corner angles
    angles = np.zeros(n)
    for i in range(n):
        p_prev = contour_pts[(i - 1) % n].astype(float)
        p_curr = contour_pts[i].astype(float)
        p_next = contour_pts[(i + 1) % n].astype(float)

        v1 = p_prev - p_curr
        v2 = p_next - p_curr
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)

        if norm1 > 0 and norm2 > 0:
            cos_a = np.clip(np.dot(v1, v2) / (norm1 * norm2), -1.0, 1.0)
            angles[i] = np.degrees(np.arccos(cos_a))

    # Find n_sides peaks (high angle = sharp corner)
    # Use non-maximum suppression with minimum separation
    min_sep = n // (n_sides * 2)
    corner_indices = []
    remaining_angles = angles.copy()

    for _ in range(n_sides):
        idx = int(np.argmax(remaining_angles))
        corner_indices.append(idx)
        # Suppress neighbours
        for j in range(max(0, idx - min_sep), min(n, idx + min_sep + 1)):
            remaining_angles[j] = 0.0

    corner_indices.sort()
    return contour_pts[corner_indices]


def _segment_contour_into_sides(
    contour: np.ndarray,
    corner_points: np.ndarray,
) -> list[np.ndarray]:
    """
    Segment a closed contour into sides between corner points.

    Finds the contour index closest to each corner point and splits
    the contour array into n_sides sub-arrays.

    Returns:
        List of contour segments, one per side.
        Each segment is an (M, 2) float array of contour points.
    """
    pts = contour.reshape(-1, 2).astype(float)
    n = len(pts)

    # Find contour index closest to each corner point
    corner_indices = []
    for cp in corner_points:
        dists = np.linalg.norm(pts - cp.astype(float), axis=1)
        corner_indices.append(int(np.argmin(dists)))

    corner_indices.sort()

    # Split contour at corner indices (wrap around for last segment)
    sides = []
    n_corners = len(corner_indices)
    for i in range(n_corners):
        start = corner_indices[i]
        end = corner_indices[(i + 1) % n_corners]
        if end > start:
            side = pts[start:end + 1]
        else:
            # Wrap around
            side = np.vstack([pts[start:], pts[:end + 1]])
        sides.append(side)

    return sides


def _compute_discrete_curvature(side_pts: np.ndarray) -> np.ndarray:
    """
    Compute discrete signed curvature along a contour side.

    For each interior point: curvature ≈ signed angle between successive
    tangent vectors. Positive = left turn (tab), Negative = right turn (blank).

    Returns:
        1D float array of curvature values (length = len(side_pts) - 2)
    """
    if len(side_pts) < 3:
        return np.zeros(1)

    curvatures = []
    for i in range(1, len(side_pts) - 1):
        p0 = side_pts[i - 1]
        p1 = side_pts[i]
        p2 = side_pts[i + 1]

        v1 = p1 - p0
        v2 = p2 - p1

        # Signed angle (cross product gives sign)
        cross = v1[0] * v2[1] - v1[1] * v2[0]
        dot = v1[0] * v2[0] + v1[1] * v2[1]
        angle = np.arctan2(cross, dot)
        curvatures.append(angle)

    return np.array(curvatures)


def _sample_curvature(curvature: np.ndarray, n_samples: int) -> np.ndarray:
    """
    Resample curvature vector to a fixed length via linear interpolation.
    This ensures all curvature profiles have the same dimensionality
    regardless of contour resolution.
    """
    if len(curvature) == 0:
        return np.zeros(n_samples)
    if len(curvature) == n_samples:
        return curvature.copy()

    x_old = np.linspace(0, 1, len(curvature))
    x_new = np.linspace(0, 1, n_samples)
    return np.interp(x_new, x_old, curvature)


def _classify_side(curvature_sampled: np.ndarray) -> tuple[bool, float]:
    """
    Classify a side as flat, tab, or blank based on its curvature profile.

    Returns:
        (is_flat, peak_value)
        is_flat:    True if the side is a straight border edge
        peak_value: Signed peak curvature — positive=tab, negative=blank
    """
    mean_curv = float(np.mean(np.abs(curvature_sampled)))
    peak = float(curvature_sampled[np.argmax(np.abs(curvature_sampled))])
    is_flat = mean_curv < _FLAT_CURVATURE_THRESHOLD
    return is_flat, peak


def analyze_contours(pieces: list[PieceCrop]) -> list[PieceCrop]:
    """
    Enrich PieceCrop objects with curvature profiles and flat_side_count.

    Modifies pieces in-place (also returns list for convenience).

    Args:
        pieces: List of PieceCrop objects from piece_extractor

    Returns:
        Same list with curvature_profiles and flat_side_count populated.
    """
    for piece in pieces:
        try:
            _analyze_single(piece)
        except Exception as e:
            log.warning(
                "contour_analysis_failed",
                piece_id=piece.piece_id,
                error=str(e),
            )
            # On failure: leave curvature_profiles empty, flat_side_count=0
            # Piece will have low confidence but won't crash the pipeline

    flat_counts = [p.flat_side_count for p in pieces]
    log.info(
        "contour_analysis_complete",
        total_pieces=len(pieces),
        corners=flat_counts.count(2),
        edges=flat_counts.count(1),
        interior=flat_counts.count(0),
    )

    return pieces


def _analyze_single(piece: PieceCrop) -> None:
    """Analyze a single piece contour and populate its curvature fields."""
    contour = piece.contour

    # Find 4 corner points
    corner_pts = _find_corner_points(contour, n_sides=_EXPECTED_SIDES)

    if len(corner_pts) < 2:
        return

    # Segment contour into sides
    sides = _segment_contour_into_sides(contour, corner_pts)

    profiles: list[CurvatureProfile] = []
    flat_count = 0

    for side_idx, side_pts in enumerate(sides):
        # Compute curvature
        raw_curv = _compute_discrete_curvature(side_pts)

        # Normalise: subtract mean to remove global rotation bias
        if len(raw_curv) > 0:
            raw_curv = raw_curv - raw_curv.mean()

        # Sample to fixed length
        sampled = _sample_curvature(raw_curv, _CURVATURE_SAMPLES)

        # Classify
        is_flat, peak = _classify_side(sampled)

        if is_flat:
            flat_count += 1

        profile = CurvatureProfile(
            side_index=side_idx,
            curvature_vector=sampled,
            is_flat=is_flat,
            peak_value=peak,
        )
        profiles.append(profile)

    piece.curvature_profiles = profiles
    piece.flat_side_count = flat_count