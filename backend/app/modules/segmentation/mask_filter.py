# Copyright (c) 2026 Harsh Dwivedi
# Licensed under the Harsh Non-Commercial Attribution License (HNCAL) v1.0
# Commercial use requires written permission. See LICENSE for details.

"""
PieceWise — Mask Filter
Filters raw SAM masks to retain only valid puzzle piece candidates.

Filters applied (in order):
  1. Area bounds       — reject masks that are too small (noise) or too large
  2. Solidity          — reject non-convex noise blobs
  3. Aspect ratio      — reject long thin streaks
  4. Overlap           — deduplicate overlapping masks, keep higher stability

All thresholds are tuned for 1000-piece puzzles on a 2048px long-edge image.
"""

from __future__ import annotations

import cv2
import numpy as np

from app.utils.logger import get_logger

log = get_logger(__name__)

# ─── Filter thresholds ───────────────────────────────────────────────────────

# Area bounds as fraction of image area
# 1000-piece puzzle: each piece ≈ 0.1% of image at 2048px
_MIN_AREA_FRACTION = 0.0002    # 0.02% — below this is noise/shadow
_MAX_AREA_FRACTION = 0.04      # 4%   — above this is background or merged blob

# Solidity: contour_area / convex_hull_area
# Puzzle pieces are roughly convex — low solidity = noise blob
_MIN_SOLIDITY = 0.60

# Aspect ratio: width / height — valid piece ratios
_MIN_ASPECT_RATIO = 0.20
_MAX_ASPECT_RATIO = 5.00

# Overlap threshold for deduplication
# If two masks overlap by more than this fraction of the smaller mask,
# keep the one with higher stability_score
_OVERLAP_THRESHOLD = 0.30


def _compute_solidity(mask: np.ndarray) -> float:
    """Compute solidity = contour area / convex hull area."""
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return 0.0
    c = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(c)
    hull = cv2.convexHull(c)
    hull_area = cv2.contourArea(hull)
    return area / hull_area if hull_area > 0 else 0.0


def _bbox_aspect_ratio(bbox: list[int]) -> float:
    """Compute width/height from SAM bbox [x, y, w, h]."""
    _, _, w, h = bbox
    return w / h if h > 0 else 0.0


def filter_by_area(
    masks: list[dict],
    image_area: int,
) -> list[dict]:
    """Filter masks outside the valid area fraction bounds."""
    min_area = int(image_area * _MIN_AREA_FRACTION)
    max_area = int(image_area * _MAX_AREA_FRACTION)
    filtered = [m for m in masks if min_area <= m["area"] <= max_area]
    log.debug(
        "filter_area",
        before=len(masks),
        after=len(filtered),
        min_area=min_area,
        max_area=max_area,
    )
    return filtered


def filter_by_solidity(masks: list[dict]) -> list[dict]:
    """Filter masks with solidity below threshold (noise blobs)."""
    valid = []
    for m in masks:
        seg = m["segmentation"].astype(np.uint8) * 255
        solidity = _compute_solidity(seg)
        if solidity >= _MIN_SOLIDITY:
            m["_solidity"] = solidity
            valid.append(m)
    log.debug("filter_solidity", before=len(masks), after=len(valid))
    return valid


def filter_by_aspect_ratio(masks: list[dict]) -> list[dict]:
    """Filter masks with extreme aspect ratios (long streaks, thin slivers)."""
    valid = []
    for m in masks:
        ratio = _bbox_aspect_ratio(m["bbox"])
        if _MIN_ASPECT_RATIO <= ratio <= _MAX_ASPECT_RATIO:
            valid.append(m)
    log.debug("filter_aspect_ratio", before=len(masks), after=len(valid))
    return valid


def deduplicate_overlapping(masks: list[dict]) -> list[dict]:
    """
    Remove duplicate/overlapping masks.
    When two masks overlap by more than OVERLAP_THRESHOLD (fraction of
    the smaller mask's area), the one with lower stability_score is removed.

    Uses a greedy approach sorted by stability score (descending) —
    the most stable masks are kept first.
    """
    # Sort by stability score descending — keep best masks first
    sorted_masks = sorted(
        masks, key=lambda m: m.get("stability_score", 0.0), reverse=True
    )

    kept: list[dict] = []
    kept_segs: list[np.ndarray] = []

    for m in sorted_masks:
        seg = m["segmentation"]
        m_area = seg.sum()

        overlaps = False
        for k_seg in kept_segs:
            intersection = np.logical_and(seg, k_seg).sum()
            smaller_area = min(m_area, k_seg.sum())
            if smaller_area > 0 and intersection / smaller_area > _OVERLAP_THRESHOLD:
                overlaps = True
                break

        if not overlaps:
            kept.append(m)
            kept_segs.append(seg)

    log.debug("deduplicate_overlap", before=len(masks), after=len(kept))
    return kept


def filter_masks(
    masks: list[dict],
    image_shape: tuple[int, int],
) -> list[dict]:
    """
    Apply all filters in sequence to raw SAM masks.

    Args:
        masks:       Raw SAM mask dicts from mask_generator.generate()
        image_shape: (H, W) of the pieces image

    Returns:
        Filtered list of valid piece mask dicts.
    """
    h, w = image_shape
    image_area = h * w

    log.info("mask_filter_start", total_masks=len(masks))

    masks = filter_by_area(masks, image_area)
    masks = filter_by_solidity(masks)
    masks = filter_by_aspect_ratio(masks)
    masks = deduplicate_overlapping(masks)

    log.info("mask_filter_complete", valid_masks=len(masks))
    return masks