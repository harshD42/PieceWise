# Copyright (c) 2026 Harsh Dwivedi
# Licensed under the Harsh Non-Commercial Attribution License (HNCAL) v1.0
# Commercial use requires written permission. See LICENSE for details.

"""
PieceWise — Mask Generator
Two-stage piece detection:

Stage 1 — Connected Components Pre-filter (cheap, fast)
  Binarise the pieces image and run connected components analysis.
  Reject components that are obviously too small or too large.
  Use surviving component centroids as SAM point prompts.
  This dramatically reduces SAM's search space on 1000-piece puzzles.

Stage 2 — SAM Automatic Mask Generation
  Run SAM with the filtered point prompts (or full automatic mode
  as fallback). Returns raw mask dicts for downstream filtering.
"""

from __future__ import annotations

import cv2
import numpy as np

from app.config import get_settings
from app.utils.logger import get_logger

log = get_logger(__name__)

# Connected components area bounds — expressed as fraction of total image area.
# A single puzzle piece on a 1000-piece puzzle typically covers
# 0.02% – 2% of the image area.
_CC_MIN_AREA_FRACTION = 0.0001   # 0.01% of image — reject dust/noise
_CC_MAX_AREA_FRACTION = 0.05     # 5% of image — reject background blob


# ─── Stage 1: Connected Components Pre-filter ────────────────────────────────

def connected_components_prefilter(
    pieces_img: np.ndarray,
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """
    Fast pre-filter using connected components to find candidate piece regions.

    Steps:
      1. Convert to grayscale
      2. Otsu thresholding to separate pieces from background
      3. Morphological closing to join fragmented piece regions
      4. Connected components analysis
      5. Filter by area fraction
      6. Return binary mask and list of centroid (x, y) points

    Args:
        pieces_img: BGR uint8 normalised pieces image.

    Returns:
        (binary_mask, centroids)
        binary_mask: uint8 binary image (255=foreground piece candidate)
        centroids:   list of (x, y) centroid coordinates for SAM prompting
    """
    settings = get_settings()
    h, w = pieces_img.shape[:2]
    total_area = h * w

    min_area = int(total_area * _CC_MIN_AREA_FRACTION)
    max_area = int(total_area * _CC_MAX_AREA_FRACTION)

    # Step 1: Grayscale
    gray = cv2.cvtColor(pieces_img, cv2.COLOR_BGR2GRAY)

    # Step 2: Otsu threshold — automatically finds optimal threshold
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Step 3: Morphological closing — joins small gaps within pieces
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # Step 4: Connected components
    n_labels, labels, stats, centroids_arr = cv2.connectedComponentsWithStats(
        closed, connectivity=8
    )

    # Step 5: Filter — skip label 0 (background)
    valid_centroids: list[tuple[int, int]] = []
    output_mask = np.zeros((h, w), dtype=np.uint8)

    for label in range(1, n_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        if min_area <= area <= max_area:
            cx, cy = centroids_arr[label]
            valid_centroids.append((int(cx), int(cy)))
            output_mask[labels == label] = 255

    log.debug(
        "cc_prefilter_complete",
        total_components=n_labels - 1,
        valid_components=len(valid_centroids),
        min_area=min_area,
        max_area=max_area,
    )

    return output_mask, valid_centroids


# ─── Stage 2: SAM Mask Generation ────────────────────────────────────────────

def generate_masks_with_sam(
    pieces_img: np.ndarray,
    centroids: list[tuple[int, int]],
) -> list[dict]:
    """
    Run SAM to generate segmentation masks on the pieces image.

    Uses the automatic mask generator (no point prompting needed —
    SAM's automatic mode already handles this efficiently).
    The centroids from the pre-filter are used for validation downstream
    but SAM runs in full automatic mode for maximum mask coverage.

    Args:
        pieces_img: BGR uint8 normalised pieces image.
        centroids:  Centroid list from connected_components_prefilter
                    (used for logging/validation, not passed to SAM directly).

    Returns:
        List of SAM mask dicts, each containing:
          'segmentation': bool H×W mask
          'area': int pixel area
          'bbox': [x, y, w, h]
          'predicted_iou': float
          'stability_score': float
          'point_coords': [[x, y]]
          'crop_box': [x, y, w, h]
    """
    from app.modules.segmentation.sam_loader import get_mask_generator

    # SAM expects RGB
    rgb_img = cv2.cvtColor(pieces_img, cv2.COLOR_BGR2RGB)

    log.info(
        "sam_generation_start",
        image_shape=pieces_img.shape,
        cc_centroid_count=len(centroids),
    )

    mask_generator = get_mask_generator()
    masks = mask_generator.generate(rgb_img)

    log.info(
        "sam_generation_complete",
        raw_mask_count=len(masks),
    )

    return masks


def generate_masks(pieces_img: np.ndarray) -> tuple[list[dict], list[tuple[int, int]]]:
    """
    Full two-stage mask generation pipeline.
    Runs CC pre-filter then SAM automatic generation.

    Returns:
        (sam_masks, centroids)
    """
    _, centroids = connected_components_prefilter(pieces_img)
    masks = generate_masks_with_sam(pieces_img, centroids)
    return masks, centroids