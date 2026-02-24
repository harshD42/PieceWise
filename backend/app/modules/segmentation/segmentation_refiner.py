# Copyright (c) 2026 Harsh Dwivedi
# Licensed under the Harsh Non-Commercial Attribution License (HNCAL) v1.0
# Commercial use requires written permission. See LICENSE for details.

"""
PieceWise — Segmentation Refiner
Detects and separates merged/touching puzzle pieces that SAM
incorrectly grouped into a single mask.

Detection: Convexity defect analysis
  A single puzzle piece has a roughly convex contour with tab/blank
  protrusions. If a mask contains very deep convexity defects
  (concave indentations), it likely contains two or more merged pieces.

Separation: Morphological erosion + Distance Transform + Watershed
  1. Erode the binary mask to separate touching regions
  2. Apply distance transform to find local maxima (piece centres)
  3. Use watershed segmentation to partition the mask
  4. Validate each resulting region with the mask filter criteria
  5. If both child regions are valid pieces, replace parent with children

This is the key robustness mechanism for handling overlapping or
touching pieces in the scattered pieces photo.
"""

from __future__ import annotations

import cv2
import numpy as np

from app.utils.logger import get_logger

log = get_logger(__name__)

# Convexity defect depth threshold (pixels)
# Defects deeper than this suggest a merged piece boundary
_DEFECT_DEPTH_THRESHOLD_FRACTION = 0.08  # 8% of contour perimeter

# Minimum area for a child mask to be considered a valid piece after split
# (expressed as fraction of parent mask area)
_MIN_CHILD_AREA_FRACTION = 0.15

# Distance transform threshold for watershed markers
# (fraction of max distance value)
_DIST_THRESH_FRACTION = 0.4


def _has_deep_convexity_defects(
    mask: np.ndarray,
    depth_threshold: float,
) -> bool:
    """
    Check if a binary mask contains deep convexity defects suggesting
    two merged pieces.

    Args:
        mask:            uint8 binary mask (0/255)
        depth_threshold: Minimum defect depth in pixels to flag as merged

    Returns:
        True if deep convexity defects are found.
    """
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return False

    contour = max(contours, key=cv2.contourArea)
    if len(contour) < 4:
        return False

    hull_indices = cv2.convexHull(contour, returnPoints=False)
    if hull_indices is None or len(hull_indices) < 3:
        return False

    try:
        defects = cv2.convexityDefects(contour, hull_indices)
    except cv2.error:
        return False

    if defects is None:
        return False

    # defects shape: (N, 1, 4) — [start, end, farthest, depth/256]
    for defect in defects[:, 0]:
        depth = defect[3] / 256.0  # Convert to pixels
        if depth > depth_threshold:
            return True

    return False


def _watershed_separate(mask: np.ndarray) -> list[np.ndarray]:
    """
    Apply distance transform + watershed to separate a merged mask
    into individual piece masks.

    Args:
        mask: uint8 binary mask (0/255) of the merged region

    Returns:
        List of child binary masks (uint8 0/255).
        Returns empty list if separation failed or produced invalid regions.
    """
    # Distance transform — bright pixels are far from background
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)

    # Threshold to find definite foreground (piece centres)
    thresh_val = _DIST_THRESH_FRACTION * dist.max()
    _, sure_fg = cv2.threshold(dist, thresh_val, 255, cv2.THRESH_BINARY)
    sure_fg = sure_fg.astype(np.uint8)

    # Sure background via dilation
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    sure_bg = cv2.dilate(mask, kernel, iterations=3)

    # Unknown region = background - foreground
    unknown = cv2.subtract(sure_bg, sure_fg)

    # Label connected components in sure foreground
    n_markers, markers = cv2.connectedComponents(sure_fg)
    if n_markers < 3:
        # Less than 2 foreground regions found — can't separate
        return []

    # Add 1 to all labels so background is 1 (not 0)
    markers = markers + 1
    # Mark unknown region as 0
    markers[unknown == 255] = 0

    # Watershed requires 3-channel image
    img_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    cv2.watershed(img_3ch, markers)

    # Extract each labelled region (skip background=1 and boundary=-1)
    child_masks = []
    for label in range(2, n_markers + 1):
        child = np.zeros_like(mask)
        child[markers == label] = 255
        child_masks.append(child)

    return child_masks


def _is_valid_child(
    child_mask: np.ndarray,
    parent_area: int,
    min_area: int,
) -> bool:
    """
    Check that a child mask from watershed separation is a valid piece.
    Must have minimum absolute area and minimum fraction of parent area.
    """
    child_area = child_mask.sum() // 255
    if child_area < min_area:
        return False
    if child_area < parent_area * _MIN_CHILD_AREA_FRACTION:
        return False
    return True


def refine_masks(
    masks: list[dict],
    image_shape: tuple[int, int],
    min_piece_area: int,
) -> list[dict]:
    """
    Attempt to split merged piece masks using convexity defect detection
    and watershed segmentation.

    For each mask:
      1. Check for deep convexity defects (merged piece indicator)
      2. If detected: apply watershed separation
      3. If separation produces ≥2 valid child masks: replace parent
      4. Otherwise: keep parent mask unchanged

    Args:
        masks:          Filtered SAM mask dicts
        image_shape:    (H, W) of the pieces image
        min_piece_area: Minimum area in pixels for a valid piece

    Returns:
        Refined list of mask dicts — may contain more masks than input.
    """
    refined: list[dict] = []
    n_separated = 0

    for m in masks:
        seg = m["segmentation"].astype(np.uint8) * 255
        contours, _ = cv2.findContours(
            seg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            refined.append(m)
            continue

        contour = max(contours, key=cv2.contourArea)
        perimeter = cv2.arcLength(contour, closed=True)
        depth_threshold = perimeter * _DEFECT_DEPTH_THRESHOLD_FRACTION

        parent_area = int(seg.sum()) // 255

        if _has_deep_convexity_defects(seg, depth_threshold):
            children = _watershed_separate(seg)

            valid_children = [
                c for c in children
                if _is_valid_child(c, parent_area, min_piece_area)
            ]

            if len(valid_children) >= 2:
                # Replace parent with children
                for i, child in enumerate(valid_children):
                    child_bool = child.astype(bool)
                    child_mask_dict = {
                        "segmentation": child_bool,
                        "area": int(child.sum()) // 255,
                        "bbox": list(cv2.boundingRect(
                            cv2.findContours(
                                child, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                            )[0][0]
                        )),
                        "predicted_iou": m.get("predicted_iou", 0.0),
                        "stability_score": m.get("stability_score", 0.0),
                        "_refined": True,
                        "_parent_id": id(m),
                    }
                    refined.append(child_mask_dict)
                n_separated += 1
                continue

        refined.append(m)

    log.info(
        "segmentation_refiner_complete",
        input_count=len(masks),
        output_count=len(refined),
        pieces_separated=n_separated,
    )

    return refined