# Copyright (c) 2026 Harsh Dwivedi
# Licensed under the Harsh Non-Commercial Attribution License (HNCAL) v1.0
# Commercial use requires written permission. See LICENSE for details.

"""
PieceWise — Piece Extractor
Converts validated SAM mask dicts into PieceCrop objects.

For each mask:
  1. Extract tight bounding box with padding
  2. Crop the pieces image to that bounding box
  3. Apply the binary alpha mask (background → transparent)
  4. Assign a sequential integer piece_id

The resulting PieceCrop objects carry both the image data and
coordinate metadata needed by the feature extraction and rendering modules.
"""

from __future__ import annotations

import cv2
import numpy as np

from app.models.piece import PieceCrop
from app.utils.logger import get_logger

log = get_logger(__name__)

# Padding around the tight bounding box (pixels)
_CROP_PAD = 8


def extract_pieces(
    masks: list[dict],
    pieces_img: np.ndarray,
) -> list[PieceCrop]:
    """
    Convert filtered SAM masks into PieceCrop objects.

    Args:
        masks:      Filtered and refined SAM mask dicts
        pieces_img: BGR uint8 normalised pieces image

    Returns:
        List of PieceCrop objects, one per valid mask.
        piece_id is assigned sequentially starting from 0.
    """
    ih, iw = pieces_img.shape[:2]
    crops: list[PieceCrop] = []

    for idx, m in enumerate(masks):
        seg = m["segmentation"]  # bool H×W

        # Convert to uint8 mask (0/255)
        mask_u8 = (seg.astype(np.uint8)) * 255

        # Find tight bounding box
        x, y, w, h = cv2.boundingRect(mask_u8)

        # Add padding — clamped to image bounds
        x1 = max(0, x - _CROP_PAD)
        y1 = max(0, y - _CROP_PAD)
        x2 = min(iw, x + w + _CROP_PAD)
        y2 = min(ih, y + h + _CROP_PAD)

        # Crop image and mask
        img_crop = pieces_img[y1:y2, x1:x2].copy()
        mask_crop = mask_u8[y1:y2, x1:x2].copy()

        # Compute contour on the cropped mask
        contours, _ = cv2.findContours(
            mask_crop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            log.warning("no_contour_for_piece", piece_id=idx)
            continue

        contour = max(contours, key=cv2.contourArea)

        # Basic shape descriptors
        area = float(cv2.contourArea(contour))
        perimeter = float(cv2.arcLength(contour, closed=True))
        if perimeter > 0:
            compactness = (4 * np.pi * area) / (perimeter ** 2)
        else:
            compactness = 0.0

        hull = cv2.convexHull(contour)
        hull_area = float(cv2.contourArea(hull))
        solidity = area / hull_area if hull_area > 0 else 0.0

        crop = PieceCrop(
            piece_id=idx,
            image=img_crop,
            alpha_mask=mask_crop,
            bbox=(x1, y1, x2 - x1, y2 - y1),
            contour=contour,
            area_px=area,
            solidity=solidity,
            compactness=compactness,
            # curvature_profiles and flat_side_count filled by contour_analyzer
            curvature_profiles=[],
            flat_side_count=0,
            pca_correction_deg=0.0,
        )
        crops.append(crop)

    log.info(
        "piece_extraction_complete",
        total_masks=len(masks),
        extracted_pieces=len(crops),
    )

    return crops