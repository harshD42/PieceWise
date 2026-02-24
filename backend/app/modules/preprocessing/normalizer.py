# Copyright (c) 2026 Harsh Dwivedi
# Licensed under the Harsh Non-Commercial Attribution License (HNCAL) v1.0
# Commercial use requires written permission. See LICENSE for details.

"""
PieceWise — Image Normalizer (Phase 2 stub)
Full implementation (histogram matching, Gaussian denoise) in Phase 2.
"""

import numpy as np
from pathlib import Path

from app.utils.image_utils import load_image_bgr, resize_long_edge
from app.utils.logger import get_logger

log = get_logger(__name__)

MAX_LONG_EDGE = 2048


def normalize_image_pair(
    ref_path: Path,
    pieces_path: Path,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Load, resize, and basic-normalise the image pair.
    Returns (ref_img_bgr, pieces_img_bgr, scale_factors).
    Phase 2 adds histogram matching and Gaussian denoising.
    """
    ref_img = load_image_bgr(ref_path)
    pieces_img = load_image_bgr(pieces_path)

    ref_img, ref_scale = resize_long_edge(ref_img, MAX_LONG_EDGE)
    pieces_img, pieces_scale = resize_long_edge(pieces_img, MAX_LONG_EDGE)

    log.debug(
        "images_normalised",
        ref_shape=ref_img.shape,
        pieces_shape=pieces_img.shape,
        ref_scale=ref_scale,
        pieces_scale=pieces_scale,
    )

    return ref_img, pieces_img, {"ref": ref_scale, "pieces": pieces_scale}