# Copyright (c) 2026 Harsh Dwivedi
# Licensed under the Harsh Non-Commercial Attribution License (HNCAL) v1.0
# Commercial use requires written permission. See LICENSE for details.

"""
PieceWise — Image Normalizer
Prepares the reference and pieces images for consistent downstream
feature extraction. Three operations are applied in sequence:

  1. Resize       — both images capped at MAX_LONG_EDGE (2048px)
  2. Denoise      — mild Gaussian blur on pieces image only
                    (reduces camera noise without blurring piece edges)
  3. Histogram    — transfer the colour distribution of the pieces image
     Match          to match the reference image using per-channel CDF
                    transfer. Critical for DINOv2 embedding consistency —
                    the same visual region must produce similar embeddings
                    regardless of lighting or white-balance differences
                    between the two photos.

All operations preserve aspect ratio and return BGR uint8 arrays.
Scale factors are returned so downstream modules can map coordinates
back to original image space if needed.
"""

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from app.utils.image_utils import resize_long_edge
from app.utils.logger import get_logger

log = get_logger(__name__)

MAX_LONG_EDGE = 2048

# Gaussian kernel size for piece denoising — must be odd
# Small enough to preserve piece edges, large enough to reduce sensor noise
_DENOISE_KERNEL = (3, 3)
_DENOISE_SIGMA = 0.8


@dataclass
class NormalisedPair:
    """Output of normalize_image_pair — carries images and metadata."""
    ref_img: np.ndarray          # BGR uint8, resized
    pieces_img: np.ndarray       # BGR uint8, resized + denoised + hist-matched
    ref_scale: float             # Scale factor applied to reference (≤ 1.0)
    pieces_scale: float          # Scale factor applied to pieces image (≤ 1.0)
    ref_original_shape: tuple    # (H, W) before resize
    pieces_original_shape: tuple # (H, W) before resize


# ─── Step 1: Resize ──────────────────────────────────────────────────────────

def resize_for_pipeline(img: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Resize image so its longest edge ≤ MAX_LONG_EDGE.
    Returns (resized_img, scale_factor).
    Images already within limit are returned as-is with scale=1.0.
    """
    return resize_long_edge(img, MAX_LONG_EDGE)


# ─── Step 2: Denoise ─────────────────────────────────────────────────────────

def denoise_pieces(img: np.ndarray) -> np.ndarray:
    """
    Apply mild Gaussian blur to reduce sensor noise on the pieces image.
    Only applied to pieces image — NOT to the reference image (which is
    typically a clean scan/print and should not be smoothed).

    Uses a small kernel (3×3, σ=0.8) to reduce high-frequency noise
    while preserving piece boundary sharpness needed for contour analysis.
    """
    return cv2.GaussianBlur(img, _DENOISE_KERNEL, _DENOISE_SIGMA)


# ─── Step 3: Histogram Matching ──────────────────────────────────────────────

def _match_channel_histogram(
    src_channel: np.ndarray,
    ref_channel: np.ndarray,
) -> np.ndarray:
    """
    Transfer the colour distribution of ref_channel onto src_channel
    using cumulative distribution function (CDF) matching.

    This is the standard histogram specification technique:
      1. Compute CDFs for source and reference channels
      2. Build a lookup table: for each source intensity, find the
         reference intensity that has the closest CDF value
      3. Apply the lookup table to remap source pixel values

    Args:
        src_channel: uint8 single-channel array (the pieces image channel)
        ref_channel: uint8 single-channel array (the reference image channel)

    Returns:
        uint8 single-channel array with src distribution matched to ref.
    """
    # Compute normalised CDFs
    src_hist, _ = np.histogram(src_channel.flatten(), 256, [0, 256])
    ref_hist, _ = np.histogram(ref_channel.flatten(), 256, [0, 256])

    src_cdf = src_hist.cumsum().astype(np.float64)
    ref_cdf = ref_hist.cumsum().astype(np.float64)

    # Normalise to [0, 1]
    src_cdf /= src_cdf[-1]
    ref_cdf /= ref_cdf[-1]

    # Build lookup table: for each source intensity i, find the reference
    # intensity j whose CDF value is closest to src_cdf[i]
    lut = np.zeros(256, dtype=np.uint8)
    j = 0
    for i in range(256):
        while j < 255 and ref_cdf[j] < src_cdf[i]:
            j += 1
        lut[i] = j

    return lut[src_channel]


def match_histogram(
    pieces_img: np.ndarray,
    ref_img: np.ndarray,
) -> np.ndarray:
    """
    Match the colour histogram of pieces_img to match ref_img.
    Applied independently per BGR channel.

    This compensates for:
      - Different lighting conditions between the two photos
      - White balance differences (phone vs. camera)
      - Exposure differences (flash on/off)
      - Different times of day when photos were taken

    Result: DINOv2 will see similar colour distributions in both images,
    producing more consistent embeddings for matching visual regions.

    Args:
        pieces_img: BGR uint8 array — the scattered pieces photo
        ref_img:    BGR uint8 array — the complete puzzle reference image

    Returns:
        BGR uint8 array — pieces image with colour matched to reference
    """
    matched = np.empty_like(pieces_img)
    for c in range(3):  # B, G, R channels
        matched[:, :, c] = _match_channel_histogram(
            pieces_img[:, :, c],
            ref_img[:, :, c],
        )
    return matched


# ─── Main Entry Point ────────────────────────────────────────────────────────

def normalize_image_pair(
    ref_path: Path,
    pieces_path: Path,
) -> NormalisedPair:
    """
    Full normalisation pipeline for the reference + pieces image pair.
    Called by the pipeline orchestrator after validation.

    Steps applied:
      ref_img:    resize only
      pieces_img: resize → denoise → histogram match to ref

    Args:
        ref_path:    Path to the validated reference image file.
        pieces_path: Path to the validated pieces image file.

    Returns:
        NormalisedPair with processed images and scale metadata.
    """
    # Load (already validated — these reads will not fail)
    ref_raw = cv2.imread(str(ref_path), cv2.IMREAD_COLOR)
    pieces_raw = cv2.imread(str(pieces_path), cv2.IMREAD_COLOR)

    ref_original_shape = ref_raw.shape[:2]
    pieces_original_shape = pieces_raw.shape[:2]

    # Step 1: Resize both
    ref_resized, ref_scale = resize_for_pipeline(ref_raw)
    pieces_resized, pieces_scale = resize_for_pipeline(pieces_raw)

    log.debug(
        "images_resized",
        ref_original=ref_original_shape,
        ref_resized=ref_resized.shape[:2],
        ref_scale=round(ref_scale, 4),
        pieces_original=pieces_original_shape,
        pieces_resized=pieces_resized.shape[:2],
        pieces_scale=round(pieces_scale, 4),
    )

    # Step 2: Denoise pieces image only
    pieces_denoised = denoise_pieces(pieces_resized)

    # Step 3: Histogram match pieces → reference
    pieces_matched = match_histogram(pieces_denoised, ref_resized)

    log.debug(
        "normalisation_complete",
        ref_shape=ref_resized.shape,
        pieces_shape=pieces_matched.shape,
    )

    return NormalisedPair(
        ref_img=ref_resized,
        pieces_img=pieces_matched,
        ref_scale=ref_scale,
        pieces_scale=pieces_scale,
        ref_original_shape=ref_original_shape,
        pieces_original_shape=pieces_original_shape,
    )


def normalize_from_arrays(
    ref_img: np.ndarray,
    pieces_img: np.ndarray,
) -> NormalisedPair:
    """
    Same as normalize_image_pair but accepts pre-loaded numpy arrays.
    Used in tests and when images are already in memory.
    """
    ref_original_shape = ref_img.shape[:2]
    pieces_original_shape = pieces_img.shape[:2]

    ref_resized, ref_scale = resize_for_pipeline(ref_img)
    pieces_resized, pieces_scale = resize_for_pipeline(pieces_img)
    pieces_denoised = denoise_pieces(pieces_resized)
    pieces_matched = match_histogram(pieces_denoised, ref_resized)

    return NormalisedPair(
        ref_img=ref_resized,
        pieces_img=pieces_matched,
        ref_scale=ref_scale,
        pieces_scale=pieces_scale,
        ref_original_shape=ref_original_shape,
        pieces_original_shape=pieces_original_shape,
    )