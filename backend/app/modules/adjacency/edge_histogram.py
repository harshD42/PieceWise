# Copyright (c) 2026 Harsh Dwivedi
# Licensed under the Harsh Non-Commercial Attribution License (HNCAL) v1.0
# Commercial use requires written permission. See LICENSE for details.

"""
PieceWise — Edge Color Histogram Comparator
Compares the color distribution along the shared boundary between
two adjacent puzzle pieces.

Correctly assembled neighbors should have matching edge colors —
the right side of piece A and the left side of piece B should both
show the same region of the image. A low histogram intersection score
strongly suggests a mis-assignment.

Strip extraction:
  For each piece, extract a thin strip (STRIP_WIDTH pixels) along the
  relevant side of its alpha-masked crop.
  Side 0 (top):    top STRIP_WIDTH rows of crop
  Side 1 (right):  rightmost STRIP_WIDTH cols of crop
  Side 2 (bottom): bottom STRIP_WIDTH rows of crop
  Side 3 (left):   leftmost STRIP_WIDTH cols of crop

Score:
  Histogram intersection normalised to [0, 1].
  1.0 = identical color distributions.
  0.0 = completely disjoint.
"""

from __future__ import annotations

import cv2
import numpy as np

from app.models.piece import PieceCrop
from app.modules.adjacency.neighbor_extractor import NeighborPair
from app.utils.logger import get_logger

log = get_logger(__name__)

# Width of boundary strip to sample (pixels)
_STRIP_WIDTH = 10

# Number of histogram bins per channel
_N_BINS = 32


def _extract_boundary_strip(
    piece: PieceCrop,
    side: int,
) -> np.ndarray:
    """
    Extract a thin strip of pixels along one side of the piece crop.
    Only pixels inside the alpha mask are included.

    Args:
        piece: PieceCrop with .image (BGR) and .alpha_mask
        side:  0=top, 1=right, 2=bottom, 3=left

    Returns:
        (N, 3) uint8 array of BGR pixels in the strip,
        filtered to only include foreground (mask > 0) pixels.
        Returns empty array if no foreground pixels in strip.
    """
    img = piece.image       # (H, W, 3) BGR
    mask = piece.alpha_mask  # (H, W) uint8

    h, w = img.shape[:2]
    sw = min(_STRIP_WIDTH, h // 4, w // 4)
    sw = max(1, sw)

    if side == 0:    # top
        strip_img = img[:sw, :, :]
        strip_mask = mask[:sw, :]
    elif side == 1:  # right
        strip_img = img[:, max(0, w - sw):, :]
        strip_mask = mask[:, max(0, w - sw):]
    elif side == 2:  # bottom
        strip_img = img[max(0, h - sw):, :, :]
        strip_mask = mask[max(0, h - sw):, :]
    else:            # left (side == 3)
        strip_img = img[:, :sw, :]
        strip_mask = mask[:, :sw]

    # Extract only foreground pixels
    fg_pixels = strip_img[strip_mask > 0]  # (N, 3)
    return fg_pixels


def _channel_histogram(pixels: np.ndarray, n_bins: int = _N_BINS) -> np.ndarray:
    """
    Compute a concatenated per-channel histogram from BGR pixel array.
    Returns normalised (3 × n_bins,) float32 array.
    """
    if len(pixels) == 0:
        return np.zeros(3 * n_bins, dtype=np.float32)

    hists = []
    for c in range(3):
        h, _ = np.histogram(pixels[:, c], bins=n_bins, range=(0, 256))
        h = h.astype(np.float32)
        total = h.sum()
        if total > 0:
            h /= total
        hists.append(h)

    return np.concatenate(hists)


def _histogram_intersection(h1: np.ndarray, h2: np.ndarray) -> float:
    """
    Compute normalised histogram intersection.
    = sum(min(h1_i, h2_i)) — value in [0, 1] when histograms are normalised.
    """
    return float(np.minimum(h1, h2).sum())


def edge_histogram_score(
    piece_a: PieceCrop,
    piece_b: PieceCrop,
    pair: NeighborPair,
) -> float:
    """
    Compute edge color compatibility between two adjacent pieces.

    Args:
        piece_a: First piece (corresponds to pair.piece_id_a)
        piece_b: Second piece (corresponds to pair.piece_id_b)
        pair:    NeighborPair specifying which sides are shared

    Returns:
        Score in [0, 1] — higher = better color compatibility.
    """
    pixels_a = _extract_boundary_strip(piece_a, pair.side_a)
    pixels_b = _extract_boundary_strip(piece_b, pair.side_b)

    if len(pixels_a) == 0 or len(pixels_b) == 0:
        # No foreground pixels in strip — return neutral score
        return 0.5

    hist_a = _channel_histogram(pixels_a)
    hist_b = _channel_histogram(pixels_b)

    score = _histogram_intersection(hist_a, hist_b)

    # Histogram intersection of two normalised histograms has a maximum of 1.0
    # but in practice rarely reaches 1.0 — scale up slightly for better
    # discrimination (effective range is roughly [0, 0.8] in practice)
    return min(1.0, score)


def score_all_pairs_histogram(
    pairs: list[NeighborPair],
    piece_map: dict[int, PieceCrop],
) -> dict[tuple[int, int], float]:
    """
    Compute edge histogram scores for all neighbor pairs.

    Args:
        pairs:     List of NeighborPair from neighbor_extractor
        piece_map: piece_id → PieceCrop lookup

    Returns:
        Dict mapping (piece_id_a, piece_id_b) → histogram score.
        Note: keys are always (min_id, max_id) for consistent lookup.
    """
    scores: dict[tuple[int, int], float] = {}

    for pair in pairs:
        pa = piece_map.get(pair.piece_id_a)
        pb = piece_map.get(pair.piece_id_b)

        if pa is None or pb is None:
            continue

        score = edge_histogram_score(pa, pb, pair)
        key = (min(pair.piece_id_a, pair.piece_id_b),
               max(pair.piece_id_a, pair.piece_id_b))
        scores[key] = score

    return scores