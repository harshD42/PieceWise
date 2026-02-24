# Copyright (c) 2026 Harsh Dwivedi
# Licensed under the Harsh Non-Commercial Attribution License (HNCAL) v1.0
# Commercial use requires written permission. See LICENSE for details.

"""
PieceWise — Grid Size Estimator
Infers the NxM puzzle grid dimensions from:
  - The number of detected pieces (from segmentation output)
  - The aspect ratio of the reference image
  - An optional user-specified override

For a 1000-piece puzzle with a roughly 4:3 reference image:
  aspect_ratio = 4/3 ≈ 1.333
  grid_cols ≈ sqrt(1000 * 1.333) ≈ 36.5 → 37
  grid_rows ≈ 1000 / 37 ≈ 27

The estimator finds the integer (rows, cols) pair whose product is
closest to piece_count while preserving the reference aspect ratio.
"""

import math

import numpy as np

from app.utils.logger import get_logger

log = get_logger(__name__)

# Tolerance: accept grid shapes whose total cell count is within
# this fraction of the detected piece count.
# E.g. 0.15 = up to 15% difference is acceptable
_COUNT_TOLERANCE = 0.15

# Hard bounds on grid dimensions to prevent degenerate outputs
_MIN_GRID_DIM = 2
_MAX_GRID_DIM = 100


def estimate_grid_shape(
    piece_count: int,
    ref_img_shape: tuple[int, int],  # (H, W)
    override: tuple[int, int] | None = None,
) -> tuple[int, int]:
    """
    Estimate the puzzle grid dimensions (n_rows, n_cols).

    Args:
        piece_count:   Number of pieces detected by the segmentation module.
        ref_img_shape: (H, W) of the normalised reference image.
        override:      Optional (n_rows, n_cols) specified by the user via API.
                       If provided, validated and returned directly.

    Returns:
        (n_rows, n_cols) — best estimated grid shape.

    Raises:
        ValueError: If override dimensions are out of valid range or
                    if piece_count is non-positive.
    """
    if override is not None:
        return _validate_override(override, piece_count)

    if piece_count <= 0:
        raise ValueError(f"piece_count must be positive, got {piece_count}")

    h, w = ref_img_shape
    aspect_ratio = w / h if h > 0 else 1.0

    # Estimate cols first: cols ≈ sqrt(piece_count * aspect_ratio)
    cols_float = math.sqrt(piece_count * aspect_ratio)
    rows_float = piece_count / cols_float

    # Search ±3 around the float estimate to find the integer pair
    # whose product is closest to piece_count
    best_rows, best_cols = _find_best_integer_pair(
        rows_float, cols_float, piece_count
    )

    total_cells = best_rows * best_cols
    diff_pct = abs(total_cells - piece_count) / piece_count * 100

    log.info(
        "grid_shape_estimated",
        piece_count=piece_count,
        aspect_ratio=round(aspect_ratio, 3),
        estimated_rows=best_rows,
        estimated_cols=best_cols,
        total_cells=total_cells,
        diff_from_piece_count_pct=round(diff_pct, 1),
    )

    if diff_pct > _COUNT_TOLERANCE * 100:
        log.warning(
            "grid_estimate_high_deviation",
            piece_count=piece_count,
            total_cells=total_cells,
            diff_pct=round(diff_pct, 1),
            advice="Consider providing a manual grid override via the API.",
        )

    return best_rows, best_cols


def _find_best_integer_pair(
    rows_float: float,
    cols_float: float,
    target: int,
    search_radius: int = 5,
) -> tuple[int, int]:
    """
    Search integer (rows, cols) combinations near the float estimates
    and return the pair whose product is closest to target.
    """
    rows_candidates = range(
        max(_MIN_GRID_DIM, int(rows_float) - search_radius),
        min(_MAX_GRID_DIM, int(rows_float) + search_radius + 1),
    )
    cols_candidates = range(
        max(_MIN_GRID_DIM, int(cols_float) - search_radius),
        min(_MAX_GRID_DIM, int(cols_float) + search_radius + 1),
    )

    best_rows, best_cols = int(round(rows_float)), int(round(cols_float))
    best_diff = abs(best_rows * best_cols - target)

    for r in rows_candidates:
        for c in cols_candidates:
            diff = abs(r * c - target)
            if diff < best_diff:
                best_diff = diff
                best_rows, best_cols = r, c
            elif diff == best_diff:
                # Tiebreak: prefer the pair with aspect ratio closer
                # to the float estimate ratio
                current_ratio = best_cols / best_rows if best_rows > 0 else 0
                candidate_ratio = c / r if r > 0 else 0
                target_ratio = cols_float / rows_float if rows_float > 0 else 1
                if abs(candidate_ratio - target_ratio) < abs(current_ratio - target_ratio):
                    best_rows, best_cols = r, c

    return best_rows, best_cols


def _validate_override(
    override: tuple[int, int],
    piece_count: int,
) -> tuple[int, int]:
    """
    Validate a user-provided grid override.
    Logs a warning if the override's cell count differs significantly
    from the detected piece count, but does not reject it.
    """
    rows, cols = override

    if rows < _MIN_GRID_DIM or cols < _MIN_GRID_DIM:
        raise ValueError(
            f"Grid dimensions must be at least {_MIN_GRID_DIM}x{_MIN_GRID_DIM}. "
            f"Got {rows}x{cols}."
        )
    if rows > _MAX_GRID_DIM or cols > _MAX_GRID_DIM:
        raise ValueError(
            f"Grid dimensions must not exceed {_MAX_GRID_DIM}. "
            f"Got {rows}x{cols}."
        )

    total_cells = rows * cols
    diff_pct = abs(total_cells - piece_count) / max(piece_count, 1) * 100

    if diff_pct > _COUNT_TOLERANCE * 100:
        log.warning(
            "grid_override_deviation",
            override_rows=rows,
            override_cols=cols,
            total_cells=total_cells,
            piece_count=piece_count,
            diff_pct=round(diff_pct, 1),
            advice=(
                "Override grid cell count differs significantly from "
                "detected piece count. Verify the override is correct."
            ),
        )
    else:
        log.info(
            "grid_override_accepted",
            rows=rows,
            cols=cols,
            total_cells=total_cells,
        )

    return rows, cols


def patch_size_px(
    img_shape: tuple[int, int],
    grid_shape: tuple[int, int],
) -> tuple[float, float]:
    """
    Compute the pixel dimensions of each grid cell.

    Args:
        img_shape:   (H, W) of the normalised reference image.
        grid_shape:  (n_rows, n_cols) of the puzzle grid.

    Returns:
        (patch_h, patch_w) as floats — may be fractional.
    """
    h, w = img_shape
    n_rows, n_cols = grid_shape
    return h / n_rows, w / n_cols