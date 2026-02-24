# Copyright (c) 2026 Harsh Dwivedi
# Licensed under the Harsh Non-Commercial Attribution License (HNCAL) v1.0
# Commercial use requires written permission. See LICENSE for details.

"""
PieceWise — Geometry Utilities
Rotation, bounding box, contour, and coordinate-mapping helpers
shared across segmentation, feature extraction, and rendering modules.
"""

import cv2
import numpy as np


# ─── Rotation ────────────────────────────────────────────────────────────────

def rotate_image(img: np.ndarray, angle_deg: float) -> np.ndarray:
    """
    Rotate image by angle_deg (counter-clockwise) around its centre.
    Uses INTER_LINEAR interpolation. Expands canvas to fit full rotated image.
    """
    h, w = img.shape[:2]
    cx, cy = w / 2.0, h / 2.0
    M = cv2.getRotationMatrix2D((cx, cy), angle_deg, 1.0)

    # Compute new bounding box dimensions
    cos_a = abs(M[0, 0])
    sin_a = abs(M[0, 1])
    new_w = int(h * sin_a + w * cos_a)
    new_h = int(h * cos_a + w * sin_a)

    # Adjust translation
    M[0, 2] += (new_w / 2.0) - cx
    M[1, 2] += (new_h / 2.0) - cy

    return cv2.warpAffine(
        img, M, (new_w, new_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(128, 128, 128),
    )


def rotate_image_90(img: np.ndarray, k: int) -> np.ndarray:
    """
    Rotate image by k * 90 degrees counter-clockwise.
    k=0: no rotation, k=1: 90°, k=2: 180°, k=3: 270°.
    Fast — uses np.rot90 with contiguous array guarantee.
    """
    return np.ascontiguousarray(np.rot90(img, k=k % 4))


ROTATION_VARIANTS: list[int] = [0, 90, 180, 270]


def rotation_deg_to_k(deg: int) -> int:
    """Convert rotation degrees (0/90/180/270) to np.rot90 k value."""
    return (deg // 90) % 4


# ─── Bounding Box ────────────────────────────────────────────────────────────

def mask_to_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    """
    Compute tight bounding box of a binary mask.
    Returns (x, y, w, h) — top-left corner + dimensions.
    Raises ValueError if mask is entirely zero.
    """
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any():
        raise ValueError("mask_to_bbox: mask is entirely zero.")
    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]
    return int(x_min), int(y_min), int(x_max - x_min + 1), int(y_max - y_min + 1)


def scale_bbox(
    bbox: tuple[int, int, int, int], scale: float
) -> tuple[int, int, int, int]:
    """Scale a (x, y, w, h) bbox by a float factor."""
    x, y, w, h = bbox
    return (
        int(round(x * scale)),
        int(round(y * scale)),
        int(round(w * scale)),
        int(round(h * scale)),
    )


def bbox_area(bbox: tuple[int, int, int, int]) -> int:
    """Return pixel area of a (x, y, w, h) bounding box."""
    return bbox[2] * bbox[3]


def bbox_aspect_ratio(bbox: tuple[int, int, int, int]) -> float:
    """Return width/height aspect ratio. Returns 0 if height is zero."""
    _, _, w, h = bbox
    return w / h if h > 0 else 0.0


# ─── Contour ─────────────────────────────────────────────────────────────────

def largest_contour(mask: np.ndarray) -> np.ndarray:
    """
    Find the largest external contour in a binary mask.
    Returns contour as (N, 1, 2) int32 array.
    Raises ValueError if no contour found.
    """
    contours, _ = cv2.findContours(
        mask.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    if not contours:
        raise ValueError("No contours found in mask.")
    return max(contours, key=cv2.contourArea)


def contour_solidity(contour: np.ndarray) -> float:
    """
    Solidity = contour_area / convex_hull_area.
    Values close to 1.0 indicate convex shapes.
    Used to filter noise blobs and validate piece masks.
    """
    area = cv2.contourArea(contour)
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    return area / hull_area if hull_area > 0 else 0.0


def contour_compactness(contour: np.ndarray) -> float:
    """
    Compactness = 4π × area / perimeter².
    Circle = 1.0, elongated shapes approach 0.
    """
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, closed=True)
    if perimeter == 0:
        return 0.0
    return (4 * np.pi * area) / (perimeter ** 2)


def pca_orientation(contour: np.ndarray) -> float:
    """
    Compute principal axis orientation of a contour via PCA.
    Returns the angle in degrees of the primary axis from horizontal.
    Used for piece orientation normalisation before DINOv2 embedding.
    """
    pts = contour.reshape(-1, 2).astype(np.float32)
    mean, eigenvectors, _ = cv2.PCACompute2(pts, mean=np.array([]))
    # Primary eigenvector angle from positive x-axis
    angle = float(np.degrees(np.arctan2(eigenvectors[0, 1], eigenvectors[0, 0])))
    return angle


def normalize_contour_orientation(
    img: np.ndarray, contour: np.ndarray
) -> tuple[np.ndarray, float]:
    """
    Rotate img so the contour's principal axis is horizontal.
    Returns (rotated_image, correction_angle_deg).
    The correction angle can be stored and later added to the
    matched rotation to report the true physical rotation to the user.
    """
    angle = pca_orientation(contour)
    rotated = rotate_image(img, -angle)
    return rotated, angle


# ─── Grid Coordinate Helpers ─────────────────────────────────────────────────

def grid_pos_to_pixel(
    row: int, col: int,
    patch_h: float, patch_w: float,
) -> tuple[int, int]:
    """Convert grid (row, col) to top-left pixel coordinate."""
    return int(col * patch_w), int(row * patch_h)


def pixel_to_grid_pos(
    x: int, y: int,
    patch_h: float, patch_w: float,
) -> tuple[int, int]:
    """Convert pixel (x, y) to nearest grid (row, col)."""
    return int(y / patch_h), int(x / patch_w)


def is_corner_pos(row: int, col: int, grid_rows: int, grid_cols: int) -> bool:
    return (row in (0, grid_rows - 1)) and (col in (0, grid_cols - 1))


def is_edge_pos(row: int, col: int, grid_rows: int, grid_cols: int) -> bool:
    if is_corner_pos(row, col, grid_rows, grid_cols):
        return False
    return row == 0 or row == grid_rows - 1 or col == 0 or col == grid_cols - 1


def expected_flat_sides(
    row: int, col: int, grid_rows: int, grid_cols: int
) -> int:
    """
    Return the number of flat (border) sides a piece at (row, col) should have.
    corner → 2, edge → 1, interior → 0.
    """
    if is_corner_pos(row, col, grid_rows, grid_cols):
        return 2
    if is_edge_pos(row, col, grid_rows, grid_cols):
        return 1
    return 0