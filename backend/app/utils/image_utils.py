# Copyright (c) 2026 Harsh Dwivedi
# Licensed under the Harsh Non-Commercial Attribution License (HNCAL) v1.0
# Commercial use requires written permission. See LICENSE for details.

"""
PieceWise — Image I/O and Conversion Utilities
Shared helpers used across preprocessing, segmentation, and rendering.
All internal processing uses BGR numpy arrays (OpenCV convention).
Conversion to/from RGB happens only at DINOv2 inference boundaries.
"""

from pathlib import Path

import cv2
import numpy as np
from PIL import Image


# ─── Load / Save ─────────────────────────────────────────────────────────────

def load_image_bgr(path: Path) -> np.ndarray:
    """
    Load an image from disk as a BGR uint8 numpy array.
    Raises FileNotFoundError if path does not exist.
    Raises ValueError if the file cannot be decoded as an image.
    """
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Could not decode image: {path}")
    return img


def load_image_rgb(path: Path) -> np.ndarray:
    """Load image as RGB uint8 numpy array (for DINOv2 / PIL consumers)."""
    return bgr_to_rgb(load_image_bgr(path))


def save_image(img: np.ndarray, path: Path, quality: int = 92) -> None:
    """
    Save a BGR numpy array as JPEG.
    Creates parent directories if they don't exist.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img, [cv2.IMWRITE_JPEG_QUALITY, quality])


def bytes_to_bgr(data: bytes) -> np.ndarray:
    """Decode raw image bytes (from upload) to BGR numpy array."""
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode uploaded image bytes.")
    return img


def bgr_to_png_bytes(img: np.ndarray) -> bytes:
    """Encode a BGR numpy array to PNG bytes (lossless)."""
    success, buf = cv2.imencode(".png", img)
    if not success:
        raise RuntimeError("Failed to encode image to PNG bytes.")
    return buf.tobytes()


# ─── Color Space ─────────────────────────────────────────────────────────────

def bgr_to_rgb(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def rgb_to_bgr(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def bgr_to_gray(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# ─── Resize ──────────────────────────────────────────────────────────────────

def resize_long_edge(img: np.ndarray, max_long_edge: int) -> tuple[np.ndarray, float]:
    """
    Resize image so its longest edge equals max_long_edge.
    Preserves aspect ratio. Returns (resized_image, scale_factor).
    Scale factor < 1.0 means the image was downscaled.
    """
    h, w = img.shape[:2]
    long_edge = max(h, w)
    if long_edge <= max_long_edge:
        return img.copy(), 1.0
    scale = max_long_edge / long_edge
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized, scale


def resize_to_square(img: np.ndarray, size: int) -> np.ndarray:
    """Resize image to size×size (used for DINOv2 input preprocessing)."""
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)


# ─── Padding / Cropping ──────────────────────────────────────────────────────

def crop_with_padding(
    img: np.ndarray,
    x: int, y: int, w: int, h: int,
    pad: int = 8,
) -> np.ndarray:
    """
    Crop a region from img with optional padding.
    Clamps to image boundaries — never raises on out-of-bounds coords.
    """
    ih, iw = img.shape[:2]
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(iw, x + w + pad)
    y2 = min(ih, y + h + pad)
    return img[y1:y2, x1:x2].copy()


def paste_on_background(
    img: np.ndarray,
    alpha: np.ndarray,
    size: int = 224,
    bg_value: int = 128,
) -> np.ndarray:
    """
    Resize piece crop and its alpha mask to size×size,
    paste onto a neutral grey background.
    Used to prepare piece images for DINOv2 inference.

    Args:
        img:       BGR crop of the piece (H×W×3)
        alpha:     Binary alpha mask (H×W), 0=background 255=piece
        size:      Target square size (DINOv2 default 224)
        bg_value:  Grey background intensity (128 ≈ ImageNet mean)

    Returns:
        BGR image of shape (size, size, 3)
    """
    piece_resized = cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)
    mask_resized = cv2.resize(alpha, (size, size), interpolation=cv2.INTER_NEAREST)

    background = np.full((size, size, 3), bg_value, dtype=np.uint8)
    mask_3ch = mask_resized[:, :, np.newaxis].astype(np.float32) / 255.0
    result = (piece_resized.astype(np.float32) * mask_3ch
              + background.astype(np.float32) * (1.0 - mask_3ch))
    return result.astype(np.uint8)


# ─── PIL Bridge ──────────────────────────────────────────────────────────────

def bgr_to_pil(img: np.ndarray) -> Image.Image:
    """Convert BGR numpy array to PIL Image (RGB mode)."""
    return Image.fromarray(bgr_to_rgb(img))


def pil_to_bgr(pil_img: Image.Image) -> np.ndarray:
    """Convert PIL Image to BGR numpy array."""
    return rgb_to_bgr(np.array(pil_img.convert("RGB")))


# ─── Validation ──────────────────────────────────────────────────────────────

def is_valid_image_bytes(data: bytes) -> bool:
    """Return True if bytes can be decoded as a valid BGR image."""
    try:
        img = bytes_to_bgr(data)
        return img is not None and img.ndim == 3 and img.shape[2] == 3
    except Exception:
        return False