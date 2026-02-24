# Copyright (c) 2026 Harsh Dwivedi
# Licensed under the Harsh Non-Commercial Attribution License (HNCAL) v1.0
# Commercial use requires written permission. See LICENSE for details.

"""
PieceWise — Image Validator
Validates uploaded image files before they enter the pipeline.
Checks format, file size, image decodability, channel count,
and minimum resolution.

Raises ImageValidationError (subclass of ValueError) on any failure
so the API error handler maps it cleanly to HTTP 422.
"""

from pathlib import Path

import cv2
import numpy as np

from app.api.middleware.error_handler import ImageValidationError
from app.config import get_settings
from app.utils.logger import get_logger

log = get_logger(__name__)

# Supported formats by magic bytes (first few bytes of file)
_MAGIC_BYTES: dict[str, bytes] = {
    "jpeg": b"\xff\xd8\xff",
    "png":  b"\x89PNG",
    "webp": b"RIFF",          # RIFF....WEBP — checked further below
}

# Minimum acceptable resolution in either dimension
_MIN_DIMENSION_PX = 64

# Maximum acceptable resolution (prevents absurdly large images slipping
# past the file-size check due to very high compression ratios)
_MAX_DIMENSION_PX = 16_000


def _detect_format(data: bytes) -> str | None:
    """
    Detect image format from magic bytes.
    Returns format string ('jpeg', 'png', 'webp') or None if unrecognised.
    """
    if data[:3] == _MAGIC_BYTES["jpeg"]:
        return "jpeg"
    if data[:4] == _MAGIC_BYTES["png"]:
        return "png"
    # WebP: RIFF????WEBP
    if data[:4] == _MAGIC_BYTES["webp"] and data[8:12] == b"WEBP":
        return "webp"
    return None


def validate_image_bytes(
    data: bytes,
    label: str = "image",
) -> np.ndarray:
    """
    Validate raw image bytes and return a decoded BGR numpy array.

    Checks performed (in order):
      1. Non-empty bytes
      2. File size within configured limit
      3. Magic byte format detection (JPEG / PNG / WebP only)
      4. OpenCV decodability
      5. Exactly 3 channels (RGB — no grayscale, no RGBA)
      6. Minimum resolution (both dimensions ≥ MIN_DIMENSION_PX)
      7. Maximum resolution guard

    Args:
        data:  Raw bytes from upload or file read.
        label: Human-readable label used in error messages ('reference image',
               'pieces image', etc.)

    Returns:
        Decoded BGR uint8 numpy array (H × W × 3).

    Raises:
        ImageValidationError: On any validation failure.
    """
    settings = get_settings()

    # 1. Non-empty
    if not data:
        raise ImageValidationError(f"The {label} file is empty.")

    # 2. File size
    size_mb = len(data) / (1024 * 1024)
    if len(data) > settings.upload_max_bytes:
        raise ImageValidationError(
            f"The {label} file is {size_mb:.1f} MB, which exceeds the "
            f"maximum allowed size of {settings.upload_max_mb} MB."
        )

    # 3. Magic bytes format check
    fmt = _detect_format(data)
    if fmt is None:
        raise ImageValidationError(
            f"The {label} file format is not supported. "
            "Please upload a JPEG, PNG, or WebP image."
        )

    # 4. OpenCV decodability
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ImageValidationError(
            f"The {label} file could not be decoded. "
            "The file may be corrupted or truncated."
        )

    h, w = img.shape[:2]

    # 5. Channel count
    if img.ndim != 3 or img.shape[2] != 3:
        raise ImageValidationError(
            f"The {label} must be a 3-channel colour image. "
            f"Got shape {img.shape}. Grayscale and RGBA images are not supported."
        )

    # 6. Minimum resolution
    if h < _MIN_DIMENSION_PX or w < _MIN_DIMENSION_PX:
        raise ImageValidationError(
            f"The {label} resolution ({w}×{h}px) is too small. "
            f"Both dimensions must be at least {_MIN_DIMENSION_PX}px."
        )

    # 7. Maximum resolution guard
    if h > _MAX_DIMENSION_PX or w > _MAX_DIMENSION_PX:
        raise ImageValidationError(
            f"The {label} resolution ({w}×{h}px) exceeds the maximum "
            f"allowed dimension of {_MAX_DIMENSION_PX}px. "
            "Please downscale the image before uploading."
        )

    log.debug(
        "image_validated",
        label=label,
        format=fmt,
        shape=img.shape,
        size_mb=round(size_mb, 2),
    )
    return img


def validate_image_file(path: Path, label: str = "image") -> np.ndarray:
    """
    Read a file from disk and validate it.
    Convenience wrapper around validate_image_bytes for pipeline use.

    Args:
        path:  Path to the image file.
        label: Human-readable label for error messages.

    Returns:
        Decoded BGR uint8 numpy array.

    Raises:
        ImageValidationError: If the file does not exist or fails validation.
    """
    if not path.exists():
        raise ImageValidationError(
            f"The {label} file was not found at path: {path}"
        )
    if not path.is_file():
        raise ImageValidationError(
            f"The {label} path does not point to a file: {path}"
        )

    data = path.read_bytes()
    return validate_image_bytes(data, label=label)


def validate_image_pair(
    ref_path: Path,
    pieces_path: Path,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Validate both the reference and pieces images in one call.
    Returns (ref_img_bgr, pieces_img_bgr).
    Raises ImageValidationError on the first failure encountered.
    """
    ref_img = validate_image_file(ref_path, label="reference image")
    pieces_img = validate_image_file(pieces_path, label="pieces image")
    return ref_img, pieces_img