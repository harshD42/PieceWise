# Copyright (c) 2026 Harsh Dwivedi
# Licensed under the Harsh Non-Commercial Attribution License (HNCAL) v1.0
# Commercial use requires written permission. See LICENSE for details.

"""
PieceWise — Image Validator (Phase 2 stub)
Full implementation delivered in Phase 2.
"""

from pathlib import Path

from app.api.middleware.error_handler import ImageValidationError
from app.utils.image_utils import load_image_bgr
from app.utils.logger import get_logger

log = get_logger(__name__)


def validate_image_file(path: Path) -> None:
    """
    Validate that a file at path is a readable, 3-channel image.
    Raises ImageValidationError on failure.
    Phase 2 will add size limits and format checks.
    """
    try:
        img = load_image_bgr(path)
        if img.ndim != 3 or img.shape[2] != 3:
            raise ImageValidationError(
                f"Image at {path} is not a 3-channel RGB image."
            )
        log.debug("image_validated", path=str(path), shape=img.shape)
    except FileNotFoundError:
        raise ImageValidationError(f"Image file not found: {path}")
    except ValueError as e:
        raise ImageValidationError(str(e))