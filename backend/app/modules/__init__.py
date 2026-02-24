# Copyright (c) 2026 Harsh Dwivedi
# Licensed under the Harsh Non-Commercial Attribution License (HNCAL) v1.0
# Commercial use requires written permission. See LICENSE for details.

"""
PieceWise — Preprocessing Module
Public API for the preprocessing stage.
"""

from app.modules.preprocessing.grid_estimator import (
    estimate_grid_shape,
    patch_size_px,
)
from app.modules.preprocessing.normalizer import (
    NormalisedPair,
    match_histogram,
    normalize_from_arrays,
    normalize_image_pair,
)
from app.modules.preprocessing.validator import (
    validate_image_bytes,
    validate_image_file,
    validate_image_pair,
)

__all__ = [
    # Validator
    "validate_image_bytes",
    "validate_image_file",
    "validate_image_pair",
    # Normalizer
    "NormalisedPair",
    "normalize_image_pair",
    "normalize_from_arrays",
    "match_histogram",
    # Grid estimator
    "estimate_grid_shape",
    "patch_size_px",
]