# Copyright (c) 2026 Harsh Dwivedi
# Licensed under the Harsh Non-Commercial Attribution License (HNCAL) v1.0
# Commercial use requires written permission. See LICENSE for details.

"""
PieceWise — Segmentation Module
Public API for the segmentation stage.
"""

from app.modules.segmentation.contour_analyzer import analyze_contours
from app.modules.segmentation.mask_filter import filter_masks
from app.modules.segmentation.mask_generator import (
    connected_components_prefilter,
    generate_masks,
)
from app.modules.segmentation.piece_extractor import extract_pieces
from app.modules.segmentation.sam_loader import (
    get_mask_generator,
    get_predictor,
    init_sam,
    is_loaded,
)
from app.modules.segmentation.segmentation_refiner import refine_masks

__all__ = [
    # SAM loader
    "init_sam",
    "get_mask_generator",
    "get_predictor",
    "is_loaded",
    # Mask generation
    "connected_components_prefilter",
    "generate_masks",
    # Mask filtering
    "filter_masks",
    # Segmentation refiner
    "refine_masks",
    # Piece extraction
    "extract_pieces",
    # Contour analysis
    "analyze_contours",
]