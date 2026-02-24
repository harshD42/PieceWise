# Copyright (c) 2026 Harsh Dwivedi
# Licensed under the Harsh Non-Commercial Attribution License (HNCAL) v1.0
# Commercial use requires written permission. See LICENSE for details.

"""
PieceWise — SAM Model Loader
Singleton loader for Meta's Segment Anything Model.
Loaded once at application startup via FastAPI lifespan.
Supports ViT-B (fast) and ViT-H (accurate, default for 1000-piece puzzles).
Auto-detects CUDA / MPS / CPU.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from app.config import get_settings
from app.utils.logger import get_logger

if TYPE_CHECKING:
    from segment_anything import SamAutomaticMaskGenerator, SamPredictor

log = get_logger(__name__)

# ─── Module-level singletons ─────────────────────────────────────────────────
_sam_model = None
_mask_generator: "SamAutomaticMaskGenerator | None" = None
_predictor: "SamPredictor | None" = None


def _get_device() -> str:
    """Auto-detect the best available compute device."""
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    log.info("device_selected", device=device)
    return device


def init_sam() -> None:
    """
    Load the SAM model and initialise the automatic mask generator.
    Called once during FastAPI lifespan startup.
    Safe to call multiple times — subsequent calls are no-ops.
    """
    global _sam_model, _mask_generator, _predictor

    if _sam_model is not None:
        log.debug("sam_already_loaded")
        return

    try:
        from segment_anything import (
            SamAutomaticMaskGenerator,
            SamPredictor,
            sam_model_registry,
        )
    except ImportError as e:
        raise ImportError(
            "segment-anything package not found. "
            "Run: pip install git+https://github.com/facebookresearch/segment-anything.git"
        ) from e

    settings = get_settings()
    checkpoint = settings.sam_checkpoint_path

    if not checkpoint.exists():
        raise FileNotFoundError(
            f"SAM checkpoint not found at {checkpoint}. "
            "Run: python scripts/download_models.py"
        )

    device = _get_device()

    log.info(
        "sam_loading",
        model_type=settings.sam_model_type,
        checkpoint=str(checkpoint),
        device=device,
    )

    _sam_model = sam_model_registry[settings.sam_model_type](
        checkpoint=str(checkpoint)
    )
    _sam_model.to(device=device)

    # Automatic mask generator — tuned for 1000-piece puzzles
    # points_per_side=64: dense prompting for small pieces
    # min_mask_region_area=300: lower threshold for small pieces
    _mask_generator = SamAutomaticMaskGenerator(
        model=_sam_model,
        points_per_side=settings.sam_points_per_side,
        pred_iou_thresh=settings.sam_pred_iou_thresh,
        stability_score_thresh=settings.sam_stability_score_thresh,
        min_mask_region_area=settings.sam_min_mask_area,
    )

    # Point-prompted predictor (used for targeted re-segmentation)
    _predictor = SamPredictor(_sam_model)

    log.info(
        "sam_loaded",
        model_type=settings.sam_model_type,
        device=device,
        points_per_side=settings.sam_points_per_side,
    )


def get_mask_generator() -> "SamAutomaticMaskGenerator":
    """
    Return the SAM automatic mask generator singleton.
    Raises RuntimeError if init_sam() has not been called.
    """
    if _mask_generator is None:
        raise RuntimeError(
            "SAM mask generator not initialised. "
            "Ensure init_sam() is called during app lifespan startup."
        )
    return _mask_generator


def get_predictor() -> "SamPredictor":
    """Return the SAM point predictor singleton."""
    if _predictor is None:
        raise RuntimeError(
            "SAM predictor not initialised. "
            "Ensure init_sam() is called during app lifespan startup."
        )
    return _predictor


def is_loaded() -> bool:
    """Return True if SAM has been successfully loaded."""
    return _sam_model is not None