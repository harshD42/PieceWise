# Copyright (c) 2026 Harsh Dwivedi
# Licensed under the Harsh Non-Commercial Attribution License (HNCAL) v1.0
# Commercial use requires written permission. See LICENSE for details.

"""
PieceWise — DINOv2 Model Loader
Singleton loader for Meta's DINOv2 ViT-B/14 model.
Loaded once at application startup via FastAPI lifespan.

Key design decisions:
- Uses a forward hook to extract the full patch token grid
  (H/14 × W/14 × 768) rather than only the CLS token.
  This preserves spatial structure needed for local region matching.
- All tensors remain on GPU throughout — no .cpu() in the hot path.
- Same device as SAM to avoid cross-device transfers.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from app.config import get_settings
from app.utils.logger import get_logger

log = get_logger(__name__)

# ─── Module-level singletons ─────────────────────────────────────────────────
_dino_model: nn.Module | None = None
_dino_processor = None
_device: str = "cpu"

# Storage for the patch tokens extracted by the forward hook
_patch_tokens: torch.Tensor | None = None


def _get_device() -> str:
    """Return the device already selected by SAM loader, or auto-detect."""
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _make_patch_hook():
    """
    Create a forward hook that captures the patch token output
    from DINOv2's final transformer block.

    DINOv2 output shape from last block: (batch, n_patches + 1, embed_dim)
    - Index 0: CLS token
    - Indices 1..: patch tokens (H/14 × W/14 spatial grid)
    """
    def hook(module: nn.Module, input: Any, output: Any) -> None:
        global _patch_tokens
        # output shape: (B, N+1, D) — keep on device
        _patch_tokens = output  # full output including CLS
    return hook


def init_dino() -> None:
    """
    Load DINOv2 ViT-B/14 and register the patch token extraction hook.
    Called once during FastAPI lifespan startup after SAM is loaded.
    Safe to call multiple times — subsequent calls are no-ops.
    """
    global _dino_model, _dino_processor, _device

    if _dino_model is not None:
        log.debug("dino_already_loaded")
        return

    try:
        from transformers import AutoImageProcessor, AutoModel
    except ImportError as e:
        raise ImportError(
            "transformers package not found. "
            "Run: pip install transformers"
        ) from e

    settings = get_settings()
    _device = _get_device()
    cache_dir = str(settings.dino_cache_dir)

    log.info(
        "dino_loading",
        model=settings.dino_model_name,
        device=_device,
        cache_dir=cache_dir,
    )

    _dino_processor = AutoImageProcessor.from_pretrained(
        settings.dino_model_name,
        cache_dir=cache_dir,
    )

    model = AutoModel.from_pretrained(
        settings.dino_model_name,
        cache_dir=cache_dir,
    )
    model.eval()
    model.to(_device)

    # Register forward hook on the last encoder layer to capture patch tokens
    # DINOv2 (ViT-B/14): encoder.layer[-1] is the final transformer block
    last_layer = model.encoder.layer[-1]
    last_layer.register_forward_hook(_make_patch_hook())

    _dino_model = model

    log.info(
        "dino_loaded",
        model=settings.dino_model_name,
        device=_device,
        patch_size=14,
    )


def get_dino_model() -> nn.Module:
    """Return the DINOv2 model singleton."""
    if _dino_model is None:
        raise RuntimeError(
            "DINOv2 model not initialised. "
            "Ensure init_dino() is called during app lifespan startup."
        )
    return _dino_model


def get_dino_processor():
    """Return the DINOv2 image processor singleton."""
    if _dino_processor is None:
        raise RuntimeError(
            "DINOv2 processor not initialised. "
            "Ensure init_dino() is called during app lifespan startup."
        )
    return _dino_processor


def get_device() -> str:
    """Return the device string used by DINOv2."""
    return _device


def is_loaded() -> bool:
    """Return True if DINOv2 has been successfully loaded."""
    return _dino_model is not None


@torch.no_grad()
def extract_tokens(
    image_tensor: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Run DINOv2 on a preprocessed image tensor and extract:
      - patch_grid: (H/14, W/14, D) spatial token grid — on device
      - cls_token:  (D,) CLS embedding — on device

    Args:
        image_tensor: Preprocessed image tensor (1, 3, H, W) on device.
                      Must be divisible by 14 in H and W dimensions.

    Returns:
        (patch_grid, cls_token) — both on the same device as the model.
    """
    global _patch_tokens
    _patch_tokens = None

    model = get_dino_model()
    _ = model(pixel_values=image_tensor)

    if _patch_tokens is None:
        raise RuntimeError("Patch token hook did not fire — check model architecture.")

    # _patch_tokens shape: (1, N+1, D) where N = (H/14)*(W/14)
    tokens = _patch_tokens[0]  # (N+1, D)
    cls_token = tokens[0]       # (D,)
    patch_tokens = tokens[1:]   # (N, D)

    # Reshape to spatial grid
    h_patches = image_tensor.shape[2] // 14
    w_patches = image_tensor.shape[3] // 14
    patch_grid = patch_tokens.reshape(h_patches, w_patches, -1)  # (H/14, W/14, D)

    return patch_grid, cls_token