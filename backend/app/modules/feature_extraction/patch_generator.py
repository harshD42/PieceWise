# Copyright (c) 2026 Harsh Dwivedi
# Licensed under the Harsh Non-Commercial Attribution License (HNCAL) v1.0
# Commercial use requires written permission. See LICENSE for details.

"""
PieceWise — Reference Patch Token Extractor
Extracts DINOv2 spatial token grids for the reference image,
subdivides them into the puzzle grid cell layout, and builds
the pre-normalised token cache used by the matching engine.

Key design decisions (from sequence diagram + architecture review):
1. Run DINOv2 on the FULL reference image — extract all patch tokens at once.
   This is more efficient than running DINOv2 per cell (no redundant
   computation at cell boundaries).
2. Subdivide the full token grid into N×M cell regions.
   Each cell gets a 10% overlap border for context.
3. Pre-normalise (L2) and pre-flatten ALL cell token grids ONCE here.
   The matching engine reads from this cache — never recomputes per piece.
4. Also extract CLS token per cell → stacked into cls_matrix for the
   coarse-to-fine filter (cheap CLS cosine similarity before spatial torch.mm).
5. All tensors stay on GPU. npz cache persists to disk for job reuse.
"""

from __future__ import annotations

import numpy as np
import torch

from app.models.grid import PatchTokenCell, PatchTokenMap
from app.modules.feature_extraction.dino_loader import (
    extract_tokens,
    get_device,
    get_dino_processor,
)
from app.modules.feature_extraction.pca_reducer import PCAReducer
from app.utils.logger import get_logger
from app.utils.storage import reference_tokens_cache_path

log = get_logger(__name__)

# DINOv2 ViT-B/14 patch size
_PATCH_SIZE = 14

# Overlap fraction for cell context borders (10% per side)
_OVERLAP_FRACTION = 0.10

# Input size for DINOv2 — must be divisible by patch_size (14)
# 518 = 37 × 14 — gives 37×37 patch grid for a square image
_DINO_INPUT_SIZE = 518


def _prepare_image_tensor(img_bgr: np.ndarray) -> torch.Tensor:
    """
    Preprocess a BGR numpy image for DINOv2 inference.
    Converts to RGB, resizes to DINO_INPUT_SIZE, normalises.
    Returns (1, 3, H, W) tensor on the correct device.
    """
    import cv2
    from PIL import Image

    device = get_device()
    processor = get_dino_processor()

    # BGR → RGB → PIL
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb).resize(
        (_DINO_INPUT_SIZE, _DINO_INPUT_SIZE), Image.BICUBIC
    )

    inputs = processor(images=pil_img, return_tensors="pt")
    return inputs["pixel_values"].to(device)


def _l2_normalise_rows(matrix: np.ndarray) -> np.ndarray:
    """L2-normalise each row of a 2D matrix."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return matrix / norms


def extract_reference_tokens(
    ref_img: np.ndarray,
    grid_shape: tuple[int, int],
    reducer: PCAReducer,
    job_id: str,
) -> PatchTokenMap:
    """
    Extract and cache DINOv2 spatial token grids for the reference image.

    Steps:
      1. Run DINOv2 on the full reference image → full patch token grid
      2. Apply PCA reduction if enabled
      3. Subdivide token grid into N×M cell regions (with overlap)
      4. Pre-normalise (L2) and pre-flatten each cell token grid → cached
      5. Extract CLS token per cell → stack into cls_matrix
      6. Save to {job_id}/cache/reference_tokens.npz

    Args:
        ref_img:     BGR uint8 normalised reference image
        grid_shape:  (n_rows, n_cols) puzzle grid dimensions
        reducer:     Fitted PCAReducer (fit called before this function)
        job_id:      Job ID for cache file path

    Returns:
        PatchTokenMap with all cells populated and cls_matrix built.
    """
    n_rows, n_cols = grid_shape
    device = get_device()

    log.info(
        "reference_token_extraction_start",
        grid_shape=grid_shape,
        pca_enabled=reducer.enabled,
        pca_dims=reducer.output_dims,
    )

    # Step 1: Run DINOv2 on full reference image
    img_tensor = _prepare_image_tensor(ref_img)
    patch_grid, _ = extract_tokens(img_tensor)
    # patch_grid: (H_tok, W_tok, D) on device — H_tok = W_tok = 37 for 518px

    H_tok, W_tok, D_orig = patch_grid.shape

    # Move to CPU numpy for PCA (sklearn operates on CPU)
    grid_np = patch_grid.cpu().numpy().reshape(-1, D_orig)  # (H*W, D)

    # Step 2: Fit PCA on reference tokens (first call — fit here)
    reduced_flat = reducer.fit_transform(grid_np)           # (H*W, D_out)
    D_out = reduced_flat.shape[1]

    # Reshape back to spatial grid
    reduced_grid = reduced_flat.reshape(H_tok, W_tok, D_out)  # (H, W, D_out)

    log.debug(
        "reference_tokens_extracted",
        token_grid_shape=reduced_grid.shape,
        output_dims=D_out,
    )

    # Step 3: Subdivide into N×M cells
    patch_token_map = PatchTokenMap(
        grid_shape=grid_shape,
        patch_size_px=(
            ref_img.shape[0] // n_rows,
            ref_img.shape[1] // n_cols,
        ),
    )

    # Token grid cell dimensions
    cell_h = H_tok / n_rows
    cell_w = W_tok / n_cols
    overlap_h = max(1, int(cell_h * _OVERLAP_FRACTION))
    overlap_w = max(1, int(cell_w * _OVERLAP_FRACTION))

    cls_vectors: list[np.ndarray] = []
    cls_index: list[tuple[int, int]] = []

    for row in range(n_rows):
        for col in range(n_cols):
            # Cell token bounds (with overlap, clamped)
            r0 = max(0, int(row * cell_h) - overlap_h)
            r1 = min(H_tok, int((row + 1) * cell_h) + overlap_h)
            c0 = max(0, int(col * cell_w) - overlap_w)
            c1 = min(W_tok, int((col + 1) * cell_w) + overlap_w)

            cell_grid = reduced_grid[r0:r1, c0:c1, :]      # (h, w, D_out)
            n_cell_tokens = (r1 - r0) * (c1 - c0)

            # Step 4: Pre-normalise + pre-flatten
            cell_flat = cell_grid.reshape(n_cell_tokens, D_out)
            cell_flat_norm = _l2_normalise_rows(cell_flat)  # (N_tokens, D_out)

            # CLS for this cell = mean of its patch tokens (no CLS token per cell)
            cls_vec = cell_flat_norm.mean(axis=0)           # (D_out,)
            cls_norm = np.linalg.norm(cls_vec)
            if cls_norm > 0:
                cls_vec = cls_vec / cls_norm

            cls_vectors.append(cls_vec)
            cls_index.append((row, col))

            cell = PatchTokenCell(
                grid_pos=(row, col),
                token_grid=cell_grid,
                token_flat_normalised=cell_flat_norm,
                cls_vector=cls_vec,
            )
            patch_token_map.set_cell(cell)

    # Step 5: Stack CLS matrix for fast coarse filtering
    patch_token_map.cls_matrix = np.stack(cls_vectors, axis=0)   # (N*M, D_out)
    patch_token_map.cls_index = cls_index

    log.info(
        "reference_token_map_built",
        total_cells=patch_token_map.total_cells,
        cls_matrix_shape=patch_token_map.cls_matrix.shape,
    )

    # Step 6: Save cache to disk
    _save_reference_cache(patch_token_map, job_id, D_out)

    return patch_token_map


def _save_reference_cache(
    token_map: PatchTokenMap,
    job_id: str,
    d_out: int,
) -> None:
    """Persist reference token map to npz for job reuse."""
    cache_path = reference_tokens_cache_path(job_id)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    # Collect all cell data
    cell_grids = {}
    cell_flats = {}
    for key, cell in token_map.cells.items():
        cell_grids[f"grid_{key}"] = cell.token_grid
        cell_flats[f"flat_{key}"] = cell.token_flat_normalised

    np.savez_compressed(
        str(cache_path),
        cls_matrix=token_map.cls_matrix,
        grid_shape=np.array(token_map.grid_shape),
        patch_size_px=np.array(token_map.patch_size_px),
        d_out=np.array([d_out]),
        **cell_grids,
        **cell_flats,
    )
    log.debug("reference_token_cache_saved", path=str(cache_path))