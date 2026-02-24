# Copyright (c) 2026 Harsh Dwivedi
# Licensed under the Harsh Non-Commercial Attribution License (HNCAL) v1.0
# Commercial use requires written permission. See LICENSE for details.

"""
PieceWise — Piece Spatial Token Embedder
Extracts DINOv2 spatial token grids for each puzzle piece
across 4 rotations (0°, 90°, 180°, 270°).

Pipeline per piece:
  1. PCA orientation normalisation — rotate piece so its principal axis
     is horizontal (reduces rotational ambiguity before 4-rotation test)
  2. Paste alpha-masked piece onto neutral grey 518×518 background
  3. Apply 4 rotations
  4. Run DINOv2 in batches of DINO_BATCH_SIZE — all tensors on GPU
  5. Apply same PCA reduction used on reference tokens
  6. Cache to {job_id}/cache/piece_tokens.npz
"""

from __future__ import annotations

import cv2
import numpy as np
import torch

from app.config import get_settings
from app.models.piece import PieceCrop, PieceEmbedding
from app.modules.feature_extraction.dino_loader import (
    extract_tokens,
    get_device,
    get_dino_processor,
)
from app.modules.feature_extraction.pca_reducer import PCAReducer
from app.utils.geometry_utils import (
    ROTATION_VARIANTS,
    normalize_contour_orientation,
    rotate_image_90,
    rotation_deg_to_k,
)
from app.utils.image_utils import paste_on_background
from app.utils.logger import get_logger
from app.utils.storage import piece_tokens_cache_path

log = get_logger(__name__)

_DINO_INPUT_SIZE = 518
_GREY_BG = 128


def _prepare_piece_tensor(
    piece_img: np.ndarray,
    alpha_mask: np.ndarray,
    rotation_deg: int,
) -> torch.Tensor:
    """
    Prepare a single piece crop for DINOv2 inference at a given rotation.

    Steps:
      1. Paste piece onto neutral grey background (removes background content)
      2. Rotate by rotation_deg (0/90/180/270)
      3. Resize to DINO_INPUT_SIZE × DINO_INPUT_SIZE
      4. Apply DINOv2 processor normalisation
      5. Move to device

    Args:
        piece_img:    BGR uint8 piece crop
        alpha_mask:   uint8 binary mask (0/255)
        rotation_deg: Rotation in degrees (0, 90, 180, 270)

    Returns:
        (1, 3, H, W) float32 tensor on device
    """
    from PIL import Image

    device = get_device()
    processor = get_dino_processor()

    # Step 1: Paste on grey background
    bg = paste_on_background(piece_img, alpha_mask, size=_DINO_INPUT_SIZE)

    # Step 2: Rotate
    k = rotation_deg_to_k(rotation_deg)
    if k > 0:
        bg = rotate_image_90(bg, k=k)

    # Step 3+4: PIL → processor
    rgb = cv2.cvtColor(bg, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)

    inputs = processor(images=pil_img, return_tensors="pt")
    return inputs["pixel_values"].to(device)


def embed_pieces(
    pieces: list[PieceCrop],
    reducer: PCAReducer,
    job_id: str,
) -> list[PieceEmbedding]:
    """
    Extract DINOv2 spatial token grids for all pieces across 4 rotations.

    Args:
        pieces:  List of PieceCrop objects from segmentation
        reducer: FITTED PCAReducer (already fitted on reference tokens)
        job_id:  Job ID for cache file path

    Returns:
        List of PieceEmbedding objects, one per piece.
    """
    settings = get_settings()
    batch_size = settings.dino_batch_size
    embeddings: list[PieceEmbedding] = []

    log.info(
        "piece_embedding_start",
        n_pieces=len(pieces),
        n_rotations=len(ROTATION_VARIANTS),
        batch_size=batch_size,
        pca_enabled=reducer.enabled,
    )

    # Build flat list of (piece_id, rotation_deg, tensor) for batch processing
    all_items: list[tuple[int, int, torch.Tensor, float]] = []

    for piece in pieces:
        # Step 1: PCA orientation normalisation
        normalised_img, correction_deg = normalize_contour_orientation(
            piece.image, piece.contour
        )
        piece.pca_correction_deg = correction_deg

        for rot_deg in ROTATION_VARIANTS:
            tensor = _prepare_piece_tensor(
                normalised_img, piece.alpha_mask, rot_deg
            )
            all_items.append((piece.piece_id, rot_deg, tensor, correction_deg))

    # Process in batches
    # Map: piece_id → PieceEmbedding
    emb_map: dict[int, PieceEmbedding] = {
        p.piece_id: PieceEmbedding(piece_id=p.piece_id)
        for p in pieces
    }

    n_items = len(all_items)
    for batch_start in range(0, n_items, batch_size):
        batch = all_items[batch_start: batch_start + batch_size]

        for piece_id, rot_deg, tensor, _ in batch:
            patch_grid, cls_token = extract_tokens(tensor)

            # Move to CPU for PCA + storage
            grid_np = patch_grid.cpu().numpy()            # (H, W, D_orig)
            cls_np = cls_token.cpu().numpy()              # (D_orig,)

            H, W, D_orig = grid_np.shape

            # Apply PCA reduction (same reducer fitted on reference)
            flat = grid_np.reshape(-1, D_orig)
            reduced_flat = reducer.transform(flat)        # (H*W, D_out)
            D_out = reduced_flat.shape[1]
            reduced_grid = reduced_flat.reshape(H, W, D_out)

            # Reduce CLS token too
            cls_reduced = reducer.transform(cls_np.reshape(1, -1))[0]

            emb_map[piece_id].token_grids[rot_deg] = reduced_grid
            emb_map[piece_id].cls_vectors[rot_deg] = cls_reduced

        log.debug(
            "embedding_batch_complete",
            batch_start=batch_start,
            batch_end=min(batch_start + batch_size, n_items),
            total=n_items,
        )

    embeddings = list(emb_map.values())

    log.info(
        "piece_embedding_complete",
        total_pieces=len(embeddings),
        output_dims=reducer.output_dims,
    )

    # Cache to disk
    _save_piece_cache(embeddings, job_id)

    return embeddings


def _save_piece_cache(
    embeddings: list[PieceEmbedding],
    job_id: str,
) -> None:
    """Persist piece embeddings to npz for job reuse."""
    cache_path = piece_tokens_cache_path(job_id)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    arrays: dict[str, np.ndarray] = {}
    piece_ids = np.array([e.piece_id for e in embeddings])
    rotations = np.array(ROTATION_VARIANTS)

    for emb in embeddings:
        pid = emb.piece_id
        for rot in ROTATION_VARIANTS:
            if rot in emb.token_grids:
                arrays[f"grid_{pid}_{rot}"] = emb.token_grids[rot]
            if rot in emb.cls_vectors:
                arrays[f"cls_{pid}_{rot}"] = emb.cls_vectors[rot]

    np.savez_compressed(
        str(cache_path),
        piece_ids=piece_ids,
        rotations=rotations,
        **arrays,
    )
    log.debug("piece_token_cache_saved", path=str(cache_path))