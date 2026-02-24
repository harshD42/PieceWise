# Copyright (c) 2026 Harsh Dwivedi
# Licensed under the Harsh Non-Commercial Attribution License (HNCAL) v1.0
# Commercial use requires written permission. See LICENSE for details.

"""
PieceWise — Spatial Token Similarity (Fine Stage)
Computes fine-grained spatial similarity between a piece's token grid
and a reference cell's pre-normalised flat token matrix.

Uses torch.mm on GPU for maximum throughput — all tensors stay on device.
Reference tokens are pre-normalised once in Phase 4 (patch_generator.py).
Piece tokens are L2-normalised here per query.

Similarity is computed as mean-pooled cosine similarity across the
spatial token grid, producing a single scalar score per (piece, cell) pair.

This is the FINE stage — only called for the top-K shortlisted cells
from the coarse CLS filter (candidate_selector.py).
"""

from __future__ import annotations

import numpy as np
import torch

from app.utils.logger import get_logger

log = get_logger(__name__)


def _l2_normalise_tensor(t: torch.Tensor) -> torch.Tensor:
    """
    L2-normalise rows of a 2D tensor.
    Safe against zero-norm rows (returns zero vector).
    """
    norms = torch.norm(t, dim=1, keepdim=True).clamp(min=1e-8)
    return t / norms


def spatial_similarity(
    piece_token_grid: np.ndarray,
    ref_cell_flat_normalised: np.ndarray,
    device: str = "cpu",
) -> float:
    """
    Compute spatial token similarity between a piece and one reference cell.

    Args:
        piece_token_grid:         (H, W, D) float32 — piece token grid
        ref_cell_flat_normalised: (N_ref, D) float32 — pre-L2-normalised
                                  reference cell tokens (cached from Phase 4)
        device:                   Compute device string

    Returns:
        Scalar float in [0, 1] — mean cosine similarity across token pairs.
        Higher = better spatial match.

    Algorithm:
        1. Flatten piece tokens → (N_piece, D)
        2. L2-normalise piece tokens (reference already normalised)
        3. torch.mm(piece_flat, ref_flat.T) → (N_piece, N_ref) similarity matrix
        4. For each piece token: take max similarity to any ref token
        5. Mean of those max similarities → scalar score

    The max-then-mean pooling gives credit for partial region matches,
    which is important when piece crops have slightly different aspect
    ratios than the corresponding grid cell.
    """
    # Flatten piece token grid
    H, W, D = piece_token_grid.shape
    n_piece = H * W

    # Move to device as float32 tensors
    piece_flat = torch.from_numpy(
        piece_token_grid.reshape(n_piece, D).astype(np.float32)
    ).to(device)

    ref_flat = torch.from_numpy(
        ref_cell_flat_normalised.astype(np.float32)
    ).to(device)

    # L2-normalise piece tokens
    piece_norm = _l2_normalise_tensor(piece_flat)   # (N_piece, D)

    # Cosine similarity matrix: (N_piece, N_ref)
    sim_matrix = torch.mm(piece_norm, ref_flat.T)

    # Max-then-mean pooling
    max_sim_per_piece = sim_matrix.max(dim=1).values  # (N_piece,)
    score = float(max_sim_per_piece.mean().item())

    # Clamp to [0, 1] — cosine similarity can be negative for dissimilar tokens
    return max(0.0, min(1.0, score))


def batch_spatial_similarity(
    piece_token_grid: np.ndarray,
    candidate_cells: list[tuple[tuple[int, int], np.ndarray]],
    device: str = "cpu",
) -> list[tuple[tuple[int, int], float]]:
    """
    Compute spatial similarity for a piece against multiple candidate cells.
    More efficient than calling spatial_similarity() in a loop because
    the piece tokens are moved to GPU only once.

    Args:
        piece_token_grid: (H, W, D) float32 piece token grid
        candidate_cells:  List of (grid_pos, ref_flat_normalised) tuples
        device:           Compute device string

    Returns:
        List of (grid_pos, similarity_score) tuples, same order as input.
    """
    H, W, D = piece_token_grid.shape
    n_piece = H * W

    # Move piece to GPU once
    piece_flat = torch.from_numpy(
        piece_token_grid.reshape(n_piece, D).astype(np.float32)
    ).to(device)
    piece_norm = _l2_normalise_tensor(piece_flat)  # (N_piece, D)

    results: list[tuple[tuple[int, int], float]] = []

    for grid_pos, ref_flat in candidate_cells:
        ref_t = torch.from_numpy(
            ref_flat.astype(np.float32)
        ).to(device)

        sim_matrix = torch.mm(piece_norm, ref_t.T)
        max_sim = sim_matrix.max(dim=1).values
        score = float(max(0.0, min(1.0, max_sim.mean().item())))
        results.append((grid_pos, score))

    return results