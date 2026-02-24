# Copyright (c) 2026 Harsh Dwivedi
# Licensed under the Harsh Non-Commercial Attribution License (HNCAL) v1.0
# Commercial use requires written permission. See LICENSE for details.

"""
PieceWise — Coarse-to-Fine Candidate Selector

Stage 1 (COARSE): CLS cosine similarity filter
  For each piece × rotation: compute cosine similarity between the
  piece CLS vector and ALL reference cell CLS vectors in one vectorised
  torch.mm call. Select top-K cells.
  Cost: O(pieces × rotations × cells) — cheap, all on GPU.

Stage 2 (FINE): Spatial token correlation
  For each piece × rotation: run spatial_similarity() against its
  top-K shortlisted cells only (not all cells).
  Cost: O(pieces × rotations × K) — K << total_cells for large puzzles.

This reduces the spatial correlation cost from:
  1000 pieces × 4 rotations × 1000 cells = 4M comparisons
to:
  1000 pieces × 4 rotations × 30 cells  = 120K comparisons
— a 33× reduction.
"""

from __future__ import annotations

import numpy as np
import torch

from app.config import get_settings
from app.modules.feature_extraction.embedding_store import EmbeddingStore
from app.modules.matching.similarity import batch_spatial_similarity
from app.utils.geometry_utils import ROTATION_VARIANTS
from app.utils.logger import get_logger

log = get_logger(__name__)


def _cosine_similarity_matrix(
    query: np.ndarray,          # (D,) L2-normalised query vector
    key_matrix: torch.Tensor,   # (N, D) L2-normalised key matrix — on device
    device: str,
) -> np.ndarray:
    """
    Compute cosine similarity between one query vector and a matrix of keys.
    Returns (N,) float32 numpy array of similarity scores.
    """
    q = torch.from_numpy(query.astype(np.float32)).to(device)
    q_norm = q / q.norm().clamp(min=1e-8)
    scores = torch.mv(key_matrix, q_norm)  # (N,)
    return scores.cpu().numpy()


def select_candidates(
    emb_store: EmbeddingStore,
    top_k: int | None = None,
    device: str = "cpu",
) -> dict[int, dict[int, list[tuple[tuple[int, int], float, float]]]]:
    """
    Run the full coarse-to-fine candidate selection for all pieces.

    Args:
        emb_store: Populated EmbeddingStore from Phase 4
        top_k:     Number of candidate cells per piece per rotation.
                   Defaults to config COARSE_TOPK (30).
        device:    Compute device string

    Returns:
        Nested dict:
          piece_id → rotation_deg → list of (grid_pos, coarse_score, fine_score)
        Sorted by fine_score descending.
    """
    settings = get_settings()
    if top_k is None:
        top_k = settings.coarse_topk

    ref_cls_gpu = emb_store.get_ref_cls_gpu().to(device)   # (N_cells, D)
    cls_index = emb_store.get_ref_cls_index()               # [(row, col), ...]
    token_map = emb_store.get_patch_token_map()

    piece_ids = emb_store.get_all_piece_ids()

    log.info(
        "candidate_selection_start",
        n_pieces=len(piece_ids),
        n_cells=emb_store.n_cells,
        top_k=top_k,
        rotations=ROTATION_VARIANTS,
    )

    results: dict[int, dict[int, list[tuple[tuple[int, int], float, float]]]] = {}

    for piece_id in piece_ids:
        results[piece_id] = {}

        for rot_deg in ROTATION_VARIANTS:
            try:
                cls_vec = emb_store.get_piece_cls_vector(piece_id, rot_deg)
            except KeyError:
                continue

            # ── Stage 1: Coarse CLS filter ────────────────────────────────
            coarse_scores = _cosine_similarity_matrix(cls_vec, ref_cls_gpu, device)
            # coarse_scores: (N_cells,) — one score per reference cell

            # Select top-K indices
            effective_k = min(top_k, len(cls_index))
            top_k_indices = np.argpartition(coarse_scores, -effective_k)[-effective_k:]
            # Sort by descending coarse score
            top_k_indices = top_k_indices[np.argsort(coarse_scores[top_k_indices])[::-1]]

            # ── Stage 2: Fine spatial correlation on shortlist ────────────
            candidate_cells = []
            coarse_scores_for_k = []

            for idx in top_k_indices:
                grid_pos = cls_index[idx]
                cell = token_map.get_cell(*grid_pos)
                if cell is None:
                    continue
                candidate_cells.append((grid_pos, cell.token_flat_normalised))
                coarse_scores_for_k.append(float(coarse_scores[idx]))

            try:
                piece_grid = emb_store.get_piece_token_grid(piece_id, rot_deg)
            except KeyError:
                continue

            fine_results = batch_spatial_similarity(
                piece_grid, candidate_cells, device=device
            )

            # Combine coarse and fine scores
            combined: list[tuple[tuple[int, int], float, float]] = []
            for (grid_pos, fine_score), coarse_score in zip(
                fine_results, coarse_scores_for_k
            ):
                combined.append((grid_pos, coarse_score, fine_score))

            # Sort by fine score descending
            combined.sort(key=lambda x: x[2], reverse=True)
            results[piece_id][rot_deg] = combined

    log.info(
        "candidate_selection_complete",
        n_pieces=len(results),
    )

    return results