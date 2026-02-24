# Copyright (c) 2026 Harsh Dwivedi
# Licensed under the Harsh Non-Commercial Attribution License (HNCAL) v1.0
# Commercial use requires written permission. See LICENSE for details.

"""
PieceWise — Embedding Store
In-memory cache for reference and piece token embeddings.
Provides clean accessors used by the matching engine.

Lifecycle:
  - Created per job in the pipeline orchestrator
  - Populated by patch_generator and piece_embedder
  - Consumed by the matching engine (Phase 5)
  - Discarded after job completion (no cross-job state)

The store holds numpy arrays (CPU) that are moved to GPU on demand
for torch.mm similarity computation in the matching engine.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from app.models.grid import PatchTokenMap
from app.models.piece import PieceEmbedding
from app.utils.geometry_utils import ROTATION_VARIANTS
from app.utils.logger import get_logger

log = get_logger(__name__)


class EmbeddingStore:
    """
    Holds all embeddings for one solve job.
    Thread-safe for read access (pipeline writes sequentially).
    """

    def __init__(self) -> None:
        self._patch_token_map: PatchTokenMap | None = None
        self._piece_embeddings: dict[int, PieceEmbedding] = {}

        # Pre-stacked GPU tensors for fast matching — built lazily
        # Shape: (N_cells, N_tokens_per_cell, D)
        self._ref_flat_gpu: torch.Tensor | None = None
        # Shape: (N_cells, D) — CLS vectors
        self._ref_cls_gpu: torch.Tensor | None = None
        # Ordered list of grid positions matching rows of _ref_cls_gpu
        self._ref_cls_index: list[tuple[int, int]] = []

    # ── Population ───────────────────────────────────────────────────────────

    def set_patch_token_map(self, token_map: PatchTokenMap) -> None:
        """Store the reference patch token map and pre-stack GPU tensors."""
        self._patch_token_map = token_map

        # Stack CLS matrix onto GPU for fast coarse filtering
        if token_map.cls_matrix is not None:
            self._ref_cls_gpu = torch.from_numpy(
                token_map.cls_matrix.astype(np.float32)
            )
            self._ref_cls_index = token_map.cls_index

        log.debug(
            "embedding_store_ref_set",
            total_cells=token_map.total_cells,
        )

    def add_piece_embedding(self, emb: PieceEmbedding) -> None:
        """Add a single piece embedding to the store."""
        self._piece_embeddings[emb.piece_id] = emb

    def set_piece_embeddings(self, embeddings: list[PieceEmbedding]) -> None:
        """Bulk-set all piece embeddings."""
        self._piece_embeddings = {e.piece_id: e for e in embeddings}
        log.debug("embedding_store_pieces_set", count=len(embeddings))

    # ── Reference Accessors ──────────────────────────────────────────────────

    def get_patch_token_map(self) -> PatchTokenMap:
        if self._patch_token_map is None:
            raise RuntimeError("Reference patch token map not set.")
        return self._patch_token_map

    def get_ref_cls_gpu(self) -> torch.Tensor:
        """
        Return stacked reference CLS matrix on GPU.
        Shape: (N_cells, D)
        Used for the coarse CLS cosine similarity filter.
        """
        if self._ref_cls_gpu is None:
            raise RuntimeError("Reference CLS matrix not built.")
        return self._ref_cls_gpu

    def get_ref_cls_index(self) -> list[tuple[int, int]]:
        """Return ordered list of (row, col) matching rows of get_ref_cls_gpu()."""
        return self._ref_cls_index

    def get_cell_flat_normalised(
        self, row: int, col: int
    ) -> np.ndarray:
        """
        Return pre-normalised flat token matrix for a reference cell.
        Shape: (N_tokens, D) — ready for torch.mm with piece tokens.
        """
        map_ = self.get_patch_token_map()
        cell = map_.get_cell(row, col)
        if cell is None:
            raise KeyError(f"No cell at ({row}, {col})")
        return cell.token_flat_normalised

    # ── Piece Accessors ──────────────────────────────────────────────────────

    def get_piece_token_grid(
        self, piece_id: int, rotation_deg: int
    ) -> np.ndarray:
        """
        Return the token grid for a piece at a given rotation.
        Shape: (H_tok, W_tok, D)
        """
        emb = self._piece_embeddings.get(piece_id)
        if emb is None:
            raise KeyError(f"No embedding for piece_id={piece_id}")
        grid = emb.token_grids.get(rotation_deg)
        if grid is None:
            raise KeyError(
                f"No token grid for piece_id={piece_id}, rotation={rotation_deg}"
            )
        return grid

    def get_piece_cls_vector(
        self, piece_id: int, rotation_deg: int
    ) -> np.ndarray:
        """Return the CLS vector for a piece at a given rotation. Shape: (D,)"""
        emb = self._piece_embeddings.get(piece_id)
        if emb is None:
            raise KeyError(f"No embedding for piece_id={piece_id}")
        cls = emb.cls_vectors.get(rotation_deg)
        if cls is None:
            raise KeyError(
                f"No CLS vector for piece_id={piece_id}, rotation={rotation_deg}"
            )
        return cls

    def get_all_piece_ids(self) -> list[int]:
        """Return all stored piece IDs."""
        return sorted(self._piece_embeddings.keys())

    # ── Diagnostics ──────────────────────────────────────────────────────────

    @property
    def n_pieces(self) -> int:
        return len(self._piece_embeddings)

    @property
    def n_cells(self) -> int:
        if self._patch_token_map is None:
            return 0
        return self._patch_token_map.total_cells

    def summary(self) -> dict:
        return {
            "n_pieces": self.n_pieces,
            "n_cells": self.n_cells,
            "ref_ready": self._patch_token_map is not None,
            "pieces_ready": self.n_pieces > 0,
        }