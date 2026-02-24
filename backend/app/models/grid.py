# Copyright (c) 2026 Harsh Dwivedi
# Licensed under the Harsh Non-Commercial Attribution License (HNCAL) v1.0
# Commercial use requires written permission. See LICENSE for details.

"""
PieceWise — Grid and Patch Token Models
Models representing the reference image's spatial token map
as produced by the DINOv2 feature extraction module.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field


class PatchTokenCell(BaseModel):
    """
    DINOv2 spatial token grid for one cell of the reference image grid.
    Stored pre-normalised and pre-flattened for fast GPU similarity lookup.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    grid_pos: tuple[int, int] = Field(..., description="(row, col) in puzzle grid")
    # Spatial token grid: numpy (H_tok × W_tok × D), D=128 or 768
    token_grid: Any = Field(..., description="np.ndarray spatial token grid")
    # Pre-L2-normalised flat version: numpy (H_tok*W_tok, D)
    # Pre-computed once and cached — never recomputed per piece
    token_flat_normalised: Any = Field(
        ..., description="np.ndarray pre-normalised flat token matrix"
    )
    # CLS vector: numpy (D,) — used for coarse shortlist filter
    cls_vector: Any = Field(..., description="np.ndarray CLS token embedding")


class PatchTokenMap(BaseModel):
    """
    Complete spatial token map for the reference image.
    Produced once by the feature extraction module and reused
    throughout the matching engine.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    grid_shape: tuple[int, int] = Field(..., description="(n_rows, n_cols)")
    patch_size_px: tuple[int, int] = Field(..., description="(patch_h, patch_w) in pixels")
    # Keyed by "row,col" string for JSON serialisability
    cells: dict[str, PatchTokenCell] = Field(default_factory=dict)

    # Stacked CLS matrix for fast coarse filtering: numpy (N*M, D)
    # Row i corresponds to grid cell (i // n_cols, i % n_cols)
    cls_matrix: Any = Field(
        None, description="np.ndarray stacked CLS vectors (N*M, D)"
    )
    # Ordered list of grid_pos matching rows of cls_matrix
    cls_index: list[tuple[int, int]] = Field(default_factory=list)

    def get_cell(self, row: int, col: int) -> PatchTokenCell | None:
        return self.cells.get(f"{row},{col}")

    def set_cell(self, cell: PatchTokenCell) -> None:
        r, c = cell.grid_pos
        self.cells[f"{r},{c}"] = cell

    @property
    def n_rows(self) -> int:
        return self.grid_shape[0]

    @property
    def n_cols(self) -> int:
        return self.grid_shape[1]

    @property
    def total_cells(self) -> int:
        return self.n_rows * self.n_cols