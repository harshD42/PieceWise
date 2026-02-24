# Copyright (c) 2026 Harsh Dwivedi
# Licensed under the Harsh Non-Commercial Attribution License (HNCAL) v1.0
# Commercial use requires written permission. See LICENSE for details.

"""
PieceWise — Neighbor Pair Extractor
Builds an adjacency map from the Hungarian-assigned PieceMatch list.

At this stage all piece→cell assignments are FIXED (post-Hungarian).
Neighbor relationships are therefore well-defined and non-circular.
This is the prerequisite for all adjacency refiner operations.

For each pair of horizontally or vertically adjacent grid cells,
we record which pieces are assigned to them and which shared side
they touch. This drives both the edge histogram comparator and
the curvature complement scorer.

Side convention (0-indexed):
  0 = top side of piece
  1 = right side of piece
  2 = bottom side of piece
  3 = left side of piece

Adjacency relationships:
  Horizontal neighbors (col, col+1): piece at (r,c) right side (1)
                                     touches piece at (r,c+1) left side (3)
  Vertical neighbors (row, row+1):   piece at (r,c) bottom side (2)
                                     touches piece at (r+1,c) top side (0)
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.models.piece import PieceMatch
from app.utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class NeighborPair:
    """A pair of adjacent pieces sharing a boundary."""
    piece_id_a: int
    piece_id_b: int
    grid_pos_a: tuple[int, int]
    grid_pos_b: tuple[int, int]
    # Side indices on each piece that touch the shared boundary
    side_a: int   # side of piece_a that faces piece_b
    side_b: int   # side of piece_b that faces piece_a


def extract_neighbor_pairs(
    matches: list[PieceMatch],
    grid_shape: tuple[int, int],
) -> list[NeighborPair]:
    """
    Extract all horizontally and vertically adjacent piece pairs.

    Args:
        matches:    Post-Hungarian PieceMatch list
        grid_shape: (n_rows, n_cols)

    Returns:
        List of NeighborPair objects for all adjacent cell pairs
        where both cells have an assigned piece.
    """
    n_rows, n_cols = grid_shape

    # Build grid_pos → PieceMatch lookup
    pos_to_match: dict[tuple[int, int], PieceMatch] = {
        m.grid_pos: m for m in matches
    }

    pairs: list[NeighborPair] = []

    for r in range(n_rows):
        for c in range(n_cols):
            match_a = pos_to_match.get((r, c))
            if match_a is None:
                continue

            # Horizontal neighbor: (r, c) → (r, c+1)
            if c + 1 < n_cols:
                match_b = pos_to_match.get((r, c + 1))
                if match_b is not None:
                    pairs.append(NeighborPair(
                        piece_id_a=match_a.piece_id,
                        piece_id_b=match_b.piece_id,
                        grid_pos_a=(r, c),
                        grid_pos_b=(r, c + 1),
                        side_a=1,  # right side of piece_a
                        side_b=3,  # left side of piece_b
                    ))

            # Vertical neighbor: (r, c) → (r+1, c)
            if r + 1 < n_rows:
                match_b = pos_to_match.get((r + 1, c))
                if match_b is not None:
                    pairs.append(NeighborPair(
                        piece_id_a=match_a.piece_id,
                        piece_id_b=match_b.piece_id,
                        grid_pos_a=(r, c),
                        grid_pos_b=(r + 1, c),
                        side_a=2,  # bottom side of piece_a
                        side_b=0,  # top side of piece_b
                    ))

    log.debug(
        "neighbor_pairs_extracted",
        total_pairs=len(pairs),
        grid_shape=grid_shape,
        assigned_cells=len(pos_to_match),
    )

    return pairs


def build_piece_neighbor_map(
    pairs: list[NeighborPair],
) -> dict[int, list[NeighborPair]]:
    """
    Index neighbor pairs by piece_id for O(1) lookup.
    Returns dict: piece_id → list of NeighborPairs involving that piece.
    """
    index: dict[int, list[NeighborPair]] = {}
    for pair in pairs:
        index.setdefault(pair.piece_id_a, []).append(pair)
        index.setdefault(pair.piece_id_b, []).append(pair)
    return index