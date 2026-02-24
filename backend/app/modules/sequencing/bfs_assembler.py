# Copyright (c) 2026 Harsh Dwivedi
# Licensed under the Harsh Non-Commercial Attribution License (HNCAL) v1.0
# Commercial use requires written permission. See LICENSE for details.

"""
PieceWise — BFS Assembly Sequencer
Produces an optimal piece-placement ordering using Breadth-First Search
from the top-left corner anchor.

Why BFS from a corner?
  - Physical stability: every piece placed has at least one already-placed
    neighbor to align against, making the physical assembly process easier
  - Natural progression: mimics how experienced puzzle solvers work —
    start from a fixed reference point and expand outward
  - Predictable: the sequence is deterministic given the same assignments

Anchor selection:
  The top-left corner of the grid (0, 0) is the canonical anchor.
  The piece assigned to (0, 0) is always placed first.
  If (0, 0) is unassigned, fall back to the highest-confidence corner.

BFS ordering within each level:
  Pieces are enqueued in this priority order:
    1. CORNER pieces (highest priority — reference anchors)
    2. EDGE pieces (border frame second)
    3. INTERIOR pieces (fill in last)
  Within each type, sort by descending composite_confidence
  so the most reliable pieces are placed before uncertain ones.

Result:
  An ordered list of piece_ids giving the recommended placement sequence.
  The step generator (step_generator.py) converts this to AssemblyStep objects.
"""

from __future__ import annotations

from collections import deque

from app.models.piece import PieceMatch, PieceType
from app.utils.logger import get_logger

log = get_logger(__name__)

# BFS priority: lower number = placed earlier
_TYPE_PRIORITY = {
    PieceType.CORNER: 0,
    PieceType.EDGE: 1,
    PieceType.INTERIOR: 2,
    PieceType.UNKNOWN: 3,
}

# 4-directional grid neighbours (row_delta, col_delta)
_NEIGHBOURS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _sort_key(
    match: PieceMatch,
    piece_type: PieceType,
) -> tuple[int, float]:
    """
    Sort key for BFS queue: (type_priority, -confidence)
    Lower tuple = placed earlier.
    """
    return (_TYPE_PRIORITY[piece_type], -match.composite_confidence)


def bfs_order(
    matches: list[PieceMatch],
    classifications: dict[int, PieceType],
    grid_shape: tuple[int, int],
) -> list[PieceMatch]:
    """
    Produce BFS-ordered placement sequence anchored at the top-left corner.

    Args:
        matches:         Post-adjacency-refined PieceMatch list
        classifications: piece_id → PieceType from piece_classifier
        grid_shape:      (n_rows, n_cols)

    Returns:
        Ordered list of PieceMatch objects — placement order.
        All input matches are included (unvisited pieces appended at end).
    """
    n_rows, n_cols = grid_shape

    # Build lookup maps
    pos_to_match: dict[tuple[int, int], PieceMatch] = {
        m.grid_pos: m for m in matches
    }

    # Find anchor: piece at (0, 0), or best corner if (0,0) unassigned
    anchor = _find_anchor(matches, classifications, pos_to_match)

    if anchor is None:
        log.warning(
            "bfs_no_anchor_found",
            advice="No corner piece found — using first match as fallback.",
        )
        return matches  # Fallback: return unsorted

    visited: set[tuple[int, int]] = set()
    ordered: list[PieceMatch] = []
    queue: deque[PieceMatch] = deque()

    # Seed BFS with anchor
    queue.append(anchor)
    visited.add(anchor.grid_pos)

    while queue:
        current = queue.popleft()
        ordered.append(current)

        # Collect unvisited grid neighbours
        r, c = current.grid_pos
        neighbours: list[PieceMatch] = []

        for dr, dc in _NEIGHBOURS:
            nr, nc = r + dr, c + dc
            if (nr, nc) in visited:
                continue
            if not (0 <= nr < n_rows and 0 <= nc < n_cols):
                continue
            nb_match = pos_to_match.get((nr, nc))
            if nb_match is None:
                continue
            neighbours.append(nb_match)
            visited.add((nr, nc))

        # Sort neighbours by type priority then confidence before enqueuing
        neighbours.sort(
            key=lambda m: _sort_key(m, classifications.get(m.piece_id, PieceType.UNKNOWN))
        )
        queue.extend(neighbours)

    # Append any pieces not reached by BFS (disconnected assignments)
    unvisited = [m for m in matches if m.grid_pos not in visited]
    if unvisited:
        log.warning(
            "bfs_unvisited_pieces",
            count=len(unvisited),
            reason="These pieces were not reachable from the BFS anchor.",
        )
        # Sort unvisited by type then confidence
        unvisited.sort(
            key=lambda m: _sort_key(m, classifications.get(m.piece_id, PieceType.UNKNOWN))
        )
        ordered.extend(unvisited)

    log.info(
        "bfs_ordering_complete",
        total_pieces=len(ordered),
        anchor_pos=anchor.grid_pos,
    )

    return ordered


def _find_anchor(
    matches: list[PieceMatch],
    classifications: dict[int, PieceType],
    pos_to_match: dict[tuple[int, int], PieceMatch],
) -> PieceMatch | None:
    """
    Find the best anchor piece for BFS.

    Priority:
      1. Piece at grid position (0, 0)
      2. Any corner piece, sorted by descending confidence
      3. Any piece (last resort)
    """
    # Primary: top-left cell
    tl = pos_to_match.get((0, 0))
    if tl is not None:
        return tl

    # Secondary: any corner piece by confidence
    corners = [
        m for m in matches
        if classifications.get(m.piece_id) == PieceType.CORNER
    ]
    if corners:
        return max(corners, key=lambda m: m.composite_confidence)

    # Fallback: highest confidence piece overall
    if matches:
        return max(matches, key=lambda m: m.composite_confidence)

    return None