# Copyright (c) 2026 Harsh Dwivedi
# Licensed under the Harsh Non-Commercial Attribution License (HNCAL) v1.0
# Commercial use requires written permission. See LICENSE for details.

"""
PieceWise — Step Generator
Converts the BFS-ordered PieceMatch list into a numbered sequence
of AssemblyStep objects consumed by the rendering module (Phase 8)
and the frontend solution viewer (Phase 9).

Each AssemblyStep carries:
  - step_num:                    1-based placement order
  - piece_id:                    which piece to place
  - grid_pos:                    where to place it (row, col)
  - rotation_deg:                how to orient it physically
  - piece_type:                  CORNER / EDGE / INTERIOR
  - composite_confidence:        matching confidence score
  - adjacency_score:             edge color compatibility score
  - curvature_complement_score:  tab/blank geometry score
  - flagged:                     True if low confidence (needs human review)
"""

from __future__ import annotations

from app.models.piece import AssemblyStep, PieceMatch, PieceType
from app.utils.logger import get_logger

log = get_logger(__name__)


def generate_steps(
    ordered_matches: list[PieceMatch],
    classifications: dict[int, PieceType],
) -> list[AssemblyStep]:
    """
    Convert an ordered PieceMatch list into AssemblyStep objects.

    Args:
        ordered_matches: BFS-ordered list from bfs_assembler.bfs_order()
        classifications: piece_id → PieceType from piece_classifier

    Returns:
        List of AssemblyStep, step_num starting at 1.
    """
    steps: list[AssemblyStep] = []

    for step_num, match in enumerate(ordered_matches, start=1):
        piece_type = classifications.get(match.piece_id, PieceType.UNKNOWN)

        step = AssemblyStep(
            step_num=step_num,
            piece_id=match.piece_id,
            grid_pos=match.grid_pos,
            rotation_deg=match.rotation_deg,
            piece_type=piece_type,
            composite_confidence=match.composite_confidence,
            adjacency_score=match.adjacency_score,
            curvature_complement_score=match.curvature_complement_score,
            flagged=match.flagged,
        )
        steps.append(step)

    # Log summary breakdown
    corners = sum(1 for s in steps if s.piece_type == PieceType.CORNER)
    edges = sum(1 for s in steps if s.piece_type == PieceType.EDGE)
    interior = sum(1 for s in steps if s.piece_type == PieceType.INTERIOR)
    flagged = sum(1 for s in steps if s.flagged)

    log.info(
        "step_generation_complete",
        total_steps=len(steps),
        corners=corners,
        edges=edges,
        interior=interior,
        flagged=flagged,
    )

    return steps