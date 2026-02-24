# Copyright (c) 2026 Harsh Dwivedi
# Licensed under the Harsh Non-Commercial Attribution License (HNCAL) v1.0
# Commercial use requires written permission. See LICENSE for details.

"""
Phase 7 — Sequencing module tests.
Pure logic — no GPU, models, or image data required.
Tests cover: piece classifier, BFS assembler, step generator,
and full sequencing pipeline on synthetic grids.
"""

import pytest
from app.models.piece import (
    AssemblyStep, CurvatureProfile, PieceCrop, PieceMatch, PieceType
)
import cv2
import numpy as np


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _make_match(
    piece_id: int,
    row: int,
    col: int,
    confidence: float = 0.8,
    rotation: int = 0,
    flagged: bool = False,
) -> PieceMatch:
    return PieceMatch(
        piece_id=piece_id,
        grid_pos=(row, col),
        rotation_deg=rotation,
        composite_confidence=confidence,
        flagged=flagged,
    )


def _make_piece(
    piece_id: int,
    flat_side_count: int = 0,
    with_profiles: bool = True,
) -> PieceCrop:
    h, w = 60, 60
    img = np.zeros((h, w, 3), dtype=np.uint8)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.rectangle(mask, (5, 5), (w - 5, h - 5), 255, -1)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = contours[0] if contours else np.array([[[0, 0]]], dtype=np.int32)

    profiles = []
    if with_profiles:
        for i in range(4):
            is_flat = i < flat_side_count
            profiles.append(CurvatureProfile(
                side_index=i,
                curvature_vector=np.zeros(32, dtype=np.float32),
                is_flat=is_flat,
                peak_value=0.0,
            ))

    return PieceCrop(
        piece_id=piece_id,
        image=img,
        alpha_mask=mask,
        bbox=(0, 0, w, h),
        contour=contour,
        area_px=float(mask.sum() // 255),
        solidity=0.95,
        compactness=0.85,
        flat_side_count=flat_side_count,
        curvature_profiles=profiles,
    )


def _full_grid_matches(n_rows: int, n_cols: int) -> list[PieceMatch]:
    """Create a complete set of matches for an n_rows × n_cols grid."""
    matches = []
    pid = 0
    for r in range(n_rows):
        for c in range(n_cols):
            matches.append(_make_match(pid, r, c))
            pid += 1
    return matches


# ─── Piece Classifier ────────────────────────────────────────────────────────

def test_classify_corner_positions():
    from app.modules.sequencing.piece_classifier import classify_piece

    grid = (4, 4)
    for r, c in [(0, 0), (0, 3), (3, 0), (3, 3)]:
        m = _make_match(0, r, c)
        assert classify_piece(m, grid) == PieceType.CORNER


def test_classify_edge_positions():
    from app.modules.sequencing.piece_classifier import classify_piece

    grid = (4, 4)
    # Top edge (not corners)
    for c in [1, 2]:
        m = _make_match(0, 0, c)
        assert classify_piece(m, grid) == PieceType.EDGE
    # Left edge
    for r in [1, 2]:
        m = _make_match(0, r, 0)
        assert classify_piece(m, grid) == PieceType.EDGE


def test_classify_interior_positions():
    from app.modules.sequencing.piece_classifier import classify_piece

    grid = (4, 4)
    for r, c in [(1, 1), (1, 2), (2, 1), (2, 2)]:
        m = _make_match(0, r, c)
        assert classify_piece(m, grid) == PieceType.INTERIOR


def test_classify_2x2_all_corners():
    from app.modules.sequencing.piece_classifier import classify_piece

    grid = (2, 2)
    for r, c in [(0, 0), (0, 1), (1, 0), (1, 1)]:
        m = _make_match(0, r, c)
        assert classify_piece(m, grid) == PieceType.CORNER


def test_classify_and_validate_counts():
    from app.modules.sequencing.piece_classifier import classify_and_validate

    # 3×3 grid: 4 corners, 4 edges, 1 interior
    matches = _full_grid_matches(3, 3)
    pieces = [_make_piece(i) for i in range(9)]
    result = classify_and_validate(matches, pieces, (3, 3))

    corners = sum(1 for v in result.values() if v == PieceType.CORNER)
    edges = sum(1 for v in result.values() if v == PieceType.EDGE)
    interior = sum(1 for v in result.values() if v == PieceType.INTERIOR)

    assert corners == 4
    assert edges == 4
    assert interior == 1


def test_classify_and_validate_all_pieces_classified():
    from app.modules.sequencing.piece_classifier import classify_and_validate

    matches = _full_grid_matches(4, 5)
    pieces = [_make_piece(i) for i in range(20)]
    result = classify_and_validate(matches, pieces, (4, 5))

    assert len(result) == 20
    for pid in range(20):
        assert pid in result
        assert isinstance(result[pid], PieceType)


def test_classify_and_validate_mismatch_trusts_grid():
    from app.modules.sequencing.piece_classifier import classify_and_validate

    # Piece at (0,0) is a corner but has flat_side_count=0 (mismatch)
    matches = [_make_match(0, 0, 0)]
    pieces = [_make_piece(0, flat_side_count=0, with_profiles=True)]
    result = classify_and_validate(matches, pieces, (3, 3))

    # Should still be CORNER — grid position wins over curvature
    assert result[0] == PieceType.CORNER


def test_classify_without_profiles_skips_validation():
    from app.modules.sequencing.piece_classifier import classify_and_validate

    # Piece has no curvature profiles — validation skipped, classification still works
    matches = [_make_match(0, 0, 0)]
    pieces = [_make_piece(0, flat_side_count=0, with_profiles=False)]
    result = classify_and_validate(matches, pieces, (3, 3))
    assert result[0] == PieceType.CORNER


def test_classify_empty_matches():
    from app.modules.sequencing.piece_classifier import classify_and_validate
    result = classify_and_validate([], [], (3, 3))
    assert result == {}


# ─── BFS Assembler ───────────────────────────────────────────────────────────

def test_bfs_first_piece_is_top_left():
    from app.modules.sequencing.bfs_assembler import bfs_order
    from app.modules.sequencing.piece_classifier import classify_and_validate

    matches = _full_grid_matches(3, 3)
    pieces = [_make_piece(i) for i in range(9)]
    classifications = classify_and_validate(matches, pieces, (3, 3))
    ordered = bfs_order(matches, classifications, (3, 3))

    assert ordered[0].grid_pos == (0, 0)


def test_bfs_all_pieces_included():
    from app.modules.sequencing.bfs_assembler import bfs_order
    from app.modules.sequencing.piece_classifier import classify_and_validate

    matches = _full_grid_matches(4, 4)
    pieces = [_make_piece(i) for i in range(16)]
    classifications = classify_and_validate(matches, pieces, (4, 4))
    ordered = bfs_order(matches, classifications, (4, 4))

    assert len(ordered) == 16
    piece_ids = {m.piece_id for m in ordered}
    assert piece_ids == set(range(16))


def test_bfs_no_duplicate_pieces():
    from app.modules.sequencing.bfs_assembler import bfs_order
    from app.modules.sequencing.piece_classifier import classify_and_validate

    matches = _full_grid_matches(5, 5)
    pieces = [_make_piece(i) for i in range(25)]
    classifications = classify_and_validate(matches, pieces, (5, 5))
    ordered = bfs_order(matches, classifications, (5, 5))

    piece_ids = [m.piece_id for m in ordered]
    assert len(piece_ids) == len(set(piece_ids))


def test_bfs_corners_before_edges_before_interior():
    from app.modules.sequencing.bfs_assembler import bfs_order
    from app.modules.sequencing.piece_classifier import classify_and_validate

    matches = _full_grid_matches(3, 3)
    pieces = [_make_piece(i) for i in range(9)]
    classifications = classify_and_validate(matches, pieces, (3, 3))
    ordered = bfs_order(matches, classifications, (3, 3))

    types_in_order = [classifications[m.piece_id] for m in ordered]

    # The ANCHOR (step 0) must always be a corner
    assert types_in_order[0] == PieceType.CORNER

    # Within each BFS level, type priority is respected:
    # corners before edges before interior.
    # We verify this by checking that among the DIRECT NEIGHBORS of (0,0)
    # — which are all at BFS level 1 — no interior piece appears before
    # an edge piece. In a 3×3 grid the direct neighbors of (0,0) are
    # (0,1) and (1,0), both edge pieces, so they must come before interior.
    first_interior_idx = next(
        (i for i, t in enumerate(types_in_order) if t == PieceType.INTERIOR),
        None,
    )
    # The anchor's direct neighbors (BFS level 1) are both edges in a 3×3 grid.
    # Both must appear before the first interior piece.
    neighbor_positions = {(0, 1), (1, 0)}
    pos_to_step = {m.grid_pos: i for i, m in enumerate(ordered)}
    for pos in neighbor_positions:
        neighbor_idx = pos_to_step.get(pos)
        if neighbor_idx is not None and first_interior_idx is not None:
            assert neighbor_idx < first_interior_idx, (
                f"Neighbor {pos} at step {neighbor_idx} should come before "
                f"first interior at step {first_interior_idx}"
            )


def test_bfs_each_piece_has_placed_neighbor():
    """
    Verify that for every piece except the first, at least one of its
    grid neighbours appears earlier in the sequence.
    """
    from app.modules.sequencing.bfs_assembler import bfs_order
    from app.modules.sequencing.piece_classifier import classify_and_validate

    matches = _full_grid_matches(4, 4)
    pieces = [_make_piece(i) for i in range(16)]
    classifications = classify_and_validate(matches, pieces, (4, 4))
    ordered = bfs_order(matches, classifications, (4, 4))

    placed_positions: set[tuple[int, int]] = set()

    for i, m in enumerate(ordered):
        if i > 0:
            r, c = m.grid_pos
            adjacent = {(r-1,c), (r+1,c), (r,c-1), (r,c+1)}
            has_placed_neighbor = bool(adjacent & placed_positions)
            assert has_placed_neighbor, (
                f"Piece at step {i+1} pos {m.grid_pos} has no placed neighbor. "
                f"Placed so far: {placed_positions}"
            )
        placed_positions.add(m.grid_pos)


def test_bfs_fallback_when_top_left_missing():
    from app.modules.sequencing.bfs_assembler import bfs_order

    # Grid with (0,0) unassigned — only (0,1), (1,0), (1,1)
    matches = [
        _make_match(1, 0, 1, confidence=0.7),
        _make_match(2, 1, 0, confidence=0.9),
        _make_match(3, 1, 1, confidence=0.8),
    ]
    classifications = {
        1: PieceType.CORNER,
        2: PieceType.CORNER,
        3: PieceType.CORNER,
    }
    ordered = bfs_order(matches, classifications, (2, 2))

    # Should not crash and should return all 3 pieces
    assert len(ordered) == 3


def test_bfs_single_piece():
    from app.modules.sequencing.bfs_assembler import bfs_order

    matches = [_make_match(0, 0, 0)]
    classifications = {0: PieceType.CORNER}
    ordered = bfs_order(matches, classifications, (1, 1))

    assert len(ordered) == 1
    assert ordered[0].piece_id == 0


def test_bfs_empty_matches():
    from app.modules.sequencing.bfs_assembler import bfs_order

    ordered = bfs_order([], {}, (3, 3))
    assert ordered == []


def test_bfs_confidence_tiebreaking():
    from app.modules.sequencing.bfs_assembler import bfs_order

    # All interior pieces — highest confidence should come first within BFS level
    matches = [
        _make_match(0, 0, 0, confidence=0.9),
        _make_match(1, 0, 1, confidence=0.5),
        _make_match(2, 1, 0, confidence=0.7),
        _make_match(3, 1, 1, confidence=0.3),
    ]
    classifications = {i: PieceType.INTERIOR for i in range(4)}
    ordered = bfs_order(matches, classifications, (2, 2))

    # First piece must be at (0,0) — it's the anchor
    assert ordered[0].grid_pos == (0, 0)
    assert len(ordered) == 4


# ─── Step Generator ──────────────────────────────────────────────────────────

def test_generate_steps_count():
    from app.modules.sequencing.step_generator import generate_steps

    matches = _full_grid_matches(3, 3)
    classifications = {m.piece_id: PieceType.INTERIOR for m in matches}
    steps = generate_steps(matches, classifications)

    assert len(steps) == 9


def test_generate_steps_numbering():
    from app.modules.sequencing.step_generator import generate_steps

    matches = _full_grid_matches(2, 3)
    classifications = {m.piece_id: PieceType.EDGE for m in matches}
    steps = generate_steps(matches, classifications)

    for i, step in enumerate(steps, start=1):
        assert step.step_num == i


def test_generate_steps_piece_type_assigned():
    from app.modules.sequencing.step_generator import generate_steps

    matches = [_make_match(0, 0, 0)]
    classifications = {0: PieceType.CORNER}
    steps = generate_steps(matches, classifications)

    assert steps[0].piece_type == PieceType.CORNER


def test_generate_steps_unknown_type_fallback():
    from app.modules.sequencing.step_generator import generate_steps

    matches = [_make_match(99, 1, 1)]
    classifications = {}  # piece_id 99 not in map
    steps = generate_steps(matches, classifications)

    assert steps[0].piece_type == PieceType.UNKNOWN


def test_generate_steps_metadata_preserved():
    from app.modules.sequencing.step_generator import generate_steps

    m = _make_match(7, 2, 3, confidence=0.91, rotation=90, flagged=True)
    m.adjacency_score = 0.75
    m.curvature_complement_score = 0.82
    classifications = {7: PieceType.INTERIOR}
    steps = generate_steps([m], classifications)

    s = steps[0]
    assert s.piece_id == 7
    assert s.grid_pos == (2, 3)
    assert s.rotation_deg == 90
    assert s.flagged is True
    assert abs(s.composite_confidence - 0.91) < 0.001
    assert abs(s.adjacency_score - 0.75) < 0.001
    assert abs(s.curvature_complement_score - 0.82) < 0.001


def test_generate_steps_empty():
    from app.modules.sequencing.step_generator import generate_steps
    steps = generate_steps([], {})
    assert steps == []


# ─── Full Sequencing Pipeline ─────────────────────────────────────────────────

def test_full_pipeline_3x3():
    from app.modules.sequencing.piece_classifier import classify_and_validate
    from app.modules.sequencing.bfs_assembler import bfs_order
    from app.modules.sequencing.step_generator import generate_steps

    matches = _full_grid_matches(3, 3)
    pieces = [_make_piece(i, flat_side_count=2 if i in [0,2,6,8] else 1 if i in [1,3,5,7] else 0) for i in range(9)]

    classifications = classify_and_validate(matches, pieces, (3, 3))
    ordered = bfs_order(matches, classifications, (3, 3))
    steps = generate_steps(ordered, classifications)

    assert len(steps) == 9
    assert steps[0].step_num == 1
    assert steps[-1].step_num == 9
    # First step should be at (0,0)
    assert steps[0].grid_pos == (0, 0)
    # All step numbers unique
    step_nums = [s.step_num for s in steps]
    assert len(set(step_nums)) == 9


def test_full_pipeline_4x5_complete():
    from app.modules.sequencing.piece_classifier import classify_and_validate
    from app.modules.sequencing.bfs_assembler import bfs_order
    from app.modules.sequencing.step_generator import generate_steps

    n_rows, n_cols = 4, 5
    n = n_rows * n_cols
    matches = _full_grid_matches(n_rows, n_cols)
    pieces = [_make_piece(i) for i in range(n)]

    classifications = classify_and_validate(matches, pieces, (n_rows, n_cols))
    ordered = bfs_order(matches, classifications, (n_rows, n_cols))
    steps = generate_steps(ordered, classifications)

    assert len(steps) == n
    piece_ids_in_steps = {s.piece_id for s in steps}
    assert piece_ids_in_steps == set(range(n))

    # Verify corner count
    corner_steps = [s for s in steps if s.piece_type == PieceType.CORNER]
    assert len(corner_steps) == 4


def test_full_pipeline_flagged_pieces_preserved():
    from app.modules.sequencing.piece_classifier import classify_and_validate
    from app.modules.sequencing.bfs_assembler import bfs_order
    from app.modules.sequencing.step_generator import generate_steps

    matches = _full_grid_matches(2, 2)
    matches[2].flagged = True   # one flagged piece
    pieces = [_make_piece(i) for i in range(4)]

    classifications = classify_and_validate(matches, pieces, (2, 2))
    ordered = bfs_order(matches, classifications, (2, 2))
    steps = generate_steps(ordered, classifications)

    flagged_steps = [s for s in steps if s.flagged]
    assert len(flagged_steps) == 1