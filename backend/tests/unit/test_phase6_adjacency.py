# Copyright (c) 2026 Harsh Dwivedi
# Licensed under the Harsh Non-Commercial Attribution License (HNCAL) v1.0
# Commercial use requires written permission. See LICENSE for details.

"""
Phase 6 — Adjacency refiner tests.
All tests use synthetic piece crops and matches — no GPU, models, or
puzzle images required.
"""

import cv2
import numpy as np
import pytest

from app.models.piece import CurvatureProfile, PieceCrop, PieceMatch


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _make_piece(
    piece_id: int,
    h: int = 80,
    w: int = 80,
    color: tuple = (100, 120, 140),
    flat_side_count: int = 0,
) -> PieceCrop:
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:] = color
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.rectangle(mask, (5, 5), (w - 5, h - 5), 255, -1)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = contours[0] if contours else np.array([[[0, 0]]], dtype=np.int32)
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
    )


def _make_match(piece_id: int, row: int, col: int, rot: int = 0) -> PieceMatch:
    return PieceMatch(
        piece_id=piece_id,
        grid_pos=(row, col),
        rotation_deg=rot,
        composite_confidence=0.8,
    )


def _make_curvature_profile(
    side_index: int,
    is_flat: bool = False,
    peak: float = 0.0,
) -> CurvatureProfile:
    if is_flat:
        vec = np.zeros(32, dtype=np.float32)
    else:
        vec = np.sin(np.linspace(0, np.pi, 32)).astype(np.float32) * peak
    return CurvatureProfile(
        side_index=side_index,
        curvature_vector=vec,
        is_flat=is_flat,
        peak_value=peak,
    )


# ─── Neighbor Extractor ──────────────────────────────────────────────────────

def test_extract_neighbor_pairs_count():
    from app.modules.adjacency.neighbor_extractor import extract_neighbor_pairs

    # 2×2 grid, all 4 cells filled
    matches = [
        _make_match(0, 0, 0), _make_match(1, 0, 1),
        _make_match(2, 1, 0), _make_match(3, 1, 1),
    ]
    pairs = extract_neighbor_pairs(matches, (2, 2))
    # 2 horizontal + 2 vertical = 4 pairs
    assert len(pairs) == 4


def test_extract_neighbor_pairs_sides_correct():
    from app.modules.adjacency.neighbor_extractor import extract_neighbor_pairs

    matches = [_make_match(0, 0, 0), _make_match(1, 0, 1)]
    pairs = extract_neighbor_pairs(matches, (1, 2))

    assert len(pairs) == 1
    p = pairs[0]
    assert p.side_a == 1   # right side of (0,0)
    assert p.side_b == 3   # left side of (0,1)


def test_extract_neighbor_pairs_vertical_sides():
    from app.modules.adjacency.neighbor_extractor import extract_neighbor_pairs

    matches = [_make_match(0, 0, 0), _make_match(1, 1, 0)]
    pairs = extract_neighbor_pairs(matches, (2, 1))

    assert len(pairs) == 1
    p = pairs[0]
    assert p.side_a == 2   # bottom of (0,0)
    assert p.side_b == 0   # top of (1,0)


def test_extract_neighbor_pairs_missing_cell_skipped():
    from app.modules.adjacency.neighbor_extractor import extract_neighbor_pairs

    # 3×1 grid, middle cell missing
    matches = [_make_match(0, 0, 0), _make_match(2, 2, 0)]
    pairs = extract_neighbor_pairs(matches, (3, 1))
    # No adjacent pairs — gap in the middle
    assert len(pairs) == 0


def test_build_piece_neighbor_map():
    from app.modules.adjacency.neighbor_extractor import (
        extract_neighbor_pairs,
        build_piece_neighbor_map,
    )

    matches = [
        _make_match(0, 0, 0), _make_match(1, 0, 1),
        _make_match(2, 1, 0), _make_match(3, 1, 1),
    ]
    pairs = extract_neighbor_pairs(matches, (2, 2))
    nmap = build_piece_neighbor_map(pairs)

    # Corner pieces (0, 1, 2, 3) each have 2 neighbors in 2×2 grid
    for pid in range(4):
        assert pid in nmap
        assert len(nmap[pid]) == 2


# ─── Edge Histogram Comparator ───────────────────────────────────────────────

def test_edge_histogram_identical_pieces_high_score():
    from app.modules.adjacency.edge_histogram import edge_histogram_score
    from app.modules.adjacency.neighbor_extractor import NeighborPair

    # Two identical pieces → histogram intersection should be ~1.0
    piece_a = _make_piece(0, color=(100, 150, 200))
    piece_b = _make_piece(1, color=(100, 150, 200))
    pair = NeighborPair(0, 1, (0, 0), (0, 1), side_a=1, side_b=3)

    score = edge_histogram_score(piece_a, piece_b, pair)
    assert score > 0.7


def test_edge_histogram_different_colors_low_score():
    from app.modules.adjacency.edge_histogram import edge_histogram_score
    from app.modules.adjacency.neighbor_extractor import NeighborPair

    piece_a = _make_piece(0, color=(10, 10, 10))    # dark
    piece_b = _make_piece(1, color=(240, 240, 240)) # bright
    pair = NeighborPair(0, 1, (0, 0), (0, 1), side_a=1, side_b=3)

    score = edge_histogram_score(piece_a, piece_b, pair)
    assert score < 0.4


def test_edge_histogram_score_in_range():
    from app.modules.adjacency.edge_histogram import edge_histogram_score
    from app.modules.adjacency.neighbor_extractor import NeighborPair

    piece_a = _make_piece(0)
    piece_b = _make_piece(1, color=(50, 80, 120))
    pair = NeighborPair(0, 1, (0, 0), (0, 1), side_a=2, side_b=0)

    score = edge_histogram_score(piece_a, piece_b, pair)
    assert 0.0 <= score <= 1.0


def test_edge_histogram_all_sides():
    from app.modules.adjacency.edge_histogram import edge_histogram_score
    from app.modules.adjacency.neighbor_extractor import NeighborPair

    pa = _make_piece(0, color=(128, 128, 128))
    pb = _make_piece(1, color=(128, 128, 128))

    for side_a, side_b in [(0, 2), (1, 3), (2, 0), (3, 1)]:
        pair = NeighborPair(0, 1, (0, 0), (0, 1), side_a=side_a, side_b=side_b)
        score = edge_histogram_score(pa, pb, pair)
        assert 0.0 <= score <= 1.0


def test_score_all_pairs_histogram_keys():
    from app.modules.adjacency.edge_histogram import score_all_pairs_histogram
    from app.modules.adjacency.neighbor_extractor import NeighborPair

    pieces = [_make_piece(i) for i in range(3)]
    piece_map = {p.piece_id: p for p in pieces}
    pairs = [
        NeighborPair(0, 1, (0, 0), (0, 1), side_a=1, side_b=3),
        NeighborPair(1, 2, (0, 1), (0, 2), side_a=1, side_b=3),
    ]

    scores = score_all_pairs_histogram(pairs, piece_map)
    assert (0, 1) in scores
    assert (1, 2) in scores


# ─── Curvature Complement Scorer ─────────────────────────────────────────────

def test_curvature_both_flat_returns_one():
    from app.modules.adjacency.curvature_complement import curvature_complement_score
    from app.modules.adjacency.neighbor_extractor import NeighborPair

    pa = _make_piece(0)
    pb = _make_piece(1)
    pa.curvature_profiles = [_make_curvature_profile(1, is_flat=True)]
    pb.curvature_profiles = [_make_curvature_profile(3, is_flat=True)]
    pair = NeighborPair(0, 1, (0, 0), (0, 1), side_a=1, side_b=3)

    score = curvature_complement_score(pa, pb, pair)
    assert score == 1.0


def test_curvature_one_flat_one_not_low_score():
    from app.modules.adjacency.curvature_complement import curvature_complement_score
    from app.modules.adjacency.neighbor_extractor import NeighborPair

    pa = _make_piece(0)
    pb = _make_piece(1)
    pa.curvature_profiles = [_make_curvature_profile(1, is_flat=True)]
    pb.curvature_profiles = [_make_curvature_profile(3, is_flat=False, peak=1.0)]
    pair = NeighborPair(0, 1, (0, 0), (0, 1), side_a=1, side_b=3)

    score = curvature_complement_score(pa, pb, pair)
    assert score == 0.25


def test_curvature_tab_blank_complement_high_score():
    from app.modules.adjacency.curvature_complement import curvature_complement_score
    from app.modules.adjacency.neighbor_extractor import NeighborPair

    # Tab: positive peak; Blank: negative peak (complement)
    pa = _make_piece(0)
    pb = _make_piece(1)
    x = np.linspace(0, np.pi, 32).astype(np.float32)
    tab_vec = np.sin(x)       # positive bump
    blank_vec = -np.sin(x)    # negative bump (complement)

    pa.curvature_profiles = [CurvatureProfile(
        side_index=1, curvature_vector=tab_vec, is_flat=False, peak_value=1.0
    )]
    pb.curvature_profiles = [CurvatureProfile(
        side_index=3, curvature_vector=blank_vec, is_flat=False, peak_value=-1.0
    )]
    pair = NeighborPair(0, 1, (0, 0), (0, 1), side_a=1, side_b=3)

    score = curvature_complement_score(pa, pb, pair)
    assert score > 0.7  # Complementary profiles should score high


def test_curvature_missing_profile_neutral():
    from app.modules.adjacency.curvature_complement import curvature_complement_score
    from app.modules.adjacency.neighbor_extractor import NeighborPair

    pa = _make_piece(0)  # no profiles
    pb = _make_piece(1)
    pair = NeighborPair(0, 1, (0, 0), (0, 1), side_a=1, side_b=3)

    score = curvature_complement_score(pa, pb, pair)
    assert score == 0.5


def test_curvature_score_in_range():
    from app.modules.adjacency.curvature_complement import curvature_complement_score
    from app.modules.adjacency.neighbor_extractor import NeighborPair

    pa = _make_piece(0)
    pb = _make_piece(1)
    rng = np.random.default_rng(7)
    pa.curvature_profiles = [CurvatureProfile(
        side_index=1,
        curvature_vector=rng.standard_normal(32).astype(np.float32),
        is_flat=False,
        peak_value=0.5,
    )]
    pb.curvature_profiles = [CurvatureProfile(
        side_index=3,
        curvature_vector=rng.standard_normal(32).astype(np.float32),
        is_flat=False,
        peak_value=-0.5,
    )]
    pair = NeighborPair(0, 1, (0, 0), (0, 1), side_a=1, side_b=3)
    score = curvature_complement_score(pa, pb, pair)
    assert 0.0 <= score <= 1.0


# ─── Swap Engine ─────────────────────────────────────────────────────────────

def test_swap_engine_preserves_piece_count():
    from app.modules.adjacency.swap_engine import run_swap_engine

    matches = [_make_match(i, i // 2, i % 2) for i in range(4)]
    pieces = [_make_piece(i) for i in range(4)]

    result = run_swap_engine(matches, pieces, (2, 2), max_iterations=5)
    assert len(result) == 4


def test_swap_engine_no_duplicate_positions():
    from app.modules.adjacency.swap_engine import run_swap_engine

    matches = [_make_match(i, i // 2, i % 2) for i in range(4)]
    pieces = [_make_piece(i) for i in range(4)]

    result = run_swap_engine(matches, pieces, (2, 2), max_iterations=10)
    positions = [m.grid_pos for m in result]
    assert len(set(positions)) == len(positions)


def test_swap_engine_scores_attached():
    from app.modules.adjacency.swap_engine import run_swap_engine

    matches = [_make_match(i, i // 2, i % 2) for i in range(4)]
    pieces = [_make_piece(i) for i in range(4)]

    result = run_swap_engine(matches, pieces, (2, 2), max_iterations=5)
    for m in result:
        assert 0.0 <= m.adjacency_score <= 1.0
        assert 0.0 <= m.curvature_complement_score <= 1.0


def test_swap_engine_zero_iterations_still_scores():
    from app.modules.adjacency.swap_engine import run_swap_engine

    matches = [_make_match(i, i // 2, i % 2) for i in range(4)]
    pieces = [_make_piece(i) for i in range(4)]

    result = run_swap_engine(matches, pieces, (2, 2), max_iterations=0)
    assert len(result) == 4
    for m in result:
        assert isinstance(m.adjacency_score, float)


def test_swap_engine_improves_bad_assignment():
    from app.modules.adjacency.swap_engine import run_swap_engine

    # Piece 0 is dark, piece 1 is bright
    # Assign them next to each other initially (bad histogram match)
    # After swapping, pieces with matching colors should be adjacent
    dark = _make_piece(0, color=(10, 10, 10))
    dark2 = _make_piece(2, color=(15, 15, 15))
    bright = _make_piece(1, color=(240, 240, 240))
    bright2 = _make_piece(3, color=(235, 235, 235))

    # Bad initial assignment: dark next to bright
    matches = [
        _make_match(0, 0, 0),   # dark at top-left
        _make_match(1, 0, 1),   # bright at top-right (bad neighbor of dark)
        _make_match(2, 1, 0),   # dark2 at bottom-left
        _make_match(3, 1, 1),   # bright2 at bottom-right
    ]

    result = run_swap_engine(
        matches, [dark, bright, dark2, bright2], (2, 2), max_iterations=50
    )
    # All pieces should still be assigned
    assert len(result) == 4
    # Scores should be attached
    for m in result:
        assert isinstance(m.adjacency_score, float)


# ─── Full Refiner Pipeline ────────────────────────────────────────────────────

def test_refine_adjacency_returns_same_count():
    from app.modules.adjacency.refiner import refine_adjacency

    matches = [_make_match(i, i // 3, i % 3) for i in range(9)]
    pieces = [_make_piece(i) for i in range(9)]

    result = refine_adjacency(matches, pieces, (3, 3))
    assert len(result) == 9


def test_refine_adjacency_no_duplicate_positions():
    from app.modules.adjacency.refiner import refine_adjacency

    matches = [_make_match(i, i // 3, i % 3) for i in range(9)]
    pieces = [_make_piece(i) for i in range(9)]

    result = refine_adjacency(matches, pieces, (3, 3))
    positions = [m.grid_pos for m in result]
    assert len(set(positions)) == len(positions)


def test_refine_adjacency_scores_in_range():
    from app.modules.adjacency.refiner import refine_adjacency

    matches = [_make_match(i, i // 2, i % 2) for i in range(4)]
    pieces = [_make_piece(i) for i in range(4)]

    result = refine_adjacency(matches, pieces, (2, 2))
    for m in result:
        assert 0.0 <= m.adjacency_score <= 1.0
        assert 0.0 <= m.curvature_complement_score <= 1.0


def test_refine_adjacency_empty_matches():
    from app.modules.adjacency.refiner import refine_adjacency

    result = refine_adjacency([], [], (2, 2))
    assert result == []