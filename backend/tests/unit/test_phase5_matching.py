# Copyright (c) 2026 Harsh Dwivedi
# Licensed under the Harsh Non-Commercial Attribution License (HNCAL) v1.0
# Commercial use requires written permission. See LICENSE for details.

"""
Phase 5 — Matching engine tests.
All tests use synthetic embeddings — no DINOv2, SAM, or GPU required.
Tests cover: spatial similarity, coarse filter, flat-side scorer,
Hungarian resolver, confidence scorer, and full pipeline integration.
"""

import numpy as np
import pytest
import torch

from app.models.piece import CandidateMatch, PieceCrop, PieceMatch


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _rand_f32(*shape) -> np.ndarray:
    rng = np.random.default_rng(42)
    return rng.standard_normal(shape).astype(np.float32)


def _l2_norm(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.where(n == 0, 1.0, n)


def _make_piece(
    piece_id: int,
    flat_side_count: int = 0,
    h: int = 60,
    w: int = 60,
) -> PieceCrop:
    import cv2
    img = np.zeros((h, w, 3), dtype=np.uint8)
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


def _make_embedding_store(
    grid_shape: tuple[int, int],
    n_pieces: int,
    d: int = 32,
):
    """Build a fully populated EmbeddingStore with synthetic data."""
    from app.modules.feature_extraction.embedding_store import EmbeddingStore
    from app.models.grid import PatchTokenMap, PatchTokenCell
    from app.models.piece import PieceEmbedding
    from app.utils.geometry_utils import ROTATION_VARIANTS

    rng = np.random.default_rng(0)
    store = EmbeddingStore()
    n_rows, n_cols = grid_shape
    n_cells = n_rows * n_cols

    # Build PatchTokenMap
    token_map = PatchTokenMap(
        grid_shape=grid_shape,
        patch_size_px=(50, 50),
    )
    cls_vecs = []
    cls_idx = []
    for r in range(n_rows):
        for c in range(n_cols):
            grid = rng.standard_normal((3, 3, d)).astype(np.float32)
            flat = grid.reshape(9, d)
            flat_norm = _l2_norm(flat)
            cls_v = _l2_norm(flat_norm.mean(axis=0, keepdims=True))[0]
            cell = PatchTokenCell(
                grid_pos=(r, c),
                token_grid=grid,
                token_flat_normalised=flat_norm,
                cls_vector=cls_v,
            )
            token_map.set_cell(cell)
            cls_vecs.append(cls_v)
            cls_idx.append((r, c))

    token_map.cls_matrix = np.stack(cls_vecs, axis=0)
    token_map.cls_index = cls_idx
    store.set_patch_token_map(token_map)

    # Build piece embeddings
    embeddings = []
    for pid in range(n_pieces):
        emb = PieceEmbedding(piece_id=pid)
        for rot in ROTATION_VARIANTS:
            grid = rng.standard_normal((3, 3, d)).astype(np.float32)
            emb.token_grids[rot] = grid
            emb.cls_vectors[rot] = _l2_norm(
                rng.standard_normal((1, d)).astype(np.float32)
            )[0]
        embeddings.append(emb)
    store.set_piece_embeddings(embeddings)

    return store


# ─── Spatial Similarity ───────────────────────────────────────────────────────

def test_spatial_similarity_identical_returns_one():
    from app.modules.matching.similarity import spatial_similarity

    d = 16
    # Identical normalised token grids → similarity should be ~1.0
    grid = _l2_norm(_rand_f32(4, 4, d).reshape(-1, d)).reshape(4, 4, d)
    ref_flat = _l2_norm(grid.reshape(16, d))

    score = spatial_similarity(grid, ref_flat, device="cpu")
    assert abs(score - 1.0) < 0.01


def test_spatial_similarity_orthogonal_returns_zero():
    from app.modules.matching.similarity import spatial_similarity

    d = 8
    rng = np.random.default_rng(1)
    # Make two orthogonal vectors
    v1 = np.zeros((1, d), dtype=np.float32)
    v2 = np.zeros((1, d), dtype=np.float32)
    v1[0, 0] = 1.0
    v2[0, 1] = 1.0

    grid = v1.reshape(1, 1, d)
    ref_flat = v2  # orthogonal to piece token

    score = spatial_similarity(grid, ref_flat, device="cpu")
    assert score == 0.0  # clamped at 0


def test_spatial_similarity_in_range():
    from app.modules.matching.similarity import spatial_similarity

    d = 32
    grid = _rand_f32(3, 3, d)
    ref_flat = _l2_norm(_rand_f32(9, d))

    score = spatial_similarity(grid, ref_flat, device="cpu")
    assert 0.0 <= score <= 1.0


def test_batch_spatial_similarity_length():
    from app.modules.matching.similarity import batch_spatial_similarity

    d = 16
    grid = _rand_f32(3, 3, d)
    candidates = [
        ((0, 0), _l2_norm(_rand_f32(9, d))),
        ((0, 1), _l2_norm(_rand_f32(9, d))),
        ((1, 0), _l2_norm(_rand_f32(9, d))),
    ]

    results = batch_spatial_similarity(grid, candidates, device="cpu")
    assert len(results) == 3
    for pos, score in results:
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0


def test_batch_spatial_similarity_grid_positions_preserved():
    from app.modules.matching.similarity import batch_spatial_similarity

    d = 8
    grid = _rand_f32(2, 2, d)
    positions = [(0, 0), (1, 2), (3, 4)]
    candidates = [(p, _l2_norm(_rand_f32(4, d))) for p in positions]

    results = batch_spatial_similarity(grid, candidates, device="cpu")
    result_positions = [r[0] for r in results]
    assert result_positions == positions


# ─── Flat-Side Scorer ────────────────────────────────────────────────────────

def test_flat_side_score_corner_exact_match():
    from app.modules.matching.flat_side_scorer import flat_side_score
    piece = _make_piece(0, flat_side_count=2)
    score = flat_side_score(piece, (0, 0), (5, 5))  # corner cell
    assert score == 1.0


def test_flat_side_score_edge_exact_match():
    from app.modules.matching.flat_side_scorer import flat_side_score
    piece = _make_piece(0, flat_side_count=1)
    score = flat_side_score(piece, (0, 2), (5, 5))  # top edge
    assert score == 1.0


def test_flat_side_score_interior_exact_match():
    from app.modules.matching.flat_side_scorer import flat_side_score
    piece = _make_piece(0, flat_side_count=0)
    score = flat_side_score(piece, (2, 2), (5, 5))  # interior
    assert score == 1.0


def test_flat_side_score_mismatch_penalised():
    from app.modules.matching.flat_side_scorer import flat_side_score
    # Corner cell expects 2 flat sides, piece has 0
    piece = _make_piece(0, flat_side_count=0)
    score = flat_side_score(piece, (0, 0), (5, 5))
    assert score < 1.0
    assert score >= 0.0


def test_flat_side_score_large_mismatch_floored():
    from app.modules.matching.flat_side_scorer import flat_side_score
    # Corner cell expects 2, piece has 0 — diff=2 → 1.0 - 0.5*2 = 0.0
    piece = _make_piece(0, flat_side_count=0)
    score = flat_side_score(piece, (0, 0), (5, 5))
    assert score == 0.0


def test_score_all_candidates_structure():
    from app.modules.matching.flat_side_scorer import score_all_candidates

    pieces = [_make_piece(0, flat_side_count=2), _make_piece(1, flat_side_count=0)]
    candidates = {
        0: {0: [((0, 0), 0.9, 0.8), ((0, 1), 0.7, 0.6)]},
        1: {0: [((1, 1), 0.8, 0.75)]},
    }
    scored = score_all_candidates(pieces, candidates, (3, 3))

    assert 0 in scored
    assert 1 in scored
    # Each tuple should now have 4 elements (added flat_side score)
    for gpos, coarse, fine, flat_s in scored[0][0]:
        assert 0.0 <= flat_s <= 1.0


# ─── Cost Matrix ─────────────────────────────────────────────────────────────

def test_build_cost_matrix_shape():
    from app.modules.matching.conflict_resolver import build_cost_matrix

    pieces = [_make_piece(i) for i in range(3)]
    scored = {
        0: {0: [((0, 0), 0.9, 0.8, 1.0), ((0, 1), 0.6, 0.5, 0.5)]},
        1: {0: [((1, 0), 0.85, 0.75, 1.0)]},
        2: {0: [((0, 1), 0.7, 0.65, 0.8)]},
    }
    grid_shape = (2, 2)  # 4 cells

    cost_matrix, piece_ids, cell_positions = build_cost_matrix(
        pieces, scored, grid_shape
    )

    assert cost_matrix.shape == (3, 4)
    assert len(piece_ids) == 3
    assert len(cell_positions) == 4


def test_build_cost_matrix_values_in_range():
    from app.modules.matching.conflict_resolver import build_cost_matrix

    pieces = [_make_piece(0)]
    scored = {0: {0: [((0, 0), 0.9, 0.8, 1.0)]}}
    cost_matrix, _, _ = build_cost_matrix(pieces, scored, (2, 2))

    assert np.all(cost_matrix >= 0.0)
    assert np.all(cost_matrix <= 1.0)


def test_build_cost_matrix_unmatched_cells_max_cost():
    from app.modules.matching.conflict_resolver import build_cost_matrix

    pieces = [_make_piece(0)]
    # Piece only has candidates for cell (0,0)
    scored = {0: {0: [((0, 0), 0.9, 0.8, 1.0)]}}
    cost_matrix, _, cell_positions = build_cost_matrix(pieces, scored, (2, 2))

    # Cell (0,0) should have low cost, others should have max cost (1.0)
    cell_to_col = {pos: i for i, pos in enumerate(cell_positions)}
    matched_col = cell_to_col[(0, 0)]

    for col in range(4):
        if col != matched_col:
            assert cost_matrix[0, col] == 1.0


# ─── Hungarian Resolver ──────────────────────────────────────────────────────

def test_resolve_returns_one_match_per_piece():
    from app.modules.matching.conflict_resolver import resolve

    n = 4
    pieces = [_make_piece(i) for i in range(n)]
    # Give each piece a unique best cell
    scored = {}
    for i in range(n):
        scored[i] = {
            0: [((i // 2, i % 2), 0.9, 0.85, 1.0)]
        }

    matches = resolve(pieces, scored, (2, 2))
    assert len(matches) == n
    piece_ids_in_matches = {m.piece_id for m in matches}
    assert piece_ids_in_matches == {0, 1, 2, 3}


def test_resolve_no_duplicate_grid_positions():
    from app.modules.matching.conflict_resolver import resolve

    pieces = [_make_piece(i) for i in range(4)]
    # All pieces prefer (0,0) — Hungarian should force unique assignments
    scored = {}
    for i in range(4):
        scored[i] = {
            0: [
                ((0, 0), 0.95, 0.9, 1.0),
                ((0, 1), 0.7, 0.65, 0.8),
                ((1, 0), 0.6, 0.55, 0.7),
                ((1, 1), 0.5, 0.45, 0.6),
            ]
        }

    matches = resolve(pieces, scored, (2, 2))
    grid_positions = [m.grid_pos for m in matches]
    # All grid positions should be unique
    assert len(set(grid_positions)) == len(grid_positions)


def test_resolve_scores_populated():
    from app.modules.matching.conflict_resolver import resolve

    pieces = [_make_piece(0, flat_side_count=2)]
    scored = {0: {0: [((0, 0), 0.9, 0.8, 1.0)]}}
    matches = resolve(pieces, scored, (2, 2))

    assert len(matches) == 1
    m = matches[0]
    assert 0.0 <= m.composite_confidence <= 1.0
    assert 0.0 <= m.spatial_score <= 1.0


def test_resolve_top3_candidates_populated():
    from app.modules.matching.conflict_resolver import resolve

    pieces = [_make_piece(0)]
    scored = {
        0: {
            0: [
                ((0, 0), 0.95, 0.9, 1.0),
                ((0, 1), 0.80, 0.75, 0.8),
                ((1, 0), 0.70, 0.65, 0.7),
                ((1, 1), 0.60, 0.55, 0.6),
            ]
        }
    }
    matches = resolve(pieces, scored, (2, 2))
    m = matches[0]
    assert len(m.top3_candidates) <= 3
    for c in m.top3_candidates:
        assert isinstance(c, CandidateMatch)
        assert 0.0 <= c.composite_score <= 1.0


# ─── Confidence Scorer ───────────────────────────────────────────────────────

def test_flag_low_confidence_marks_below_threshold():
    from app.modules.matching.confidence_scorer import flag_low_confidence

    matches = [
        PieceMatch(piece_id=0, grid_pos=(0, 0), rotation_deg=0,
                   composite_confidence=0.8),
        PieceMatch(piece_id=1, grid_pos=(0, 1), rotation_deg=0,
                   composite_confidence=0.3),
        PieceMatch(piece_id=2, grid_pos=(1, 0), rotation_deg=0,
                   composite_confidence=0.55),
    ]

    result = flag_low_confidence(matches, threshold=0.55)
    assert result[0].flagged is False   # 0.8 >= 0.55
    assert result[1].flagged is True    # 0.3 < 0.55
    assert result[2].flagged is False   # 0.55 == threshold (not below)


def test_flag_low_confidence_none_flagged():
    from app.modules.matching.confidence_scorer import flag_low_confidence

    matches = [
        PieceMatch(piece_id=i, grid_pos=(0, i), rotation_deg=0,
                   composite_confidence=0.9)
        for i in range(5)
    ]
    result = flag_low_confidence(matches, threshold=0.5)
    assert all(not m.flagged for m in result)


def test_flag_low_confidence_all_flagged():
    from app.modules.matching.confidence_scorer import flag_low_confidence

    matches = [
        PieceMatch(piece_id=i, grid_pos=(0, i), rotation_deg=0,
                   composite_confidence=0.1)
        for i in range(3)
    ]
    result = flag_low_confidence(matches, threshold=0.9)
    assert all(m.flagged for m in result)


def test_compute_confidence_stats():
    from app.modules.matching.confidence_scorer import compute_confidence_stats

    matches = [
        PieceMatch(piece_id=0, grid_pos=(0, 0), rotation_deg=0,
                   composite_confidence=0.8, flagged=False),
        PieceMatch(piece_id=1, grid_pos=(0, 1), rotation_deg=0,
                   composite_confidence=0.4, flagged=True),
        PieceMatch(piece_id=2, grid_pos=(1, 0), rotation_deg=0,
                   composite_confidence=0.6, flagged=False),
    ]

    stats = compute_confidence_stats(matches)
    assert abs(stats["mean"] - (0.8 + 0.4 + 0.6) / 3) < 0.001
    assert stats["min"] == pytest.approx(0.4, abs=0.001)
    assert stats["max"] == pytest.approx(0.8, abs=0.001)
    assert stats["flagged_count"] == 1


def test_compute_confidence_stats_empty():
    from app.modules.matching.confidence_scorer import compute_confidence_stats
    stats = compute_confidence_stats([])
    assert stats["mean"] == 0.0
    assert stats["flagged_count"] == 0


# ─── Full Pipeline Integration (synthetic embeddings) ────────────────────────

def test_match_pieces_returns_one_per_piece():
    from app.modules.matching.matcher import match_pieces

    grid_shape = (2, 3)  # 6 cells
    n_pieces = 6
    store = _make_embedding_store(grid_shape, n_pieces, d=16)
    pieces = [_make_piece(i, flat_side_count=i % 3) for i in range(n_pieces)]

    with __import__("unittest.mock", fromlist=["patch"]).patch(
        "app.modules.matching.matcher.get_device", return_value="cpu"
    ):
        matches = match_pieces(store, pieces, grid_shape)

    assert len(matches) == n_pieces
    piece_ids = {m.piece_id for m in matches}
    assert piece_ids == set(range(n_pieces))


def test_match_pieces_no_duplicate_assignments():
    from app.modules.matching.matcher import match_pieces

    grid_shape = (3, 3)
    n_pieces = 9
    store = _make_embedding_store(grid_shape, n_pieces, d=16)
    pieces = [_make_piece(i) for i in range(n_pieces)]

    with __import__("unittest.mock", fromlist=["patch"]).patch(
        "app.modules.matching.matcher.get_device", return_value="cpu"
    ):
        matches = match_pieces(store, pieces, grid_shape)

    grid_positions = [m.grid_pos for m in matches]
    assert len(set(grid_positions)) == len(grid_positions)


def test_match_pieces_scores_in_range():
    from app.modules.matching.matcher import match_pieces

    grid_shape = (2, 2)
    store = _make_embedding_store(grid_shape, n_pieces=4, d=16)
    pieces = [_make_piece(i) for i in range(4)]

    with __import__("unittest.mock", fromlist=["patch"]).patch(
        "app.modules.matching.matcher.get_device", return_value="cpu"
    ):
        matches = match_pieces(store, pieces, grid_shape)

    for m in matches:
        assert 0.0 <= m.composite_confidence <= 1.0
        assert isinstance(m.flagged, bool)
        assert m.rotation_deg in (0, 90, 180, 270)