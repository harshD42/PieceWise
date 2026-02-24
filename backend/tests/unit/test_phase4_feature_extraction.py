# Copyright (c) 2026 Harsh Dwivedi
# Licensed under the Harsh Non-Commercial Attribution License (HNCAL) v1.0
# Commercial use requires written permission. See LICENSE for details.

"""
Phase 4 — Feature extraction module tests.
All DINOv2 inference is mocked — no model weights or GPU required.
Tests cover: PCA reducer, embedding store, token grid operations,
patch generator logic, and piece embedder helpers.
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest
import torch


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _make_bgr(h: int, w: int, color=(100, 120, 140)) -> np.ndarray:
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:] = color
    return img


def _make_token_grid(h: int, w: int, d: int = 768) -> np.ndarray:
    """Random float32 token grid (H, W, D)."""
    rng = np.random.default_rng(42)
    return rng.standard_normal((h, w, d)).astype(np.float32)


def _make_piece_crop(piece_id: int = 0, h: int = 80, w: int = 80):
    """Create a minimal PieceCrop for testing."""
    from app.models.piece import PieceCrop
    import cv2, numpy as np

    img = _make_bgr(h, w)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (w // 2, h // 2), min(h, w) // 3, 255, -1)
    contour_pts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = contour_pts[0] if contour_pts else np.array([[[0, 0]]], dtype=np.int32)

    return PieceCrop(
        piece_id=piece_id,
        image=img,
        alpha_mask=mask,
        bbox=(0, 0, w, h),
        contour=contour,
        area_px=float(mask.sum() // 255),
        solidity=0.9,
        compactness=0.8,
    )


# ─── PCA Reducer ─────────────────────────────────────────────────────────────

def test_pca_reducer_disabled_passthrough():
    from app.modules.feature_extraction.pca_reducer import PCAReducer
    reducer = PCAReducer(n_components=128, enabled=False)
    data = np.random.randn(100, 768).astype(np.float32)
    result = reducer.transform(data)
    # Pass-through — shape unchanged
    assert result.shape == data.shape
    assert reducer.output_dims == 768


def test_pca_reducer_fit_reduces_dims():
    from app.modules.feature_extraction.pca_reducer import PCAReducer
    reducer = PCAReducer(n_components=32, enabled=True)
    data = np.random.randn(200, 64).astype(np.float32)
    reduced = reducer.fit_transform(data)
    assert reduced.shape == (200, 32)
    assert reducer.output_dims == 32


def test_pca_reducer_transform_without_fit_raises():
    from app.modules.feature_extraction.pca_reducer import PCAReducer
    reducer = PCAReducer(n_components=32, enabled=True)
    data = np.random.randn(10, 64).astype(np.float32)
    with pytest.raises(RuntimeError, match="fit"):
        reducer.transform(data)


def test_pca_reducer_fit_then_transform_consistent():
    from app.modules.feature_extraction.pca_reducer import PCAReducer
    reducer = PCAReducer(n_components=16, enabled=True)
    train = np.random.randn(100, 32).astype(np.float32)
    reducer.fit(train)
    test = np.random.randn(20, 32).astype(np.float32)
    result = reducer.transform(test)
    assert result.shape == (20, 16)


def test_pca_reducer_save_load():
    from app.modules.feature_extraction.pca_reducer import PCAReducer
    reducer = PCAReducer(n_components=8, enabled=True)
    data = np.random.randn(100, 32).astype(np.float32)
    original_result = reducer.fit_transform(data)

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "pca_model.pkl"
        reducer.save(path)
        assert path.exists()

        # Load into new reducer and verify same transform
        reducer2 = PCAReducer(n_components=8, enabled=True)
        loaded = reducer2.load(path)
        assert loaded is True
        result2 = reducer2.transform(data)
        np.testing.assert_allclose(original_result, result2, rtol=1e-5)


def test_pca_reducer_save_disabled_noop():
    from app.modules.feature_extraction.pca_reducer import PCAReducer
    reducer = PCAReducer(n_components=8, enabled=False)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "pca_model.pkl"
        reducer.save(path)
        assert not path.exists()


def test_pca_reducer_load_missing_file_returns_false():
    from app.modules.feature_extraction.pca_reducer import PCAReducer
    reducer = PCAReducer(n_components=8, enabled=True)
    result = reducer.load(Path("/nonexistent/path.pkl"))
    assert result is False


def test_make_reducer_uses_config():
    from app.modules.feature_extraction.pca_reducer import make_reducer
    from app.config import Settings
    with patch("app.modules.feature_extraction.pca_reducer.get_settings") as mock_cfg:
        mock_cfg.return_value = Settings(
            _env_file=None,
            enable_pca_reduction=True,
            pca_target_dims=64,
        )
        reducer = make_reducer()
    assert reducer.enabled is True
    assert reducer.n_components == 64


# ─── DINOv2 Loader (state only — no model loaded) ────────────────────────────

def test_dino_loader_not_loaded_initially():
    from app.modules.feature_extraction.dino_loader import is_loaded
    import app.modules.feature_extraction.dino_loader as dl
    original = dl._dino_model
    dl._dino_model = None
    assert is_loaded() is False
    dl._dino_model = original


def test_get_dino_model_raises_before_init():
    from app.modules.feature_extraction.dino_loader import get_dino_model
    import app.modules.feature_extraction.dino_loader as dl
    original = dl._dino_model
    dl._dino_model = None
    with pytest.raises(RuntimeError, match="not initialised"):
        get_dino_model()
    dl._dino_model = original


def test_get_dino_processor_raises_before_init():
    from app.modules.feature_extraction.dino_loader import get_dino_processor
    import app.modules.feature_extraction.dino_loader as dl
    original = dl._dino_processor
    dl._dino_processor = None
    with pytest.raises(RuntimeError, match="not initialised"):
        get_dino_processor()
    dl._dino_processor = original


# ─── Embedding Store ─────────────────────────────────────────────────────────

def test_embedding_store_empty_summary():
    from app.modules.feature_extraction.embedding_store import EmbeddingStore
    store = EmbeddingStore()
    s = store.summary()
    assert s["n_pieces"] == 0
    assert s["n_cells"] == 0
    assert s["ref_ready"] is False


def test_embedding_store_set_pieces():
    from app.modules.feature_extraction.embedding_store import EmbeddingStore
    from app.models.piece import PieceEmbedding
    store = EmbeddingStore()

    emb = PieceEmbedding(piece_id=0)
    emb.token_grids[0] = _make_token_grid(4, 4, 32)
    emb.cls_vectors[0] = np.random.randn(32).astype(np.float32)

    store.set_piece_embeddings([emb])
    assert store.n_pieces == 1
    assert 0 in store.get_all_piece_ids()


def test_embedding_store_get_token_grid():
    from app.modules.feature_extraction.embedding_store import EmbeddingStore
    from app.models.piece import PieceEmbedding
    store = EmbeddingStore()

    grid = _make_token_grid(4, 4, 32)
    emb = PieceEmbedding(piece_id=5)
    emb.token_grids[90] = grid
    store.add_piece_embedding(emb)

    result = store.get_piece_token_grid(5, 90)
    np.testing.assert_array_equal(result, grid)


def test_embedding_store_missing_piece_raises():
    from app.modules.feature_extraction.embedding_store import EmbeddingStore
    store = EmbeddingStore()
    with pytest.raises(KeyError):
        store.get_piece_token_grid(999, 0)


def test_embedding_store_missing_rotation_raises():
    from app.modules.feature_extraction.embedding_store import EmbeddingStore
    from app.models.piece import PieceEmbedding
    store = EmbeddingStore()
    emb = PieceEmbedding(piece_id=1)
    store.add_piece_embedding(emb)
    with pytest.raises(KeyError):
        store.get_piece_token_grid(1, 45)  # 45° not a valid rotation


def test_embedding_store_ref_cls_gpu():
    from app.modules.feature_extraction.embedding_store import EmbeddingStore
    from app.models.grid import PatchTokenMap, PatchTokenCell
    store = EmbeddingStore()

    # Build minimal PatchTokenMap
    token_map = PatchTokenMap(
        grid_shape=(2, 2),
        patch_size_px=(100, 100),
    )
    cls_matrix = np.random.randn(4, 32).astype(np.float32)
    token_map.cls_matrix = cls_matrix
    token_map.cls_index = [(0,0),(0,1),(1,0),(1,1)]

    # Add cells
    for r in range(2):
        for c in range(2):
            cell = PatchTokenCell(
                grid_pos=(r, c),
                token_grid=_make_token_grid(3, 3, 32),
                token_flat_normalised=np.random.randn(9, 32).astype(np.float32),
                cls_vector=np.random.randn(32).astype(np.float32),
            )
            token_map.set_cell(cell)

    store.set_patch_token_map(token_map)
    assert store.n_cells == 4

    cls_gpu = store.get_ref_cls_gpu()
    assert isinstance(cls_gpu, torch.Tensor)
    assert cls_gpu.shape == (4, 32)


def test_embedding_store_get_cell_flat():
    from app.modules.feature_extraction.embedding_store import EmbeddingStore
    from app.models.grid import PatchTokenMap, PatchTokenCell
    store = EmbeddingStore()

    flat = np.random.randn(9, 32).astype(np.float32)
    cell = PatchTokenCell(
        grid_pos=(0, 0),
        token_grid=_make_token_grid(3, 3, 32),
        token_flat_normalised=flat,
        cls_vector=np.random.randn(32).astype(np.float32),
    )
    token_map = PatchTokenMap(grid_shape=(1, 1), patch_size_px=(100, 100))
    token_map.set_cell(cell)
    token_map.cls_matrix = np.random.randn(1, 32).astype(np.float32)
    token_map.cls_index = [(0, 0)]
    store.set_patch_token_map(token_map)

    result = store.get_cell_flat_normalised(0, 0)
    np.testing.assert_array_equal(result, flat)


# ─── L2 Normalisation (used in patch_generator) ──────────────────────────────

def test_l2_normalise_rows():
    from app.modules.feature_extraction.patch_generator import _l2_normalise_rows
    data = np.array([[3.0, 4.0], [1.0, 0.0], [0.0, 0.0]], dtype=np.float32)
    normed = _l2_normalise_rows(data)
    # Row 0: norm=5, should become [0.6, 0.8]
    np.testing.assert_allclose(normed[0], [0.6, 0.8], atol=1e-6)
    # Row 1: norm=1, unchanged
    np.testing.assert_allclose(normed[1], [1.0, 0.0], atol=1e-6)
    # Row 2: zero row, kept as zero (no div-by-zero)
    np.testing.assert_allclose(normed[2], [0.0, 0.0], atol=1e-6)


def test_l2_normalise_rows_all_unit():
    from app.modules.feature_extraction.patch_generator import _l2_normalise_rows
    data = np.random.randn(50, 32).astype(np.float32)
    normed = _l2_normalise_rows(data)
    norms = np.linalg.norm(normed, axis=1)
    np.testing.assert_allclose(norms, np.ones(50), atol=1e-5)


# ─── Image preparation helpers ────────────────────────────────────────────────

def test_prepare_image_tensor_mocked():
    """Verify _prepare_image_tensor produces correct tensor shape when mocked."""
    from app.modules.feature_extraction.patch_generator import _prepare_image_tensor

    mock_processor = MagicMock()
    mock_processor.return_value = {
        "pixel_values": torch.zeros(1, 3, 518, 518)
    }

    with patch("app.modules.feature_extraction.patch_generator.get_dino_processor",
               return_value=mock_processor), \
         patch("app.modules.feature_extraction.patch_generator.get_device",
               return_value="cpu"):
        img = _make_bgr(400, 600)
        tensor = _prepare_image_tensor(img)

    assert tensor.shape == (1, 3, 518, 518)


def test_paste_on_background_grey():
    from app.utils.image_utils import paste_on_background
    img = _make_bgr(80, 80, (200, 100, 50))
    mask = np.zeros((80, 80), dtype=np.uint8)
    cv2.circle(mask, (40, 40), 30, 255, -1)

    result = paste_on_background(img, mask, size=224)
    assert result.shape == (224, 224, 3)
    assert result.dtype == np.uint8
    # Background pixels (mask=0) should be grey (128)
    # Check corner pixels (far from centre circle)
    assert abs(int(result[0, 0, 0]) - 128) < 10


# ─── Rotation helpers (used in piece_embedder) ───────────────────────────────

def test_rotation_variants_coverage():
    from app.utils.geometry_utils import ROTATION_VARIANTS
    assert set(ROTATION_VARIANTS) == {0, 90, 180, 270}
    assert len(ROTATION_VARIANTS) == 4


def test_rotation_deg_to_k():
    from app.utils.geometry_utils import rotation_deg_to_k
    assert rotation_deg_to_k(0) == 0
    assert rotation_deg_to_k(90) == 1
    assert rotation_deg_to_k(180) == 2
    assert rotation_deg_to_k(270) == 3
    assert rotation_deg_to_k(360) == 0


def test_rotate_image_90_all_rotations():
    from app.utils.geometry_utils import rotate_image_90
    img = np.zeros((100, 200, 3), dtype=np.uint8)
    img[0, 0] = [255, 0, 0]  # marker pixel

    r90 = rotate_image_90(img, k=1)
    assert r90.shape == (200, 100, 3)

    r180 = rotate_image_90(img, k=2)
    assert r180.shape == (100, 200, 3)

    r270 = rotate_image_90(img, k=3)
    assert r270.shape == (200, 100, 3)


# ─── PCA reducer + store integration ─────────────────────────────────────────

def test_pca_reducer_store_roundtrip():
    """Fit PCA on synthetic reference tokens, transform piece tokens,
    verify dimensions are consistent in the embedding store."""
    from app.modules.feature_extraction.pca_reducer import PCAReducer
    from app.modules.feature_extraction.embedding_store import EmbeddingStore
    from app.models.piece import PieceEmbedding

    D_in, D_out = 64, 16
    reducer = PCAReducer(n_components=D_out, enabled=True)

    # Simulate reference token fitting
    ref_tokens = np.random.randn(200, D_in).astype(np.float32)
    reducer.fit(ref_tokens)

    # Simulate piece embedding
    piece_tokens = np.random.randn(50, D_in).astype(np.float32)
    reduced_pieces = reducer.transform(piece_tokens)
    assert reduced_pieces.shape == (50, D_out)

    # Store in EmbeddingStore
    store = EmbeddingStore()
    emb = PieceEmbedding(piece_id=0)
    emb.token_grids[0] = reduced_pieces.reshape(5, 10, D_out)
    emb.cls_vectors[0] = reduced_pieces[0]
    store.add_piece_embedding(emb)

    grid = store.get_piece_token_grid(0, 0)
    assert grid.shape == (5, 10, D_out)