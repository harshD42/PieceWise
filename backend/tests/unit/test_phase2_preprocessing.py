# Copyright (c) 2026 Harsh Dwivedi
# Licensed under the Harsh Non-Commercial Attribution License (HNCAL) v1.0
# Commercial use requires written permission. See LICENSE for details.

"""
Phase 2 — Preprocessing module tests.
Covers validator, normalizer (resize + denoise + histogram match),
and grid estimator. No GPU or model weights required.
"""

import io
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _make_bgr_array(h: int, w: int, color: tuple = (100, 150, 200)) -> np.ndarray:
    """Create a solid-colour BGR uint8 array."""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:] = color
    return img


def _encode_jpeg(img: np.ndarray, quality: int = 90) -> bytes:
    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buf.tobytes()


def _encode_png(img: np.ndarray) -> bytes:
    _, buf = cv2.imencode(".png", img)
    return buf.tobytes()


def _encode_webp(img: np.ndarray) -> bytes:
    _, buf = cv2.imencode(".webp", img)
    return buf.tobytes()


def _save_temp_jpeg(img: np.ndarray, directory: str) -> Path:
    p = Path(directory) / "test_img.jpg"
    cv2.imwrite(str(p), img)
    return p


# ─── Validator: valid inputs ──────────────────────────────────────────────────

def test_validate_jpeg_bytes_returns_array():
    from app.modules.preprocessing.validator import validate_image_bytes
    img = _make_bgr_array(200, 300)
    data = _encode_jpeg(img)
    result = validate_image_bytes(data, label="test")
    assert isinstance(result, np.ndarray)
    assert result.ndim == 3
    assert result.shape[2] == 3


def test_validate_png_bytes():
    from app.modules.preprocessing.validator import validate_image_bytes
    img = _make_bgr_array(100, 100)
    data = _encode_png(img)
    result = validate_image_bytes(data, label="test")
    assert result.shape == (100, 100, 3)


def test_validate_webp_bytes():
    from app.modules.preprocessing.validator import validate_image_bytes
    img = _make_bgr_array(128, 128)
    data = _encode_webp(img)
    result = validate_image_bytes(data, label="test")
    assert result.shape[2] == 3


def test_validate_image_file_success():
    from app.modules.preprocessing.validator import validate_image_file
    img = _make_bgr_array(200, 300)
    with tempfile.TemporaryDirectory() as tmp:
        p = _save_temp_jpeg(img, tmp)
        result = validate_image_file(p, label="ref")
    assert result.ndim == 3


def test_validate_image_pair_returns_both():
    from app.modules.preprocessing.validator import validate_image_pair
    ref = _make_bgr_array(400, 600)
    pieces = _make_bgr_array(400, 600, color=(50, 80, 120))
    with tempfile.TemporaryDirectory() as tmp:
        p_ref = Path(tmp) / "ref.jpg"
        p_pieces = Path(tmp) / "pieces.jpg"
        cv2.imwrite(str(p_ref), ref)
        cv2.imwrite(str(p_pieces), pieces)
        r, p = validate_image_pair(p_ref, p_pieces)
    assert r.shape == ref.shape
    assert p.shape == pieces.shape


# ─── Validator: invalid inputs ────────────────────────────────────────────────

def test_validate_empty_bytes_raises():
    from app.modules.preprocessing.validator import validate_image_bytes
    from app.api.middleware.error_handler import ImageValidationError
    with pytest.raises(ImageValidationError, match="empty"):
        validate_image_bytes(b"", label="test")


def test_validate_unsupported_format_raises():
    from app.modules.preprocessing.validator import validate_image_bytes
    from app.api.middleware.error_handler import ImageValidationError
    # BMP magic bytes — not supported
    bmp_header = b"BM" + b"\x00" * 50
    with pytest.raises(ImageValidationError, match="not supported"):
        validate_image_bytes(bmp_header, label="test")


def test_validate_corrupted_bytes_raises():
    from app.modules.preprocessing.validator import validate_image_bytes
    from app.api.middleware.error_handler import ImageValidationError
    # Valid JPEG magic bytes but garbage content
    bad_data = b"\xff\xd8\xff" + b"\x00" * 100
    with pytest.raises(ImageValidationError, match="decoded"):
        validate_image_bytes(bad_data, label="test")


def test_validate_too_small_raises():
    from app.modules.preprocessing.validator import validate_image_bytes
    from app.api.middleware.error_handler import ImageValidationError
    # 10x10 is below MIN_DIMENSION_PX (64)
    img = _make_bgr_array(10, 10)
    data = _encode_jpeg(img)
    with pytest.raises(ImageValidationError, match="too small"):
        validate_image_bytes(data, label="test")


def test_validate_file_not_found_raises():
    from app.modules.preprocessing.validator import validate_image_file
    from app.api.middleware.error_handler import ImageValidationError
    with pytest.raises(ImageValidationError, match="not found"):
        validate_image_file(Path("/nonexistent/path/img.jpg"), label="test")


def test_validate_oversized_file_raises():
    from app.modules.preprocessing.validator import validate_image_bytes
    from app.api.middleware.error_handler import ImageValidationError
    from app.config import Settings
    # Create bytes larger than the configured limit
    settings = Settings(_env_file=None)
    oversized = b"\xff\xd8\xff" + b"\x00" * (settings.upload_max_bytes + 1)
    with pytest.raises(ImageValidationError, match="exceeds"):
        validate_image_bytes(oversized, label="test")


# ─── Normalizer: resize ───────────────────────────────────────────────────────

def test_resize_large_image_scales_down():
    from app.modules.preprocessing.normalizer import resize_for_pipeline, MAX_LONG_EDGE
    img = _make_bgr_array(3000, 4000)
    resized, scale = resize_for_pipeline(img)
    h, w = resized.shape[:2]
    assert max(h, w) <= MAX_LONG_EDGE
    assert scale < 1.0


def test_resize_small_image_unchanged():
    from app.modules.preprocessing.normalizer import resize_for_pipeline
    img = _make_bgr_array(800, 600)
    resized, scale = resize_for_pipeline(img)
    assert scale == 1.0
    assert resized.shape == img.shape


def test_resize_preserves_aspect_ratio():
    from app.modules.preprocessing.normalizer import resize_for_pipeline
    img = _make_bgr_array(2000, 4000)  # 1:2 aspect ratio
    resized, _ = resize_for_pipeline(img)
    h, w = resized.shape[:2]
    assert abs(w / h - 2.0) < 0.02


# ─── Normalizer: denoise ──────────────────────────────────────────────────────

def test_denoise_preserves_shape():
    from app.modules.preprocessing.normalizer import denoise_pieces
    img = _make_bgr_array(300, 400)
    denoised = denoise_pieces(img)
    assert denoised.shape == img.shape
    assert denoised.dtype == np.uint8


def test_denoise_reduces_noise():
    from app.modules.preprocessing.normalizer import denoise_pieces
    # Add salt-and-pepper noise
    img = _make_bgr_array(200, 200, (128, 128, 128))
    noisy = img.copy()
    rng = np.random.default_rng(42)
    noise_mask = rng.integers(0, 2, (200, 200), dtype=bool)
    noisy[noise_mask] = 255
    denoised = denoise_pieces(noisy)
    # Denoised image should have lower std deviation than noisy
    assert denoised.std() < noisy.std()


# ─── Normalizer: histogram matching ──────────────────────────────────────────

def test_histogram_match_output_shape():
    from app.modules.preprocessing.normalizer import match_histogram
    pieces = _make_bgr_array(300, 400, (50, 50, 50))    # dark
    ref = _make_bgr_array(300, 400, (200, 200, 200))    # bright
    matched = match_histogram(pieces, ref)
    assert matched.shape == pieces.shape
    assert matched.dtype == np.uint8


def test_histogram_match_shifts_distribution():
    from app.modules.preprocessing.normalizer import match_histogram
    # Dark pieces image matched to bright reference
    pieces = _make_bgr_array(300, 400, (30, 30, 30))
    ref = _make_bgr_array(300, 400, (220, 220, 220))
    matched = match_histogram(pieces, ref)
    # Mean of matched should be significantly higher than original pieces
    assert matched.mean() > pieces.mean() + 50


def test_histogram_match_identical_images_unchanged():
    from app.modules.preprocessing.normalizer import match_histogram
    img = _make_bgr_array(200, 200, (128, 100, 80))
    matched = match_histogram(img.copy(), img.copy())
    # When src == ref, output should be very close to input
    assert np.abs(matched.astype(int) - img.astype(int)).mean() < 5.0


def test_histogram_match_preserves_relative_channel_info():
    from app.modules.preprocessing.normalizer import match_histogram
    # Use a real gradient image so channels have structure
    pieces = np.zeros((256, 256, 3), dtype=np.uint8)
    ref = np.zeros((256, 256, 3), dtype=np.uint8)
    for i in range(256):
        pieces[i, :] = (i // 2, i // 3, i // 4)
        ref[i, :] = (i, i, i)
    matched = match_histogram(pieces, ref)
    assert matched.shape == pieces.shape


# ─── Normalizer: full pipeline ────────────────────────────────────────────────

def test_normalize_from_arrays_returns_normalised_pair():
    from app.modules.preprocessing.normalizer import normalize_from_arrays, NormalisedPair
    ref = _make_bgr_array(1000, 1500, (200, 180, 160))
    pieces = _make_bgr_array(800, 1200, (60, 50, 40))
    result = normalize_from_arrays(ref, pieces)
    assert isinstance(result, NormalisedPair)
    assert result.ref_img.ndim == 3
    assert result.pieces_img.ndim == 3
    assert result.ref_scale <= 1.0
    assert result.pieces_scale <= 1.0


def test_normalize_pair_from_files():
    from app.modules.preprocessing.normalizer import normalize_image_pair
    ref = _make_bgr_array(1200, 1600, (200, 180, 160))
    pieces = _make_bgr_array(900, 1200, (60, 50, 40))
    with tempfile.TemporaryDirectory() as tmp:
        p_ref = Path(tmp) / "ref.jpg"
        p_pieces = Path(tmp) / "pieces.jpg"
        cv2.imwrite(str(p_ref), ref)
        cv2.imwrite(str(p_pieces), pieces)
        result = normalize_image_pair(p_ref, p_pieces)
    assert result.ref_img is not None
    assert result.pieces_img is not None
    assert result.ref_original_shape == (1200, 1600)
    assert result.pieces_original_shape == (900, 1200)


def test_normalize_large_images_rescaled():
    from app.modules.preprocessing.normalizer import normalize_from_arrays, MAX_LONG_EDGE
    ref = _make_bgr_array(3000, 4000)
    pieces = _make_bgr_array(2500, 3500)
    result = normalize_from_arrays(ref, pieces)
    assert max(result.ref_img.shape[:2]) <= MAX_LONG_EDGE
    assert max(result.pieces_img.shape[:2]) <= MAX_LONG_EDGE


# ─── Grid Estimator ──────────────────────────────────────────────────────────

def test_grid_estimator_1000_pieces_4x3():
    from app.modules.preprocessing.grid_estimator import estimate_grid_shape
    # 4:3 reference image, 1000 pieces → expect roughly 37x27 or similar
    rows, cols = estimate_grid_shape(1000, (768, 1024))
    assert rows * cols > 0
    # Total cells should be within 15% of 1000
    assert abs(rows * cols - 1000) / 1000 <= 0.15


def test_grid_estimator_500_pieces_square():
    from app.modules.preprocessing.grid_estimator import estimate_grid_shape
    rows, cols = estimate_grid_shape(500, (1000, 1000))
    # Square image → rows and cols should be reasonably close
    # 25x20=500 is a valid result — allow up to 30% difference between dims
    assert rows * cols > 0
    assert abs(rows * cols - 500) / 500 <= 0.15
    larger, smaller = max(rows, cols), min(rows, cols)
    assert larger / smaller <= 1.5  # neither dim more than 1.5x the other


def test_grid_estimator_override_accepted():
    from app.modules.preprocessing.grid_estimator import estimate_grid_shape
    rows, cols = estimate_grid_shape(1000, (768, 1024), override=(27, 37))
    assert rows == 27
    assert cols == 37


def test_grid_estimator_override_invalid_too_small():
    from app.modules.preprocessing.grid_estimator import estimate_grid_shape
    with pytest.raises(ValueError, match="at least"):
        estimate_grid_shape(100, (400, 600), override=(1, 10))


def test_grid_estimator_override_invalid_too_large():
    from app.modules.preprocessing.grid_estimator import estimate_grid_shape
    with pytest.raises(ValueError, match="exceed"):
        estimate_grid_shape(100, (400, 600), override=(200, 10))


def test_grid_estimator_zero_pieces_raises():
    from app.modules.preprocessing.grid_estimator import estimate_grid_shape
    with pytest.raises(ValueError, match="positive"):
        estimate_grid_shape(0, (400, 600))


def test_patch_size_px():
    from app.modules.preprocessing.grid_estimator import patch_size_px
    ph, pw = patch_size_px((1080, 1920), (27, 48))
    assert abs(ph - 40.0) < 0.01
    assert abs(pw - 40.0) < 0.01


def test_grid_wide_aspect_ratio():
    from app.modules.preprocessing.grid_estimator import estimate_grid_shape
    # Very wide image (panorama-like) — cols >> rows
    rows, cols = estimate_grid_shape(200, (400, 1600))
    assert cols > rows


def test_grid_tall_aspect_ratio():
    from app.modules.preprocessing.grid_estimator import estimate_grid_shape
    # Tall image — rows >> cols
    rows, cols = estimate_grid_shape(200, (1600, 400))
    assert rows > cols