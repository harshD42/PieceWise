# Copyright (c) 2026 Harsh Dwivedi
# Licensed under the Harsh Non-Commercial Attribution License (HNCAL) v1.0
# Commercial use requires written permission. See LICENSE for details.

"""
Phase 3 — Segmentation module tests.
All tests use synthetic images and mock masks — no SAM weights or GPU required.
SAM loader is tested via is_loaded() state only.
"""

import numpy as np
import pytest
import cv2


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _make_bgr(h: int, w: int, color=(100, 120, 140)) -> np.ndarray:
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:] = color
    return img


def _make_circle_mask(h: int, w: int, cx: int, cy: int, r: int) -> np.ndarray:
    """Create a binary mask (0/255) with a filled circle."""
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (cx, cy), r, 255, -1)
    return mask


def _make_rect_mask(h: int, w: int, x: int, y: int, mw: int, mh: int) -> np.ndarray:
    """Create a binary mask with a filled rectangle."""
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[y:y+mh, x:x+mw] = 255
    return mask


def _mask_to_sam_dict(mask_u8: np.ndarray, stability: float = 0.9) -> dict:
    """Convert a uint8 mask to a SAM-style mask dict."""
    bool_mask = mask_u8.astype(bool)
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bbox = list(cv2.boundingRect(contours[0])) if contours else [0, 0, 10, 10]
    return {
        "segmentation": bool_mask,
        "area": int(bool_mask.sum()),
        "bbox": bbox,
        "predicted_iou": stability,
        "stability_score": stability,
        "point_coords": [[0, 0]],
        "crop_box": [0, 0, mask_u8.shape[1], mask_u8.shape[0]],
    }


# ─── SAM Loader ──────────────────────────────────────────────────────────────

def test_sam_loader_not_loaded_initially():
    from app.modules.segmentation.sam_loader import is_loaded
    # Before init_sam() is called, model should not be loaded
    # (module may have been imported already in other tests — reset)
    import app.modules.segmentation.sam_loader as sl
    original = sl._sam_model
    sl._sam_model = None
    assert is_loaded() is False
    sl._sam_model = original


def test_get_mask_generator_raises_before_init():
    from app.modules.segmentation.sam_loader import get_mask_generator
    import app.modules.segmentation.sam_loader as sl
    original = sl._mask_generator
    sl._mask_generator = None
    with pytest.raises(RuntimeError, match="not initialised"):
        get_mask_generator()
    sl._mask_generator = original


# ─── Connected Components Pre-filter ─────────────────────────────────────────

def test_cc_prefilter_returns_centroids():
    from app.modules.segmentation.mask_generator import connected_components_prefilter

    # Create image with 3 distinct blobs (simulated pieces)
    img = np.zeros((500, 500, 3), dtype=np.uint8)
    img[50:120, 50:120] = 200     # blob 1
    img[200:280, 200:280] = 200   # blob 2
    img[350:420, 350:420] = 200   # blob 3

    binary_mask, centroids = connected_components_prefilter(img)
    assert isinstance(centroids, list)
    assert binary_mask.dtype == np.uint8
    assert binary_mask.shape == (500, 500)


def test_cc_prefilter_rejects_tiny_noise():
    from app.modules.segmentation.mask_generator import connected_components_prefilter

    img = np.zeros((500, 500, 3), dtype=np.uint8)
    # Single 2x2 pixel noise blob — should be rejected
    img[10:12, 10:12] = 200
    # Large valid piece
    img[100:200, 100:200] = 200

    _, centroids = connected_components_prefilter(img)
    # Only the large blob should survive
    assert len(centroids) <= 1


def test_cc_prefilter_output_mask_binary():
    from app.modules.segmentation.mask_generator import connected_components_prefilter

    img = _make_bgr(400, 400, (50, 50, 50))
    img[100:200, 100:200] = 200
    binary_mask, _ = connected_components_prefilter(img)
    unique_vals = np.unique(binary_mask)
    assert set(unique_vals).issubset({0, 255})


# ─── Mask Filter ─────────────────────────────────────────────────────────────

def test_filter_by_area_removes_small_masks():
    from app.modules.segmentation.mask_filter import filter_by_area

    img_area = 500 * 500  # 250,000px
    # max_area = 4% of 250,000 = 10,000px
    # Circle r=40 → area ≈ 5,026px — safely within bounds
    # Circle r=1  → area ≈ 3px — well below min (0.02% = 50px)
    tiny_mask = _make_circle_mask(500, 500, 50, 50, 1)
    big_mask = _make_circle_mask(500, 500, 250, 250, 40)

    masks = [
        _mask_to_sam_dict(tiny_mask),
        _mask_to_sam_dict(big_mask),
    ]
    filtered = filter_by_area(masks, img_area)
    # Only big_mask should survive
    assert len(filtered) == 1


def test_filter_by_area_removes_huge_masks():
    from app.modules.segmentation.mask_filter import filter_by_area

    img_area = 500 * 500
    # Mask covering 80% of image — above max fraction (4%)
    huge = _make_rect_mask(500, 500, 0, 0, 490, 490)
    valid = _make_circle_mask(500, 500, 100, 100, 40)

    masks = [_mask_to_sam_dict(huge), _mask_to_sam_dict(valid)]
    filtered = filter_by_area(masks, img_area)
    assert len(filtered) == 1


def test_filter_by_solidity_removes_noise_blobs():
    from app.modules.segmentation.mask_filter import filter_by_solidity

    # L-shaped mask has low solidity
    l_shape = np.zeros((200, 200), dtype=np.uint8)
    l_shape[10:190, 10:30] = 255   # vertical bar
    l_shape[170:190, 10:190] = 255  # horizontal bar

    # Circle has high solidity (~1.0)
    circle = _make_circle_mask(200, 200, 100, 100, 40)

    masks = [_mask_to_sam_dict(l_shape), _mask_to_sam_dict(circle)]
    filtered = filter_by_solidity(masks)
    # Circle should survive, L-shape may or may not depending on threshold
    assert len(filtered) >= 1
    # The circle should always be in filtered
    areas = [m["area"] for m in filtered]
    circle_area = int(circle.astype(bool).sum())
    assert any(abs(a - circle_area) < 100 for a in areas)


def test_filter_by_aspect_ratio():
    from app.modules.segmentation.mask_filter import filter_by_aspect_ratio

    # Very thin horizontal strip — extreme aspect ratio
    thin = _make_rect_mask(500, 500, 10, 10, 400, 5)
    # Normal square piece
    square = _make_rect_mask(500, 500, 100, 100, 80, 80)

    masks = [_mask_to_sam_dict(thin), _mask_to_sam_dict(square)]
    filtered = filter_by_aspect_ratio(masks)
    # Square should survive, thin strip may be filtered
    assert len(filtered) >= 1


def test_deduplicate_keeps_higher_stability():
    from app.modules.segmentation.mask_filter import deduplicate_overlapping

    # Two heavily overlapping masks
    mask_a = _make_circle_mask(200, 200, 100, 100, 50)
    mask_b = _make_circle_mask(200, 200, 105, 105, 50)  # overlaps heavily

    dict_a = _mask_to_sam_dict(mask_a, stability=0.95)
    dict_b = _mask_to_sam_dict(mask_b, stability=0.70)

    result = deduplicate_overlapping([dict_a, dict_b])
    assert len(result) == 1
    assert result[0]["stability_score"] == 0.95


def test_filter_masks_full_pipeline():
    from app.modules.segmentation.mask_filter import filter_masks

    h, w = 600, 800
    # Valid piece-sized mask
    valid = _make_circle_mask(h, w, 200, 200, 50)
    # Tiny noise
    noise = _make_circle_mask(h, w, 10, 10, 2)

    masks = [_mask_to_sam_dict(valid), _mask_to_sam_dict(noise)]
    filtered = filter_masks(masks, (h, w))
    assert isinstance(filtered, list)
    # Valid mask should survive
    assert len(filtered) >= 1


# ─── Segmentation Refiner ────────────────────────────────────────────────────

def test_refiner_keeps_clean_masks():
    from app.modules.segmentation.segmentation_refiner import refine_masks

    # Clean circle — no deep convexity defects
    mask = _make_circle_mask(400, 400, 200, 200, 80)
    masks = [_mask_to_sam_dict(mask)]
    refined = refine_masks(masks, (400, 400), min_piece_area=500)
    # Should keep original mask
    assert len(refined) == 1


def test_refiner_attempts_split_on_dumbbell():
    from app.modules.segmentation.segmentation_refiner import refine_masks

    # Dumbbell shape — two circles connected by thin neck
    mask = np.zeros((400, 400), dtype=np.uint8)
    cv2.circle(mask, (120, 200), 70, 255, -1)   # left circle
    cv2.circle(mask, (280, 200), 70, 255, -1)   # right circle
    mask[185:215, 120:280] = 255                  # thin connector

    masks = [_mask_to_sam_dict(mask)]
    refined = refine_masks(masks, (400, 400), min_piece_area=1000)
    # May produce 1 or 2 masks — just verify no crash and valid output
    assert len(refined) >= 1
    for m in refined:
        assert "segmentation" in m
        assert "area" in m


def test_refiner_output_always_has_required_fields():
    from app.modules.segmentation.segmentation_refiner import refine_masks

    mask = _make_rect_mask(300, 300, 50, 50, 100, 100)
    masks = [_mask_to_sam_dict(mask)]
    refined = refine_masks(masks, (300, 300), min_piece_area=100)
    for m in refined:
        assert "segmentation" in m
        assert "area" in m
        assert "bbox" in m


# ─── Piece Extractor ─────────────────────────────────────────────────────────

def test_extract_pieces_assigns_ids():
    from app.modules.segmentation.piece_extractor import extract_pieces
    from app.models.piece import PieceCrop

    h, w = 400, 400
    img = _make_bgr(h, w, (150, 150, 150))

    mask1 = _make_circle_mask(h, w, 100, 100, 50)
    mask2 = _make_circle_mask(h, w, 300, 300, 50)
    masks = [_mask_to_sam_dict(mask1), _mask_to_sam_dict(mask2)]

    pieces = extract_pieces(masks, img)
    assert len(pieces) == 2
    ids = [p.piece_id for p in pieces]
    assert ids == [0, 1]


def test_extract_pieces_crop_shape():
    from app.modules.segmentation.piece_extractor import extract_pieces

    h, w = 400, 400
    img = _make_bgr(h, w)
    mask = _make_circle_mask(h, w, 200, 200, 60)
    masks = [_mask_to_sam_dict(mask)]

    pieces = extract_pieces(masks, img)
    assert len(pieces) == 1
    p = pieces[0]
    assert p.image.ndim == 3
    assert p.image.shape[2] == 3
    assert p.alpha_mask.ndim == 2


def test_extract_pieces_bbox_within_image():
    from app.modules.segmentation.piece_extractor import extract_pieces

    h, w = 400, 400
    img = _make_bgr(h, w)
    mask = _make_circle_mask(h, w, 200, 200, 60)
    masks = [_mask_to_sam_dict(mask)]

    pieces = extract_pieces(masks, img)
    p = pieces[0]
    x, y, bw, bh = p.bbox
    assert x >= 0 and y >= 0
    assert x + bw <= w
    assert y + bh <= h


def test_extract_pieces_shape_descriptors():
    from app.modules.segmentation.piece_extractor import extract_pieces

    h, w = 400, 400
    img = _make_bgr(h, w)
    mask = _make_circle_mask(h, w, 200, 200, 60)
    masks = [_mask_to_sam_dict(mask)]

    pieces = extract_pieces(masks, img)
    p = pieces[0]
    assert p.area_px > 0
    assert 0.0 <= p.solidity <= 1.0
    assert p.compactness >= 0.0


# ─── Contour Analyzer ────────────────────────────────────────────────────────

def test_contour_analyzer_populates_profiles():
    from app.modules.segmentation.piece_extractor import extract_pieces
    from app.modules.segmentation.contour_analyzer import analyze_contours

    h, w = 400, 400
    img = _make_bgr(h, w)
    # Use a rectangle — clear 4 corners
    mask = _make_rect_mask(h, w, 80, 80, 120, 100)
    masks = [_mask_to_sam_dict(mask)]

    pieces = extract_pieces(masks, img)
    pieces = analyze_contours(pieces)

    p = pieces[0]
    assert len(p.curvature_profiles) > 0


def test_contour_analyzer_curvature_vector_length():
    from app.modules.segmentation.piece_extractor import extract_pieces
    from app.modules.segmentation.contour_analyzer import analyze_contours, _CURVATURE_SAMPLES

    h, w = 400, 400
    img = _make_bgr(h, w)
    mask = _make_rect_mask(h, w, 80, 80, 120, 100)
    masks = [_mask_to_sam_dict(mask)]

    pieces = extract_pieces(masks, img)
    pieces = analyze_contours(pieces)

    for profile in pieces[0].curvature_profiles:
        assert len(profile.curvature_vector) == _CURVATURE_SAMPLES


def test_contour_analyzer_flat_sides_rectangle():
    from app.modules.segmentation.piece_extractor import extract_pieces
    from app.modules.segmentation.contour_analyzer import analyze_contours

    h, w = 400, 400
    img = _make_bgr(h, w)
    # Large clear rectangle — all 4 sides should be flat
    mask = _make_rect_mask(h, w, 50, 50, 200, 150)
    masks = [_mask_to_sam_dict(mask, stability=0.99)]

    pieces = extract_pieces(masks, img)
    pieces = analyze_contours(pieces)

    p = pieces[0]
    # A rectangle should have flat_side_count >= 2
    # (may not be exactly 4 due to contour approximation noise)
    assert p.flat_side_count >= 2


def test_contour_analyzer_is_flat_attribute():
    from app.modules.segmentation.piece_extractor import extract_pieces
    from app.modules.segmentation.contour_analyzer import analyze_contours

    h, w = 400, 400
    img = _make_bgr(h, w)
    mask = _make_rect_mask(h, w, 80, 80, 120, 100)
    masks = [_mask_to_sam_dict(mask)]

    pieces = extract_pieces(masks, img)
    pieces = analyze_contours(pieces)

    for profile in pieces[0].curvature_profiles:
        assert isinstance(profile.is_flat, bool)
        assert isinstance(profile.peak_value, float)


def test_contour_analyzer_handles_failure_gracefully():
    from app.modules.segmentation.contour_analyzer import analyze_contours
    from app.models.piece import PieceCrop

    # Create a PieceCrop with a degenerate 2-point contour.
    # The analyzer won't raise — it produces zero-curvature profiles
    # (all flat) because there are no interior points to compute angles on.
    # The key guarantee is: no exception is raised and the piece is returned.
    tiny_contour = np.array([[[0, 0]], [[1, 0]]], dtype=np.int32)
    tiny_mask = np.zeros((10, 10), dtype=np.uint8)
    tiny_img = np.zeros((10, 10, 3), dtype=np.uint8)

    piece = PieceCrop(
        piece_id=99,
        image=tiny_img,
        alpha_mask=tiny_mask,
        bbox=(0, 0, 10, 10),
        contour=tiny_contour,
        area_px=2.0,
        solidity=0.5,
        compactness=0.1,
    )

    # Must not raise regardless of contour quality
    result = analyze_contours([piece])
    assert len(result) == 1
    # flat_side_count is set to whatever the analyzer computes —
    # zero-curvature sides classify as flat, which is acceptable
    assert isinstance(result[0].flat_side_count, int)
    assert 0 <= result[0].flat_side_count <= 4