# Copyright (c) 2026 Harsh Dwivedi
# Licensed under the Harsh Non-Commercial Attribution License (HNCAL) v1.0
# Commercial use requires written permission. See LICENSE for details.

"""
Phase 8 — Rendering module tests.
Pure OpenCV / numpy — no GPU, models, or network required.
Uses temp directories so no real storage paths are needed.
"""

import json
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest

from app.models.piece import AssemblyStep, PieceCrop, PieceType, PieceMatch


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _make_bgr(h=200, w=300, color=(100, 120, 140)) -> np.ndarray:
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:] = color
    return img


def _make_piece(piece_id: int, h=60, w=60) -> PieceCrop:
    img = _make_bgr(h, w, color=(150, 180, 200))
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.rectangle(mask, (5, 5), (w-5, h-5), 255, -1)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = contours[0] if contours else np.array([[[0, 0]]], dtype=np.int32)
    return PieceCrop(
        piece_id=piece_id,
        image=img,
        alpha_mask=mask,
        bbox=(piece_id * 70, 10, w, h),
        contour=contour,
        area_px=float(mask.sum() // 255),
        solidity=0.95,
        compactness=0.85,
    )


def _make_step(
    step_num: int,
    piece_id: int,
    row: int,
    col: int,
    piece_type=PieceType.INTERIOR,
    confidence=0.8,
    rotation=0,
    flagged=False,
) -> AssemblyStep:
    s = AssemblyStep(
        step_num=step_num,
        piece_id=piece_id,
        grid_pos=(row, col),
        rotation_deg=rotation,
        piece_type=piece_type,
        composite_confidence=confidence,
        flagged=flagged,
    )
    s.adjacency_score = 0.75
    s.curvature_complement_score = 0.70
    return s


def _make_steps_3x3() -> list[AssemblyStep]:
    types = {
        (0,0): PieceType.CORNER, (0,1): PieceType.EDGE, (0,2): PieceType.CORNER,
        (1,0): PieceType.EDGE,   (1,1): PieceType.INTERIOR, (1,2): PieceType.EDGE,
        (2,0): PieceType.CORNER, (2,1): PieceType.EDGE, (2,2): PieceType.CORNER,
    }
    steps = []
    for i, (pos, pt) in enumerate(types.items()):
        steps.append(_make_step(i+1, i, pos[0], pos[1], piece_type=pt))
    return steps


# ─── Reference Overlay ───────────────────────────────────────────────────────

def test_reference_overlay_returns_same_shape():
    from app.modules.rendering.reference_overlay import render_reference_overlay

    ref = _make_bgr(300, 400)
    steps = _make_steps_3x3()

    with tempfile.TemporaryDirectory() as tmp:
        with __import__("unittest.mock", fromlist=["patch"]).patch(
            "app.modules.rendering.reference_overlay.overlay_reference_path",
            return_value=Path(tmp) / "ref.jpg"
        ):
            result = render_reference_overlay(ref, steps, (3, 3), "test_job")

    assert result.shape == ref.shape
    assert result.dtype == np.uint8


def test_reference_overlay_does_not_modify_original():
    from app.modules.rendering.reference_overlay import render_reference_overlay

    ref = _make_bgr(300, 400)
    original = ref.copy()
    steps = _make_steps_3x3()

    with tempfile.TemporaryDirectory() as tmp:
        with __import__("unittest.mock", fromlist=["patch"]).patch(
            "app.modules.rendering.reference_overlay.overlay_reference_path",
            return_value=Path(tmp) / "ref.jpg"
        ):
            render_reference_overlay(ref, steps, (3, 3), "test_job")

    np.testing.assert_array_equal(ref, original)


def test_reference_overlay_draws_on_image():
    from app.modules.rendering.reference_overlay import render_reference_overlay

    ref = _make_bgr(300, 400, color=(100, 100, 100))
    steps = _make_steps_3x3()

    with tempfile.TemporaryDirectory() as tmp:
        with __import__("unittest.mock", fromlist=["patch"]).patch(
            "app.modules.rendering.reference_overlay.overlay_reference_path",
            return_value=Path(tmp) / "ref.jpg"
        ):
            result = render_reference_overlay(ref, steps, (3, 3), "test_job")

    # Grid lines and text should change some pixels
    assert not np.array_equal(result, ref)


def test_reference_overlay_saves_file():
    from app.modules.rendering.reference_overlay import render_reference_overlay

    ref = _make_bgr(300, 400)
    steps = _make_steps_3x3()

    with tempfile.TemporaryDirectory() as tmp:
        out_path = Path(tmp) / "ref.jpg"
        with __import__("unittest.mock", fromlist=["patch"]).patch(
            "app.modules.rendering.reference_overlay.overlay_reference_path",
            return_value=out_path
        ):
            render_reference_overlay(ref, steps, (3, 3), "test_job")

        assert out_path.exists()
        assert out_path.stat().st_size > 0


def test_reference_overlay_empty_steps():
    from app.modules.rendering.reference_overlay import render_reference_overlay

    ref = _make_bgr(200, 200)
    with tempfile.TemporaryDirectory() as tmp:
        with __import__("unittest.mock", fromlist=["patch"]).patch(
            "app.modules.rendering.reference_overlay.overlay_reference_path",
            return_value=Path(tmp) / "ref.jpg"
        ):
            result = render_reference_overlay(ref, [], (3, 3), "test_job")
    assert result.shape == ref.shape


# ─── Pieces Overlay ──────────────────────────────────────────────────────────

def test_pieces_overlay_returns_same_shape():
    from app.modules.rendering.pieces_overlay import render_pieces_overlay

    pieces_img = _make_bgr(300, 600)
    pieces = [_make_piece(i) for i in range(4)]
    steps = [_make_step(i+1, i, i//2, i%2) for i in range(4)]

    with tempfile.TemporaryDirectory() as tmp:
        with __import__("unittest.mock", fromlist=["patch"]).patch(
            "app.modules.rendering.pieces_overlay.overlay_pieces_path",
            return_value=Path(tmp) / "pieces.jpg"
        ):
            result = render_pieces_overlay(pieces_img, pieces, steps, "test_job")

    assert result.shape == pieces_img.shape


def test_pieces_overlay_flagged_piece_annotated():
    from app.modules.rendering.pieces_overlay import render_pieces_overlay

    pieces_img = _make_bgr(200, 400, color=(50, 50, 50))
    pieces = [_make_piece(0)]
    steps = [_make_step(1, 0, 0, 0, flagged=True)]

    with tempfile.TemporaryDirectory() as tmp:
        with __import__("unittest.mock", fromlist=["patch"]).patch(
            "app.modules.rendering.pieces_overlay.overlay_pieces_path",
            return_value=Path(tmp) / "pieces.jpg"
        ):
            result = render_pieces_overlay(pieces_img, pieces, steps, "test_job")

    # Flagged piece should have orange pixels drawn on it
    assert not np.array_equal(result, pieces_img)


def test_pieces_overlay_saves_file():
    from app.modules.rendering.pieces_overlay import render_pieces_overlay

    pieces_img = _make_bgr(200, 400)
    pieces = [_make_piece(i) for i in range(2)]
    steps = [_make_step(i+1, i, 0, i) for i in range(2)]

    with tempfile.TemporaryDirectory() as tmp:
        out_path = Path(tmp) / "pieces.jpg"
        with __import__("unittest.mock", fromlist=["patch"]).patch(
            "app.modules.rendering.pieces_overlay.overlay_pieces_path",
            return_value=out_path
        ):
            render_pieces_overlay(pieces_img, pieces, steps, "test_job")

        assert out_path.exists()


# ─── Piece Crop Saver ────────────────────────────────────────────────────────

def test_save_piece_crops_creates_files():
    from app.modules.rendering.piece_crop_saver import save_piece_crops

    pieces = [_make_piece(i) for i in range(3)]
    steps = [_make_step(i+1, i, 0, i, rotation=i*90) for i in range(3)]

    with tempfile.TemporaryDirectory() as tmp:
        with __import__("unittest.mock", fromlist=["patch"]).patch(
            "app.modules.rendering.piece_crop_saver.outputs_dir",
            return_value=Path(tmp)
        ):
            urls = save_piece_crops(pieces, steps, "test_job")

    assert len(urls) == 3
    for pid, url in urls.items():
        assert f"piece_{pid:04d}.jpg" in url


def test_save_piece_crops_url_format():
    from app.modules.rendering.piece_crop_saver import save_piece_crops

    pieces = [_make_piece(0)]
    steps = [_make_step(1, 0, 0, 0)]

    with tempfile.TemporaryDirectory() as tmp:
        with __import__("unittest.mock", fromlist=["patch"]).patch(
            "app.modules.rendering.piece_crop_saver.outputs_dir",
            return_value=Path(tmp)
        ):
            urls = save_piece_crops(pieces, steps, "job123")

    assert urls[0].startswith("/assets/job123/")
    assert "piece_0000.jpg" in urls[0]


def test_save_piece_crops_all_rotations():
    from app.modules.rendering.piece_crop_saver import save_piece_crops

    pieces = [_make_piece(i) for i in range(4)]
    # Test all 4 rotations
    steps = [_make_step(i+1, i, 0, i, rotation=i*90) for i in range(4)]

    with tempfile.TemporaryDirectory() as tmp:
        with __import__("unittest.mock", fromlist=["patch"]).patch(
            "app.modules.rendering.piece_crop_saver.outputs_dir",
            return_value=Path(tmp)
        ):
            urls = save_piece_crops(pieces, steps, "test_job")

    assert len(urls) == 4


# ─── Step Card Generator ─────────────────────────────────────────────────────

def test_generate_step_cards_count():
    from app.modules.rendering.step_card_generator import generate_step_cards

    ref = _make_bgr(300, 400)
    pieces = [_make_piece(i) for i in range(4)]
    steps = [_make_step(i+1, i, i//2, i%2) for i in range(4)]

    with tempfile.TemporaryDirectory() as tmp:
        with __import__("unittest.mock", fromlist=["patch"]).patch(
            "app.modules.rendering.step_card_generator.step_cards_dir",
            return_value=Path(tmp)
        ), __import__("unittest.mock", fromlist=["patch"]).patch(
            "app.modules.rendering.step_card_generator.step_card_path",
            side_effect=lambda jid, n: Path(tmp) / f"step_{n:04d}.jpg"
        ):
            urls = generate_step_cards(steps, pieces, ref, (2, 2), "test_job")

    assert len(urls) == 4


def test_generate_step_cards_shape():
    from app.modules.rendering.step_card_generator import (
        generate_step_cards, _CARD_W, _CARD_H
    )

    ref = _make_bgr(300, 400)
    pieces = [_make_piece(0)]
    steps = [_make_step(1, 0, 0, 0)]

    with tempfile.TemporaryDirectory() as tmp:
        with __import__("unittest.mock", fromlist=["patch"]).patch(
            "app.modules.rendering.step_card_generator.step_cards_dir",
            return_value=Path(tmp)
        ), __import__("unittest.mock", fromlist=["patch"]).patch(
            "app.modules.rendering.step_card_generator.step_card_path",
            side_effect=lambda jid, n: Path(tmp) / f"step_{n:04d}.jpg"
        ):
            generate_step_cards(steps, pieces, ref, (2, 2), "test_job")

        # Load the saved card and check dimensions
        saved = cv2.imread(str(Path(tmp) / "step_0001.jpg"))
        assert saved is not None
        assert saved.shape == (_CARD_H, _CARD_W, 3)


def test_generate_step_cards_flagged_has_border():
    from app.modules.rendering.step_card_generator import generate_step_cards

    ref = _make_bgr(300, 400)
    pieces = [_make_piece(0)]
    normal_step = _make_step(1, 0, 0, 0, flagged=False)
    flagged_step = _make_step(1, 0, 0, 0, flagged=True)

    with tempfile.TemporaryDirectory() as tmp:
        out_normal = Path(tmp) / "normal.jpg"
        out_flagged = Path(tmp) / "flagged.jpg"

        with __import__("unittest.mock", fromlist=["patch"]).patch(
            "app.modules.rendering.step_card_generator.step_cards_dir",
            return_value=Path(tmp)
        ), __import__("unittest.mock", fromlist=["patch"]).patch(
            "app.modules.rendering.step_card_generator.step_card_path",
            return_value=out_normal
        ):
            generate_step_cards([normal_step], pieces, ref, (2, 2), "test_job")

        with __import__("unittest.mock", fromlist=["patch"]).patch(
            "app.modules.rendering.step_card_generator.step_cards_dir",
            return_value=Path(tmp)
        ), __import__("unittest.mock", fromlist=["patch"]).patch(
            "app.modules.rendering.step_card_generator.step_card_path",
            return_value=out_flagged
        ):
            generate_step_cards([flagged_step], pieces, ref, (2, 2), "test_job")

        normal = cv2.imread(str(out_normal))
        flagged = cv2.imread(str(out_flagged))

    # Flagged card should differ from normal (orange border drawn)
    assert normal is not None and flagged is not None
    assert not np.array_equal(normal, flagged)


def test_generate_step_cards_url_format():
    from app.modules.rendering.step_card_generator import generate_step_cards

    ref = _make_bgr(200, 300)
    pieces = [_make_piece(0)]
    steps = [_make_step(1, 0, 0, 0)]

    with tempfile.TemporaryDirectory() as tmp:
        with __import__("unittest.mock", fromlist=["patch"]).patch(
            "app.modules.rendering.step_card_generator.step_cards_dir",
            return_value=Path(tmp)
        ), __import__("unittest.mock", fromlist=["patch"]).patch(
            "app.modules.rendering.step_card_generator.step_card_path",
            side_effect=lambda jid, n: Path(tmp) / f"step_{n:04d}.jpg"
        ):
            urls = generate_step_cards(steps, pieces, ref, (2, 2), "job_abc")

    assert "step_0001.jpg" in urls[0]
    assert "job_abc" in urls[0]


# ─── Manifest Builder ────────────────────────────────────────────────────────

def test_build_manifest_structure():
    from app.modules.rendering.manifest_builder import build_manifest

    steps = _make_steps_3x3()
    crop_urls = {s.piece_id: f"/assets/j/pieces/piece_{s.piece_id:04d}.jpg" for s in steps}
    card_urls = [f"/assets/j/step_cards/step_{s.step_num:04d}.jpg" for s in steps]

    manifest = build_manifest("job_x", steps, (3, 3), crop_urls, card_urls)

    assert manifest.job_id == "job_x"
    assert manifest.grid_shape == (3, 3)
    assert manifest.total_pieces == 9
    assert len(manifest.steps) == 9
    assert manifest.corner_count == 4
    assert manifest.edge_count == 4
    assert manifest.interior_count == 1


def test_build_manifest_confidence_stats():
    from app.modules.rendering.manifest_builder import build_manifest

    steps = [
        _make_step(1, 0, 0, 0, confidence=0.9),
        _make_step(2, 1, 0, 1, confidence=0.6),
        _make_step(3, 2, 1, 0, confidence=0.3, flagged=True),
    ]
    manifest = build_manifest("j", steps, (2, 2), {}, [])

    assert abs(manifest.mean_confidence - 0.6) < 0.01
    assert abs(manifest.min_confidence - 0.3) < 0.01
    assert abs(manifest.max_confidence - 0.9) < 0.01
    assert manifest.flagged_count == 1


def test_write_manifest_creates_valid_json():
    from app.modules.rendering.manifest_builder import build_manifest, write_manifest

    steps = _make_steps_3x3()
    manifest = build_manifest("job_w", steps, (3, 3), {}, [])

    with tempfile.TemporaryDirectory() as tmp:
        with __import__("unittest.mock", fromlist=["patch"]).patch(
            "app.modules.rendering.manifest_builder.solution_manifest_path",
            return_value=Path(tmp) / "solution.json"
        ):
            url = write_manifest(manifest, "job_w")

        saved = json.loads((Path(tmp) / "solution.json").read_text())

    assert saved["job_id"] == "job_w"
    assert saved["total_pieces"] == 9
    assert len(saved["steps"]) == 9
    assert "solution.json" in url


def test_build_output_bundle():
    from app.modules.rendering.manifest_builder import (
        build_manifest, build_output_bundle
    )

    steps = _make_steps_3x3()
    manifest = build_manifest("job_b", steps, (3, 3), {}, [])
    bundle = build_output_bundle("job_b", manifest, [])

    assert "overlay_reference" in bundle.overlay_reference_url
    assert "overlay_pieces" in bundle.overlay_pieces_url
    assert "solution.json" in bundle.solution_manifest_url
    assert bundle.total_pieces == 9


def test_manifest_step_entry_has_urls():
    from app.modules.rendering.manifest_builder import build_manifest

    steps = [_make_step(1, 0, 0, 0)]
    crop_urls = {0: "/assets/j/pieces/piece_0000.jpg"}
    card_urls = ["/assets/j/step_cards/step_0001.jpg"]

    manifest = build_manifest("j", steps, (1, 1), crop_urls, card_urls)
    entry = manifest.steps[0]

    assert entry.piece_crop_url == crop_urls[0]
    assert entry.step_card_url == card_urls[0]