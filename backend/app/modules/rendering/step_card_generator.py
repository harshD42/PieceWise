# Copyright (c) 2026 Harsh Dwivedi
# Licensed under the Harsh Non-Commercial Attribution License (HNCAL) v1.0
# Commercial use requires written permission. See LICENSE for details.

"""
PieceWise — Step Card Generator
Generates a 400×200px side-by-side card for each assembly step:

  Left half  (200×200): piece crop with rotation applied, on grey background
  Right half (200×200): reference image with target cell highlighted in yellow
  Bottom bar (400×24):  "Step N — Row X, Col Y — Rotate Z° — [type] — conf: 0.87"
                        Flagged pieces: orange border + "⚠ Uncertain" label

Output: {job_id}/outputs/step_cards/step_{step_num:04d}.jpg
"""

from __future__ import annotations

import cv2
import numpy as np

from app.models.piece import AssemblyStep, PieceCrop
from app.utils.geometry_utils import rotate_image_90, rotation_deg_to_k
from app.utils.image_utils import paste_on_background
from app.utils.logger import get_logger
from app.utils.storage import step_card_path, step_cards_dir

log = get_logger(__name__)

# Card dimensions
_CARD_W = 400
_CARD_H = 224    # 200 content + 24 label bar
_HALF_W = 200
_CONTENT_H = 200
_BAR_H = 24

# Visual constants
_BG_GREY       = (45, 45, 45)
_BAR_BG        = (30, 30, 30)
_TEXT_WHITE    = (240, 240, 240)
_TEXT_GREY     = (160, 160, 160)
_HIGHLIGHT_COL = (0, 215, 255)    # yellow (BGR)
_FLAGGED_COL   = (30, 100, 240)   # orange (BGR)
_FONT = cv2.FONT_HERSHEY_SIMPLEX


def _highlight_cell(
    ref_img: np.ndarray,
    grid_pos: tuple[int, int],
    grid_shape: tuple[int, int],
    target_size: int,
) -> np.ndarray:
    """
    Resize reference image to target_size×target_size and draw a
    yellow rectangle over the target grid cell.
    """
    thumb = cv2.resize(ref_img, (target_size, target_size), interpolation=cv2.INTER_AREA)
    n_rows, n_cols = grid_shape
    r, c = grid_pos

    cell_h = target_size / n_rows
    cell_w = target_size / n_cols
    x1 = int(c * cell_w)
    y1 = int(r * cell_h)
    x2 = int((c + 1) * cell_w)
    y2 = int((r + 1) * cell_h)

    # Semi-transparent yellow fill
    overlay = thumb.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), _HIGHLIGHT_COL, -1)
    cv2.addWeighted(overlay, 0.35, thumb, 0.65, 0, thumb)
    # Solid border
    cv2.rectangle(thumb, (x1, y1), (x2, y2), _HIGHLIGHT_COL, 2)
    return thumb


def _make_card(
    step: AssemblyStep,
    piece: PieceCrop | None,
    ref_img: np.ndarray,
    grid_shape: tuple[int, int],
) -> np.ndarray:
    """Render a single 400×224px step card."""
    card = np.full((_CARD_H, _CARD_W, 3), _BG_GREY, dtype=np.uint8)

    # ── Left: piece crop ─────────────────────────────────────────────────────
    if piece is not None:
        k = rotation_deg_to_k(step.rotation_deg)
        rot_img = rotate_image_90(piece.image, k=k) if k > 0 else piece.image
        rot_mask = (
            rotate_image_90(piece.alpha_mask[:, :, np.newaxis], k=k)[:, :, 0]
            if k > 0 else piece.alpha_mask
        )
        thumb = paste_on_background(rot_img, rot_mask, size=_CONTENT_H)
        card[:_CONTENT_H, :_HALF_W] = thumb

    # ── Right: reference with highlighted cell ────────────────────────────────
    ref_thumb = _highlight_cell(ref_img, step.grid_pos, grid_shape, _CONTENT_H)
    card[:_CONTENT_H, _HALF_W:_CARD_W] = ref_thumb

    # ── Divider line ─────────────────────────────────────────────────────────
    cv2.line(card, (_HALF_W, 0), (_HALF_W, _CONTENT_H), (60, 60, 60), 1)

    # ── Bottom label bar ─────────────────────────────────────────────────────
    card[_CONTENT_H:, :] = _BAR_BG

    label = (
        f"Step {step.step_num}  "
        f"({step.grid_pos[0]},{step.grid_pos[1]})  "
        f"R{step.rotation_deg}deg  "
        f"{step.piece_type.value}  "
        f"conf:{step.composite_confidence:.2f}"
    )
    cv2.putText(
        card, label,
        (6, _CONTENT_H + 16),
        _FONT, 0.30, _TEXT_GREY, 1, cv2.LINE_AA,
    )

    # ── Flagged border + label ────────────────────────────────────────────────
    if step.flagged:
        cv2.rectangle(card, (0, 0), (_CARD_W - 1, _CARD_H - 1), _FLAGGED_COL, 2)
        cv2.putText(
            card, "? Uncertain",
            (_CARD_W - 82, _CONTENT_H + 16),
            _FONT, 0.30, _FLAGGED_COL, 1, cv2.LINE_AA,
        )

    return card


def generate_step_cards(
    steps: list[AssemblyStep],
    pieces: list[PieceCrop],
    ref_img: np.ndarray,
    grid_shape: tuple[int, int],
    job_id: str,
) -> list[str]:
    """
    Generate a step card image for every assembly step.

    Args:
        steps:      Ordered AssemblyStep list from Phase 7
        pieces:     PieceCrop list from Phase 3
        ref_img:    BGR uint8 normalised reference image
        grid_shape: (n_rows, n_cols)
        job_id:     Job ID for output paths

    Returns:
        List of asset URL strings, one per step, in step order.
    """
    out_dir = step_cards_dir(job_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    pid_to_piece: dict[int, PieceCrop] = {p.piece_id: p for p in pieces}
    urls: list[str] = []

    for step in steps:
        piece = pid_to_piece.get(step.piece_id)
        card = _make_card(step, piece, ref_img, grid_shape)

        card_path = step_card_path(job_id, step.step_num)
        cv2.imwrite(str(card_path), card, [cv2.IMWRITE_JPEG_QUALITY, 90])
        urls.append(f"/assets/{job_id}/step_cards/step_{step.step_num:04d}.jpg")

    log.info("step_cards_generated", count=len(urls), dir=str(out_dir))
    return urls