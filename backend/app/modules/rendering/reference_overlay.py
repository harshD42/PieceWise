# Copyright (c) 2026 Harsh Dwivedi
# Licensed under the Harsh Non-Commercial Attribution License (HNCAL) v1.0
# Commercial use requires written permission. See LICENSE for details.

"""
PieceWise — Reference Image Overlay Renderer
Draws an annotated grid over the reference image showing:
  - Grid lines dividing the puzzle into NxM cells
  - Step number (large, bold) in each cell
  - Piece ID (small) in each cell
  - Confidence dot (green / yellow / red) indicating match quality

Confidence colour coding:
  Green  (≥ 0.75): high confidence — place with certainty
  Yellow (≥ 0.55): medium confidence — double-check placement
  Red    (< 0.55): low confidence — flagged for human review

Output: overlay_reference.jpg saved to {job_id}/outputs/
"""

from __future__ import annotations

import cv2
import numpy as np

from app.models.piece import AssemblyStep, PieceType
from app.utils.logger import get_logger
from app.utils.storage import overlay_reference_path

log = get_logger(__name__)

# Confidence colour thresholds (BGR)
_COLOUR_GREEN  = (34, 197, 94)    # green
_COLOUR_YELLOW = (59, 186, 234)   # amber/yellow
_COLOUR_RED    = (60, 60, 220)    # red

_CONF_HIGH = 0.75
_CONF_MED  = 0.55

# Visual constants
_GRID_COLOUR    = (80, 80, 80)     # dark grey grid lines
_GRID_THICKNESS = 1
_TEXT_COLOUR    = (255, 255, 255)  # white text
_SHADOW_COLOUR  = (0, 0, 0)        # black text shadow
_DOT_RADIUS     = 6
_FONT            = cv2.FONT_HERSHEY_SIMPLEX


def _confidence_colour(confidence: float) -> tuple[int, int, int]:
    if confidence >= _CONF_HIGH:
        return _COLOUR_GREEN
    if confidence >= _CONF_MED:
        return _COLOUR_YELLOW
    return _COLOUR_RED


def _draw_text_with_shadow(
    img: np.ndarray,
    text: str,
    pos: tuple[int, int],
    font_scale: float,
    thickness: int,
    colour: tuple,
) -> None:
    """Draw text with a 1px black shadow for readability on any background."""
    x, y = pos
    # Shadow
    cv2.putText(img, text, (x + 1, y + 1), _FONT,
                font_scale, _SHADOW_COLOUR, thickness + 1, cv2.LINE_AA)
    # Foreground
    cv2.putText(img, text, (x, y), _FONT,
                font_scale, colour, thickness, cv2.LINE_AA)


def render_reference_overlay(
    ref_img: np.ndarray,
    steps: list[AssemblyStep],
    grid_shape: tuple[int, int],
    job_id: str,
) -> np.ndarray:
    """
    Draw annotated grid overlay on the reference image.

    Args:
        ref_img:    BGR uint8 normalised reference image
        steps:      Ordered AssemblyStep list from Phase 7
        grid_shape: (n_rows, n_cols)
        job_id:     Job ID for output path

    Returns:
        Annotated BGR uint8 image (copy — original unchanged).
    """
    overlay = ref_img.copy()
    h, w = overlay.shape[:2]
    n_rows, n_cols = grid_shape

    cell_h = h / n_rows
    cell_w = w / n_cols

    # Build step lookup: grid_pos → AssemblyStep
    pos_to_step: dict[tuple[int, int], AssemblyStep] = {
        s.grid_pos: s for s in steps
    }

    # Draw grid lines
    for r in range(n_rows + 1):
        y = int(r * cell_h)
        cv2.line(overlay, (0, y), (w, y), _GRID_COLOUR, _GRID_THICKNESS)
    for c in range(n_cols + 1):
        x = int(c * cell_w)
        cv2.line(overlay, (x, 0), (x, h), _GRID_COLOUR, _GRID_THICKNESS)

    # Annotate each cell
    for r in range(n_rows):
        for c in range(n_cols):
            step = pos_to_step.get((r, c))
            if step is None:
                continue

            cx = int(c * cell_w + cell_w / 2)
            cy = int(r * cell_h + cell_h / 2)

            # Confidence dot (top-right of cell)
            dot_x = int((c + 1) * cell_w) - _DOT_RADIUS - 3
            dot_y = int(r * cell_h) + _DOT_RADIUS + 3
            dot_colour = _confidence_colour(step.composite_confidence)
            cv2.circle(overlay, (dot_x, dot_y), _DOT_RADIUS, dot_colour, -1)

            # Step number — scale font to cell size
            font_scale = max(0.3, min(0.7, cell_w / 120))
            step_text = str(step.step_num)
            (tw, th), _ = cv2.getTextSize(step_text, _FONT, font_scale, 2)
            _draw_text_with_shadow(
                overlay, step_text,
                (cx - tw // 2, cy + th // 2),
                font_scale, 2, _TEXT_COLOUR,
            )

            # Piece ID — smaller, below step number
            id_scale = max(0.2, font_scale * 0.55)
            id_text = f"#{step.piece_id}"
            (iw, ih), _ = cv2.getTextSize(id_text, _FONT, id_scale, 1)
            _draw_text_with_shadow(
                overlay, id_text,
                (cx - iw // 2, cy + th // 2 + ih + 2),
                id_scale, 1, (200, 200, 200),
            )

    # Save
    out_path = overlay_reference_path(job_id)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), overlay, [cv2.IMWRITE_JPEG_QUALITY, 92])

    log.info(
        "reference_overlay_saved",
        path=str(out_path),
        grid_shape=grid_shape,
    )

    return overlay