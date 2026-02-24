# Copyright (c) 2026 Harsh Dwivedi
# Licensed under the Harsh Non-Commercial Attribution License (HNCAL) v1.0
# Commercial use requires written permission. See LICENSE for details.

"""
PieceWise — Scattered Pieces Image Overlay Renderer
Draws annotated bounding boxes over each detected piece in the
scattered pieces photo, showing:
  - Bounding box with colour coded by piece type
  - Step number (large) inside the box
  - Rotation arrow indicator
  - Red/orange border + warning label for flagged pieces

Piece type colour coding (BGR):
  Green  — CORNER piece
  Blue   — EDGE piece
  White  — INTERIOR piece
  Orange — flagged (low confidence, any type)

Output: overlay_pieces.jpg saved to {job_id}/outputs/
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from app.models.piece import AssemblyStep, PieceCrop, PieceType
from app.utils.logger import get_logger
from app.utils.storage import overlay_pieces_path

log = get_logger(__name__)

# Bbox border colours by piece type (BGR)
_TYPE_COLOURS = {
    PieceType.CORNER:   (34, 197, 94),    # green
    PieceType.EDGE:     (220, 100, 30),   # blue
    PieceType.INTERIOR: (200, 200, 200),  # white/grey
    PieceType.UNKNOWN:  (150, 150, 150),  # grey
}
_FLAGGED_COLOUR = (30, 100, 240)   # orange-red
_BOX_THICKNESS  = 2
_FONT = cv2.FONT_HERSHEY_SIMPLEX
_TEXT_COLOUR = (255, 255, 255)
_SHADOW_COLOUR = (0, 0, 0)


def _draw_rotation_arrow(
    img: np.ndarray,
    centre: tuple[int, int],
    rotation_deg: int,
    radius: int,
    colour: tuple,
) -> None:
    """Draw a small arrow indicating piece rotation direction."""
    cx, cy = centre
    # Arrow points from centre outward in the rotation direction
    angle_rad = math.radians(-rotation_deg)  # negative = clockwise in image coords
    end_x = int(cx + radius * math.cos(angle_rad))
    end_y = int(cy + radius * math.sin(angle_rad))
    cv2.arrowedLine(img, (cx, cy), (end_x, end_y),
                    colour, 1, tipLength=0.4)


def _draw_text_shadow(
    img: np.ndarray, text: str, pos: tuple[int, int],
    scale: float, thickness: int, colour: tuple,
) -> None:
    x, y = pos
    cv2.putText(img, text, (x+1, y+1), _FONT, scale, _SHADOW_COLOUR, thickness+1, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), _FONT, scale, colour, thickness, cv2.LINE_AA)


def render_pieces_overlay(
    pieces_img: np.ndarray,
    pieces: list[PieceCrop],
    steps: list[AssemblyStep],
    job_id: str,
) -> np.ndarray:
    """
    Draw annotated bounding boxes on the scattered pieces image.

    Args:
        pieces_img: BGR uint8 normalised pieces image
        pieces:     PieceCrop list from Phase 3 (carry bbox)
        steps:      AssemblyStep list from Phase 7 (carry step_num, type, etc.)
        job_id:     Job ID for output path

    Returns:
        Annotated BGR uint8 image.
    """
    overlay = pieces_img.copy()

    # Build piece_id → step lookup
    pid_to_step: dict[int, AssemblyStep] = {s.piece_id: s for s in steps}
    pid_to_piece: dict[int, PieceCrop] = {p.piece_id: p for p in pieces}

    for step in steps:
        piece = pid_to_piece.get(step.piece_id)
        if piece is None:
            continue

        x, y, bw, bh = piece.bbox
        cx = x + bw // 2
        cy = y + bh // 2

        # Choose border colour
        if step.flagged:
            box_colour = _FLAGGED_COLOUR
        else:
            box_colour = _TYPE_COLOURS.get(step.piece_type, _TYPE_COLOURS[PieceType.UNKNOWN])

        # Draw bounding box
        cv2.rectangle(overlay, (x, y), (x + bw, y + bh), box_colour, _BOX_THICKNESS)

        # Step number inside box
        font_scale = max(0.3, min(0.7, bw / 100))
        step_text = str(step.step_num)
        (tw, th), _ = cv2.getTextSize(step_text, _FONT, font_scale, 2)
        _draw_text_shadow(
            overlay, step_text,
            (cx - tw // 2, cy + th // 2),
            font_scale, 2, _TEXT_COLOUR,
        )

        # Rotation arrow (small, top-left corner of box)
        arrow_cx = x + 10
        arrow_cy = y + 10
        _draw_rotation_arrow(overlay, (arrow_cx, arrow_cy),
                              step.rotation_deg, 7, box_colour)

        # Flagged warning label
        if step.flagged:
            warn_scale = max(0.2, font_scale * 0.5)
            _draw_text_shadow(
                overlay, "?",
                (x + bw - 12, y + 12),
                warn_scale, 1, _FLAGGED_COLOUR,
            )

    # Save
    out_path = overlay_pieces_path(job_id)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), overlay, [cv2.IMWRITE_JPEG_QUALITY, 92])

    log.info("pieces_overlay_saved", path=str(out_path))
    return overlay