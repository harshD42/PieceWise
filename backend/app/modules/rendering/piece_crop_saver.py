# Copyright (c) 2026 Harsh Dwivedi
# Licensed under the Harsh Non-Commercial Attribution License (HNCAL) v1.0
# Commercial use requires written permission. See LICENSE for details.

"""
PieceWise — Piece Crop Saver
Saves individual piece crop images to disk so they can be referenced
in the solution manifest and served as assets by the API.

Each crop is saved with the winning rotation applied so the frontend
shows the piece in the orientation the user should physically hold it.

Output: {job_id}/outputs/pieces/piece_{piece_id:04d}.jpg
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from app.models.piece import AssemblyStep, PieceCrop
from app.utils.geometry_utils import rotate_image_90, rotation_deg_to_k
from app.utils.image_utils import paste_on_background
from app.utils.logger import get_logger
from app.utils.storage import outputs_dir

log = get_logger(__name__)

_CROP_SIZE = 120   # px — thumbnail size for step cards and manifest


def _pieces_dir(job_id: str) -> Path:
    return outputs_dir(job_id) / "pieces"


def piece_crop_url(job_id: str, piece_id: int) -> str:
    return f"/assets/{job_id}/pieces/piece_{piece_id:04d}.jpg"


def save_piece_crops(
    pieces: list[PieceCrop],
    steps: list[AssemblyStep],
    job_id: str,
) -> dict[int, str]:
    """
    Save each piece crop with its winning rotation applied.

    Args:
        pieces: PieceCrop list from Phase 3
        steps:  AssemblyStep list from Phase 7 (carries rotation_deg)
        job_id: Job ID for output path

    Returns:
        Dict mapping piece_id → asset URL string.
    """
    out_dir = _pieces_dir(job_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    pid_to_step: dict[int, AssemblyStep] = {s.piece_id: s for s in steps}
    urls: dict[int, str] = {}

    for piece in pieces:
        step = pid_to_step.get(piece.piece_id)
        rot_deg = step.rotation_deg if step else 0

        # Apply rotation
        k = rotation_deg_to_k(rot_deg)
        if k > 0:
            rotated_img = rotate_image_90(piece.image, k=k)
            rotated_mask = rotate_image_90(piece.alpha_mask[:, :, np.newaxis], k=k)[:, :, 0]
        else:
            rotated_img = piece.image
            rotated_mask = piece.alpha_mask

        # Paste on grey background at thumbnail size
        thumb = paste_on_background(rotated_img, rotated_mask, size=_CROP_SIZE)

        fname = f"piece_{piece.piece_id:04d}.jpg"
        fpath = out_dir / fname
        cv2.imwrite(str(fpath), thumb, [cv2.IMWRITE_JPEG_QUALITY, 90])

        urls[piece.piece_id] = piece_crop_url(job_id, piece.piece_id)

    log.info("piece_crops_saved", count=len(urls), dir=str(out_dir))
    return urls