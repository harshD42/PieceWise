# Copyright (c) 2026 Harsh Dwivedi
# Licensed under the Harsh Non-Commercial Attribution License (HNCAL) v1.0
# Commercial use requires written permission. See LICENSE for details.

"""
PieceWise — Rendering Module
Public API for the output rendering stage.
"""

from app.modules.rendering.manifest_builder import (
    build_manifest,
    build_output_bundle,
    write_manifest,
)
from app.modules.rendering.piece_crop_saver import save_piece_crops
from app.modules.rendering.pieces_overlay import render_pieces_overlay
from app.modules.rendering.reference_overlay import render_reference_overlay
from app.modules.rendering.step_card_generator import generate_step_cards

__all__ = [
    "render_reference_overlay",
    "render_pieces_overlay",
    "save_piece_crops",
    "generate_step_cards",
    "build_manifest",
    "write_manifest",
    "build_output_bundle",
]