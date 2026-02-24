# Copyright (c) 2026 Harsh Dwivedi
# Licensed under the Harsh Non-Commercial Attribution License (HNCAL) v1.0
# Commercial use requires written permission. See LICENSE for details.

"""
PieceWise — Solution Manifest Builder
Builds and writes solution.json — the complete output artifact consumed
by the frontend solution viewer.

The manifest contains:
  - Job metadata (grid shape, piece counts, confidence stats)
  - Per-step entries with all scores + asset URLs
  - Top-3 candidate cells for flagged pieces (human-in-the-loop UI)
  - Named asset URLs (overlays, step cards, manifest itself)
"""

from __future__ import annotations

import json

import numpy as np

from app.models.job import OutputBundle
from app.models.output import PieceCropManifestEntry, SolutionManifest, StepManifestEntry
from app.models.piece import AssemblyStep, PieceType
from app.utils.logger import get_logger
from app.utils.storage import (
    get_asset_url,
    get_step_card_url,
    overlay_pieces_path,
    overlay_reference_path,
    solution_manifest_path,
)

log = get_logger(__name__)


def build_manifest(
    job_id: str,
    steps: list[AssemblyStep],
    grid_shape: tuple[int, int],
    crop_urls: dict[int, str],
    step_card_urls: list[str],
) -> SolutionManifest:
    """
    Assemble the SolutionManifest from all rendering outputs.

    Args:
        job_id:         Job ID
        steps:          Ordered AssemblyStep list
        grid_shape:     (n_rows, n_cols)
        crop_urls:      piece_id → crop asset URL
        step_card_urls: List of step card URLs in step order

    Returns:
        Populated SolutionManifest (not yet written to disk).
    """
    confidences = [s.composite_confidence for s in steps]
    mean_conf = float(np.mean(confidences)) if confidences else 0.0
    min_conf  = float(np.min(confidences))  if confidences else 0.0
    max_conf  = float(np.max(confidences))  if confidences else 0.0

    flagged_count  = sum(1 for s in steps if s.flagged)
    corner_count   = sum(1 for s in steps if s.piece_type == PieceType.CORNER)
    edge_count     = sum(1 for s in steps if s.piece_type == PieceType.EDGE)
    interior_count = sum(1 for s in steps if s.piece_type == PieceType.INTERIOR)

    # Build per-step manifest entries
    step_entries: list[StepManifestEntry] = []
    for i, step in enumerate(steps):
        crop_url = crop_urls.get(step.piece_id, "")
        card_url = step_card_urls[i] if i < len(step_card_urls) else ""

        entry = StepManifestEntry(
            step_num=step.step_num,
            piece_id=step.piece_id,
            grid_pos=step.grid_pos,
            rotation_deg=step.rotation_deg,
            piece_type=step.piece_type,
            composite_confidence=step.composite_confidence,
            adjacency_score=step.adjacency_score,
            curvature_complement_score=step.curvature_complement_score,
            flagged=step.flagged,
            piece_crop_url=crop_url,
            step_card_url=card_url,
            top3_candidates=[],   # populated from PieceMatch.top3_candidates in renderer
        )
        step_entries.append(entry)

    # Asset URL map
    asset_urls = {
        "overlay_reference": get_asset_url(job_id, "overlay_reference.jpg"),
        "overlay_pieces":    get_asset_url(job_id, "overlay_pieces.jpg"),
        "solution_manifest": get_asset_url(job_id, "solution.json"),
        "step_cards":        step_card_urls,
    }

    manifest = SolutionManifest(
        job_id=job_id,
        grid_shape=grid_shape,
        total_pieces=len(steps),
        flagged_count=flagged_count,
        mean_confidence=round(mean_conf, 4),
        min_confidence=round(min_conf, 4),
        max_confidence=round(max_conf, 4),
        steps=step_entries,
        asset_urls=asset_urls,
        corner_count=corner_count,
        edge_count=edge_count,
        interior_count=interior_count,
    )

    return manifest


def write_manifest(manifest: SolutionManifest, job_id: str) -> str:
    """
    Serialise the manifest to solution.json and write to disk.

    Returns:
        Asset URL string for the written manifest file.
    """
    out_path = solution_manifest_path(job_id)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Pydantic v2 serialisation → dict → JSON
    manifest_dict = manifest.model_dump(mode="json")
    out_path.write_text(
        json.dumps(manifest_dict, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    log.info(
        "manifest_written",
        path=str(out_path),
        total_pieces=manifest.total_pieces,
        flagged=manifest.flagged_count,
    )

    return get_asset_url(job_id, "solution.json")


def build_output_bundle(
    job_id: str,
    manifest: SolutionManifest,
    step_card_urls: list[str],
) -> OutputBundle:
    """
    Build the OutputBundle stored on the Job record on completion.
    This is what GET /status returns in the result field.
    """
    return OutputBundle(
        overlay_reference_url=get_asset_url(job_id, "overlay_reference.jpg"),
        overlay_pieces_url=get_asset_url(job_id, "overlay_pieces.jpg"),
        solution_manifest_url=get_asset_url(job_id, "solution.json"),
        step_card_urls=step_card_urls,
        total_pieces=manifest.total_pieces,
        flagged_count=manifest.flagged_count,
        mean_confidence=manifest.mean_confidence,
    )