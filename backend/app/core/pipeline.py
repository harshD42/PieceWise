# Copyright (c) 2026 Harsh Dwivedi
# Licensed under the Harsh Non-Commercial Attribution License (HNCAL) v1.0
# Commercial use requires written permission. See LICENSE for details.

"""
PieceWise — Pipeline Orchestrator
Wires all modules in the correct dependency order.
Manages async parallel execution of independent branches,
progress reporting, and structured error handling per stage.

Execution order (from sequence diagram):
  1. Preprocessing
  2. PARALLEL: reference embedding + piece segmentation
  3. Piece embedding (depends on segmentation output)
  4. Matching (coarse-to-fine + Hungarian)
  5. Adjacency refinement
  6. Sequencing
  7. Rendering
"""

from __future__ import annotations

import asyncio
import traceback
from pathlib import Path

import structlog

from app.core.job_store import JobStore
from app.models.job import JobStage, JobStatus, OutputBundle
from app.utils.logger import get_logger
from app.utils.storage import (
    init_job_dirs,
    outputs_dir,
    pieces_upload_path,
    reference_upload_path,
)

log = get_logger(__name__)


async def run_pipeline(job_id: str, store: JobStore) -> None:
    """
    Main pipeline coroutine. Runs as a FastAPI background task.
    All progress updates flow through store.advance_stage().
    On any unhandled exception the job is marked FAILED with the error message.
    """
    structlog.contextvars.bind_contextvars(job_id=job_id)

    try:
        await _run(job_id, store)
    except Exception as exc:
        err_msg = f"{type(exc).__name__}: {exc}"
        log.error(
            "pipeline_fatal_error",
            job_id=job_id,
            error=err_msg,
            traceback=traceback.format_exc(),
        )
        store.fail_job(job_id, err_msg)
    finally:
        structlog.contextvars.clear_contextvars()


async def _run(job_id: str, store: JobStore) -> None:
    ref_path = reference_upload_path(job_id)
    pieces_path = pieces_upload_path(job_id)

    # ── Stage 1: Preprocessing ───────────────────────────────────────────────
    store.advance_stage(job_id, JobStage.PREPROCESSING)
    log.info("stage_start", stage="preprocessing")

    ref_img, pieces_img = await asyncio.to_thread(
        _preprocess, job_id, ref_path, pieces_path
    )
    log.info("stage_complete", stage="preprocessing")

    # ── Stage 2: PARALLEL — segmentation + reference embedding ───────────────
    store.advance_stage(job_id, JobStage.SEGMENTATION)
    log.info("stage_start", stage="parallel_seg_ref_embed")

    piece_crops, patch_token_map = await asyncio.gather(
        asyncio.to_thread(_segment_pieces, job_id, pieces_img),
        asyncio.to_thread(_embed_reference, job_id, ref_img),
    )

    log.info(
        "stage_complete",
        stage="parallel_seg_ref_embed",
        piece_count=len(piece_crops),
    )

    # ── Stage 3: Piece Embedding ─────────────────────────────────────────────
    store.advance_stage(job_id, JobStage.FEATURE_EXTRACTION)
    log.info("stage_start", stage="piece_embedding")

    piece_embeddings = await asyncio.to_thread(
        _embed_pieces, job_id, piece_crops
    )
    log.info("stage_complete", stage="piece_embedding")

    # ── Stage 4: Matching ────────────────────────────────────────────────────
    store.advance_stage(job_id, JobStage.MATCHING)
    log.info("stage_start", stage="matching")

    piece_matches = await asyncio.to_thread(
        _match_pieces, job_id, piece_embeddings, patch_token_map, piece_crops
    )
    log.info(
        "stage_complete",
        stage="matching",
        flagged=sum(1 for m in piece_matches if m.flagged),
    )

    # ── Stage 5: Adjacency Refinement ────────────────────────────────────────
    store.advance_stage(job_id, JobStage.ADJACENCY_REFINEMENT)
    log.info("stage_start", stage="adjacency_refinement")

    piece_matches = await asyncio.to_thread(
        _refine_adjacency, job_id, piece_matches, piece_crops
    )
    log.info("stage_complete", stage="adjacency_refinement")

    # ── Stage 6: Sequencing ──────────────────────────────────────────────────
    store.advance_stage(job_id, JobStage.SEQUENCING)
    log.info("stage_start", stage="sequencing")

    assembly_steps = await asyncio.to_thread(
        _sequence, job_id, piece_matches, patch_token_map.grid_shape
    )
    log.info("stage_complete", stage="sequencing", steps=len(assembly_steps))

    # ── Stage 7: Rendering ───────────────────────────────────────────────────
    store.advance_stage(job_id, JobStage.RENDERING)
    log.info("stage_start", stage="rendering")

    output_bundle = await asyncio.to_thread(
        _render,
        job_id,
        ref_img,
        pieces_img,
        piece_crops,
        piece_matches,
        assembly_steps,
    )
    log.info("stage_complete", stage="rendering")

    # ── Done ─────────────────────────────────────────────────────────────────
    store.advance_stage(job_id, JobStage.DONE)
    store.update_job(job_id, result=output_bundle)
    log.info("pipeline_complete", job_id=job_id)


# ─── Stage Runner Stubs ───────────────────────────────────────────────────────
# Each stub will be replaced by real module calls as phases are implemented.
# They raise NotImplementedError with a clear message so partial pipeline
# runs fail loudly rather than silently producing incorrect results.

def _preprocess(job_id: str, ref_path: Path, pieces_path: Path):
    """Phase 2 — full preprocessing: validate + normalize + histogram match."""
    from app.modules.preprocessing.validator import validate_image_pair
    from app.modules.preprocessing.normalizer import normalize_image_pair

    validate_image_pair(ref_path, pieces_path)
    normalised = normalize_image_pair(ref_path, pieces_path)
    return normalised.ref_img, normalised.pieces_img


def _segment_pieces(job_id: str, pieces_img):
    """Phase 3 — segmentation module."""
    raise NotImplementedError(
        "Segmentation module not yet implemented (Phase 3). "
        "Run download_models.py and implement Phase 3 before executing the pipeline."
    )


def _embed_reference(job_id: str, ref_img):
    """Phase 4 — DINOv2 reference spatial token extraction."""
    raise NotImplementedError(
        "Feature extraction module not yet implemented (Phase 4)."
    )


def _embed_pieces(job_id: str, piece_crops, ):
    """Phase 4 — DINOv2 piece embedding."""
    raise NotImplementedError(
        "Feature extraction module not yet implemented (Phase 4)."
    )


def _match_pieces(job_id: str, piece_embeddings, patch_token_map, piece_crops):
    """Phase 5 — matching engine."""
    raise NotImplementedError(
        "Matching engine not yet implemented (Phase 5)."
    )


def _refine_adjacency(job_id: str, piece_matches, piece_crops):
    """Phase 6 — adjacency refiner."""
    raise NotImplementedError(
        "Adjacency refiner not yet implemented (Phase 6)."
    )


def _sequence(job_id: str, piece_matches, grid_shape):
    """Phase 7 — sequencing module."""
    raise NotImplementedError(
        "Sequencing module not yet implemented (Phase 7)."
    )


def _render(job_id, ref_img, pieces_img, piece_crops, piece_matches, assembly_steps):
    """Phase 8 — rendering module."""
    raise NotImplementedError(
        "Rendering module not yet implemented (Phase 8)."
    )


# ─── Correction Re-run ───────────────────────────────────────────────────────

async def rerun_from_correction(
    job_id: str,
    piece_id: int,
    corrected_grid_pos: tuple[int, int],
    store: JobStore,
) -> None:
    """
    Re-run sequencing and rendering after a human-in-the-loop correction.
    Loads existing match results from cache, applies the correction,
    then re-runs stages 6 (adjacency), 7 (sequencing), and 8 (rendering).
    Full segmentation and matching are NOT re-run — too expensive.
    """
    structlog.contextvars.bind_contextvars(job_id=job_id, correction_piece=piece_id)
    log.info(
        "correction_rerun_start",
        piece_id=piece_id,
        corrected_pos=corrected_grid_pos,
    )

    try:
        # Stub — Phase 8 will wire this fully
        raise NotImplementedError(
            "Correction re-run not yet fully implemented. "
            "Complete Phase 8 (rendering) first."
        )
    except Exception as exc:
        store.fail_job(job_id, f"Correction failed: {exc}")
        log.error("correction_rerun_failed", error=str(exc))
    finally:
        structlog.contextvars.clear_contextvars()