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

    # Shared PCA reducer — fitted on reference, applied to pieces
    from app.modules.feature_extraction.pca_reducer import make_reducer
    from app.modules.feature_extraction.embedding_store import EmbeddingStore
    from app.utils.storage import pca_model_cache_path

    reducer = make_reducer()
    emb_store = EmbeddingStore()

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
        asyncio.to_thread(_embed_reference, job_id, ref_img, reducer),
    )

    emb_store.set_patch_token_map(patch_token_map)
    reducer.save(pca_model_cache_path(job_id))

    log.info(
        "stage_complete",
        stage="parallel_seg_ref_embed",
        piece_count=len(piece_crops),
    )

    # ── Stage 3: Piece Embedding ─────────────────────────────────────────────
    store.advance_stage(job_id, JobStage.FEATURE_EXTRACTION)
    log.info("stage_start", stage="piece_embedding")

    piece_embeddings = await asyncio.to_thread(
        _embed_pieces, job_id, piece_crops, reducer
    )
    emb_store.set_piece_embeddings(piece_embeddings)
    log.info("stage_complete", stage="piece_embedding")

    # ── Stage 4: Matching ────────────────────────────────────────────────────
    store.advance_stage(job_id, JobStage.MATCHING)
    log.info("stage_start", stage="matching")

    piece_matches = await asyncio.to_thread(
        _match_pieces, job_id, emb_store, piece_crops
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
    """Phase 3 — full segmentation pipeline."""
    from app.modules.segmentation.mask_generator import generate_masks
    from app.modules.segmentation.mask_filter import filter_masks
    from app.modules.segmentation.segmentation_refiner import refine_masks
    from app.modules.segmentation.piece_extractor import extract_pieces
    from app.modules.segmentation.contour_analyzer import analyze_contours
    from app.config import get_settings

    settings = get_settings()
    h, w = pieces_img.shape[:2]

    # Stage 1+2: CC pre-filter + SAM mask generation
    masks, _ = generate_masks(pieces_img)

    # Stage 3: Filter invalid masks
    masks = filter_masks(masks, (h, w))

    # Stage 4: Separate merged/touching pieces
    masks = refine_masks(masks, (h, w), settings.sam_min_mask_area)

    # Stage 5: Extract PieceCrop objects
    pieces = extract_pieces(masks, pieces_img)

    # Stage 6: Curvature encoding + flat_side_count
    pieces = analyze_contours(pieces)

    return pieces


def _embed_reference(job_id: str, ref_img, reducer):
    """Phase 4 — DINOv2 reference spatial token extraction."""
    from app.modules.feature_extraction.patch_generator import extract_reference_tokens
    from app.modules.preprocessing.grid_estimator import estimate_grid_shape

    # Grid shape will be refined after segmentation in production
    # For now estimate from a nominal 1000-piece 4:3 puzzle
    # Phase 5 will receive the real piece count from segmentation
    h, w = ref_img.shape[:2]
    grid_shape = estimate_grid_shape(1000, (h, w))

    return extract_reference_tokens(ref_img, grid_shape, reducer, job_id)


def _embed_pieces(job_id: str, piece_crops, reducer):
    """Phase 4 — DINOv2 piece embedding."""
    from app.modules.feature_extraction.piece_embedder import embed_pieces
    return embed_pieces(piece_crops, reducer, job_id)


def _match_pieces(job_id: str, emb_store, piece_crops):
    """Phase 5 — full coarse-to-fine matching engine."""
    from app.modules.matching.matcher import match_pieces
    from app.modules.preprocessing.grid_estimator import estimate_grid_shape

    # Derive grid shape from piece count and reference image dimensions
    n_pieces = len(piece_crops)
    token_map = emb_store.get_patch_token_map()
    grid_shape = token_map.grid_shape

    return match_pieces(emb_store, piece_crops, grid_shape)


def _refine_adjacency(job_id: str, piece_matches, piece_crops):
    """Phase 6 — adjacency refiner."""
    from app.modules.adjacency.refiner import refine_adjacency
    from app.modules.feature_extraction.embedding_store import EmbeddingStore

    # Recover grid shape from matches
    if not piece_matches:
        return piece_matches

    rows = max(m.grid_pos[0] for m in piece_matches) + 1
    cols = max(m.grid_pos[1] for m in piece_matches) + 1
    grid_shape = (rows, cols)

    return refine_adjacency(piece_matches, piece_crops, grid_shape)


def _sequence(job_id: str, piece_matches, grid_shape):
    """Phase 7 — sequencing module."""
    from app.modules.sequencing.piece_classifier import classify_and_validate
    from app.modules.sequencing.bfs_assembler import bfs_order
    from app.modules.sequencing.step_generator import generate_steps

    # Phase 7 has no PieceCrop list in this call — pass empty for classifier
    # The classifier still works: without curvature profiles it skips
    # cross-validation and classifies purely from grid_pos (correct behaviour)
    classifications = classify_and_validate(piece_matches, [], grid_shape)
    ordered = bfs_order(piece_matches, classifications, grid_shape)
    return generate_steps(ordered, classifications)


def _render(job_id, ref_img, pieces_img, piece_crops, piece_matches, assembly_steps):
    """Phase 8 — full rendering pipeline."""
    from app.modules.rendering.reference_overlay import render_reference_overlay
    from app.modules.rendering.pieces_overlay import render_pieces_overlay
    from app.modules.rendering.piece_crop_saver import save_piece_crops
    from app.modules.rendering.step_card_generator import generate_step_cards
    from app.modules.rendering.manifest_builder import (
        build_manifest, write_manifest, build_output_bundle
    )

    # Recover grid shape from steps
    if assembly_steps:
        rows = max(s.grid_pos[0] for s in assembly_steps) + 1
        cols = max(s.grid_pos[1] for s in assembly_steps) + 1
        grid_shape = (rows, cols)
    else:
        grid_shape = (1, 1)

    # Ensure output dirs exist
    from app.utils.storage import init_job_dirs
    init_job_dirs(job_id)

    # Reference overlay
    render_reference_overlay(ref_img, assembly_steps, grid_shape, job_id)

    # Pieces overlay
    render_pieces_overlay(pieces_img, piece_crops, assembly_steps, job_id)

    # Save piece crop thumbnails
    crop_urls = save_piece_crops(piece_crops, assembly_steps, job_id)

    # Step cards
    step_card_urls = generate_step_cards(
        assembly_steps, piece_crops, ref_img, grid_shape, job_id
    )

    # Manifest
    manifest = build_manifest(
        job_id, assembly_steps, grid_shape, crop_urls, step_card_urls
    )
    write_manifest(manifest, job_id)

    return build_output_bundle(job_id, manifest, step_card_urls)


# ─── Correction Re-run ───────────────────────────────────────────────────────

async def rerun_from_correction(
    job_id: str,
    piece_id: int,
    corrected_grid_pos: tuple[int, int],
    store: JobStore,
) -> None:
    """
    Re-run sequencing and rendering after a human-in-the-loop correction.
    Loads the existing manifest, applies the correction to the match list,
    then re-runs stages 7 (sequencing) and 8 (rendering).
    Full segmentation and matching are NOT re-run.
    """
    structlog.contextvars.bind_contextvars(job_id=job_id, correction_piece=piece_id)
    log.info(
        "correction_rerun_start",
        piece_id=piece_id,
        corrected_pos=corrected_grid_pos,
    )

    try:
        import json
        from app.utils.storage import solution_manifest_path
        from app.models.piece import PieceMatch
        from app.modules.rendering.manifest_builder import (
            build_manifest, write_manifest, build_output_bundle
        )

        manifest_path = solution_manifest_path(job_id)
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"solution.json not found for job {job_id}. "
                "Cannot apply correction before initial solve completes."
            )

        # Load existing manifest to recover match list
        manifest_data = json.loads(manifest_path.read_text())
        steps_data = manifest_data.get("steps", [])

        # Rebuild match list from manifest, applying the correction
        matches = []
        for s in steps_data:
            gpos = tuple(s["grid_pos"])
            if s["piece_id"] == piece_id:
                gpos = corrected_grid_pos   # apply correction
            matches.append(PieceMatch(
                piece_id=s["piece_id"],
                grid_pos=gpos,
                rotation_deg=s["rotation_deg"],
                composite_confidence=s["composite_confidence"],
                adjacency_score=s.get("adjacency_score", 0.5),
                curvature_complement_score=s.get("curvature_complement_score", 0.5),
                flagged=s.get("flagged", False),
            ))

        grid_shape_list = manifest_data.get("grid_shape", [1, 1])
        grid_shape = tuple(grid_shape_list)

        # Re-run sequencing
        store.update_job(job_id, stage="sequencing_correction", progress=88)
        assembly_steps = _sequence(job_id, matches, grid_shape)

        # Re-run rendering (no images available in correction path —
        # load from uploads)
        from app.utils.storage import reference_upload_path, pieces_upload_path
        from app.utils.image_utils import load_image_bgr

        ref_img = load_image_bgr(reference_upload_path(job_id))
        pieces_img = load_image_bgr(pieces_upload_path(job_id))

        store.update_job(job_id, stage="rendering_correction", progress=94)
        output_bundle = _render(
            job_id, ref_img, pieces_img, [], matches, assembly_steps
        )

        store.update_job(
            job_id,
            status=__import__("app.models.job", fromlist=["JobStatus"]).JobStatus.DONE,
            progress=100,
            result=output_bundle,
        )
        log.info("correction_rerun_complete", job_id=job_id)

    except Exception as exc:
        store.fail_job(job_id, f"Correction failed: {exc}")
        log.error("correction_rerun_failed", error=str(exc))
    finally:
        structlog.contextvars.clear_contextvars()