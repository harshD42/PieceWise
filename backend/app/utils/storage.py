# Copyright (c) 2026 Harsh Dwivedi
# Licensed under the Harsh Non-Commercial Attribution License (HNCAL) v1.0
# Commercial use requires written permission. See LICENSE for details.

"""
PieceWise — Per-Job Namespaced Storage
All file I/O is scoped to storage/{job_id}/ to prevent
concurrency collisions across simultaneous solve requests.

Layout per job:
    storage/{job_id}/
        uploads/
            reference.jpg
            pieces.jpg
        outputs/
            overlay_reference.jpg
            overlay_pieces.jpg
            step_cards/
                step_0001.jpg ...
            solution.json
        cache/
            reference_tokens.npz
            piece_tokens.npz
            pca_model.pkl
"""

import shutil
from pathlib import Path

from app.config import get_settings


def _root() -> Path:
    return get_settings().storage_root


# ─── Job Directory Builders ──────────────────────────────────────────────────

def job_dir(job_id: str) -> Path:
    return _root() / job_id


def uploads_dir(job_id: str) -> Path:
    return job_dir(job_id) / "uploads"


def outputs_dir(job_id: str) -> Path:
    return job_dir(job_id) / "outputs"


def step_cards_dir(job_id: str) -> Path:
    return outputs_dir(job_id) / "step_cards"


def cache_dir(job_id: str) -> Path:
    return job_dir(job_id) / "cache"


# ─── Upload Paths ────────────────────────────────────────────────────────────

def reference_upload_path(job_id: str) -> Path:
    return uploads_dir(job_id) / "reference.jpg"


def pieces_upload_path(job_id: str) -> Path:
    return uploads_dir(job_id) / "pieces.jpg"


# ─── Output Paths ────────────────────────────────────────────────────────────

def overlay_reference_path(job_id: str) -> Path:
    return outputs_dir(job_id) / "overlay_reference.jpg"


def overlay_pieces_path(job_id: str) -> Path:
    return outputs_dir(job_id) / "overlay_pieces.jpg"


def step_card_path(job_id: str, step_num: int) -> Path:
    return step_cards_dir(job_id) / f"step_{step_num:04d}.jpg"


def solution_manifest_path(job_id: str) -> Path:
    return outputs_dir(job_id) / "solution.json"


# ─── Cache Paths ─────────────────────────────────────────────────────────────

def reference_tokens_cache_path(job_id: str) -> Path:
    return cache_dir(job_id) / "reference_tokens.npz"


def piece_tokens_cache_path(job_id: str) -> Path:
    return cache_dir(job_id) / "piece_tokens.npz"


def pca_model_cache_path(job_id: str) -> Path:
    return cache_dir(job_id) / "pca_model.pkl"


# ─── Lifecycle Helpers ───────────────────────────────────────────────────────

def init_job_dirs(job_id: str) -> None:
    """
    Create all required subdirectories for a new job.
    Safe to call multiple times (exist_ok=True).
    """
    for d in [
        uploads_dir(job_id),
        outputs_dir(job_id),
        step_cards_dir(job_id),
        cache_dir(job_id),
    ]:
        d.mkdir(parents=True, exist_ok=True)


def cleanup_job(job_id: str) -> None:
    """
    Remove all files and directories for a completed or failed job.
    Used for housekeeping — does not raise if directory doesn't exist.
    """
    d = job_dir(job_id)
    if d.exists():
        shutil.rmtree(d)


def job_exists(job_id: str) -> bool:
    """Return True if the job directory has been initialised."""
    return job_dir(job_id).exists()


def get_asset_url(job_id: str, filename: str) -> str:
    """Build the public URL for a job output asset."""
    return f"/assets/{job_id}/{filename}"


def get_step_card_url(job_id: str, step_num: int) -> str:
    """Build the public URL for a specific step card image."""
    return f"/assets/{job_id}/step_cards/step_{step_num:04d}.jpg"