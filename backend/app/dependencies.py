# Copyright (c) 2026 Harsh Dwivedi
# Licensed under the Harsh Non-Commercial Attribution License (HNCAL) v1.0
# Commercial use requires written permission. See LICENSE for details.

"""
PieceWise — FastAPI Dependencies
Singleton providers for the JobStore and (later) model loaders.
All heavy objects are instantiated once at startup via the lifespan
event in main.py and stored here as module-level singletons.
Route handlers access them via FastAPI's Depends() injection.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends

from app.config import get_settings
from app.core.job_store import InMemoryJobStore, JobStore, RedisJobStore
from app.utils.logger import get_logger

log = get_logger(__name__)

# ─── JobStore Singleton ───────────────────────────────────────────────────────

_job_store: JobStore | None = None


def init_job_store() -> None:
    """
    Initialise the JobStore singleton based on JOB_STORE_BACKEND config.
    Called once during application lifespan startup.
    """
    global _job_store
    settings = get_settings()

    if settings.job_store_backend == "redis":
        log.info("init_job_store", backend="redis", url=settings.redis_url)
        _job_store = RedisJobStore(
            redis_url=settings.redis_url,
            ttl_seconds=settings.job_ttl_seconds,
        )
    else:
        log.info("init_job_store", backend="memory")
        _job_store = InMemoryJobStore()


def get_job_store() -> JobStore:
    """
    FastAPI dependency: inject the JobStore singleton into route handlers.

    Usage in a route:
        @router.get("/status/{job_id}")
        def get_status(job_id: str, store: JobStoreDep):
            job = store.get_job(job_id)
            ...
    """
    if _job_store is None:
        raise RuntimeError(
            "JobStore has not been initialised. "
            "Ensure init_job_store() is called during app lifespan startup."
        )
    return _job_store


# Annotated type alias for clean route signatures
JobStoreDep = Annotated[JobStore, Depends(get_job_store)]


# ─── Model Loader Stubs ───────────────────────────────────────────────────────
# These will be populated in Phase 3 (SAM) and Phase 4 (DINOv2).
# Defined here as None so main.py lifespan can call warm-up functions
# without import errors at Phase 1.

_sam_model = None
_dino_model = None
_dino_processor = None


def get_sam_model():
    """Return SAM model singleton. Populated in Phase 3."""
    if _sam_model is None:
        raise RuntimeError(
            "SAM model not loaded. Implement Phase 3 and call init_sam() at startup."
        )
    return _sam_model


def get_dino_model():
    """Return DINOv2 model singleton. Populated in Phase 4."""
    if _dino_model is None:
        raise RuntimeError(
            "DINOv2 model not loaded. Implement Phase 4 and call init_dino() at startup."
        )
    return _dino_model