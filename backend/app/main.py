# Copyright (c) 2026 Harsh Dwivedi
# Licensed under the Harsh Non-Commercial Attribution License (HNCAL) v1.0
# Commercial use requires written permission. See LICENSE for details.

"""
PieceWise — FastAPI Application Entry Point
Creates the app, registers lifespan events, CORS, routers,
and global error handlers.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.middleware.error_handler import register_error_handlers
from app.api.routes import assets, solve, status
from app.config import get_settings
from app.dependencies import init_job_store
from app.utils.logger import configure_logging, get_logger

log = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.
    Startup: configure logging, initialise JobStore, warm up model stubs.
    Shutdown: clean up resources.
    """
    # ── Startup ──────────────────────────────────────────────────────────────
    configure_logging()
    settings = get_settings()

    log.info(
        "piecewise_startup",
        version="1.0.0",
        sam_model=settings.sam_model_type,
        pca_enabled=settings.enable_pca_reduction,
        pca_dims=settings.pca_target_dims,
        job_store=settings.job_store_backend,
        coarse_topk=settings.coarse_topk,
    )

    # Initialise job store (memory or Redis)
    init_job_store()

    # Model warm-up — SAM loaded here, DINOv2 loaded in Phase 4
    try:
        from app.modules.segmentation.sam_loader import init_sam
        init_sam()
        log.info("sam_warmup_complete")
    except FileNotFoundError as e:
        log.warning(
            "sam_checkpoint_missing",
            error=str(e),
            advice="Run python scripts/download_models.py to download model weights.",
        )
    except Exception as e:
        log.warning("sam_warmup_failed", error=str(e))

    log.info("piecewise_ready")
    yield

    # ── Shutdown ─────────────────────────────────────────────────────────────
    log.info("piecewise_shutdown")


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title="PieceWise",
        summary="AI-powered jigsaw puzzle solver — because every piece has a place.",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # ── CORS ─────────────────────────────────────────────────────────────────
    # Allow the Vite dev server (port 5173) and production frontend
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:5173",   # Vite dev server
            "http://localhost:3000",   # Alternative dev port
            "http://localhost:80",     # Docker nginx
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Error Handlers ───────────────────────────────────────────────────────
    register_error_handlers(app)

    # ── Routers ──────────────────────────────────────────────────────────────
    app.include_router(solve.router)
    app.include_router(status.router)
    app.include_router(assets.router)

    # ── Health Check ─────────────────────────────────────────────────────────
    @app.get("/health", tags=["health"], summary="Health check")
    async def health() -> dict:
        return {
            "status": "ok",
            "service": "piecewise",
            "version": "1.0.0",
            "sam_model": settings.sam_model_type,
            "job_store": settings.job_store_backend,
        }

    return app


# Module-level app instance for uvicorn
app = create_app()