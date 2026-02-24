# Copyright (c) 2026 Harsh Dwivedi
# Licensed under the Harsh Non-Commercial Attribution License (HNCAL) v1.0
# Commercial use requires written permission. See LICENSE for details.

"""
PieceWise — Application Configuration
All settings are loaded from environment variables with 1000-piece
optimised defaults. Override via backend/.env or environment.
"""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ─── Model Configuration ─────────────────────────────────────────────────
    sam_model_type: Literal["vit_b", "vit_h"] = "vit_h"
    dino_model_name: str = "facebook/dinov2-base"

    # ─── Storage ─────────────────────────────────────────────────────────────
    storage_root: Path = Path("./storage")
    upload_max_mb: int = 50

    # ─── Job Store ───────────────────────────────────────────────────────────
    job_store_backend: Literal["memory", "redis"] = "memory"
    redis_url: str = "redis://localhost:6379/0"
    job_ttl_seconds: int = 86400  # 24 hours

    # ─── Matching Engine ─────────────────────────────────────────────────────
    confidence_threshold: float = 0.55
    # CLS coarse shortlist size — wider for large grids (1000-piece ~ 32x32)
    coarse_topk: int = 30

    # ─── PCA Token Reduction ─────────────────────────────────────────────────
    # On by default — essential for 500–2000 piece puzzles
    enable_pca_reduction: bool = True
    pca_target_dims: int = 128

    # ─── Segmentation (SAM) ──────────────────────────────────────────────────
    # Denser prompting for smaller pieces in 1000-piece puzzles
    sam_points_per_side: int = 64
    sam_pred_iou_thresh: float = 0.88
    sam_stability_score_thresh: float = 0.92
    # Lower min area — pieces are physically smaller at 1000-piece density
    sam_min_mask_area: int = 300

    # ─── Feature Extraction (DINOv2) ─────────────────────────────────────────
    # Smaller batch to manage GPU memory at scale
    dino_batch_size: int = 16

    # ─── Adjacency Refiner ───────────────────────────────────────────────────
    adjacency_swap_max_iter: int = 500

    # ─── Logging ─────────────────────────────────────────────────────────────
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    # ─── Server ──────────────────────────────────────────────────────────────
    host: str = "0.0.0.0"
    port: int = 8000

    # ─── Derived helpers ─────────────────────────────────────────────────────
    @property
    def upload_max_bytes(self) -> int:
        return self.upload_max_mb * 1024 * 1024

    @property
    def sam_checkpoint_path(self) -> Path:
        filename = f"sam_{self.sam_model_type}.pth"
        return self.storage_root / "models" / filename

    @property
    def dino_cache_dir(self) -> Path:
        return self.storage_root / "models" / "dinov2_vitb14"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached singleton Settings instance."""
    return Settings()