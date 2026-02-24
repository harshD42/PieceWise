# Copyright (c) 2026 Harsh Dwivedi
# Licensed under the Harsh Non-Commercial Attribution License (HNCAL) v1.0
# Commercial use requires written permission. See LICENSE for details.

"""
PieceWise — Feature Extraction Module
Public API for the DINOv2-based feature extraction stage.
"""

from app.modules.feature_extraction.dino_loader import (
    extract_tokens,
    get_device,
    get_dino_model,
    get_dino_processor,
    init_dino,
    is_loaded,
)
from app.modules.feature_extraction.embedding_store import EmbeddingStore
from app.modules.feature_extraction.patch_generator import (
    extract_reference_tokens,
)
from app.modules.feature_extraction.pca_reducer import PCAReducer, make_reducer
from app.modules.feature_extraction.piece_embedder import embed_pieces

__all__ = [
    # DINOv2 loader
    "init_dino",
    "get_dino_model",
    "get_dino_processor",
    "get_device",
    "extract_tokens",
    "is_loaded",
    # PCA reducer
    "PCAReducer",
    "make_reducer",
    # Reference tokens
    "extract_reference_tokens",
    # Piece embedder
    "embed_pieces",
    # Store
    "EmbeddingStore",
]