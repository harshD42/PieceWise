# Copyright (c) 2026 Harsh Dwivedi
# Licensed under the Harsh Non-Commercial Attribution License (HNCAL) v1.0
# Commercial use requires written permission. See LICENSE for details.

"""
PieceWise — PCA Token Dimensionality Reducer
Config-gated reduction of DINOv2 token vectors from 768 → 128 dims.

Design decisions:
- PCA is fitted ONCE on the reference image token grid vectors.
- The SAME fitted PCA is applied to ALL piece token grids.
  Consistency is mandatory — reference and piece tokens must live
  in the same reduced embedding space for similarity to be meaningful.
- Fitted PCA model is serialised to {job_id}/cache/pca_model.pkl
  so it can be reloaded if the pipeline is restarted mid-job.
- When ENABLE_PCA_REDUCTION=false, the reducer is a no-op pass-through.

Scaling rationale:
  1000-piece puzzle: 500 cells × 500 pieces × 4 rotations
  Without PCA: similarity matrix uses 768-dim vectors
  With PCA:    similarity matrix uses 128-dim vectors → 6× less memory,
               significantly faster torch.mm on GPU
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

from app.config import get_settings
from app.utils.logger import get_logger

log = get_logger(__name__)


class PCAReducer:
    """
    Wraps sklearn PCA with fit/transform/save/load lifecycle.
    When PCA is disabled via config, transform() is a no-op pass-through.
    """

    def __init__(self, n_components: int, enabled: bool) -> None:
        self.n_components = n_components
        self.enabled = enabled
        self._pca = None
        self._fitted = False

    def fit(self, token_matrix: np.ndarray) -> "PCAReducer":
        """
        Fit PCA on a matrix of token vectors.

        Args:
            token_matrix: (N, D) float32 array of token vectors.
                          Typically all patch tokens from the reference image.

        Returns:
            self (for chaining)
        """
        if not self.enabled:
            log.debug("pca_disabled_skipping_fit")
            return self

        try:
            from sklearn.decomposition import PCA
        except ImportError as e:
            raise ImportError(
                "scikit-learn not found. Run: pip install scikit-learn"
            ) from e

        n_samples, n_features = token_matrix.shape
        effective_components = min(self.n_components, n_samples, n_features)

        if effective_components != self.n_components:
            log.warning(
                "pca_components_adjusted",
                requested=self.n_components,
                effective=effective_components,
                reason="insufficient samples or features",
            )

        self._pca = PCA(n_components=effective_components, random_state=42)
        self._pca.fit(token_matrix.astype(np.float32))
        self._fitted = True

        explained = float(self._pca.explained_variance_ratio_.sum())
        log.info(
            "pca_fitted",
            n_samples=n_samples,
            input_dims=n_features,
            output_dims=effective_components,
            explained_variance=round(explained, 4),
        )

        return self

    def transform(self, token_matrix: np.ndarray) -> np.ndarray:
        """
        Apply PCA reduction to a token matrix.

        Args:
            token_matrix: (N, D) float32 array

        Returns:
            (N, n_components) float32 array if PCA enabled,
            original array unchanged if PCA disabled.
        """
        if not self.enabled:
            return token_matrix

        if not self._fitted or self._pca is None:
            raise RuntimeError(
                "PCAReducer.fit() must be called before transform()."
            )

        return self._pca.transform(
            token_matrix.astype(np.float32)
        ).astype(np.float32)

    def fit_transform(self, token_matrix: np.ndarray) -> np.ndarray:
        """Convenience: fit then transform in one call."""
        self.fit(token_matrix)
        return self.transform(token_matrix)

    @property
    def output_dims(self) -> int:
        """Return the output dimensionality after reduction."""
        if not self.enabled:
            return 768  # DINOv2 ViT-B/14 default
        if self._fitted and self._pca is not None:
            return self._pca.n_components_
        return self.n_components

    def save(self, path: Path) -> None:
        """Serialise the fitted PCA model to disk."""
        if not self.enabled or not self._fitted:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self._pca, f)
        log.debug("pca_model_saved", path=str(path))

    def load(self, path: Path) -> bool:
        """
        Load a previously fitted PCA model from disk.
        Returns True if loaded successfully, False if file not found.
        """
        if not self.enabled:
            return False
        if not path.exists():
            return False
        with open(path, "rb") as f:
            self._pca = pickle.load(f)
        self._fitted = True
        log.info("pca_model_loaded", path=str(path))
        return True


def make_reducer() -> PCAReducer:
    """
    Factory: create a PCAReducer using current config settings.
    Use this instead of constructing PCAReducer directly in production code.
    """
    settings = get_settings()
    return PCAReducer(
        n_components=settings.pca_target_dims,
        enabled=settings.enable_pca_reduction,
    )