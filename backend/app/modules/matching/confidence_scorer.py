# Copyright (c) 2026 Harsh Dwivedi
# Licensed under the Harsh Non-Commercial Attribution License (HNCAL) v1.0
# Commercial use requires written permission. See LICENSE for details.

"""
PieceWise — Confidence Scorer and Flagger
Post-Hungarian: attaches final confidence metadata and flags pieces
whose composite_confidence falls below the configured threshold.

Flagged pieces are surfaced to the user in the human-in-the-loop UI
(Phase 9) with their top-3 candidate cells as clickable alternatives.

Also computes summary statistics used in the solution manifest.
"""

from __future__ import annotations

import numpy as np

from app.config import get_settings
from app.models.piece import PieceMatch
from app.utils.logger import get_logger

log = get_logger(__name__)


def flag_low_confidence(
    matches: list[PieceMatch],
    threshold: float | None = None,
) -> list[PieceMatch]:
    """
    Mark pieces with composite_confidence below threshold as flagged.
    Modifies matches in-place and returns the list.

    Args:
        matches:   List of PieceMatch from conflict_resolver
        threshold: Confidence threshold. Defaults to config CONFIDENCE_THRESHOLD.

    Returns:
        Same list with flagged field set appropriately.
    """
    if threshold is None:
        threshold = get_settings().confidence_threshold

    n_flagged = 0
    for m in matches:
        if m.composite_confidence < threshold:
            m.flagged = True
            n_flagged += 1
        else:
            m.flagged = False

    log.info(
        "confidence_flagging_complete",
        total=len(matches),
        flagged=n_flagged,
        threshold=threshold,
    )

    return matches


def compute_confidence_stats(matches: list[PieceMatch]) -> dict:
    """
    Compute summary confidence statistics for logging and manifest output.

    Returns:
        Dict with keys: mean, min, max, flagged_count, flagged_fraction
    """
    if not matches:
        return {
            "mean": 0.0, "min": 0.0, "max": 0.0,
            "flagged_count": 0, "flagged_fraction": 0.0,
        }

    scores = np.array([m.composite_confidence for m in matches], dtype=np.float32)
    flagged = sum(1 for m in matches if m.flagged)

    stats = {
        "mean": float(scores.mean()),
        "min": float(scores.min()),
        "max": float(scores.max()),
        "std": float(scores.std()),
        "flagged_count": flagged,
        "flagged_fraction": flagged / len(matches),
    }

    log.info(
        "confidence_stats",
        mean=round(stats["mean"], 3),
        min=round(stats["min"], 3),
        max=round(stats["max"], 3),
        flagged=flagged,
        total=len(matches),
    )

    return stats