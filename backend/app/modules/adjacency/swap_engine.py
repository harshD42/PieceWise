# Copyright (c) 2026 Harsh Dwivedi
# Licensed under the Harsh Non-Commercial Attribution License (HNCAL) v1.0
# Commercial use requires written permission. See LICENSE for details.

"""
PieceWise — Local Swap Engine
Post-assignment local search that improves the overall adjacency score
by attempting pairwise swaps of piece assignments.

Algorithm:
  1. Score all neighbor pairs (histogram + curvature combined)
  2. Identify pairs below the compatibility threshold
  3. For each low-score pair: find candidate swap partners
     (pieces that might improve the neighborhood if swapped)
  4. Attempt the swap: re-score the affected neighborhood
  5. Accept swap if total neighborhood score improves
  6. Repeat up to ADJACENCY_SWAP_MAX_ITER times

Swap validity constraint:
  All swaps maintain the 1:1 piece→cell bijection.
  A swap exchanges two pieces' grid_pos assignments — no cell ever
  has zero or two pieces after a swap.

Combined adjacency score:
  combined = 0.6 × histogram_score + 0.4 × curvature_complement_score

This weighting reflects that histogram evidence is stronger (it's pixel-level)
while curvature complement is more noise-prone at 1000-piece resolution.
"""

from __future__ import annotations

import numpy as np

from app.config import get_settings
from app.models.piece import PieceCrop, PieceMatch
from app.modules.adjacency.curvature_complement import (
    curvature_complement_score,
    score_all_pairs_curvature,
)
from app.modules.adjacency.edge_histogram import (
    edge_histogram_score,
    score_all_pairs_histogram,
)
from app.modules.adjacency.neighbor_extractor import (
    NeighborPair,
    build_piece_neighbor_map,
    extract_neighbor_pairs,
)
from app.utils.logger import get_logger

log = get_logger(__name__)

# Combined score weights
_W_HISTOGRAM = 0.6
_W_CURVATURE = 0.4

# Pairs below this threshold are candidates for swapping
_SWAP_CANDIDATE_THRESHOLD = 0.4


def _combined_score(hist: float, curv: float) -> float:
    return _W_HISTOGRAM * hist + _W_CURVATURE * curv


def _pair_key(id_a: int, id_b: int) -> tuple[int, int]:
    return (min(id_a, id_b), max(id_a, id_b))


def _score_pair(
    pair: NeighborPair,
    piece_map: dict[int, PieceCrop],
    hist_scores: dict,
    curv_scores: dict,
) -> float:
    """Return combined adjacency score for a neighbor pair."""
    key = _pair_key(pair.piece_id_a, pair.piece_id_b)
    hist = hist_scores.get(key, 0.5)
    curv = curv_scores.get(key, 0.5)
    return _combined_score(hist, curv)


def _neighborhood_score(
    piece_id: int,
    neighbor_map: dict[int, list[NeighborPair]],
    hist_scores: dict,
    curv_scores: dict,
) -> float:
    """Return the mean combined score of all neighbor pairs for a piece."""
    pairs = neighbor_map.get(piece_id, [])
    if not pairs:
        return 1.0  # No neighbors = no penalty
    scores = [
        _combined_score(
            hist_scores.get(_pair_key(p.piece_id_a, p.piece_id_b), 0.5),
            curv_scores.get(_pair_key(p.piece_id_a, p.piece_id_b), 0.5),
        )
        for p in pairs
    ]
    return float(np.mean(scores))


def _recompute_pair_scores(
    affected_ids: set[int],
    matches: list[PieceMatch],
    piece_map: dict[int, PieceCrop],
    grid_shape: tuple[int, int],
    hist_scores: dict,
    curv_scores: dict,
) -> tuple[dict, dict]:
    """
    Recompute histogram and curvature scores for all pairs involving
    the affected piece IDs after a swap.

    Returns updated copies of hist_scores and curv_scores.
    """
    pos_to_match = {m.grid_pos: m for m in matches}
    new_hist = dict(hist_scores)
    new_curv = dict(curv_scores)

    for pid in affected_ids:
        match = next((m for m in matches if m.piece_id == pid), None)
        if match is None:
            continue

        r, c = match.grid_pos
        n_rows, n_cols = grid_shape

        # Re-score all neighbors of this piece
        neighbors = []
        for dr, dc, side_a, side_b in [
            (-1, 0, 0, 2), (1, 0, 2, 0),  # vertical
            (0, -1, 3, 1), (0, 1, 1, 3),  # horizontal
        ]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < n_rows and 0 <= nc < n_cols:
                nb_match = pos_to_match.get((nr, nc))
                if nb_match:
                    pair = NeighborPair(
                        piece_id_a=pid,
                        piece_id_b=nb_match.piece_id,
                        grid_pos_a=(r, c),
                        grid_pos_b=(nr, nc),
                        side_a=side_a,
                        side_b=side_b,
                    )
                    key = _pair_key(pid, nb_match.piece_id)
                    pa = piece_map.get(pid)
                    pb = piece_map.get(nb_match.piece_id)
                    if pa and pb:
                        new_hist[key] = edge_histogram_score(pa, pb, pair)
                        new_curv[key] = curvature_complement_score(pa, pb, pair)

    return new_hist, new_curv


def run_swap_engine(
    matches: list[PieceMatch],
    pieces: list[PieceCrop],
    grid_shape: tuple[int, int],
    max_iterations: int | None = None,
) -> list[PieceMatch]:
    """
    Run the local swap engine to improve adjacency scores.

    Args:
        matches:        Post-Hungarian PieceMatch list (modified in-place)
        pieces:         All PieceCrop objects
        grid_shape:     (n_rows, n_cols)
        max_iterations: Cap on swap attempts. Defaults to config value.

    Returns:
        Refined PieceMatch list with updated adjacency scores.
    """
    settings = get_settings()
    if max_iterations is None:
        max_iterations = settings.adjacency_swap_max_iter

    piece_map = {p.piece_id: p for p in pieces}

    # Initial scoring
    pairs = extract_neighbor_pairs(matches, grid_shape)
    neighbor_map = build_piece_neighbor_map(pairs)
    hist_scores = score_all_pairs_histogram(pairs, piece_map)
    curv_scores = score_all_pairs_curvature(pairs, piece_map)

    n_swaps = 0
    n_attempts = 0
    iterations_run = 0

    log.info(
        "swap_engine_start",
        n_pieces=len(matches),
        n_pairs=len(pairs),
        max_iterations=max_iterations,
    )

    for iteration in range(max_iterations):
        iterations_run = iteration + 1
        # Find the pair with the worst combined score
        worst_score = float("inf")
        worst_pair: NeighborPair | None = None

        for pair in pairs:
            score = _combined_score(
                hist_scores.get(_pair_key(pair.piece_id_a, pair.piece_id_b), 0.5),
                curv_scores.get(_pair_key(pair.piece_id_a, pair.piece_id_b), 0.5),
            )
            if score < worst_score:
                worst_score = score
                worst_pair = pair

        # Stop if worst pair is already above threshold
        if worst_pair is None or worst_score >= _SWAP_CANDIDATE_THRESHOLD:
            log.debug(
                "swap_engine_early_stop",
                iteration=iteration,
                worst_score=round(worst_score, 3),
            )
            break

        # Try swapping piece_a with each other piece (greedy search)
        improved = False
        match_a = next(
            (m for m in matches if m.piece_id == worst_pair.piece_id_a), None
        )
        if match_a is None:
            break

        score_before = (
            _neighborhood_score(worst_pair.piece_id_a, neighbor_map, hist_scores, curv_scores)
            + _neighborhood_score(worst_pair.piece_id_b, neighbor_map, hist_scores, curv_scores)
        )

        for match_b in matches:
            if match_b.piece_id == match_a.piece_id:
                continue
            if match_b.piece_id == worst_pair.piece_id_b:
                continue

            n_attempts += 1

            # Tentatively swap grid positions
            match_a.grid_pos, match_b.grid_pos = match_b.grid_pos, match_a.grid_pos

            # Recompute scores for affected pieces
            affected = {match_a.piece_id, match_b.piece_id}
            new_hist, new_curv = _recompute_pair_scores(
                affected, matches, piece_map, grid_shape, hist_scores, curv_scores
            )

            score_after = (
                _neighborhood_score(match_a.piece_id, neighbor_map, new_hist, new_curv)
                + _neighborhood_score(match_b.piece_id, neighbor_map, new_hist, new_curv)
            )

            if score_after > score_before:
                # Accept swap
                hist_scores = new_hist
                curv_scores = new_curv
                # Rebuild neighbor structures
                pairs = extract_neighbor_pairs(matches, grid_shape)
                neighbor_map = build_piece_neighbor_map(pairs)
                n_swaps += 1
                improved = True
                break
            else:
                # Revert swap
                match_a.grid_pos, match_b.grid_pos = match_b.grid_pos, match_a.grid_pos

        if not improved:
            # No swap improved this pair — stop iterating
            break

    log.info(
        "swap_engine_complete",
        n_swaps=n_swaps,
        n_attempts=n_attempts,
        iterations_run=iterations_run,
    )

    # Attach final adjacency and curvature scores to each PieceMatch
    _attach_scores(matches, pairs, hist_scores, curv_scores)

    return matches


def _attach_scores(
    matches: list[PieceMatch],
    pairs: list[NeighborPair],
    hist_scores: dict,
    curv_scores: dict,
) -> None:
    """
    Attach mean adjacency_score and curvature_complement_score
    to each PieceMatch based on its neighbor pairs.
    """
    # Build piece → list of pair scores
    piece_hist: dict[int, list[float]] = {}
    piece_curv: dict[int, list[float]] = {}

    for pair in pairs:
        key = _pair_key(pair.piece_id_a, pair.piece_id_b)
        h = hist_scores.get(key, 0.5)
        cv = curv_scores.get(key, 0.5)
        for pid in (pair.piece_id_a, pair.piece_id_b):
            piece_hist.setdefault(pid, []).append(h)
            piece_curv.setdefault(pid, []).append(cv)

    for m in matches:
        hist_vals = piece_hist.get(m.piece_id, [])
        curv_vals = piece_curv.get(m.piece_id, [])
        m.adjacency_score = float(np.mean(hist_vals)) if hist_vals else 0.5
        m.curvature_complement_score = float(np.mean(curv_vals)) if curv_vals else 0.5