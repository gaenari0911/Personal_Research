"""Explicit trajectory-local ranking and GT-rank lookup."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


def compute_gt_rank(scores: Sequence[float], positive_indices: Sequence[int]) -> int:
    """Return the one-based rank of the first positive in a stable descending ranking."""
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 1 or not len(values):
        raise ValueError("scores must be a nonempty one-dimensional array")
    positives = {int(index) for index in positive_indices}
    if not positives:
        raise ValueError("positive_indices cannot be empty")
    if min(positives) < 0 or max(positives) >= len(values):
        raise IndexError("positive candidate outside candidate set")
    order = np.argsort(-values, kind="stable")
    return next(rank for rank, candidate in enumerate(order, start=1) if int(candidate) in positives)


def recall_at_1_from_rank(rank: int) -> float:
    if rank < 1:
        raise ValueError("rank must be one-based")
    return float(rank == 1)


def reciprocal_rank(rank: int) -> float:
    if rank < 1:
        raise ValueError("rank must be one-based")
    return 1.0 / float(rank)


@dataclass(frozen=True)
class RetrievalResult:
    rank: int
    recall_at_1: float
    reciprocal_rank: float
    predicted_index: int


def rank_scores(scores: Sequence[float], positive_indices: Sequence[int]) -> RetrievalResult:
    values = np.asarray(scores, dtype=np.float64)
    rank = compute_gt_rank(values, positive_indices)
    predicted = int(np.argsort(-values, kind="stable")[0])
    return RetrievalResult(rank, recall_at_1_from_rank(rank), reciprocal_rank(rank), predicted)
