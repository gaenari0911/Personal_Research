"""Low-capacity trajectory-local retrieval metrics for the R2 protocol."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


def _normalize(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64)
    norm = np.linalg.norm(value, axis=-1, keepdims=True)
    return value / np.maximum(norm, 1e-12)


def cosine_scores(query: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    query = _normalize(np.asarray(query).reshape(1, -1))[0]
    candidates = _normalize(np.asarray(candidates))
    if candidates.ndim != 2 or candidates.shape[1] != query.shape[0]:
        raise ValueError("query/candidate embedding dimensions do not match")
    return candidates @ query


def retrieval_result(
    query: np.ndarray,
    candidates: np.ndarray,
    positive_indices: Sequence[int],
) -> tuple[float, float, int]:
    if not positive_indices:
        raise ValueError("positive_indices cannot be empty")
    scores = cosine_scores(query, candidates)
    positives = set(int(value) for value in positive_indices)
    if min(positives) < 0 or max(positives) >= len(scores):
        raise IndexError("positive candidate outside candidate set")
    order = np.argsort(-scores, kind="stable")
    rank = next(index + 1 for index, candidate in enumerate(order) if candidate in positives)
    return (float(order[0] in positives), 1.0 / rank, rank)


@dataclass(frozen=True)
class ScoredRetrieval:
    trajectory_id: str
    recall_at_1: float
    reciprocal_rank: float
    bin_name: str = "overall"


def trajectory_macro_metrics(values: Iterable[ScoredRetrieval]) -> dict:
    grouped: dict[str, list[ScoredRetrieval]] = {}
    for value in values:
        grouped.setdefault(value.trajectory_id, []).append(value)
    if not grouped:
        return {"trajectory_count": 0, "sample_count": 0, "recall_at_1": None, "mrr": None}
    trajectory_recall = []
    trajectory_mrr = []
    sample_count = 0
    for samples in grouped.values():
        sample_count += len(samples)
        trajectory_recall.append(np.mean([item.recall_at_1 for item in samples]))
        trajectory_mrr.append(np.mean([item.reciprocal_rank for item in samples]))
    return {
        "trajectory_count": len(grouped),
        "sample_count": sample_count,
        "recall_at_1": float(np.mean(trajectory_recall)),
        "mrr": float(np.mean(trajectory_mrr)),
    }


def metric_curve(values: Iterable[ScoredRetrieval], bin_order: Sequence[str]) -> dict:
    values = list(values)
    return {
        bin_name: trajectory_macro_metrics(
            item for item in values if item.bin_name == bin_name
        )
        for bin_name in bin_order
    }


def trajectory_bootstrap_ci(
    values: Iterable[ScoredRetrieval],
    metric: str = "recall_at_1",
    confidence: float = 0.95,
    resamples: int = 2000,
    seed: int = 42,
) -> dict:
    grouped: dict[str, list[ScoredRetrieval]] = {}
    for value in values:
        grouped.setdefault(value.trajectory_id, []).append(value)
    if not grouped:
        raise ValueError("cannot bootstrap an empty sample")
    if metric not in {"recall_at_1", "reciprocal_rank"}:
        raise ValueError("unsupported metric")
    trajectory_values = np.asarray(
        [np.mean([getattr(item, metric) for item in samples]) for samples in grouped.values()]
    )
    rng = np.random.default_rng(seed)
    draws = rng.choice(trajectory_values, size=(resamples, len(trajectory_values)), replace=True)
    estimates = np.mean(draws, axis=1)
    alpha = (1.0 - confidence) / 2.0
    return {
        "estimate": float(np.mean(trajectory_values)),
        "confidence": confidence,
        "low": float(np.quantile(estimates, alpha)),
        "high": float(np.quantile(estimates, 1.0 - alpha)),
        "bootstrap_unit": "trajectory",
        "resamples": resamples,
        "seed": seed,
    }

