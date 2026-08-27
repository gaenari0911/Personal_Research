"""R2 hierarchical trajectory-macro aggregation and trajectory bootstrap."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class MetricRow:
    trajectory_id: str
    step_index: int
    distance_bin: str
    transition_bin: str
    recall_at_1: float
    reciprocal_rank: float


def _trajectory_values(rows: Iterable[MetricRow], field: str) -> dict[str, float]:
    """cell mean -> Step bin mean -> trajectory Step mean, as frozen in R2."""
    cells: dict[tuple[str, int, str], list[float]] = defaultdict(list)
    for row in rows:
        cells[(row.trajectory_id, row.step_index, row.distance_bin)].append(float(getattr(row, field)))
    steps: dict[tuple[str, int], list[float]] = defaultdict(list)
    for (trajectory_id, step_index, _distance_bin), values in cells.items():
        steps[(trajectory_id, step_index)].append(float(np.mean(values)))
    trajectories: dict[str, list[float]] = defaultdict(list)
    for (trajectory_id, _step_index), values in steps.items():
        trajectories[trajectory_id].append(float(np.mean(values)))
    return {trajectory_id: float(np.mean(values)) for trajectory_id, values in trajectories.items()}


def _bootstrap(values: dict[str, float], resamples: int, seed: int) -> tuple[float, float, float]:
    if not values:
        return float("nan"), float("nan"), float("nan")
    array = np.asarray(list(values.values()), dtype=np.float64)
    estimate = float(array.mean())
    rng = np.random.default_rng(seed)
    draws = rng.choice(array, size=(resamples, len(array)), replace=True).mean(axis=1)
    return estimate, float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def aggregate_rows(
    rows: Iterable[MetricRow], *, resamples: int = 2000, seed: int = 42
) -> dict:
    rows = list(rows)
    if not rows:
        return {
            "trajectory_count": 0,
            "sample_count": 0,
            "recall_at_1": None,
            "recall_at_1_ci_low": None,
            "recall_at_1_ci_high": None,
            "mrr": None,
            "mrr_ci_low": None,
            "mrr_ci_high": None,
            "aggregation": "trajectory_macro_cell_step_hierarchy",
            "bootstrap_unit": "trajectory",
            "bootstrap_resamples": resamples,
            "bootstrap_seed": seed,
        }
    recall = _trajectory_values(rows, "recall_at_1")
    mrr = _trajectory_values(rows, "reciprocal_rank")
    recall_estimate, recall_low, recall_high = _bootstrap(recall, resamples, seed)
    mrr_estimate, mrr_low, mrr_high = _bootstrap(mrr, resamples, seed)
    return {
        "trajectory_count": len(recall),
        "sample_count": len(rows),
        "recall_at_1": recall_estimate,
        "recall_at_1_ci_low": recall_low,
        "recall_at_1_ci_high": recall_high,
        "mrr": mrr_estimate,
        "mrr_ci_low": mrr_low,
        "mrr_ci_high": mrr_high,
        "aggregation": "trajectory_macro_cell_step_hierarchy",
        "bootstrap_unit": "trajectory",
        "bootstrap_resamples": resamples,
        "bootstrap_seed": seed,
    }


def aggregate_curve(
    rows: Iterable[MetricRow], field: str, order: Iterable[str], *, resamples: int = 2000, seed: int = 42
) -> list[dict]:
    rows = list(rows)
    result = []
    for bin_name in order:
        subset = [row for row in rows if getattr(row, field) == bin_name]
        result.append({"bin": bin_name, **aggregate_rows(subset, resamples=resamples, seed=seed)})
    return result


def paired_trajectory_bootstrap_difference(
    left: Iterable[MetricRow], right: Iterable[MetricRow], *, field: str = "recall_at_1", resamples: int = 2000, seed: int = 42
) -> dict:
    left_values = _trajectory_values(left, field)
    right_values = _trajectory_values(right, field)
    shared = sorted(set(left_values) & set(right_values))
    if not shared:
        raise ValueError("paired bootstrap requires shared trajectory IDs")
    differences = np.asarray([left_values[key] - right_values[key] for key in shared])
    rng = np.random.default_rng(seed)
    draws = rng.choice(differences, size=(resamples, len(differences)), replace=True).mean(axis=1)
    return {
        "estimate": float(differences.mean()),
        "ci_low": float(np.quantile(draws, 0.025)),
        "ci_high": float(np.quantile(draws, 0.975)),
        "trajectory_count": len(shared),
        "bootstrap_unit": "trajectory_paired",
        "bootstrap_resamples": resamples,
        "bootstrap_seed": seed,
    }
