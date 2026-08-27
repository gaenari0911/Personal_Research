"""Deterministic subset, anchor, and representation-sanity helpers for R4."""

from __future__ import annotations

import hashlib
import math
import random
from collections import Counter
from typing import Iterable, Mapping, Sequence

import numpy as np

from .sampling import build_balanced_samples


def _stable_tie(seed: int, value: str) -> int:
    return int.from_bytes(
        hashlib.sha256(f"{seed}:{value}".encode()).digest()[:8], "big"
    )


def select_stratified_subset(
    records_by_id: Mapping[str, Mapping[str, object]],
    allowed_ids: Sequence[str],
    quotas: Mapping[str, int],
    seed: int = 42,
) -> list[str]:
    """Balance source scene and cover duration/Step-count ranks deterministically."""
    selected: list[str] = []
    for scene, quota in sorted(quotas.items()):
        candidates = [records_by_id[value] for value in allowed_ids if records_by_id[value]["scene"] == scene]
        if len(candidates) < quota:
            raise ValueError(f"not enough {scene} trajectories for quota {quota}")
        frame_order = sorted(candidates, key=lambda x: (int(x["num_frames"]), _stable_tie(seed, str(x["trajectory_id"]))))
        step_order = sorted(candidates, key=lambda x: (int(x["num_steps"]), _stable_tie(seed + 1, str(x["trajectory_id"]))))
        frame_rank = {str(x["trajectory_id"]): i / max(1, len(candidates) - 1) for i, x in enumerate(frame_order)}
        step_rank = {str(x["trajectory_id"]): i / max(1, len(candidates) - 1) for i, x in enumerate(step_order)}
        frame_targets = [(i + 0.5) / quota for i in range(quota)]
        step_targets = list(frame_targets)
        random.Random(seed + sum(map(ord, scene))).shuffle(step_targets)
        remaining = {str(x["trajectory_id"]) for x in candidates}
        for frame_target, step_target in zip(frame_targets, step_targets):
            choice = min(
                remaining,
                key=lambda value: (
                    abs(frame_rank[value] - frame_target)
                    + abs(step_rank[value] - step_target),
                    _stable_tie(seed, value),
                ),
            )
            selected.append(choice)
            remaining.remove(choice)
    return sorted(selected)


def build_episode_anchors(
    episode: Mapping[str, object],
    split: str,
    count: int,
    horizon: int = 20,
) -> list[dict]:
    """Subsample the pre-registered R2 balanced anchor grid uniformly."""
    candidates = [
        item
        for item in build_balanced_samples([episode], split)
        if item.frame + horizon < int(episode["num_frames"])
    ]
    candidates.sort(key=lambda item: (item.frame, item.step_index, item.distance_bin))
    if not candidates:
        raise ValueError(f"no valid future anchors for {episode['trajectory_id']}")
    take = min(count, len(candidates))
    indices = np.linspace(0, len(candidates) - 1, num=take, dtype=int)
    unique_indices = list(dict.fromkeys(indices.tolist()))
    return [
        {
            "frame": candidates[index].frame,
            "target_frame": candidates[index].frame + horizon,
            "step_index": candidates[index].step_index,
            "distance_bin": candidates[index].distance_bin,
            "current_target": candidates[index].current_target,
            "previous_1_target": candidates[index].previous_1_target,
        }
        for index in unique_indices
    ]


def subset_statistics(
    ids: Iterable[str], records_by_id: Mapping[str, Mapping[str, object]]
) -> dict:
    rows = [records_by_id[value] for value in ids]
    return {
        "count": len(rows),
        "scene_counts": dict(sorted(Counter(str(x["scene"]) for x in rows).items())),
        "frames": {
            "total": sum(int(x["num_frames"]) for x in rows),
            "min": min(int(x["num_frames"]) for x in rows),
            "mean": sum(int(x["num_frames"]) for x in rows) / len(rows),
            "max": max(int(x["num_frames"]) for x in rows),
        },
        "steps": {
            "min": min(int(x["num_steps"]) for x in rows),
            "mean": sum(int(x["num_steps"]) for x in rows) / len(rows),
            "max": max(int(x["num_steps"]) for x in rows),
        },
    }


def collapse_statistics(representations: "object") -> dict:
    """Return deterministic collapse diagnostics for a [N,D] torch tensor."""
    import torch
    from torch.nn import functional as F

    if representations.ndim != 2 or representations.shape[0] < 2:
        raise ValueError("representations must have shape [N,D] with N>=2")
    values = representations.detach().float()
    normalized = F.normalize(values, dim=-1)
    similarity = normalized @ normalized.transpose(0, 1)
    off_diagonal = similarity[~torch.eye(len(values), dtype=torch.bool, device=values.device)]
    per_dim_std = values.std(dim=0, unbiased=False)
    mean_std = float(per_dim_std.mean())
    pair_std = float(off_diagonal.std(unbiased=False))
    return {
        "mean_norm": float(values.norm(dim=-1).mean()),
        "mean_per_dimension_std": mean_std,
        "pairwise_cosine_mean": float(off_diagonal.mean()),
        "pairwise_cosine_std": pair_std,
        "collapsed": bool(mean_std < 1e-3 or pair_std < 1e-3),
        "thresholds": {
            "minimum_mean_per_dimension_std": 1e-3,
            "minimum_pairwise_cosine_std": 1e-3,
        },
    }
