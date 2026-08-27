#!/usr/bin/env python3
"""Create the deterministic Stage A schedule and common initialization."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from robocerebra_memory.models import MemoryExperimentModel  # noqa: E402
from robocerebra_memory.pilot import build_episode_anchors  # noqa: E402
from robocerebra_memory.stage_a import (  # noqa: E402
    VARIANTS,
    atomic_json,
    atomic_torch_save,
    state_dict_sha256,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/stage_a_representation.yaml")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    episodes = json.loads((ROOT / config["dataset"]["episode_index"]).read_text())["episodes"]
    by_id = {row["trajectory_id"]: row for row in episodes}
    splits = {
        split: json.loads((ROOT / config["dataset"][f"{split}_split"]).read_text())
        for split in ("train", "val")
    }
    test_ids = set(json.loads((ROOT / config["dataset"]["forbidden_test_split"]).read_text()))
    if len(splits["train"]) != 734 or len(splits["val"]) != 85:
        raise RuntimeError("Stage A split count mismatch")
    if set(splits["train"]) & set(splits["val"]) or (set(splits["train"]) | set(splits["val"])) & test_ids:
        raise RuntimeError("Stage A split contamination")
    anchors = {}
    for split, ids in splits.items():
        anchors[split] = {}
        for trajectory_id in ids:
            rows = build_episode_anchors(by_id[trajectory_id], split, 64, 20)
            if not 8 <= len(rows) <= 64 or any(row["frame"] + 20 >= by_id[trajectory_id]["num_frames"] for row in rows):
                raise RuntimeError(f"invalid multi-anchor contract for {trajectory_id}")
            anchors[split][trajectory_id] = rows
    epoch_orders = {}
    for epoch in range(1, 4):
        order = list(splits["train"])
        random.Random(42 + epoch - 1).shuffle(order)
        epoch_orders[str(epoch)] = order
    pilot_train = json.loads((ROOT / "splits/robocerebra_r4_pilot_train.json").read_text())["trajectory_ids"]
    smoke_id = max(pilot_train, key=lambda value: (by_id[value]["num_frames"], value))
    schedule = {
        "schema_version": "stage-a-v1",
        "seed": 42,
        "future_horizon": 20,
        "anchors_per_trajectory_cap": 64,
        "minimum_anchors_per_trajectory": 8,
        "train_ids": splits["train"],
        "val_ids": splits["val"],
        "epoch_orders": epoch_orders,
        "anchors": anchors,
        "smoke": {
            "trajectory_id": smoke_id,
            "num_frames": by_id[smoke_id]["num_frames"],
            "anchors": anchors["train"][smoke_id][:8],
            "selection": "longest deterministic trajectory in existing R4 pilot train split",
        },
        "test_split_used": False,
    }
    atomic_json(ROOT / config["dataset"]["schedule"], schedule)

    torch.manual_seed(42)
    base = MemoryExperimentModel("B0")
    state = {key: value.detach().cpu() for key, value in base.state_dict().items()}
    common_hash = state_dict_sha256(state)
    for variant in VARIANTS:
        candidate = MemoryExperimentModel(variant)
        candidate.load_state_dict(state, strict=True)
        if state_dict_sha256(candidate.state_dict()) != common_hash:
            raise RuntimeError(f"common initialization mismatch for {variant}")
    common = {
        "schema_version": "stage-a-v1",
        "seed": 42,
        "architecture": "shared B0/B1/B2/B3 trainable architecture before training",
        "state_dict_sha256": common_hash,
        "model_state_dict": state,
        "trained": False,
        "source_variant_only_for_parameter_construction": "B0",
    }
    common_path = ROOT / config["training"]["common_initialization"]
    atomic_torch_save(common_path, common)
    reopened = torch.load(common_path, map_location="cpu", weights_only=False)
    if reopened["state_dict_sha256"] != common_hash:
        raise RuntimeError("common initialization atomic re-open failed")
    fairness = {
        "schema_version": "stage-a-v1",
        "status": "PLANNED_PASS",
        "common_initialization_sha256": common_hash,
        "same_train_split": True,
        "same_val_split": True,
        "same_epoch_orders": True,
        "same_anchor_ids": True,
        "same_optimizer": config["optimizer"],
        "same_objective": config["objective"],
        "same_epoch_budget": True,
        "checkpoint_independence": True,
        "weight_continuation": False,
        "test_split_used": False,
    }
    atomic_json(ROOT / "analysis/stage_a_fairness_audit.json", fairness)
    print(json.dumps({
        "status": "PASS",
        "schedule": config["dataset"]["schedule"],
        "smoke": schedule["smoke"],
        "common_init": str(common_path),
        "common_initialization_sha256": common_hash,
    }, indent=2))


if __name__ == "__main__":
    main()
