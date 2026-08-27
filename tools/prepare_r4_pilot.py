#!/usr/bin/env python3
"""Select R4 pilot trajectories and write shared deterministic anchors."""

from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from robocerebra_memory.pilot import (  # noqa: E402
    build_episode_anchors,
    select_stratified_subset,
    subset_statistics,
)


def write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    records = json.loads(
        (ROOT / "analysis/robocerebra_memory_episode_index.json").read_text()
    )["episodes"]
    by_id = {item["trajectory_id"]: item for item in records}
    parent = {
        split: json.loads((ROOT / f"splits/robocerebra_memory_{split}.json").read_text())
        for split in ("train", "val", "test")
    }
    pilot = {
        "train": select_stratified_subset(
            by_id,
            parent["train"],
            {"coffee_table": 8, "kitchen_table": 4, "study_table": 4},
            42,
        ),
        "val": select_stratified_subset(
            by_id,
            parent["val"],
            {"coffee_table": 2, "kitchen_table": 1, "study_table": 1},
            42,
        ),
    }
    if set(pilot["train"]) & set(pilot["val"]):
        raise RuntimeError("pilot train/val overlap")
    if (set(pilot["train"]) | set(pilot["val"])) & set(parent["test"]):
        raise RuntimeError("R4 pilot includes a forbidden test trajectory")
    for split in ("train", "val"):
        write(
            ROOT / f"splits/robocerebra_r4_pilot_{split}.json",
            {
                "schema_version": "r4-v1",
                "seed": 42,
                "parent_split": split,
                "trajectory_ids": pilot[split],
                "statistics": subset_statistics(pilot[split], by_id),
            },
        )
    anchors = {
        "schema_version": "r4-v1",
        "horizon": 20,
        "anchors_per_episode_cap": 64,
        "source": "R2 balanced trajectory_x_step_x_distance_bin samples",
        "splits": {},
    }
    for split, ids in pilot.items():
        anchors["splits"][split] = {
            trajectory_id: build_episode_anchors(by_id[trajectory_id], split, 64, 20)
            for trajectory_id in ids
        }
    write(ROOT / "analysis/r4_anchor_manifest.json", anchors)
    download = [
        {
            "trajectory_id": trajectory_id,
            "relative_path": by_id[trajectory_id]["visual_source"]["official_relative_path"],
            "local_path": by_id[trajectory_id]["visual_source"]["local_path"],
            "url": "https://huggingface.co/datasets/qiukingballball/RoboCerebra/resolve/main/"
            + by_id[trajectory_id]["visual_source"]["official_relative_path"],
        }
        for trajectory_id in pilot["train"] + pilot["val"]
    ]
    write(ROOT / "analysis/r4_video_download_manifest.json", download)
    print(json.dumps({split: subset_statistics(ids, by_id) for split, ids in pilot.items()}, indent=2))


if __name__ == "__main__":
    main()
