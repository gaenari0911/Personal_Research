#!/usr/bin/env python3
"""Build the deterministic split and Task-4 interface statistics."""

import argparse
import csv
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean, median
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from libero_phase1.interface import (  # noqa: E402
    METHODS,
    METHOD_HOLD,
    AnnotationStore,
    Phase1TrajectoryInterface,
    build_action_timeline,
)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def build_split(store, path, seed, force):
    if path.exists() and not force:
        raise FileExistsError(
            f"refusing to mutate existing split {path}; use a new version/path or --force"
        )
    by_task = defaultdict(list)
    for annotation in store.iter_annotations(eligible_only=True):
        by_task[int(annotation["task_id"])].append(annotation["demo_id"])
    rng = random.Random(seed)
    manifest = {
        "split_version": "libero10_phase1_v1",
        "seed": seed,
        "creation_timestamp": datetime.now(ZoneInfo("Asia/Seoul")).isoformat(),
        "unit": "trajectory",
        "eligibility_rule": "conditioning_eligible == true AND needs_review == false",
        "annotation_version": store.version,
        "annotation_manifest_sha256": sha256(store.manifest_path),
        "counts": {"train": 120, "val": 15, "test": 15},
    }
    for task_id in sorted(by_task):
        demo_ids = sorted(by_task[task_id], key=lambda x: int(x.rsplit("_", 1)[-1]))
        if len(demo_ids) != 50:
            raise ValueError(f"task {task_id} has {len(demo_ids)} eligible demos, expected 50")
        rng.shuffle(demo_ids)
        manifest[f"task_{task_id}"] = {
            "train": sorted(demo_ids[:40], key=lambda x: int(x.rsplit("_", 1)[-1])),
            "val": sorted(demo_ids[40:45], key=lambda x: int(x.rsplit("_", 1)[-1])),
            "test": sorted(demo_ids[45:], key=lambda x: int(x.rsplit("_", 1)[-1])),
        }
    write_json(path, manifest)
    return manifest


def build_stats(interface, output_json, output_csv):
    per_task = {}
    for annotation in interface.annotations.iter_annotations(eligible_only=True):
        task_id = int(annotation["task_id"])
        task = per_task.setdefault(
            task_id,
            {
                "demonstrations": 0,
                "total_action_steps": 0,
                "method_semantic_type_counts": {method: Counter() for method in METHODS},
                "current_subtask_action_counts": Counter(),
                "hold_semantic_events": 0,
                "hold_ratios": [],
                "retention_lengths": [],
                "boundary_crossing_chunks": 0,
                "policy_decision_steps": 0,
            },
        )
        trajectory = interface.load_trajectory(
            task_id,
            annotation["demo_id"],
            METHOD_HOLD,
            purpose="inspection",
            include_observations=False,
        )
        full_instruction = trajectory["full_instruction"]
        length = int(annotation["trajectory_length"])
        task["demonstrations"] += 1
        task["total_action_steps"] += length
        for method in METHODS:
            timeline = build_action_timeline(annotation, method, full_instruction)
            task["method_semantic_type_counts"][method].update(
                item["semantic_type"] for item in timeline
            )
            if method == "current_subinstruction":
                task["current_subtask_action_counts"].update(
                    str(item["current_subtask_id"]) for item in timeline
                )
        event_count = len(annotation["subtasks"])
        task["hold_semantic_events"] += event_count
        task["hold_ratios"].append(event_count / length)
        s2_start = int(annotation["subtasks"][1]["action_start"])
        retention = length - s2_start
        recorded = int(annotation["transition_metadata"]["retention_length_steps"])
        if retention != recorded:
            raise ValueError(
                f"retention mismatch task_{task_id}/{annotation['demo_id']}: {retention} != {recorded}"
            )
        task["retention_lengths"].append(retention)
        task["boundary_crossing_chunks"] += int(
            trajectory["boundary_crossing_horizon"].sum()
        )
        task["policy_decision_steps"] += int(trajectory["policy_length"])

    rows = []
    serializable = {
        "oracle_status": interface.validation_status()["oracle_status"],
        "action_horizon": interface.action_horizon,
        "sampling_hz": 20,
        "policy_alignment": "obs[t] -> action[t+1]",
        "statistics_unit": "raw action timeline unless field says policy decision",
        "tasks": {},
    }
    for task_id in sorted(per_task):
        task = per_task[task_id]
        method_counts = {
            method: dict(sorted(counts.items()))
            for method, counts in task["method_semantic_type_counts"].items()
        }
        hold_ratio_mean = mean(task["hold_ratios"])
        hold_ratio_median = median(task["hold_ratios"])
        retention_mean = mean(task["retention_lengths"])
        retention_median = median(task["retention_lengths"])
        crossing_ratio = task["boundary_crossing_chunks"] / task["policy_decision_steps"]
        value = {
            "demonstrations": task["demonstrations"],
            "total_action_steps": task["total_action_steps"],
            "method_semantic_type_counts": method_counts,
            "current_subtask_action_counts": dict(
                sorted(task["current_subtask_action_counts"].items())
            ),
            "hold_semantic_event_steps": task["hold_semantic_events"],
            "hold_semantic_event_steps_per_demo": task["hold_semantic_events"]
            / task["demonstrations"],
            "hold_sparse_event_ratio_mean": hold_ratio_mean,
            "hold_sparse_event_ratio_median": hold_ratio_median,
            "retention_length_mean_steps": retention_mean,
            "retention_length_median_steps": retention_median,
            "boundary_crossing_chunks": task["boundary_crossing_chunks"],
            "policy_decision_steps": task["policy_decision_steps"],
            "boundary_crossing_chunk_ratio": crossing_ratio,
        }
        serializable["tasks"][f"task_{task_id}"] = value
        rows.append({"task_id": task_id, **value})
    write_json(output_json, serializable)

    columns = [
        "task_id",
        "demonstrations",
        "total_action_steps",
        "hold_semantic_event_steps",
        "hold_semantic_event_steps_per_demo",
        "hold_sparse_event_ratio_mean",
        "hold_sparse_event_ratio_median",
        "retention_length_mean_steps",
        "retention_length_median_steps",
        "boundary_crossing_chunks",
        "policy_decision_steps",
        "boundary_crossing_chunk_ratio",
    ]
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as sink:
        writer = csv.DictWriter(sink, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return serializable


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    annotation_root = ROOT / "annotations/libero10_semantic"
    split_path = ROOT / "splits/libero10_phase1_split.json"
    store = AnnotationStore(annotation_root)
    build_split(store, split_path, args.seed, args.force)
    interface = Phase1TrajectoryInterface(
        dataset_root=Path("/ssd1/itaein/datasets/LIBERO/libero_10"),
        annotation_root=annotation_root,
        validation_status_path=annotation_root / "validation_status.json",
        split_manifest_path=split_path,
        action_horizon=10,
    )
    stats = build_stats(
        interface,
        ROOT / "analysis/conditioning_interface_stats.json",
        ROOT / "analysis/conditioning_interface_stats.csv",
    )
    print(
        f"wrote {split_path}; tasks={len(stats['tasks'])}; "
        f"demos={sum(x['demonstrations'] for x in stats['tasks'].values())}"
    )


if __name__ == "__main__":
    main()
