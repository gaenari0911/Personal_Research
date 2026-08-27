#!/usr/bin/env python3
"""Build the R1.5 reconciliation baseline without fabricating annotations.

When the official annotation archive is unavailable, this records deterministic
fingerprints for every existing episode.  Those fingerprints are the matching
keys to use when the official endpoint becomes available; episode indices alone
must never be assumed to be stable.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


REQUIRED_FIELDS = (
    "annotation.human.task_description",
    "annotation.human.task_name",
    "annotation.human.subtask",
    "annotation.human.subtask_name",
    "annotation.human.subtask_stage",
    "subtask_idx",
)


def _array(table, name: str, dtype=None) -> np.ndarray:
    return np.asarray(table[name].to_pylist(), dtype=dtype)


def episode_fingerprint(path: Path) -> dict:
    table = pq.read_table(
        path,
        columns=[
            "action",
            "frame_index",
            "episode_index",
            "timestamp",
            "annotation.human.task_description",
            "annotation.human.task_name",
        ],
    )
    action = np.ascontiguousarray(_array(table, "action", np.float64))
    frame_index = np.ascontiguousarray(_array(table, "frame_index", np.int64))
    timestamp = np.ascontiguousarray(_array(table, "timestamp", np.float32))
    episode_values = np.unique(_array(table, "episode_index", np.int64))
    if len(episode_values) != 1:
        raise ValueError(f"{path}: multiple episode_index values {episode_values}")
    episode_id = int(episode_values[0])
    if not np.array_equal(frame_index, np.arange(len(table), dtype=np.int64)):
        raise ValueError(f"{path}: frame_index is not contiguous")
    action_hash = hashlib.sha256(action.tobytes(order="C")).hexdigest()
    identity_hash = hashlib.sha256(
        action.tobytes(order="C") + frame_index.tobytes(order="C") + timestamp.tobytes(order="C")
    ).hexdigest()
    task_description = _array(table, "annotation.human.task_description")
    task_name = _array(table, "annotation.human.task_name")
    return {
        "old_episode_id": episode_id,
        "old_num_frames": len(table),
        "old_task_description_id": int(task_description[0]),
        "old_task_name_id": int(task_name[0]),
        "old_action_sha256": action_hash,
        "old_identity_sha256": identity_hash,
    }


def build_blocked_mapping(dataset_root: Path, output: Path) -> None:
    parquet_paths = sorted((dataset_root / "data").glob("*/episode_*.parquet"))
    rows = [episode_fingerprint(path) for path in parquet_paths]
    if len(rows) != 507 or [row["old_episode_id"] for row in rows] != list(range(507)):
        raise ValueError("expected the established 507 episodes indexed 0..506")
    fields = [
        "old_episode_id",
        "old_num_frames",
        "old_task_description_id",
        "old_task_name_id",
        "old_action_sha256",
        "old_identity_sha256",
        "new_episode_id",
        "new_num_frames",
        "new_action_sha256",
        "mapping_status",
        "reason",
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    **row,
                    "new_episode_id": "",
                    "new_num_frames": "",
                    "new_action_sha256": "",
                    "mapping_status": "BLOCKED",
                    "reason": "official_annotation_endpoint_http_404",
                }
            )


def write_empty_runs(output: Path) -> None:
    fields = [
        "episode_id",
        "run_id",
        "start_frame",
        "end_frame",
        "subtask_idx",
        "subtask_name",
        "subtask_stage",
        "subtask_instruction",
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as stream:
        csv.DictWriter(stream, fieldnames=fields).writeheader()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--mapping-output", type=Path, required=True)
    parser.add_argument("--runs-output", type=Path, required=True)
    args = parser.parse_args()
    build_blocked_mapping(args.dataset_root, args.mapping_output)
    write_empty_runs(args.runs_output)
    print(json.dumps({"mapping_rows": 507, "annotation_runs": 0, "status": "BLOCKED"}))


if __name__ == "__main__":
    main()
