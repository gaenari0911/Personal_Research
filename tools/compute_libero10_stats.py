#!/usr/bin/env python3
"""Compute trajectory/action statistics without modifying LIBERO HDF5 files."""

import argparse
import csv
import json
from pathlib import Path

import h5py
import numpy as np


TASKS = {
    0: "LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket",
    1: "LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket",
    2: "KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it",
    3: "KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it",
    4: "LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate",
    5: "STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy",
    6: "LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate",
    7: "LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket",
    8: "KITCHEN_SCENE8_put_both_moka_pots_on_the_stove",
    9: "KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it",
}


def find_file(dataset_dir, task_name):
    matches = sorted(dataset_dir.glob(f"{task_name}_demo.hdf5"))
    if len(matches) != 1:
        raise FileNotFoundError(f"expected one HDF5 for {task_name}, found {len(matches)}")
    return matches[0]


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def numeric_demo_sort(keys):
    return sorted(keys, key=lambda value: int(value.rsplit("_", 1)[1]))


def representative_indices(demos):
    ordered = sorted(demos, key=lambda item: (item["length"], item["demo_id"]))
    return {
        "short": ordered[0],
        "median": ordered[len(ordered) // 2],
        "long": ordered[-1],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    trajectory_rows = []
    action_rows = []
    stage_proxy_rows = []
    detail = {"dataset_dir": str(args.dataset_dir.resolve()), "tasks": {}}
    representatives = {"dataset_dir": str(args.dataset_dir.resolve()), "tasks": {}}

    for task_id, task_name in TASKS.items():
        path = find_file(args.dataset_dir, task_name)
        all_actions = []
        demos = []
        with h5py.File(path, "r") as handle:
            data = handle["data"]
            env_args = json.loads(data.attrs["env_args"])
            demo_ids = numeric_demo_sort(key for key in data.keys() if key.startswith("demo_"))
            for demo_id in demo_ids:
                demo = data[demo_id]
                actions = demo["actions"][()]
                rewards = demo["rewards"][()]
                dones = demo["dones"][()]
                all_actions.append(actions)
                gripper = actions[:, 6]
                sign = np.where(gripper >= 0.0, 1, -1)
                transitions = np.flatnonzero(sign[1:] != sign[:-1]) + 1
                positive_reward = np.flatnonzero(rewards > 0)
                demos.append(
                    {
                        "demo_id": demo_id,
                        "length": int(actions.shape[0]),
                        "duration_seconds_at_20hz": float(actions.shape[0] / 20.0),
                        "gripper_transition_count": int(transitions.size),
                        "gripper_transition_indices": transitions.tolist(),
                        "first_positive_reward": (
                            int(positive_reward[0]) if positive_reward.size else None
                        ),
                        "positive_reward_count": int(positive_reward.size),
                        "done_indices": np.flatnonzero(dones > 0).tolist(),
                    }
                )
            actions = np.concatenate(all_actions, axis=0)
            lengths = np.asarray([item["length"] for item in demos], dtype=np.float64)
            transitions_per_demo = np.asarray(
                [item["gripper_transition_count"] for item in demos], dtype=np.float64
            )
            trajectory_rows.append(
                {
                    "task_id": task_id,
                    "task_name": task_name,
                    "num_demos": len(demos),
                    "total_steps": int(lengths.sum()),
                    "min": int(lengths.min()),
                    "p25": float(np.percentile(lengths, 25)),
                    "mean": float(lengths.mean()),
                    "median": float(np.median(lengths)),
                    "p75": float(np.percentile(lengths, 75)),
                    "max": int(lengths.max()),
                    "std": float(lengths.std()),
                    "mean_seconds_at_20hz": float(lengths.mean() / 20.0),
                    "gripper_transitions_mean": float(transitions_per_demo.mean()),
                    "gripper_transitions_min": int(transitions_per_demo.min()),
                    "gripper_transitions_max": int(transitions_per_demo.max()),
                }
            )
            for dim in range(actions.shape[1]):
                values = actions[:, dim]
                action_rows.append(
                    {
                        "task_id": task_id,
                        "task_name": task_name,
                        "dimension": dim,
                        "count": int(values.size),
                        "min": float(values.min()),
                        "p01": float(np.percentile(values, 1)),
                        "p05": float(np.percentile(values, 5)),
                        "p25": float(np.percentile(values, 25)),
                        "mean": float(values.mean()),
                        "median": float(np.median(values)),
                        "p75": float(np.percentile(values, 75)),
                        "p95": float(np.percentile(values, 95)),
                        "p99": float(np.percentile(values, 99)),
                        "max": float(values.max()),
                        "std": float(values.std()),
                    }
                )
            unique_gripper, gripper_counts = np.unique(actions[:, 6], return_counts=True)
            detail["tasks"][str(task_id)] = {
                "task_name": task_name,
                "file": str(path.resolve()),
                "file_size_bytes": path.stat().st_size,
                "language": json.loads(data.attrs["problem_info"])["language_instruction"],
                "env_args": env_args,
                "observation_keys": sorted(data[demo_ids[0]]["obs"].keys()),
                "gripper_unique": unique_gripper.tolist(),
                "gripper_unique_counts": gripper_counts.tolist(),
                "demos": demos,
            }
            if task_id in {0, 3, 4, 6, 9}:
                for item in demos:
                    transitions = item["gripper_transition_indices"]
                    # For these tasks, the second gripper command transition is the
                    # release after the first object placement. It is only a review
                    # proxy, never an Oracle boundary.
                    boundary = transitions[1] if len(transitions) >= 2 else None
                    length = item["length"]
                    crossing = None
                    valid_horizon_starts = max(length - 9, 0)
                    if boundary is not None:
                        crossing = sum(
                            1
                            for start in range(valid_horizon_starts)
                            if start < boundary <= start + 9
                        )
                    stage_proxy_rows.append(
                        {
                            "task_id": task_id,
                            "task_name": task_name,
                            "demo_id": item["demo_id"],
                            "length": length,
                            "proxy_boundary_second_gripper_transition": boundary,
                            "proxy_stage1_steps": boundary,
                            "proxy_stage2_steps": (length - boundary if boundary is not None else None),
                            "proxy_stage1_seconds_at_20hz": (
                                boundary / 20.0 if boundary is not None else None
                            ),
                            "proxy_stage2_seconds_at_20hz": (
                                (length - boundary) / 20.0 if boundary is not None else None
                            ),
                            "full_horizon_start_count": valid_horizon_starts,
                            "horizon10_crossing_start_count": crossing,
                            "horizon10_crossing_fraction": (
                                crossing / valid_horizon_starts
                                if crossing is not None and valid_horizon_starts
                                else None
                            ),
                        }
                    )
            representatives["tasks"][str(task_id)] = {
                "task_name": task_name,
                **representative_indices(demos),
            }

    write_csv(
        args.output_dir / "libero10_trajectory_stats.csv",
        trajectory_rows,
        list(trajectory_rows[0]),
    )
    write_csv(
        args.output_dir / "libero10_action_stats.csv",
        action_rows,
        list(action_rows[0]),
    )
    write_csv(
        args.output_dir / "libero10_preliminary_stage_proxy_stats.csv",
        stage_proxy_rows,
        list(stage_proxy_rows[0]),
    )
    (args.output_dir / "libero10_demo_details.json").write_text(
        json.dumps(detail, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.output_dir / "representative_demos.json").write_text(
        json.dumps(representatives, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
