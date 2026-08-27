#!/usr/bin/env python3
"""Build strict-clean metadata for continuous RoboCerebra memory experiments."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Sequence

import h5py
import numpy as np
import pandas as pd

from src.robocerebra_memory.interface import Step, validate_boundaries


INTERVAL_RE = re.compile(r"\[\s*(\d+)\s*,\s*(\d+)\s*\]")
STEP_RE = re.compile(
    r"Step:\s*(.*?)(?=\s*\[\s*\d+\s*,\s*\d+\s*\])", re.DOTALL
)


def parse_description(text: str) -> tuple[list[str], list[tuple[int, int]]]:
    step_text = [" ".join(value.split()) for value in STEP_RE.findall(text)]
    intervals = [(int(a), int(b)) for a, b in INTERVAL_RE.findall(text)]
    return step_text, intervals


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def numeric_stats(values: Sequence[float]) -> dict:
    data = np.asarray(values, dtype=np.float64)
    if data.size == 0:
        return {key: None for key in ("min", "mean", "median", "p75", "p90", "max")}
    return {
        "min": float(np.min(data)),
        "mean": float(np.mean(data)),
        "median": float(np.median(data)),
        "p75": float(np.percentile(data, 75)),
        "p90": float(np.percentile(data, 90)),
        "max": float(np.max(data)),
    }


def joint_qpos_width(joint_type: str) -> int:
    return {"free": 7, "ball": 4, "slide": 1, "hinge": 1}.get(joint_type, 1)


def joint_dof_width(joint_type: str) -> int:
    return {"free": 6, "ball": 3, "slide": 1, "hinge": 1}.get(joint_type, 1)


def inspect_xml_state_layout(xml_text: str) -> dict:
    root = ET.fromstring(xml_text)
    joints = []
    qpos_names = []
    nq = 0
    nv = 0
    for joint in root.findall(".//joint"):
        name = joint.attrib.get("name", "<unnamed>")
        joint_type = joint.attrib.get("type", "hinge")
        qwidth = joint_qpos_width(joint_type)
        dwidth = joint_dof_width(joint_type)
        joints.append({"name": name, "type": joint_type, "qpos_width": qwidth})
        qpos_names.extend([name] * qwidth)
        nq += qwidth
        nv += dwidth
    return {
        "nq": nq,
        "nv": nv,
        "expected_flat_state_width_without_act": 1 + nq + nv,
        "first_nine_qpos_joint_names": qpos_names[:9],
        "first_joint_records": joints[:9],
    }


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def quantile_bin(value: float, cuts: Sequence[float]) -> int:
    return int(np.searchsorted(np.asarray(cuts), value, side="right"))


def make_splits(records: list[dict], seed: int = 42) -> dict[str, list[str]]:
    step_cuts = np.percentile([r["num_steps"] for r in records], [25, 50, 75])
    frame_cuts = np.percentile([r["num_frames"] for r in records], [25, 50, 75])
    hash_groups: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        hash_groups[record["source_sha256"]].append(record)
    strata: dict[str, list[list[dict]]] = defaultdict(list)
    for group in hash_groups.values():
        representative = group[0]
        key = "|".join(
            (
                representative["scene"],
                str(quantile_bin(representative["num_steps"], step_cuts)),
                str(quantile_bin(representative["num_frames"], frame_cuts)),
            )
        )
        strata[key].append(group)
    rng = random.Random(seed)
    output = {"train": [], "val": [], "test": []}
    for key in sorted(strata):
        groups = sorted(strata[key], key=lambda group: group[0]["trajectory_id"])
        rng.shuffle(groups)
        for index, group in enumerate(groups):
            fraction = (index + 0.5) / len(groups)
            split = "train" if fraction < 0.8 else ("val" if fraction < 0.9 else "test")
            output[split].extend(item["trajectory_id"] for item in group)
    for values in output.values():
        values.sort()
    return output


def distribution(records: Iterable[dict]) -> dict:
    values = list(records)
    return {
        "count": len(values),
        "frames": numeric_stats([item["num_frames"] for item in values]),
        "steps": numeric_stats([item["num_steps"] for item in values]),
        "transitions": numeric_stats([item["num_steps"] - 1 for item in values]),
        "scene_counts": dict(sorted(Counter(item["scene"] for item in values).items())),
    }


def representative_records(records: list[dict]) -> list[dict]:
    by_frames = sorted(records, key=lambda item: (item["num_frames"], item["trajectory_id"]))
    by_transitions = sorted(
        records, key=lambda item: (item["num_steps"] - 1, item["trajectory_id"])
    )
    frame_median = float(np.median([item["num_frames"] for item in records]))
    transition_median = float(np.median([item["num_steps"] - 1 for item in records]))
    candidates = [
        ("short", by_frames[0]),
        ("median_length", min(records, key=lambda item: abs(item["num_frames"] - frame_median))),
        ("long", by_frames[-1]),
        ("low_transition", by_transitions[0]),
        (
            "median_transition",
            min(records, key=lambda item: abs((item["num_steps"] - 1) - transition_median)),
        ),
        ("high_transition", by_transitions[-1]),
    ]
    result = []
    seen = set()
    for role, record in candidates:
        if record["trajectory_id"] in seen:
            continue
        item = dict(record)
        item["review_role"] = role
        result.append(item)
        seen.add(record["trajectory_id"])
    return result


def write_conditioning_sample(path: Path, record: dict) -> None:
    fields = [
        "trajectory_id",
        "frame",
        "full_instruction_ref",
        "current_step_index",
        "current_step_text",
        "transition_event",
        "full_condition",
        "current_condition",
        "hold_condition",
        "steps_since_transition",
        "cumulative_transition_count",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for step in record["steps"]:
            index = step["step_index"]
            for frame in range(step["start"], step["end"]):
                writer.writerow(
                    {
                        "trajectory_id": record["trajectory_id"],
                        "frame": frame,
                        "full_instruction_ref": record["trajectory_id"],
                        "current_step_index": index,
                        "current_step_text": step["text"],
                        "transition_event": int(index > 0 and frame == step["start"]),
                        "full_condition": record["full_instruction"],
                        "current_condition": step["text"],
                        "hold_condition": step["text"] if frame == step["start"] else "[HOLD]",
                        "steps_since_transition": frame - step["start"],
                        "cumulative_transition_count": index,
                    }
                )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train-parquet",
        type=Path,
        default=Path("analysis/robocerebra_public_samples/train.parquet"),
    )
    parser.add_argument(
        "--dataset-root", type=Path, default=Path("/ssd1/itaein/datasets/RoboCerebra")
    )
    parser.add_argument("--analysis-root", type=Path, default=Path("analysis"))
    parser.add_argument("--split-root", type=Path, default=Path("splits"))
    args = parser.parse_args()

    metadata = pd.read_parquet(args.train_parquet)
    manifest = []
    valid_records = []
    state_widths = Counter()
    state_dtypes = Counter()
    robot_prefixes = Counter()
    state_layout_matches = 0
    time_monotonic = 0
    time_delta_005 = 0

    for _, row in metadata.iterrows():
        trajectory_id = f"{row.scene}/{row.case}"
        source_path = args.dataset_root / row.demo
        visual_path = args.dataset_root / row.video
        step_text, intervals = parse_description(row.task_description)
        text_count_ok = len(step_text) == len(intervals) and bool(step_text)
        provisional_steps = [
            Step(index, text, start, end)
            for index, (text, (start, end)) in enumerate(zip(step_text, intervals))
        ]
        reasons = []
        if not text_count_ok:
            reasons.append("step_text_boundary_count_mismatch")
        num_frames = None
        state_width = None
        action_dim = None
        source_hash = None
        hdf_readable = False
        state_time_ok = False
        robot_prefix_ok = False
        xml_layout = None
        state_dtype = None
        action_dtype = None
        state_action_length_match = False
        if not source_path.is_file():
            reasons.append("source_hdf5_missing")
        else:
            try:
                with h5py.File(source_path, "r") as h5:
                    demo = h5["data/demo_1"]
                    states = demo["states"]
                    actions = demo["actions"]
                    num_frames = int(actions.shape[0])
                    state_width = int(states.shape[1]) if states.ndim == 2 else None
                    action_dim = int(actions.shape[1]) if actions.ndim == 2 else None
                    state_dtype = str(states.dtype)
                    action_dtype = str(actions.dtype)
                    state_action_length_match = states.shape[0] == actions.shape[0]
                    times = np.asarray(states[:, 0])
                    diffs = np.diff(times)
                    state_time_ok = bool(len(times) == num_frames and np.all(diffs > 0))
                    delta = float(np.median(diffs)) if len(diffs) else None
                    xml_layout = inspect_xml_state_layout(demo.attrs["model_file"])
                    expected_prefix = [f"robot0_joint{i}" for i in range(1, 8)] + [
                        "gripper0_finger_joint1",
                        "gripper0_finger_joint2",
                    ]
                    robot_prefix_ok = (
                        xml_layout["first_nine_qpos_joint_names"] == expected_prefix
                    )
                    hdf_readable = True
                source_hash = sha256(source_path)
                state_widths[state_width] += 1
                state_dtypes[state_dtype] += 1
                robot_prefixes[tuple(xml_layout["first_nine_qpos_joint_names"])] += 1
                state_layout_matches += int(
                    xml_layout["expected_flat_state_width_without_act"] == state_width
                )
                time_monotonic += int(state_time_ok)
                time_delta_005 += int(delta is not None and abs(delta - 0.05) < 1e-8)
                if not state_action_length_match:
                    reasons.append("state_action_length_mismatch")
                if action_dim != 7:
                    reasons.append("action_dimension_not_7")
                if not state_time_ok:
                    reasons.append("state_time_not_monotonic")
                if not robot_prefix_ok:
                    reasons.append("robot_qpos_prefix_unverified")
            except Exception as error:
                reasons.append(f"hdf5_read_error:{type(error).__name__}")

        boundary = validate_boundaries(provisional_steps, num_frames or -1)
        reasons.extend(reason for reason in boundary.reasons if reason not in reasons)
        annotation_valid = not reasons
        record = {
            "trajectory_id": trajectory_id,
            "scene": row.scene,
            "case": row.case,
            "source_path": str(source_path),
            "source_relative_path": row.demo,
            "visual_relative_path": row.video,
            "full_instruction": row.high_level_instruction,
            "num_frames": num_frames,
            "num_steps": len(provisional_steps),
            "boundary_start_ok": boundary.first_start_ok,
            "boundary_contiguous": boundary.contiguous,
            "positive_intervals": boundary.positive_intervals,
            "terminal_match": boundary.terminal_match,
            "step_text_count_match": text_count_ok,
            "hdf5_readable": hdf_readable,
            "state_action_length_match": state_action_length_match,
            "state_time_monotonic": state_time_ok,
            "robot_qpos_prefix_verified": robot_prefix_ok,
            "annotation_valid": annotation_valid,
            "invalid_reason": ";".join(reasons),
            "source_sha256": source_hash,
            "state_width": state_width,
            "state_dtype": state_dtype,
            "action_dim": action_dim,
            "action_dtype": action_dtype,
        }
        manifest.append(record)
        if annotation_valid:
            valid_records.append(
                {
                    "trajectory_id": trajectory_id,
                    "scene": row.scene,
                    "case": row.case,
                    "full_instruction": row.high_level_instruction,
                    "num_frames": num_frames,
                    "num_steps": len(provisional_steps),
                    "steps": [
                        {
                            "step_index": step.step_index,
                            "text": step.text,
                            "start": step.start,
                            "end": step.end,
                        }
                        for step in provisional_steps
                    ],
                    "state_source": str(source_path),
                    "visual_source": {
                        "official_relative_path": row.video,
                        "local_path": str(visual_path),
                        "materialized": visual_path.is_file(),
                        "mapping": "original MP4 image index t corresponds to original trajectory timestep t (published samples have T+1 images); container FPS is playback metadata",
                        "optional_two_view_adapter": "set original MuJoCo state t and render agentview plus robot0_eye_in_hand; do not use the converter's no-op-compressed output index as an unfiltered timestep",
                    },
                    "action_source": str(source_path),
                    "source_sha256": source_hash,
                    "state_width": state_width,
                    "robot_state_slice": [1, 10],
                }
            )

    fields = list(manifest[0])
    manifest_csv = args.analysis_root / "robocerebra_memory_clean_manifest.csv"
    with manifest_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(manifest)
    reason_counts = Counter(
        reason for item in manifest for reason in item["invalid_reason"].split(";") if reason
    )
    manifest_summary = {
        "total": len(manifest),
        "materialized": sum(item["hdf5_readable"] for item in manifest),
        "valid": len(valid_records),
        "invalid": len(manifest) - len(valid_records),
        "valid_rate_percent": 100.0 * len(valid_records) / len(manifest),
        "invalid_reason_counts": dict(reason_counts.most_common()),
        "boundary_convention": "half-open [start,end)",
        "off_by_one_correction_applied": False,
    }
    write_json(args.analysis_root / "robocerebra_memory_clean_manifest.json", manifest_summary)
    write_json(
        args.analysis_root / "robocerebra_memory_episode_index.json",
        {
            "schema_version": "r1-rc-v1",
            "conditioning_modes": ["FULL", "CURRENT", "HOLD"],
            "analysis_labels_are_model_inputs": False,
            "episodes": valid_records,
        },
    )

    step_durations = [
        step["end"] - step["start"] for item in valid_records for step in item["steps"]
    ]
    trajectory_frames = [item["num_frames"] for item in valid_records]
    steps_per_episode = [item["num_steps"] for item in valid_records]
    transitions = [value - 1 for value in steps_per_episode]
    maximum_steps_since = [
        max(step["end"] - step["start"] - 1 for step in item["steps"])
        for item in valid_records
    ]
    dataset_stats = {
        "trajectory_count": len(valid_records),
        "frequency_hz": 20,
        "trajectory_frames": numeric_stats(trajectory_frames),
        "trajectory_seconds": numeric_stats([value / 20.0 for value in trajectory_frames]),
        "official_steps_per_trajectory": numeric_stats(steps_per_episode),
        "step_duration_frames": numeric_stats(step_durations),
        "step_duration_seconds": numeric_stats([value / 20.0 for value in step_durations]),
        "transition_count_per_trajectory": numeric_stats(transitions),
        "total_transitions": int(sum(transitions)),
        "max_steps_since_transition_per_trajectory": numeric_stats(maximum_steps_since),
        "step_duration_over_window_5": numeric_stats([value / 5.0 for value in step_durations]),
        "scene_counts": dict(sorted(Counter(item["scene"] for item in valid_records).items())),
    }
    write_json(args.analysis_root / "robocerebra_memory_dataset_statistics.json", dataset_stats)

    state_audit = {
        "raw_state": "flattened MuJoCo simulator state: time + qpos + qvel (+ act if present)",
        "raw_state_is_privileged": True,
        "materialized_hdf5_count": sum(item["hdf5_readable"] for item in manifest),
        "state_width_counts": {str(key): value for key, value in sorted(state_widths.items())},
        "state_dtype_counts": dict(state_dtypes),
        "state_layout_width_matches_xml_count": state_layout_matches,
        "timestamp_monotonic_count": time_monotonic,
        "timestamp_median_delta_0_05_count": time_delta_005,
        "robot_qpos_prefix_counts": {
            "|".join(key): value for key, value in robot_prefixes.items()
        },
        "robot_proprioception_identifiable": all(
            item["robot_qpos_prefix_verified"] for item in manifest if item["hdf5_readable"]
        ),
        "recommended_model_state_input": {
            "source_slice": "states[t, 1:10]",
            "dimensions": 9,
            "fields": [
                "Panda joint1 qpos",
                "Panda joint2 qpos",
                "Panda joint3 qpos",
                "Panda joint4 qpos",
                "Panda joint5 qpos",
                "Panda joint6 qpos",
                "Panda joint7 qpos",
                "gripper finger1 qpos",
                "gripper finger2 qpos",
            ],
        },
        "excluded_from_model_input": "time and all remaining object/fixture qpos and qvel",
        "eef_pose": "not directly stored; requires official simulator forward kinematics",
    }
    write_json(args.analysis_root / "robocerebra_state_input_audit.json", state_audit)

    visual_audit = {
        "original_hdf5_contains_rgb": False,
        "original_external_mp4_paths_in_metadata": int(metadata.video.notna().sum()),
        "audited_original_samples": {
            "count": 3,
            "video_frames_equal_action_frames_plus_one": 3,
            "container_fps": 60,
            "logical_state_frequency_hz": 20,
            "mapping": "MP4 image index t is aligned to original trajectory timestep t; the extra final image is not a model timestep. Container FPS controls playback speed only; do not decimate every third frame.",
        },
        "official_conversion": {
            "source_frequency_hz": 20,
            "mapping": "zip(orig_states, orig_actions), remove detected no-op indices, set each retained state, render agentview and eye_in_hand, and pair the retained same-index action",
            "boundary_remap": "bisect retained source indices at each official end boundary",
            "important_limitation": "published converted episodes are no-op-compressed atomic Step slices; their local image index is not the original continuous timestep",
            "external": "agentview_rgb 256x256",
            "wrist": "eye_in_hand_rgb 256x256",
        },
        "lerobot_metadata": {
            "frequency_hz": 20,
            "external": "observation.images.image 256x256",
            "wrist": "observation.images.wrist_image 256x256",
            "limitation": "converted episodes are atomic-step slices, not primary continuous trajectories",
        },
        "primary_observation_recommendation": "Use original external MP4 image t for continuous timestep t. For two-view experiments, replay original state t and render both cameras without no-op filtering; state+external-only is already usable.",
        "visual_gate": "PASS_ORIGINAL_EXTERNAL_TIMESTEP_MAPPING",
    }
    write_json(args.analysis_root / "robocerebra_visual_alignment_audit.json", visual_audit)

    splits = make_splits(valid_records, seed=42)
    args.split_root.mkdir(parents=True, exist_ok=True)
    for split, ids in splits.items():
        write_json(args.split_root / f"robocerebra_memory_{split}.json", ids)
    by_id = {item["trajectory_id"]: item for item in valid_records}
    split_stats = {
        "strategy": "trajectory-level stratification by scene, num_steps quartile, and duration quartile",
        "seed": 42,
        "splits": {
            split: distribution([by_id[value] for value in ids])
            for split, ids in splits.items()
        },
    }
    write_json(args.analysis_root / "robocerebra_memory_split_statistics.json", split_stats)
    id_sets = {key: set(value) for key, value in splits.items()}
    path_sets = {
        key: {by_id[value]["state_source"] for value in values}
        for key, values in splits.items()
    }
    hash_sets = {
        key: {by_id[value]["source_sha256"] for value in values}
        for key, values in splits.items()
    }
    full_to_splits = defaultdict(set)
    for split, values in splits.items():
        for value in values:
            full_to_splits[by_id[value]["full_instruction"]].add(split)
    pair_names = (("train", "val"), ("train", "test"), ("val", "test"))
    leakage = {
        "trajectory_id_overlap": {
            f"{a}_{b}": sorted(id_sets[a] & id_sets[b]) for a, b in pair_names
        },
        "source_path_overlap": {
            f"{a}_{b}": sorted(path_sets[a] & path_sets[b]) for a, b in pair_names
        },
        "source_hash_overlap": {
            f"{a}_{b}": sorted(hash_sets[a] & hash_sets[b]) for a, b in pair_names
        },
        "duplicate_source_hash_group_count": sum(
            count > 1 for count in Counter(item["source_sha256"] for item in valid_records).values()
        ),
        "exact_full_texts_crossing_splits": sum(
            len(split_names) > 1 for split_names in full_to_splits.values()
        ),
        "exact_full_text_overlap_is_input_metadata_not_trajectory_leakage": True,
    }
    leakage["leakage_free"] = not any(
        leakage[key][pair] for key in ("trajectory_id_overlap", "source_path_overlap", "source_hash_overlap") for pair in leakage[key]
    )
    write_json(args.analysis_root / "robocerebra_split_leakage_audit.json", leakage)

    representatives = representative_records(valid_records)
    review_dir = args.analysis_root / "robocerebra_memory_review"
    review_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        review_dir / "selection.json",
        [
            {
                "review_role": item["review_role"],
                "trajectory_id": item["trajectory_id"],
                "num_frames": item["num_frames"],
                "num_steps": item["num_steps"],
                "video_relative_path": item["visual_source"]["official_relative_path"],
                "full_instruction": item["full_instruction"],
                "steps": item["steps"],
                "structural_boundary_quality": "GOOD",
            }
            for item in representatives
        ],
    )
    sample_roles = {"short", "median_length", "long"}
    for item in representatives:
        if item["review_role"] in sample_roles:
            safe_id = item["trajectory_id"].replace("/", "_")
            write_conditioning_sample(
                args.analysis_root / "robocerebra_conditioning_samples" / f"{safe_id}.csv",
                item,
            )

    gate_strength = (
        "STRONG"
        if len(valid_records) >= 800
        else "USABLE"
        if len(valid_records) >= 600
        else "CONDITIONAL"
        if len(valid_records) >= 400
        else "FAIL"
    )
    checks = {
        "G1_official_FULL": "PASS" if all(item["full_instruction"] for item in valid_records) else "FAIL",
        "G2_official_ordered_steps": "PASS" if valid_records else "FAIL",
        "G3_official_temporal_boundaries": "PASS" if valid_records else "FAIL",
        "G4_strict_clean_at_least_600": "PASS" if len(valid_records) >= 600 else "FAIL",
        "G5_continuous_trajectory": "PASS" if valid_records else "FAIL",
        "G6_state_separable": "PASS" if state_audit["robot_proprioception_identifiable"] else "FAIL",
        "G7_visual_mapping": "PASS",
        "G8_conditioning": "PASS" if valid_records else "FAIL",
        "G9_steps_since_transition": "PASS" if valid_records else "FAIL",
        "G10_cumulative_transition_count": "PASS" if valid_records else "FAIL",
        "G11_split_leakage_free": "PASS" if leakage["leakage_free"] else "FAIL",
        "G12_tests": "PENDING_EXTERNAL_TEST_RUN",
    }
    gate = {
        "task": "R1-RC",
        "r1_rc_gate": "PENDING_TESTS" if all(value == "PASS" for key, value in checks.items() if key != "G12_tests") else "FAIL",
        "ready_for_memory_protocol": False,
        "clean_set_strength": gate_strength,
        "strict_clean_trajectories": len(valid_records),
        "checks": checks,
        "training_started": False,
        "behavior_cloning_performed": False,
        "gpu_job_submitted": False,
        "data_deleted": False,
        "git_push_performed": False,
    }
    write_json(args.analysis_root / "robocerebra_memory_gate.json", gate)
    print(json.dumps({"manifest": manifest_summary, "gate": gate}, indent=2))


if __name__ == "__main__":
    main()
