#!/usr/bin/env python3
"""Audit a minimal official RoboCerebra sample without simulator replay.

This tool treats the public task-description intervals as atomic-step annotations.
It never promotes them to semantic stages or invents a grouping between them.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import cv2
import h5py
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


INTERVAL_RE = re.compile(r"\[\s*(\d+)\s*,\s*(\d+)\s*\]")
STEP_RE = re.compile(
    r"Step:\s*(.*?)(?=\s*\[\s*\d+\s*,\s*\d+\s*\])", re.DOTALL
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize(text: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9 ]", " ", text.lower()).split())


def atomic_verb(text: str) -> str:
    value = normalize(text)
    prefixes = (
        ("pick up", "pick"),
        ("place down", "place"),
        ("put down", "place"),
        ("turn on", "turn_on"),
        ("turn off", "turn_off"),
        ("pour out", "pour"),
        ("place", "place"),
        ("pour", "pour"),
        ("open", "open"),
        ("close", "close"),
        ("move", "move"),
        ("store", "store"),
        ("return", "return"),
        ("heat", "heat"),
    )
    for prefix, category in prefixes:
        if value.startswith(prefix):
            return category
    return value.split()[0] if value else ""


def parse_description(text: str) -> tuple[list[str], list[tuple[int, int]]]:
    instructions = [" ".join(item.split()) for item in STEP_RE.findall(text)]
    intervals = [(int(start), int(end)) for start, end in INTERVAL_RE.findall(text)]
    if len(instructions) != len(intervals):
        raise ValueError(
            f"step/interval count mismatch: {len(instructions)} != {len(intervals)}"
        )
    return instructions, intervals


def hdf5_tree(path: Path) -> tuple[list[dict], dict]:
    tree: list[dict] = []
    with h5py.File(path, "r") as h5:
        root_attrs = {key: str(value) for key, value in h5.attrs.items()}

        def visitor(name, obj):
            if isinstance(obj, h5py.Dataset):
                tree.append(
                    {
                        "path": f"/{name}",
                        "kind": "dataset",
                        "shape": list(obj.shape),
                        "dtype": str(obj.dtype),
                        "compression": obj.compression,
                    }
                )
            else:
                tree.append(
                    {
                        "path": f"/{name}",
                        "kind": "group",
                        "attributes": sorted(obj.attrs.keys()),
                    }
                )

        h5.visititems(visitor)
    return tree, root_attrs


def inspect_video(path: Path) -> dict:
    cap = cv2.VideoCapture(str(path))
    result = {
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": float(cap.get(cv2.CAP_PROP_FPS)),
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    }
    cap.release()
    result["duration_seconds"] = (
        result["frame_count"] / result["fps"] if result["fps"] else None
    )
    return result


def percentile(values: list[int], q: float) -> float:
    return float(np.percentile(np.asarray(values), q))


def make_review_sheet(
    video_path: Path,
    output_path: Path,
    full_instruction: str,
    instructions: list[str],
    intervals: list[tuple[int, int]],
    annotation_valid: bool,
) -> None:
    cap = cv2.VideoCapture(str(video_path))
    video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    panels: list[Image.Image] = []
    font = ImageFont.load_default()
    for transition_index, (start, _) in enumerate(intervals[1:], start=1):
        for frame_index, side in ((start - 1, "BEFORE"), (start, "AFTER")):
            if frame_index < 0 or frame_index >= video_frames:
                continue
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = cap.read()
            if not ok:
                continue
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame)
            image.thumbnail((320, 256))
            panel = Image.new("RGB", (320, 300), "white")
            panel.paste(image, ((320 - image.width) // 2, 44))
            draw = ImageDraw.Draw(panel)
            draw.text(
                (5, 3),
                f"{side} S{transition_index}->S{transition_index + 1} frame={frame_index}",
                fill="black",
                font=font,
            )
            draw.text(
                (5, 18),
                instructions[transition_index][:48],
                fill="black",
                font=font,
            )
            panels.append(panel)
    cap.release()
    columns = 4
    rows = max(1, (len(panels) + columns - 1) // columns)
    header_height = 72
    sheet = Image.new("RGB", (columns * 320, header_height + rows * 300), "white")
    draw = ImageDraw.Draw(sheet)
    status = "BOUNDARY AUDIT PASS" if annotation_valid else "BOUNDARY LENGTH MISMATCH"
    draw.text((5, 5), status, fill="black", font=font)
    line = full_instruction
    draw.text((5, 22), f"FULL: {line[:175]}", fill="black", font=font)
    if len(line) > 175:
        draw.text((5, 38), line[175:350], fill="black", font=font)
    draw.text(
        (5, 54),
        "NOTE: S labels below are official atomic steps, not research semantic stages.",
        fill="red",
        font=font,
    )
    for index, panel in enumerate(panels):
        x = (index % columns) * 320
        y = header_height + (index // columns) * 300
        sheet.paste(panel, (x, y))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, quality=88)


def write_atomic_timeline(
    path: Path,
    episode_id: str,
    frame_count: int,
    full_instruction: str,
    instructions: list[str],
    intervals: list[tuple[int, int]],
) -> None:
    fields = [
        "episode_id",
        "frame",
        "full_instruction",
        "subtask_id",
        "subtask_instruction",
        "hold_instruction",
        "transition_event",
        "steps_since_transition",
        "cumulative_transition_count",
        "annotation_level",
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for stage_index, (instruction, (start, end)) in enumerate(
            zip(instructions, intervals)
        ):
            if start < 0 or end > frame_count or end <= start:
                raise ValueError(f"invalid timeline interval {(start, end)}")
            for frame in range(start, end):
                writer.writerow(
                    {
                        "episode_id": episode_id,
                        "frame": frame,
                        "full_instruction": full_instruction,
                        "subtask_id": stage_index,
                        "subtask_instruction": instruction,
                        "hold_instruction": instruction if frame == start else "HOLD",
                        "transition_event": (
                            "START_S1"
                            if frame == 0
                            else (
                                f"S{stage_index}_TO_S{stage_index + 1}"
                                if frame == start
                                else ""
                            )
                        ),
                        "steps_since_transition": frame - start,
                        "cumulative_transition_count": stage_index,
                        "annotation_level": "OFFICIAL_ATOMIC_STEP_NOT_SEMANTIC_STAGE",
                    }
                )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sample-root",
        type=Path,
        default=Path("analysis/robocerebra_public_samples"),
    )
    parser.add_argument("--analysis-root", type=Path, default=Path("analysis"))
    args = parser.parse_args()

    train_path = args.sample_root / "train.parquet"
    test_path = args.sample_root / "test.parquet"
    lerobot_info_path = args.sample_root / "lerobot_meta" / "info.json"
    lerobot_tasks_path = args.sample_root / "lerobot_meta" / "tasks.parquet"
    train = pd.read_parquet(train_path)
    test = pd.read_parquet(test_path)
    lerobot_info = json.loads(lerobot_info_path.read_text(encoding="utf-8"))
    lerobot_tasks = pd.read_parquet(lerobot_tasks_path)
    parsed = [parse_description(text) for text in train["task_description"]]
    counts = [len(intervals) for _, intervals in parsed]

    exact_instruction_counts = Counter(train["high_level_instruction"])
    normalized_plans = [tuple(normalize(item) for item in steps) for steps, _ in parsed]
    exact_plan_counts = Counter(normalized_plans)
    verb_patterns = [tuple(atomic_verb(item) for item in steps) for steps, _ in parsed]
    verb_pattern_counts = Counter(verb_patterns)

    anomaly_rows: list[dict] = []
    valid_annotation_trajectory_lengths: list[int] = []
    valid_annotation_atomic_durations: list[int] = []
    start_zero = 0
    contiguous = 0
    positive = 0
    for (_, row), (_, intervals) in zip(train.iterrows(), parsed):
        reasons = []
        if intervals and intervals[0][0] == 0:
            start_zero += 1
        else:
            reasons.append("start_not_zero")
        if intervals and all(
            previous_end == next_start
            for (_, previous_end), (next_start, _) in zip(intervals, intervals[1:])
        ):
            contiguous += 1
        else:
            reasons.append("not_contiguous")
        if intervals and all(end > start for start, end in intervals):
            positive += 1
        else:
            reasons.append("nonpositive_interval")
        if reasons:
            anomaly_rows.append(
                {"scene": row.scene, "case": row.case, "reasons": reasons}
            )
        else:
            valid_annotation_trajectory_lengths.append(intervals[-1][1])
            valid_annotation_atomic_durations.extend(
                end - start for start, end in intervals
            )

    sample_cases = ("case1", "case2", "case10")
    samples: list[dict] = []
    all_actions: list[np.ndarray] = []
    for case in sample_cases:
        directory = args.sample_root / "coffee_table" / case
        h5_path = directory / "demo.hdf5"
        txt_path = directory / "task_description.txt"
        video_path = directory / f"{case}.mp4"
        row = train[(train.scene == "coffee_table") & (train.case == case)].iloc[0]
        instructions, intervals = parse_description(txt_path.read_text(encoding="utf-8"))
        tree, root_attrs = hdf5_tree(h5_path)
        with h5py.File(h5_path, "r") as h5:
            actions = h5["data/demo_1/actions"][()]
            states = h5["data/demo_1/states"][()]
            data_attrs = {key: str(value) for key, value in h5["data"].attrs.items()}
        all_actions.append(actions)
        video = inspect_video(video_path)
        end_matches = intervals[-1][1] == len(actions)
        sample = {
            "episode_id": f"coffee_table/{case}",
            "hdf5_file": str(h5_path),
            "task_description_file": str(txt_path),
            "video_file": str(video_path),
            "downloaded_bytes": h5_path.stat().st_size
            + txt_path.stat().st_size
            + video_path.stat().st_size,
            "sha256": {
                "hdf5": sha256(h5_path),
                "task_description": sha256(txt_path),
                "video": sha256(video_path),
            },
            "hdf5_tree": tree,
            "root_attributes": root_attrs,
            "data_attributes": {
                "env": data_attrs.get("env"),
                "repository_version": data_attrs.get("repository_version"),
                "problem_info": data_attrs.get("problem_info"),
            },
            "action_shape": list(actions.shape),
            "state_shape": list(states.shape),
            "state_time": {
                "start": float(states[0, 0]),
                "end": float(states[-1, 0]),
                "median_delta": float(np.median(np.diff(states[:, 0]))),
                "strictly_monotonic": bool(np.all(np.diff(states[:, 0]) > 0)),
            },
            "video": video,
            "video_frames_equal_actions_plus_one": video["frame_count"]
            == len(actions) + 1,
            "full_instruction": row.high_level_instruction,
            "official_atomic_steps": [
                {
                    "id": index,
                    "instruction": instruction,
                    "start": start,
                    "end": end,
                }
                for index, (instruction, (start, end)) in enumerate(
                    zip(instructions, intervals)
                )
            ],
            "official_atomic_step_count": len(intervals),
            "verified_semantic_stage_count": None,
            "annotation_end": intervals[-1][1],
            "annotation_end_matches_action_length": end_matches,
            "continuous_time_state": bool(np.all(np.diff(states[:, 0]) > 0)),
        }
        samples.append(sample)
        make_review_sheet(
            video_path,
            args.analysis_root / "robocerebra_review" / f"coffee_table_{case}.jpg",
            row.high_level_instruction,
            instructions,
            intervals,
            end_matches,
        )

    action_stack = np.concatenate(all_actions, axis=0)
    schema = {
        "source": "official qiukingballball/RoboCerebra public release",
        "dataset_commit": "5d2e1e361bf65aabbe4d18179515f5a10936cc96",
        "format": "HDF5 plus external task_description.txt and MP4",
        "downloaded_size_bytes": sum(
            path.stat().st_size for path in args.sample_root.rglob("*") if path.is_file()
        ),
        "episodes_inspected": 3,
        "samples": samples,
        "key_search_result": {
            "present_in_hdf5": ["data", "demo_1", "actions", "states"],
            "absent_in_hdf5": [
                "language",
                "instruction",
                "subtask",
                "segment",
                "boundary",
                "annotation",
                "rgb",
                "camera",
            ],
            "external_annotation_fields": ["Task:", "Step:", "[start, end]"],
        },
    }
    (args.analysis_root / "robocerebra_schema.json").write_text(
        json.dumps(schema, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    boundary_audit = {
        "paper_annotation": "EXISTS",
        "public_dataset_annotation": "PARTIAL",
        "case": "CASE_D_EXTERNAL_ANNOTATION",
        "representation": "START_END_IN_EXTERNAL_TASK_DESCRIPTION",
        "exact_fields": ["Task:", "Step:", "[start, end]"],
        "can_map_actions_to_official_atomic_step": True,
        "can_map_actions_to_research_semantic_subtask": False,
        "reason": (
            "The released intervals label atomic pick/place/open/pour steps. No official "
            "field groups them into semantic subgoals such as complete object relocation."
        ),
        "training_metadata_rows": len(train),
        "all_rows_have_step_intervals": len(counts),
        "start_at_zero_rows": start_zero,
        "strictly_contiguous_rows": contiguous,
        "all_positive_interval_rows": positive,
        "rows_with_any_interval_anomaly": len(anomaly_rows),
        "anomaly_examples": anomaly_rows[:25],
        "sample_hdf5_end_matches": sum(
            item["annotation_end_matches_action_length"] for item in samples
        ),
        "sample_hdf5_count": len(samples),
        "sample_atomic_step_counts": [
            item["official_atomic_step_count"] for item in samples
        ],
        "verified_semantic_stage_counts": [None, None, None],
        "semantic_ge4_trajectories_verified": 0,
        "atomic_ge4_trajectories": sum(count >= 4 for count in counts),
        "atomic_step_statistics": {
            "min": min(counts),
            "mean": float(np.mean(counts)),
            "median": float(np.median(counts)),
            "p75": percentile(counts, 75),
            "p90": percentile(counts, 90),
            "max": max(counts),
        },
        "trajectory_final_end_statistics": {
            "min": min(intervals[-1][1] for _, intervals in parsed),
            "mean": float(np.mean([intervals[-1][1] for _, intervals in parsed])),
            "median": float(np.median([intervals[-1][1] for _, intervals in parsed])),
            "p75": percentile([intervals[-1][1] for _, intervals in parsed], 75),
            "p90": percentile([intervals[-1][1] for _, intervals in parsed], 90),
            "max": max(intervals[-1][1] for _, intervals in parsed),
        },
        "valid_annotation_trajectory_statistics_20hz": {
            "count": len(valid_annotation_trajectory_lengths),
            "frames": {
                "min": min(valid_annotation_trajectory_lengths),
                "mean": float(np.mean(valid_annotation_trajectory_lengths)),
                "median": float(np.median(valid_annotation_trajectory_lengths)),
                "p75": percentile(valid_annotation_trajectory_lengths, 75),
                "p90": percentile(valid_annotation_trajectory_lengths, 90),
                "max": max(valid_annotation_trajectory_lengths),
            },
            "seconds": {
                "min": min(valid_annotation_trajectory_lengths) / 20.0,
                "mean": float(np.mean(valid_annotation_trajectory_lengths)) / 20.0,
                "median": float(np.median(valid_annotation_trajectory_lengths)) / 20.0,
                "p75": percentile(valid_annotation_trajectory_lengths, 75) / 20.0,
                "p90": percentile(valid_annotation_trajectory_lengths, 90) / 20.0,
                "max": max(valid_annotation_trajectory_lengths) / 20.0,
            },
        },
        "valid_atomic_duration_statistics_20hz": {
            "count": len(valid_annotation_atomic_durations),
            "frames": {
                "min": min(valid_annotation_atomic_durations),
                "mean": float(np.mean(valid_annotation_atomic_durations)),
                "median": float(np.median(valid_annotation_atomic_durations)),
                "p75": percentile(valid_annotation_atomic_durations, 75),
                "p90": percentile(valid_annotation_atomic_durations, 90),
                "max": max(valid_annotation_atomic_durations),
            },
            "seconds": {
                "min": min(valid_annotation_atomic_durations) / 20.0,
                "mean": float(np.mean(valid_annotation_atomic_durations)) / 20.0,
                "median": float(np.median(valid_annotation_atomic_durations)) / 20.0,
                "p75": percentile(valid_annotation_atomic_durations, 75) / 20.0,
                "p90": percentile(valid_annotation_atomic_durations, 90) / 20.0,
                "max": max(valid_annotation_atomic_durations) / 20.0,
            },
        },
    }
    (args.analysis_root / "robocerebra_boundary_audit.json").write_text(
        json.dumps(boundary_audit, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    repetition_fields = [
        "scene",
        "case",
        "full_instruction",
        "exact_high_level_repetitions",
        "exact_atomic_plan_repetitions",
        "atomic_verb_pattern",
        "atomic_verb_pattern_repetitions",
        "semantic_plan_repetitions",
        "research_repetition_strength",
    ]
    with (args.analysis_root / "robocerebra_task_repetition.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=repetition_fields)
        writer.writeheader()
        for (_, row), normalized_plan, verb_pattern in zip(
            train.iterrows(), normalized_plans, verb_patterns
        ):
            exact_repetitions = exact_instruction_counts[row.high_level_instruction]
            writer.writerow(
                {
                    "scene": row.scene,
                    "case": row.case,
                    "full_instruction": row.high_level_instruction,
                    "exact_high_level_repetitions": exact_repetitions,
                    "exact_atomic_plan_repetitions": exact_plan_counts[normalized_plan],
                    "atomic_verb_pattern": "->".join(verb_pattern),
                    "atomic_verb_pattern_repetitions": verb_pattern_counts[verb_pattern],
                    "semantic_plan_repetitions": "UNKNOWN_NO_OFFICIAL_GROUPING",
                    "research_repetition_strength": (
                        "INSUFFICIENT" if exact_repetitions < 10 else "POC"
                    ),
                }
            )

    candidate_fields = [
        "rank",
        "episode_id",
        "status",
        "full_instruction",
        "official_atomic_steps",
        "verified_semantic_stages",
        "transitions",
        "action_frames",
        "dependency",
        "boundary_source",
        "boundary_matches_hdf5",
        "rejection_reason",
    ]
    with (args.analysis_root / "robocerebra_candidate_tasks.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=candidate_fields)
        writer.writeheader()
        for rank, sample in enumerate(samples, start=1):
            writer.writerow(
                {
                    "rank": rank,
                    "episode_id": sample["episode_id"],
                    "status": "AUDIT_SAMPLE_NOT_RESEARCH_CANDIDATE",
                    "full_instruction": sample["full_instruction"],
                    "official_atomic_steps": sample["official_atomic_step_count"],
                    "verified_semantic_stages": "UNKNOWN",
                    "transitions": sample["official_atomic_step_count"] - 1,
                    "action_frames": sample["action_shape"][0],
                    "dependency": "MEDIUM_ATOMIC_SEQUENCE",
                    "boundary_source": "task_description.txt [start,end]",
                    "boundary_matches_hdf5": sample[
                        "annotation_end_matches_action_length"
                    ],
                    "rejection_reason": "No official semantic-stage grouping",
                }
            )

    compatibility = {
        "action": {
            "dimension": 7,
            "translation": "action[0:3] Cartesian OSC delta command",
            "rotation": "action[3:6] axis-angle OSC delta command",
            "gripper": "action[6], observed {-1,+1}",
            "controller": "OSC_POSE",
            "sample_min": action_stack.min(axis=0).tolist(),
            "sample_max": action_stack.max(axis=0).tolist(),
            "normalization": "Raw values are not consistently confined to [-1,1]",
            "mail_verdict": "ADAPTER_ONLY",
        },
        "alignment": {
            "verdict": "OBS_T_TO_ACTION_T_IN_OFFICIAL_CONVERTER",
            "evidence": (
                "The converter zips orig_states and orig_actions at the same index, sets the "
                "simulator state, reads observations, then stores the paired action."
            ),
        },
        "camera": {
            "raw_hdf5_rgb": False,
            "public_external_mp4": True,
            "sample_external_resolutions": sorted(
                {f"{item['video']['width']}x{item['video']['height']}" for item in samples}
            ),
            "sample_external_fps": sorted(
                {item["video"]["fps"] for item in samples}
            ),
            "raw_wrist_rgb": False,
            "official_converter_outputs": [
                "obs/agentview_rgb uint8 256x256x3",
                "obs/eye_in_hand_rgb uint8 256x256x3",
            ],
            "lerobot_metadata_two_view": True,
            "lerobot_resolution": "256x256",
            "lerobot_fps": lerobot_info["fps"],
            "two_view_video_bytes_inspected": False,
        },
        "stack": ["LIBERO", "robosuite", "MuJoCo", "Franka Panda"],
        "raw_primary_source_required": True,
        "converted_release_limitation": (
            "The official conversion splits each atomic interval into a separate HDF5/RLDS "
            "episode and retains only the atomic instruction, losing the continuous FULL plan."
        ),
        "lerobot_secondary_audit": {
            "total_episodes": lerobot_info["total_episodes"],
            "total_frames": lerobot_info["total_frames"],
            "total_task_indices": lerobot_info["total_tasks"],
            "tasks_parquet_columns": list(lerobot_tasks.columns),
            "instruction_text_preserved_in_tasks_parquet": "task" in lerobot_tasks.columns,
            "verdict": "TWO_VIEW_ACTION_PRESERVED_BUT_HIERARCHY_NOT_PRESERVED",
        },
    }
    (args.analysis_root / "robocerebra_mail_compatibility.json").write_text(
        json.dumps(compatibility, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    gate = {
        "task": "R0-E RoboCerebra Fast Final Compatibility Gate",
        "final": "REJECT",
        "checks": {
            "G1_FULL": "PASS",
            "G2_ordered_semantic_subtasks": "FAIL_ATOMIC_ONLY",
            "G3_temporal_boundary": "PASS_WITH_PUBLIC_QUALITY_ANOMALIES",
            "G4_semantic_stages_ge4": "FAIL_NOT_ANNOTATED",
            "G5_continuous_trajectory": "PASS_RAW_HDF5",
            "G6_RGB_action": "PASS_EXTERNAL_MP4_PLUS_HDF5_ACTION",
            "G7_7D_compatibility": "PASS_ADAPTER_ONLY",
            "G8_external_wrist": "PASS_IN_LEROBOT_METADATA",
            "G9_repetition": "FAIL_EXACT_TASK_MAX_2",
            "G10_no_manual_annotation": "FAIL_SEMANTIC_GROUPING_REQUIRED",
        },
        "exact_blocker": (
            "Public temporal intervals are recoverable only for atomic action steps. The "
            "release provides no official mapping from those steps to >=4 semantic subgoals. "
            "Creating that layer would restart manual boundary/grouping reconstruction."
        ),
        "paper_vs_public": {
            "paper_boundaries": True,
            "public_atomic_boundaries": True,
            "public_semantic_boundaries": False,
        },
        "repetition": {
            "metadata_rows": len(train),
            "exact_high_level_unique": len(exact_instruction_counts),
            "exact_high_level_max_repetitions": max(exact_instruction_counts.values()),
            "exact_atomic_plan_unique": len(exact_plan_counts),
            "exact_atomic_plan_max_repetitions": max(exact_plan_counts.values()),
            "abstract_atomic_verb_family_max": max(verb_pattern_counts.values()),
            "official_semantic_plan_family": "ABSENT",
        },
        "dataset_evaluation_separation": {
            "training_rows": len(train),
            "evaluation_rows": len(test),
            "evaluation_task_types": test["task_type"].value_counts().to_dict(),
        },
        "downloaded_bytes": schema["downloaded_size_bytes"],
        "training_started": False,
        "gpu_job_submitted": False,
        "data_deleted": False,
        "git_push_performed": False,
    }
    (args.analysis_root / "robocerebra_gate.json").write_text(
        json.dumps(gate, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    case2 = samples[1]
    write_atomic_timeline(
        args.analysis_root / "robocerebra_sample_timeline.csv",
        case2["episode_id"],
        case2["action_shape"][0],
        case2["full_instruction"],
        [item["instruction"] for item in case2["official_atomic_steps"]],
        [(item["start"], item["end"]) for item in case2["official_atomic_steps"]],
    )

    summary = {
        "final": gate["final"],
        "episodes_inspected": len(samples),
        "downloaded_bytes": schema["downloaded_size_bytes"],
        "atomic_boundaries_recoverable": True,
        "semantic_boundaries_recoverable": False,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
