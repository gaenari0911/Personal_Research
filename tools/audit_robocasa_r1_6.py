#!/usr/bin/env python3
"""R1.6 small reconstruction audit; never generates heuristic labels."""

from __future__ import annotations

import argparse
import csv
import gzip
import importlib.util
import json
import math
import os
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np


SAMPLE_EPISODES = (205, 379, 189, 341, 60)
QPOS_WIDTH = {"free": 7, "ball": 4, "hinge": 1, "slide": 1}
QVEL_WIDTH = {"free": 6, "ball": 3, "hinge": 1, "slide": 1}


def _accessible(path: str) -> bool:
    try:
        return Path(path).is_file() and os.access(path, os.R_OK)
    except OSError:
        return False


def _joint_layout(xml_root):
    joints = [(node.get("name"), node.get("type", "hinge")) for node in xml_root.iter("joint")]
    offset = 1  # MjSimState.flatten begins with simulation time
    mapping = {}
    for name, joint_type in joints:
        mapping[name] = (offset, offset + QPOS_WIDTH[joint_type])
        offset += QPOS_WIDTH[joint_type]
    nq = sum(QPOS_WIDTH[joint_type] for _, joint_type in joints)
    nv = sum(QVEL_WIDTH[joint_type] for _, joint_type in joints)
    return mapping, nq, nv


def _water_on(value: float) -> bool:
    value = value % (2 * math.pi)
    return 0.40 < value < math.pi


def _spout_orientation(value: float) -> str:
    value = value % (2 * math.pi)
    if math.pi <= value <= 2 * math.pi - math.pi / 6:
        return "left"
    if math.pi / 6 <= value <= math.pi:
        return "right"
    return "center"


def inspect_episode(extras_root: Path, episode_id: int, timeline_dir: Path) -> dict:
    episode_root = extras_root / f"episode_{episode_id:06d}"
    states = np.load(episode_root / "states.npz")["states"]
    xml_bytes = gzip.open(episode_root / "model.xml.gz", "rb").read()
    xml_root = ET.fromstring(xml_bytes)
    ep_meta = json.loads((episode_root / "ep_meta.json").read_text())
    mapping, nq, nv = _joint_layout(xml_root)
    expected_width = 1 + nq + nv
    handle_name = next(
        name for name in mapping if name.endswith("handle_joint") and not name.endswith("temp_joint")
    )
    spout_name = next(name for name in mapping if name.endswith("spout_joint"))
    handle = states[:, mapping[handle_name][0]]
    spout = states[:, mapping[spout_name][0]]
    referenced = sorted({node.get("file") for node in xml_root.iter() if node.get("file")})
    inaccessible = [path for path in referenced if not _accessible(path)]

    timeline_dir.mkdir(parents=True, exist_ok=True)
    timeline_path = timeline_dir / f"episode_{episode_id:06d}_diagnostic.csv"
    fields = [
        "episode_id",
        "t",
        "P1",
        "P2",
        "P3",
        "P4",
        "current_semantic_stage",
        "transition_event",
        "handle_joint_qpos",
        "water_on",
        "spout_joint_qpos",
        "spout_orientation",
        "evaluation_status",
        "failure_reason",
    ]
    with timeline_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for t in range(len(states)):
            writer.writerow(
                {
                    "episode_id": episode_id,
                    "t": t,
                    "P1": "UNKNOWN",
                    "P2": "UNKNOWN",
                    "P3": "UNKNOWN",
                    "P4": "UNKNOWN",
                    "current_semantic_stage": "UNAVAILABLE",
                    "transition_event": "",
                    "handle_joint_qpos": f"{handle[t]:.17g}",
                    "water_on": str(_water_on(float(handle[t]))).lower(),
                    "spout_joint_qpos": f"{spout[t]:.17g}",
                    "spout_orientation": _spout_orientation(float(spout[t])),
                    "evaluation_status": "FAILED_RECONSTRUCTION",
                    "failure_reason": "official_contact_predicates_unavailable",
                }
            )
    objects = [
        {"name": item["name"], "category": item["info"]["cat"], "mjcf_path": item["info"]["mjcf_path"]}
        for item in ep_meta["object_cfgs"]
    ]
    return {
        "episode_id": episode_id,
        "frame_count": int(len(states)),
        "state_width": int(states.shape[1]),
        "xml_nq": nq,
        "xml_nv": nv,
        "flattened_state_width_expected": expected_width,
        "state_layout_match": int(states.shape[1]) == expected_width,
        "model_xml_present": True,
        "ep_meta_present": True,
        "objects": objects,
        "referenced_asset_files": len(referenced),
        "inaccessible_asset_files": len(inaccessible),
        "inaccessible_asset_examples": inaccessible[:10],
        "handle_joint": handle_name,
        "water_first_on_frame": int(np.argmax([_water_on(float(v)) for v in handle]))
        if any(_water_on(float(v)) for v in handle)
        else None,
        "spout_joint": spout_name,
        "spout_min": float(spout.min()),
        "spout_max": float(spout.max()),
        "spout_changed_over_0_1_rad": bool(np.ptp(spout) > 0.1),
        "predicate_evaluation": "FAILED",
        "failure_reason": "Exact official P1/P2 contact and geometry evaluation requires the unavailable compiled MuJoCo model assets.",
        "timeline": str(timeline_path),
    }


def write_empty_boundaries(path: Path):
    fields = [
        "episode_id",
        "frame_count",
        "transition_c1",
        "transition_c2",
        "transition_c3",
        "terminal_completion",
        "stage_duration_s1",
        "stage_duration_s2",
        "stage_duration_s3",
        "stage_duration_s4",
        "predicate_version",
        "valid",
        "failure_reason",
    ]
    with path.open("w", newline="") as stream:
        csv.DictWriter(stream, fieldnames=fields).writeheader()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--analysis-root", type=Path, required=True)
    args = parser.parse_args()
    extras = args.dataset_root / "extras"
    timelines = args.analysis_root / "r1_6_small_predicate_timelines"
    samples = [inspect_episode(extras, episode_id, timelines) for episode_id in SAMPLE_EPISODES]
    result = {
        "task": "WashFruitColander",
        "sample_selection": {
            "seed": 42,
            "short": 205,
            "median": 379,
            "long": 189,
            "random_a": 341,
            "random_b": 60,
        },
        "route": "FAILED",
        "mujoco_3_3_1_available": bool(importlib.util.find_spec("mujoco")),
        "robosuite_1_5_2_available": bool(importlib.util.find_spec("robosuite")),
        "recorded_full_state_present": True,
        "recorded_parquet_state_sufficient": False,
        "parquet_state_reason": "16D state contains only base, EEF, and gripper; no object/contact/fixture state.",
        "samples": samples,
        "successful_initial_reconstructions": 0,
        "successful_action_replays": 0,
        "successful_ordered_timelines": 0,
        "final_successes": 0,
        "small_gate": "FAIL",
        "small_gate_reason": "Exact source predicates require MuJoCo contact/geometry, but the saved XML references inaccessible assets, including Lightwheel colanders absent from the official HF asset release; heuristic reconstruction is forbidden.",
    }
    (args.analysis_root / "r1_6_reconstruction_audit.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    write_empty_boundaries(args.analysis_root / "washfruitcolander_semantic_boundaries.csv")
    print(json.dumps({"episodes": list(SAMPLE_EPISODES), "small_gate": "FAIL"}))


if __name__ == "__main__":
    main()
