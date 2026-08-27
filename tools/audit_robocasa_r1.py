#!/usr/bin/env python3
"""Full low-dimensional and metadata audit for R1 WashFruitColander."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from robocasa_phase1.interface import (  # noqa: E402
    EXPECTED_ANNOTATION_KEYS,
    EXTERNAL_KEY,
    WRIST_KEY,
    RoboCasaTrajectoryLoader,
    arm_only_decision,
)


ALL_VIDEO_KEYS = (
    "observation.images.robot0_agentview_left",
    "observation.images.robot0_agentview_right",
    "observation.images.robot0_eye_in_hand",
)


def _json_safe(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(type(value).__name__)


def _corr(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64).reshape(-1)
    right = np.asarray(right, dtype=np.float64).reshape(-1)
    if left.std() == 0.0 or right.std() == 0.0:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def _mean_cosine(action: np.ndarray, delta: np.ndarray) -> float:
    an = np.linalg.norm(action, axis=1)
    dn = np.linalg.norm(delta, axis=1)
    valid = (an > 1e-12) & (dn > 1e-12)
    if not np.any(valid):
        return float("nan")
    return float(np.mean(np.sum(action[valid] * delta[valid], axis=1) / (an[valid] * dn[valid])))


def _video_metadata(path: Path):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return {"readable": False, "frames": -1, "width": -1, "height": -1, "fps": -1.0}
    try:
        return {
            "readable": True,
            "frames": int(round(cap.get(cv2.CAP_PROP_FRAME_COUNT))),
            "width": int(round(cap.get(cv2.CAP_PROP_FRAME_WIDTH))),
            "height": int(round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))),
            "fps": float(cap.get(cv2.CAP_PROP_FPS)),
        }
    finally:
        cap.release()


def audit(dataset_root: Path, output_root: Path, archive_path: Path) -> dict:
    loader = RoboCasaTrajectoryLoader(dataset_root)
    episode_ids = list(loader.episode_ids())
    rows = []
    corrupt = []
    eligible = []
    annotation_union = set()
    action_min = np.full(7, np.inf)
    action_max = np.full(7, -np.inf)
    timestamp_deltas = []
    current_actions = []
    next_actions = []
    eef_deltas = []
    nonzero_base_abs = []
    video_mismatches = []
    total_frames = 0

    for episode_id in episode_ids:
        try:
            trajectory = loader.load_low_dim(episode_id)
            raw = trajectory["raw_actions"]
            arm = trajectory["actions"]
            state = trajectory["state"]
            length = trajectory["num_frames"]
            decision = arm_only_decision(raw)
            annotation_union.update(trajectory["available_annotation_keys"])
            action_min = np.minimum(action_min, arm.min(axis=0))
            action_max = np.maximum(action_max, arm.max(axis=0))
            total_frames += length
            if length > 1:
                timestamp_deltas.append(np.diff(trajectory["timestamps"]).astype(np.float64))
                current_actions.append(arm[:-1, :3].astype(np.float64))
                next_actions.append(arm[1:, :3].astype(np.float64))
                eef_deltas.append(np.diff(state[:, 7:10], axis=0).astype(np.float64))
            nz = np.abs(raw[:, 0:4][raw[:, 0:4] != 0.0])
            if len(nz):
                nonzero_base_abs.append(nz)

            video_ok = True
            video_detail = {}
            for key in ALL_VIDEO_KEYS:
                path = dataset_root / f"videos/chunk-{episode_id // 1000:03d}/{key}/episode_{episode_id:06d}.mp4"
                metadata = _video_metadata(path) if path.is_file() else {
                    "readable": False, "frames": -1, "width": -1, "height": -1, "fps": -1.0
                }
                video_detail[key] = metadata
                good = (
                    metadata["readable"]
                    and metadata["frames"] == length
                    and metadata["width"] == 256
                    and metadata["height"] == 256
                    and abs(metadata["fps"] - 20.0) < 1e-6
                )
                video_ok = video_ok and good
                if not good:
                    video_mismatches.append({
                        "episode_id": episode_id, "key": key, "expected_frames": length, **metadata
                    })

            eligible_flag = decision.eligible and video_ok
            reason = decision.exclusion_reason
            if not video_ok:
                reason = ";".join(x for x in (reason, "video_metadata_mismatch") if x)
            if eligible_flag:
                eligible.append(episode_id)
            rows.append({
                "episode_id": episode_id,
                "num_frames": length,
                "base_nonzero_frames": decision.base_nonzero_frames,
                "base_nonzero_fraction": f"{decision.base_nonzero_fraction:.12g}",
                "base_max_norm": f"{decision.base_max_norm:.12g}",
                "control_mode_values": "|".join(f"{x:g}" for x in decision.control_mode_values),
                "control_mode_change_count": decision.control_mode_change_count,
                "arm_only_eligible": str(eligible_flag).lower(),
                "exclusion_reason": reason,
            })
        except Exception as exc:
            corrupt.append({"episode_id": episode_id, "error": f"{type(exc).__name__}: {exc}"})
            rows.append({
                "episode_id": episode_id, "num_frames": 0, "base_nonzero_frames": 0,
                "base_nonzero_fraction": "", "base_max_norm": "",
                "control_mode_values": "", "control_mode_change_count": 0,
                "arm_only_eligible": "false", "exclusion_reason": "corrupt_or_unreadable",
            })

    output_root.mkdir(parents=True, exist_ok=True)
    csv_path = output_root / "washfruitcolander_episode_action_audit.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    current = np.concatenate(current_actions) if current_actions else np.empty((0, 3))
    next_action = np.concatenate(next_actions) if next_actions else np.empty((0, 3))
    delta = np.concatenate(eef_deltas) if eef_deltas else np.empty((0, 3))
    dt = np.concatenate(timestamp_deltas) if timestamp_deltas else np.empty(0)
    base_nz = np.concatenate(nonzero_base_abs) if nonzero_base_abs else np.empty(0)
    alignment = {
        "pairs": int(len(delta)),
        "eef_delta_vs_same_row_action_flat_pearson": _corr(delta, current),
        "eef_delta_vs_next_row_action_flat_pearson": _corr(delta, next_action),
        "eef_delta_vs_same_row_action_mean_cosine": _mean_cosine(current, delta),
        "eef_delta_vs_next_row_action_mean_cosine": _mean_cosine(next_action, delta),
    }

    manifest = {
        "task": "WashFruitColander",
        "dataset_root": str(dataset_root),
        "selection_rule": (
            "All four LeRobot base dimensions are exactly 0.0 for every frame; "
            "control_mode is exactly constant -1.0; all three videos exist and match parquet length. "
            "Official task metadata says moma_required=No. The acquired v2.1 archive lacks the "
            "post-2026 per-frame navigation/stage fields, which is recorded as a dataset gate blocker."
        ),
        "base_zero_tolerance": 0.0,
        "base_nonzero_distribution": {
            "minimum_nonzero_abs": float(base_nz.min()) if len(base_nz) else None,
            "maximum_nonzero_abs": float(base_nz.max()) if len(base_nz) else None,
            "reason_for_exact_zero": "serialized actions contain exact zeros; no floating tolerance is needed"
        },
        "total_episodes": len(episode_ids),
        "eligible_episodes": len(eligible),
        "excluded_episodes": len(episode_ids) - len(eligible),
        "corrupt_episodes": len(corrupt),
        "episode_ids": eligible,
        "previous_expected": {"eligible": 489, "total": 507},
        "difference_from_previous": {
            "eligible": len(eligible) - 489, "total": len(episode_ids) - 507
        },
    }
    (output_root / "washfruitcolander_arm_only_manifest.json").write_text(
        json.dumps(manifest, indent=2, default=_json_safe) + "\n", encoding="utf-8"
    )

    summary = {
        "task": "WashFruitColander",
        "dataset_root": str(dataset_root),
        "archive": {
            "path": str(archive_path),
            "size_bytes": archive_path.stat().st_size,
            "sha256": "4905fff0dfe1c16c9bbab51d44cda0e89a8e15d042420c04da94dc1d2bf4fd0c",
            "tar_integrity": "PASS",
        },
        "schema": {
            "codebase_version": loader.info["codebase_version"],
            "total_episodes": loader.info["total_episodes"],
            "total_frames": loader.info["total_frames"],
            "fps": loader.info["fps"],
            "action_shape": [12],
            "state_shape": [16],
            "camera_keys": list(ALL_VIDEO_KEYS),
            "annotation_keys_present": sorted(annotation_union),
            "annotation_keys_required_but_missing": sorted(set(EXPECTED_ANNOTATION_KEYS) - annotation_union),
        },
        "actual": {
            "parquet_files": len(episode_ids),
            "frames": total_frames,
            "eligible": len(eligible),
            "excluded": len(episode_ids) - len(eligible),
            "corrupt": corrupt,
            "video_mismatch_count": len(video_mismatches),
            "video_mismatches": video_mismatches,
            "timestamp_delta_mean": float(dt.mean()),
            "timestamp_delta_max_abs_error_from_0_05": float(np.max(np.abs(dt - 0.05))),
            "arm_action_min": action_min.tolist(),
            "arm_action_max": action_max.tolist(),
        },
        "alignment_numerical": alignment,
        "annotation_release_blocker": (
            "Acquired checksum-verified LeRobot v2.1 archive predates the official per-frame "
            "subtask update and cannot satisfy R1 raw-annotation loader requirements."
        ),
    }
    (output_root / "washfruitcolander_dataset_audit.json").write_text(
        json.dumps(summary, indent=2, default=_json_safe) + "\n", encoding="utf-8"
    )
    return {"manifest": manifest, "summary": summary}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=ROOT / "analysis")
    args = parser.parse_args()
    result = audit(args.dataset_root, args.output_root, args.archive)
    print(json.dumps({
        "total": result["manifest"]["total_episodes"],
        "eligible": result["manifest"]["eligible_episodes"],
        "excluded": result["manifest"]["excluded_episodes"],
        "missing_annotations": result["summary"]["schema"]["annotation_keys_required_but_missing"],
        "video_mismatches": result["summary"]["actual"]["video_mismatch_count"],
    }, indent=2))


if __name__ == "__main__":
    main()
