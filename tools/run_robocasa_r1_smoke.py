#!/usr/bin/env python3
"""Run short/median/long loader smoke checks and create camera contact sheets."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from robocasa_phase1.interface import EXTERNAL_KEY, WRIST_KEY, RoboCasaTrajectoryLoader  # noqa: E402


DATASET_ROOT = Path("/ssd1/itaein/datasets/RoboCasa365/WashFruitColander/lerobot")
MANIFEST = ROOT / "analysis/washfruitcolander_arm_only_manifest.json"
OUTPUT = ROOT / "analysis/r1_review/camera_samples"


def _select(loader, ids):
    lengths = [(loader.load_low_dim(ep)["num_frames"], ep) for ep in ids]
    lengths.sort()
    median_length = float(np.median([x[0] for x in lengths]))
    middle = min(lengths, key=lambda x: (abs(x[0] - median_length), x[1]))
    return [("short", *lengths[0]), ("median", *middle), ("long", *lengths[-1])]


def _contact_sheet(loader, label, episode_id, length):
    indices = np.linspace(0, length - 1, 5).round().astype(np.int64)
    external = loader._decode_frames(loader.video_path(episode_id, EXTERNAL_KEY), indices)
    wrist = loader._decode_frames(loader.video_path(episode_id, WRIST_KEY), indices)
    rows = []
    for view_name, frames in (("external", external), ("wrist", wrist)):
        cells = []
        for index, rgb in zip(indices, frames):
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            cv2.rectangle(bgr, (0, 0), (127, 22), (0, 0, 0), thickness=-1)
            cv2.putText(
                bgr, f"{view_name} f={int(index)}", (4, 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA,
            )
            cells.append(bgr)
        rows.append(np.concatenate(cells, axis=1))
    sheet = np.concatenate(rows, axis=0)
    path = OUTPUT / f"{label}_episode_{episode_id:06d}_two_view_5frame.png"
    if not cv2.imwrite(str(path), sheet):
        raise RuntimeError(f"failed to write {path}")
    return path, indices


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(MANIFEST.read_text())
    loader = RoboCasaTrajectoryLoader(DATASET_ROOT, manifest_path=MANIFEST)
    selected = _select(loader, manifest["episode_ids"])
    results = []
    all_finite = True
    for label, length, episode_id in selected:
        trajectory = loader.load_trajectory(episode_id)
        policy_index = int(trajectory["valid_indices"][len(trajectory["valid_indices"]) // 2])
        sample = loader.get_sample(episode_id, policy_index)
        sheet_path, contact_indices = _contact_sheet(loader, label, episode_id, length)
        finite = all(
            np.isfinite(sample[key]).all() for key in ("external_rgb", "wrist_rgb", "actions")
        )
        all_finite = all_finite and finite
        results.append({
            "selection": label,
            "episode_id": episode_id,
            "num_frames": length,
            "policy_observation_index": policy_index,
            "observation_indices": sample["observation_indices"].tolist(),
            "target_action_indices": sample["target_action_indices"].tolist(),
            "external_rgb_shape": list(sample["external_rgb"].shape),
            "wrist_rgb_shape": list(sample["wrist_rgb"].shape),
            "action_chunk_shape": list(sample["actions"].shape),
            "valid_action_count": int(sample["valid_action_mask"].sum()),
            "raw_annotation_keys": sorted(sample["raw_annotations"]),
            "finite": bool(finite),
            "contact_sheet": str(sheet_path.relative_to(ROOT)),
            "contact_frame_indices": contact_indices.tolist(),
        })

    scaler = loader.fit_scaler([x[2] for x in selected])
    probe = loader.load_low_dim(selected[0][2])["actions"][:128]
    roundtrip_error = float(np.max(np.abs(probe - scaler.inverse_transform(scaler.transform(probe)))))
    report = {
        "dataset": str(DATASET_ROOT),
        "sample_episodes": results,
        "all_finite": bool(all_finite),
        "temporary_smoke_scaler_only": True,
        "scaler_fit_episode_ids": [x[2] for x in selected],
        "scaler_roundtrip_max_abs_error": roundtrip_error,
        "final_scaler_saved": False,
    }
    (ROOT / "analysis/r1_loader_smoke.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
