#!/usr/bin/env python3
"""Render selected LIBERO trajectories directly from read-only HDF5 arrays."""

import argparse
import json
from pathlib import Path

import cv2
import h5py
import numpy as np


def overlay(frame, lines):
    output = frame.copy()
    height = 24 + 22 * len(lines)
    cv2.rectangle(output, (0, 0), (output.shape[1], height), (0, 0, 0), -1)
    for index, line in enumerate(lines):
        cv2.putText(
            output,
            line,
            (8, 22 + index * 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return output


def combined_frame(agent, eye, lines, flip_vertical):
    if flip_vertical:
        agent = np.flipud(agent)
        eye = np.flipud(eye)
    agent = cv2.cvtColor(agent, cv2.COLOR_RGB2BGR)
    eye = cv2.cvtColor(eye, cv2.COLOR_RGB2BGR)
    combined = np.concatenate([agent, eye], axis=1)
    return overlay(combined, lines)


def render_demo(path, demo_id, task_id, task_name, label, output_dir, fps, flip_vertical):
    with h5py.File(path, "r") as handle:
        demo = handle[f"data/{demo_id}"]
        agent = demo["obs/agentview_rgb"]
        eye = demo["obs/eye_in_hand_rgb"]
        actions = demo["actions"]
        rewards = demo["rewards"]
        length = int(actions.shape[0])
        output_dir.mkdir(parents=True, exist_ok=True)
        stem = f"task_{task_id}_{label}_{demo_id}_T{length}"
        video_path = output_dir / f"{stem}.mp4"
        writer = cv2.VideoWriter(
            str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (256, 128)
        )
        if not writer.isOpened():
            raise RuntimeError(f"failed to open video writer: {video_path}")
        try:
            for t in range(length):
                frame = combined_frame(
                    agent[t],
                    eye[t],
                    [
                        f"Task {task_id}: {task_name}",
                        f"{label} {demo_id} | frame {t}/{length - 1}",
                        f"gripper_action={actions[t, 6]:+.2f} reward={int(rewards[t])}",
                    ],
                    flip_vertical,
                )
                writer.write(frame)
        finally:
            writer.release()

        indices = np.linspace(0, length - 1, 16, dtype=int)
        thumbs = []
        for t in indices:
            frame = combined_frame(
                agent[t], eye[t], [f"t={t} g={actions[t, 6]:+.0f}"], flip_vertical
            )
            thumbs.append(frame)
        rows = [np.concatenate(thumbs[i : i + 4], axis=1) for i in range(0, 16, 4)]
        sheet = np.concatenate(rows, axis=0)
        sheet_path = output_dir / f"{stem}_contact.png"
        cv2.imwrite(str(sheet_path), sheet)
    return {"video": str(video_path.resolve()), "contact_sheet": str(sheet_path.resolve())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("representatives", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--task-ids", type=int, nargs="+", default=[3, 4, 9])
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--no-flip-vertical", action="store_true")
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()

    spec = json.loads(args.representatives.read_text(encoding="utf-8"))
    dataset_dir = Path(spec["dataset_dir"])
    outputs = {"fps": args.fps, "tasks": {}}
    for task_id in args.task_ids:
        task = spec["tasks"][str(task_id)]
        task_name = task["task_name"]
        path = dataset_dir / f"{task_name}_demo.hdf5"
        outputs["tasks"][str(task_id)] = {}
        for label in ("short", "median", "long"):
            item = task[label]
            outputs["tasks"][str(task_id)][label] = {
                **item,
                **render_demo(
                    path,
                    item["demo_id"],
                    task_id,
                    task_name,
                    label,
                    args.output_dir / f"task_{task_id}",
                    args.fps,
                    not args.no_flip_vertical,
                ),
            }
    manifest = args.manifest or args.output_dir / "visualization_manifest.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(outputs, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
