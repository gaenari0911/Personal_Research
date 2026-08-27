#!/usr/bin/env python3
"""Render predicate-aware videos and ±10-frame semantic boundary sheets."""

import argparse
import json
from pathlib import Path

import cv2
import h5py
import numpy as np

from semantic_annotation_core import TASK_SPECS, infer_boundaries


DATA_ROOT = Path("/ssd1/itaein/datasets/LIBERO/libero_10")


def frame_pair(demo, t, lines):
    agent = cv2.cvtColor(np.flipud(demo["obs/agentview_rgb"][t]), cv2.COLOR_RGB2BGR)
    eye = cv2.cvtColor(np.flipud(demo["obs/eye_in_hand_rgb"][t]), cv2.COLOR_RGB2BGR)
    image = np.concatenate([agent, eye], axis=1)
    cv2.rectangle(image, (0, 0), (image.shape[1], 19 * len(lines) + 5), (0, 0, 0), -1)
    for i, line in enumerate(lines):
        cv2.putText(
            image,
            line,
            (5, 16 + 19 * i),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.39,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return image


def signal_text(row):
    pred = ",".join(f"{k}={int(v)}" for k, v in row["predicates"].items())
    grasp = ",".join(f"{k}={int(v)}" for k, v in row["grasps"].items())
    return pred, grasp


def render_trace(trace, output_root, fps, reasons=None):
    inferred = infer_boundaries(trace)
    demo_id = trace["demo_id"]
    task = trace["task_id"]
    target = output_root / f"task_{task}"
    target.mkdir(parents=True, exist_ok=True)
    outputs = {}
    with h5py.File(DATA_ROOT / trace["hdf5"], "r") as source:
        demo = source["data"][demo_id]
        video = target / f"task_{task}_{demo_id}_semantic.mp4"
        writer = cv2.VideoWriter(
            str(video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (256, 128)
        )
        if not writer.isOpened():
            raise RuntimeError(f"cannot create {video}")
        try:
            recovery_indices = {
                event.get("candidate_boundary", event.get("approx_start"))
                for event in inferred["recovery"].get("events", [])
            }
            for t, row in enumerate(trace["rows"]):
                pred, grasp = signal_text(row)
                marker = []
                if t == inferred["s1"]["completion"]:
                    marker.append("S1 COMPLETE")
                if t == inferred["s2"]["completion"]:
                    marker.append("S2 COMPLETE")
                if t == inferred["s1"]["completion"] + 1:
                    marker.append("[TRANSITION] S2 START")
                if t in recovery_indices:
                    marker.append("RECOVERY CANDIDATE")
                current = "S1" if t <= inferred["s1"]["completion"] else "S2"
                instruction = (
                    TASK_SPECS[task]["s1_instruction"]
                    if current == "S1"
                    else TASK_SPECS[task]["s2_instruction"]
                )
                writer.write(
                    frame_pair(
                        demo,
                        t,
                        [
                            f"Task {task} {demo_id} t={t}/{trace['T']-1} {' '.join(marker)}",
                            f"{current}: {instruction}",
                            f"{pred} | {grasp} | g={row['action_gripper']:+.0f}",
                        ],
                    )
                )
        finally:
            writer.release()
        outputs["video"] = str(video.resolve())

        for stage in ("s1", "s2"):
            c = inferred[stage]["completion"]
            indices = (
                list(range(max(0, c - 10), min(trace["T"], c + 11)))
                if c is not None
                else list(range(max(0, trace["T"] - 21), trace["T"]))
            )
            thumbs = []
            for t in indices:
                pred, grasp = signal_text(trace["rows"][t])
                thumbs.append(
                    frame_pair(
                        demo,
                        t,
                        [
                            (
                                f"t={t} delta={t-c:+d}{' BOUNDARY' if t == c else ''}"
                                if c is not None
                                else f"t={t} terminal={inferred[stage]['completion_status']}"
                            ),
                            pred,
                            grasp,
                        ],
                    )
                )
            blank = np.zeros_like(thumbs[0])
            while len(thumbs) < 21:
                thumbs.append(blank)
            sheet = np.concatenate(
                [np.concatenate(thumbs[i : i + 7], axis=1) for i in range(0, 21, 7)],
                axis=0,
            )
            path = target / f"task_{task}_{demo_id}_{stage.upper()}_boundary_pm10.png"
            cv2.imwrite(str(path), sheet)
            outputs[f"{stage}_sheet"] = str(path.resolve())
    return {
        "task_id": task,
        "demo_id": demo_id,
        "T": trace["T"],
        "boundaries": {
            "S1": inferred["s1"]["completion"],
            "S2": inferred["s2"]["completion"],
        },
        "confidence": inferred["confidence"],
        "recovery": inferred["recovery"],
        "terminal_completion_status": inferred["s2"]["completion_status"],
        "selection_reasons": reasons or [],
        **outputs,
    }


def render_terminal_ambiguity(trace, output_root, error):
    demo_id = trace["demo_id"]
    task = trace["task_id"]
    target = output_root / f"task_{task}"
    target.mkdir(parents=True, exist_ok=True)
    with h5py.File(DATA_ROOT / trace["hdf5"], "r") as source:
        demo = source["data"][demo_id]
        start = max(0, trace["T"] - 21)
        thumbs = []
        for t in range(start, trace["T"]):
            pred, grasp = signal_text(trace["rows"][t])
            thumbs.append(
                frame_pair(
                    demo,
                    t,
                    [f"t={t} terminal review", pred, grasp],
                )
            )
        blank = np.zeros_like(thumbs[0])
        while len(thumbs) < 21:
            thumbs.insert(0, blank)
        sheet = np.concatenate(
            [np.concatenate(thumbs[i : i + 7], axis=1) for i in range(0, 21, 7)],
            axis=0,
        )
        path = target / f"task_{task}_{demo_id}_ambiguous_terminal_last21.png"
        cv2.imwrite(str(path), sheet)
    return {
        "task_id": task,
        "demo_id": demo_id,
        "T": trace["T"],
        "status": "needs_review",
        "error": str(error),
        "terminal_sheet": str(path.resolve()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("traces", type=Path, nargs="+")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--only-demos", help="optional comma-separated demo ids, e.g. 1,6,7")
    parser.add_argument("--terminal-on-error", action="store_true")
    parser.add_argument("--selection", type=Path, help="JSON selection from automatic annotator")
    args = parser.parse_args()
    manifest = {"stable_window_frames": 5, "review_window": "boundary ±10 frames", "items": []}
    selected = None
    if args.only_demos:
        selected = {f"demo_{int(x)}" for x in args.only_demos.split(",")}
    selection_reasons = {}
    if args.selection:
        selection = json.loads(args.selection.read_text(encoding="utf-8"))
        selection_reasons = {
            (int(item["task_id"]), item["demo_id"]): item.get("reasons", [])
            for item in selection["items"]
        }
    for path in args.traces:
        for line in path.read_text(encoding="utf-8").splitlines():
            trace = json.loads(line)
            if selected is not None and trace["demo_id"] not in selected:
                continue
            if selection_reasons and (int(trace["task_id"]), trace["demo_id"]) not in selection_reasons:
                continue
            try:
                item = render_trace(
                    trace,
                    args.output_dir,
                    args.fps,
                    selection_reasons.get((int(trace["task_id"]), trace["demo_id"]), []),
                )
            except ValueError as error:
                if not args.terminal_on_error:
                    raise
                item = render_terminal_ambiguity(trace, args.output_dir, error)
            manifest["items"].append(item)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
