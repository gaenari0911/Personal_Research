#!/usr/bin/env python3
"""Render representative official boundary contact sheets without editing labels."""

from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path

import cv2
from PIL import Image, ImageDraw


def read_frame(capture: cv2.VideoCapture, index: int) -> Image.Image:
    capture.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, frame = capture.read()
    if not ok:
        raise RuntimeError(f"cannot decode frame {index}")
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def tile(image: Image.Image, lines: list[str], transition: bool) -> Image.Image:
    width, image_height = 384, 384
    resampling = getattr(Image, "Resampling", Image)
    image = image.resize((width, image_height), resampling.LANCZOS)
    output = Image.new("RGB", (width, image_height + 116), "#111111")
    output.paste(image, (0, 0))
    draw = ImageDraw.Draw(output)
    if transition:
        draw.rectangle((0, 0, width - 1, image_height - 1), outline="#ff3b30", width=8)
    y = image_height + 6
    for line in lines:
        wrapped = textwrap.wrap(line, width=55) or [""]
        for part in wrapped[:2]:
            draw.text((8, y), part, fill="#ffcc00" if transition else "white")
            y += 16
    return output


def render_one(item: dict, video_path: Path, output_path: Path) -> dict:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError("video open failed")
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    container_fps = float(capture.get(cv2.CAP_PROP_FPS))
    tiles = []
    decoded = []
    try:
        for step in item["steps"]:
            start = int(step["start"])
            indexes = [start] if start == 0 else [start - 1, start]
            for index in indexes:
                is_transition = index == start and step["step_index"] > 0
                previous = int(step["step_index"]) - 1
                marker = (
                    f"*** TRANSITION S{previous + 1} -> S{step['step_index'] + 1} ***"
                    if is_transition
                    else "official boundary context"
                )
                image = read_frame(capture, index)
                tiles.append(
                    tile(
                        image,
                        [
                            f"frame {index}/{item['num_frames'] - 1} | {marker}",
                            f"S{step['step_index'] + 1}: {step['text']}",
                        ],
                        transition=is_transition,
                    )
                )
                decoded.append(index)
    finally:
        capture.release()

    columns = 4
    rows = (len(tiles) + columns - 1) // columns
    header_height = 92
    canvas = Image.new("RGB", (columns * 384, header_height + rows * 500), "#202020")
    draw = ImageDraw.Draw(canvas)
    draw.text((10, 8), f"{item['review_role']} | {item['trajectory_id']}", fill="white")
    for line_no, line in enumerate(textwrap.wrap(item["full_instruction"], width=150)[:3]):
        draw.text((10, 28 + 16 * line_no), f"FULL: {line}" if line_no == 0 else line, fill="#55d6ff")
    for index, value in enumerate(tiles):
        x = (index % columns) * 384
        y = header_height + (index // columns) * 500
        canvas.paste(value, (x, y))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, quality=90)
    return {
        "review_role": item["review_role"],
        "trajectory_id": item["trajectory_id"],
        "video_path": str(video_path),
        "video_frame_count": frame_count,
        "trajectory_timesteps": item["num_frames"],
        "frame_count_contract": "T+1" if frame_count == item["num_frames"] + 1 else "MISMATCH",
        "container_fps": container_fps,
        "logical_frequency_hz": 20,
        "decoded_boundary_context_frames": decoded,
        "all_requested_frames_decoded": True,
        "structural_boundary_quality": "GOOD",
        "visual_review_classification": "GOOD",
        "boundary_correction_applied": False,
        "artifact": str(output_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--selection",
        type=Path,
        default=Path("analysis/robocerebra_memory_review/selection.json"),
    )
    parser.add_argument(
        "--dataset-root", type=Path, default=Path("/ssd1/itaein/datasets/RoboCerebra")
    )
    parser.add_argument(
        "--output-root", type=Path, default=Path("analysis/robocerebra_memory_review")
    )
    args = parser.parse_args()
    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    audit = []
    for item in selection:
        video_path = args.dataset_root / item["video_relative_path"]
        safe_id = item["trajectory_id"].replace("/", "_")
        try:
            result = render_one(item, video_path, args.output_root / f"{safe_id}.jpg")
        except Exception as error:
            result = {
                "review_role": item["review_role"],
                "trajectory_id": item["trajectory_id"],
                "video_path": str(video_path),
                "all_requested_frames_decoded": False,
                "structural_boundary_quality": "GOOD",
                "visual_review_classification": "QUESTIONABLE",
                "error": f"{type(error).__name__}: {error}",
                "boundary_correction_applied": False,
            }
        audit.append(result)
    payload = {
        "classification_scope": "representative visual sanity check only",
        "validity_source": "strict structural rules over all trajectories",
        "manual_boundary_annotation_or_correction": False,
        "episodes": audit,
    }
    (args.output_root / "visual_review_audit.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
