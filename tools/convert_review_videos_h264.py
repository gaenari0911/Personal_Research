#!/usr/bin/env python3
"""Create browser/VS Code-compatible H.264 copies of review videos."""

import argparse
import json
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--crf", type=int, default=20)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    converted = []
    for item in manifest["items"]:
        source = Path(item["video"])
        task_dir = args.output_dir / f"task_{item['task_id']}"
        task_dir.mkdir(parents=True, exist_ok=True)
        target = task_dir / source.name
        subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(source),
                "-map",
                "0:v:0",
                "-c:v",
                "libx264",
                "-preset",
                "medium",
                "-crf",
                str(args.crf),
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                "-an",
                str(target),
            ],
            check=True,
        )
        updated = dict(item)
        updated["source_video_mp4v"] = item["video"]
        updated["video"] = str(target.resolve())
        updated["video_codec"] = "h264"
        converted.append(updated)
        print(f"converted {source.name}")

    manifest["items"] = converted
    manifest["video_note"] = "H.264/yuv420p copies for VS Code and browser playback"
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.output_manifest.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {args.output_manifest}")


if __name__ == "__main__":
    main()
