#!/usr/bin/env python3
"""Precompute frozen CLIP features for only the selected R4 pilot episodes."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import cv2
import h5py
import numpy as np
import torch
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from robocerebra_memory.models import FrozenCLIPFeatureEncoder  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def decode_batches(path: Path, expected: int, batch_size: int):
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open video {path}")
    count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if count != expected + 1:
        capture.release()
        raise RuntimeError(f"{path}: expected T+1={expected + 1} images, found {count}")
    batch = []
    for frame in range(expected):
        ok, bgr = capture.read()
        if not ok:
            capture.release()
            raise RuntimeError(f"{path}: cannot decode image {frame}")
        batch.append(Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))
        if len(batch) == batch_size:
            yield batch
            batch = []
    capture.release()
    if batch:
        yield batch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clip-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("/ssd1/itaein/datasets/RoboCerebra/r4_clip_cache"))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("R4 feature precompute requested CUDA but no GPU is visible")
    index = json.loads((ROOT / "analysis/robocerebra_memory_episode_index.json").read_text())["episodes"]
    by_id = {x["trajectory_id"]: x for x in index}
    ids = []
    for split in ("train", "val"):
        payload = json.loads((ROOT / f"splits/robocerebra_r4_pilot_{split}.json").read_text())
        ids.extend(payload["trajectory_ids"])
    test_ids = set(json.loads((ROOT / "splits/robocerebra_memory_test.json").read_text()))
    if set(ids) & test_ids:
        raise RuntimeError("test split contamination")
    if len(ids) != 20 or len(set(ids)) != 20:
        raise RuntimeError("R4 cache scope must be exactly 20 unique episodes")
    args.output.mkdir(parents=True, exist_ok=True)
    encoder = FrozenCLIPFeatureEncoder(
        cache_dir=args.clip_cache,
        local_files_only=True,
        device=args.device,
    )
    if not all(not parameter.requires_grad for parameter in encoder.parameters()):
        raise RuntimeError("CLIP parameters are not frozen")
    start_time = time.monotonic()
    records = []
    for number, trajectory_id in enumerate(ids, 1):
        episode = by_id[trajectory_id]
        safe = trajectory_id.replace("/", "__")
        target = args.output / f"{safe}.pt"
        video = Path(episode["visual_source"]["local_path"])
        if not video.is_file():
            raise RuntimeError(f"pilot video disappeared: {video}")
        chunks = []
        for images in decode_batches(video, int(episode["num_frames"]), args.batch_size):
            chunks.append(encoder.encode_images(images).half().cpu())
        visual = torch.cat(chunks, dim=0)
        with h5py.File(episode["state_source"], "r") as source:
            qpos = torch.from_numpy(np.asarray(source["data/demo_1/states"][:, 1:10], dtype=np.float32))
        texts = [episode["full_instruction"]] + [step["text"] for step in episode["steps"]]
        text_features = encoder.encode_texts(texts).half().cpu()
        if visual.shape != (episode["num_frames"], 512) or qpos.shape != (episode["num_frames"], 9):
            raise RuntimeError(f"cache alignment failure for {trajectory_id}")
        payload = {
            "schema_version": "r4-v1",
            "trajectory_id": trajectory_id,
            "num_frames": episode["num_frames"],
            "visual_features": visual,
            "robot_qpos": qpos,
            "full_text_feature": text_features[0],
            "step_text_features": text_features[1:],
            "step_boundaries": [(step["start"], step["end"]) for step in episode["steps"]],
            "normalized": True,
            "visual_dtype": str(visual.dtype),
            "image_index_mapping": "image t equals timestep t; extra image T excluded",
            "source_video": str(video),
            "source_video_sha256": sha256(video),
        }
        temporary = target.with_suffix(".pt.r4-part")
        torch.save(payload, temporary)
        os.replace(temporary, target)
        records.append({"trajectory_id": trajectory_id, "frames": episode["num_frames"], "cache": str(target), "bytes": target.stat().st_size})
        print(f"[{number}/20] {trajectory_id} {episode['num_frames']} frames", flush=True)
        del visual, qpos, text_features, chunks
        gc.collect()
    summary = {
        "schema_version": "r4-v1",
        "status": "PASS",
        "scope": "pilot train16 plus val4 only",
        "trajectory_count": 20,
        "frame_count": sum(x["frames"] for x in records),
        "dtype": "torch.float16",
        "normalized": True,
        "mapping": "video image index t equals timestep t; T+1 extra final image excluded",
        "clip_checkpoint": "openai/clip-vit-base-patch32",
        "clip_all_parameters_frozen": True,
        "device": args.device,
        "runtime_seconds": time.monotonic() - start_time,
        "gpu_peak_memory_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0,
        "records": records,
        "test_split_used": False,
    }
    (ROOT / "analysis/r4_clip_cache_audit.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({key: summary[key] for key in ("status", "trajectory_count", "frame_count", "runtime_seconds", "gpu_peak_memory_bytes")}, indent=2))


if __name__ == "__main__":
    main()
