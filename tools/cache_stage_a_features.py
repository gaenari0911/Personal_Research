#!/usr/bin/env python3
"""Resume-safe frozen CLIP cache for train+val or gated final-test data."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import cv2
import h5py
import numpy as np
import torch
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from robocerebra_memory.models import FrozenCLIPFeatureEncoder  # noqa: E402
from robocerebra_memory.stage_a import (  # noqa: E402
    CACHE_ROOT,
    atomic_json,
    atomic_torch_save,
    cache_path,
    read_json,
    validate_cache_payload,
)
from robocerebra_memory.eval.representation_extractor import guard_split_access  # noqa: E402


REVISION = "5d2e1e361bf65aabbe4d18179515f5a10936cc96"
BASE_URL = f"https://huggingface.co/datasets/qiukingballball/RoboCerebra/resolve/{REVISION}/"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_download(url: str, target: Path, attempts: int = 3) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".stage-a-video-part")
    if attempts <= 0:
        raise ValueError("download attempts must be positive")
    for attempt in range(1, attempts + 1):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "stage-a-robocerebra/1.0"})
            with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as stream:
                while True:
                    block = response.read(4 * 1024 * 1024)
                    if not block:
                        break
                    stream.write(block)
                stream.flush()
                os.fsync(stream.fileno())
            if temporary.stat().st_size == 0:
                raise RuntimeError(f"empty video download: {url}")
            os.replace(temporary, target)
            return
        except (OSError, urllib.error.URLError, RuntimeError):
            temporary.unlink(missing_ok=True)
            if attempt == attempts:
                raise
            time.sleep(2 ** attempt)


def video_frame_count(path: Path) -> int:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        return -1
    count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()
    return count


def ensure_video(episode: dict) -> tuple[Path, str]:
    path = Path(episode["visual_source"]["local_path"])
    expected = int(episode["num_frames"]) + 1
    if path.is_file():
        count = video_frame_count(path)
        if count != expected:
            raise RuntimeError(f"existing source video alignment failure: {path} has {count}, expected {expected}")
        return path, "existing"
    relative = episode["visual_source"]["official_relative_path"]
    atomic_download(BASE_URL + relative, path)
    count = video_frame_count(path)
    if count != expected:
        raise RuntimeError(f"downloaded source video alignment failure: {path} has {count}, expected {expected}")
    return path, "downloaded"


def decode_batches(path: Path, expected: int, batch_size: int):
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open video {path}")
    if int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) != expected + 1:
        capture.release()
        raise RuntimeError(f"invalid T+1 alignment: {path}")
    batch = []
    for frame in range(expected):
        ok, bgr = capture.read()
        if not ok:
            capture.release()
            raise RuntimeError(f"cannot decode {path} image {frame}")
        batch.append(Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))
        if len(batch) == batch_size:
            yield batch
            batch = []
    capture.release()
    if batch:
        yield batch


def cache_is_valid(path: Path, episode: dict) -> tuple[bool, list[str]]:
    if not path.is_file():
        return False, ["missing"]
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        return not (errors := validate_cache_payload(payload, episode)), errors
    except Exception as error:  # corrupted files are invalid, never silently accepted
        return False, [f"load_error:{type(error).__name__}:{error}"]


def audit_cache(
    ids: list[str],
    episodes_by_id: dict[str, dict],
    runtime: float,
    device: str,
    records: list[dict],
    *,
    split_scope: str = "train-val",
) -> dict:
    completed, missing, invalid = [], [], []
    frames = 0
    total_bytes = 0
    for trajectory_id in ids:
        path = cache_path(trajectory_id)
        valid, errors = cache_is_valid(path, episodes_by_id[trajectory_id])
        if valid:
            completed.append(trajectory_id)
            frames += int(episodes_by_id[trajectory_id]["num_frames"])
            total_bytes += path.stat().st_size
        elif path.exists():
            invalid.append({"trajectory_id": trajectory_id, "errors": errors})
        else:
            missing.append(trajectory_id)
    expected = len(ids)
    status = "PASS" if len(completed) == expected and not missing and not invalid else "IN_PROGRESS"
    is_test = split_scope == "test"
    return {
        "schema_version": "stage-a-v1",
        "status": status,
        "cache_gate": "PASS" if status == "PASS" else "FAIL",
        "split_scope": split_scope,
        "expected_episodes": expected,
        "completed": len(completed),
        "missing": missing,
        "invalid": invalid,
        "frames": frames,
        "bytes": total_bytes,
        "dtype": "torch.float16",
        "dimension": 512,
        "normalized": True,
        "alignment": "video image t equals model timestep t; extra T image excluded",
        "clip_checkpoint": "openai/clip-vit-base-patch32",
        "clip_all_parameters_frozen": True,
        "text_features": "FULL and official Step frozen CLIP embeddings embedded per trajectory",
        "cache_root": str(CACHE_ROOT),
        "runtime_seconds": runtime,
        "device": device,
        "gpu_peak_memory_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0,
        "test_split_used": is_test,
        "final_test_gate": is_test,
        "records_this_run": records,
    }


def parallel_gpu_ids() -> list[str]:
    status_path = ROOT / "analysis/stage_b/pipeline_status.json"
    config_path = ROOT / "configs/stage_b_memory_eval.yaml"
    if not status_path.is_file() or not config_path.is_file() or not os.environ.get("PBS_JOBID"):
        return []
    import yaml

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not config.get("execution", {}).get("parallel_test_cache", False):
        return []
    status = read_json(status_path)
    gpu_ids = status.get("plan", {}).get("gpu_ids", [])
    return [str(gpu_id) for gpu_id in gpu_ids[:2]] if len(gpu_ids) >= 2 else []


def run_parallel_test_cache(args, ids: list[str], episodes_by_id: dict[str, dict], gpu_ids: list[str]) -> None:
    """Split test95 across the two allocated GPUs and merge one strict audit."""
    processes = []
    part_audits = []
    for shard_index, gpu_id in enumerate(gpu_ids):
        part_audit = args.audit.with_name(f"{args.audit.stem}.part{shard_index}{args.audit.suffix}")
        part_audits.append(part_audit)
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu_id
        command = [
            sys.executable, str(Path(__file__).resolve()),
            "--model-cache", str(args.model_cache), "--batch-size", str(args.batch_size),
            "--device", "cuda", "--audit", str(part_audit),
            "--split-scope", "test", "--final-test",
            "--test-num-shards", str(len(gpu_ids)), "--test-shard-index", str(shard_index),
        ]
        print(f"TEST95_CLIP_CACHE_PARALLEL_START shard={shard_index} gpu={gpu_id}", flush=True)
        processes.append((shard_index, subprocess.Popen(command, cwd=ROOT, env=env)))
    failures = []
    for shard_index, process in processes:
        if process.wait() != 0:
            failures.append(shard_index)
    if failures:
        raise RuntimeError(f"parallel test95 CLIP cache failed for shards {failures}")
    records = []
    for path in part_audits:
        payload = read_json(path)
        if payload.get("status") != "PASS":
            raise RuntimeError(f"parallel test95 cache part audit failed: {path}")
        records.extend(payload.get("records_this_run", []))
    audit = audit_cache(ids, episodes_by_id, 0.0, "cuda:parallel-2", records, split_scope="test")
    if audit["status"] != "PASS" or audit["completed"] != 95:
        raise RuntimeError("merged parallel test95 CLIP cache audit failed")
    atomic_json(args.audit, audit)
    print(json.dumps({"status": "PASS", "completed": 95, "parallel_gpus": 2}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-cache", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--audit", type=Path, default=ROOT / "analysis/stage_a_clip_cache_audit.json")
    parser.add_argument("--split-scope", choices=("train-val", "test"), default="train-val")
    parser.add_argument("--final-test", action="store_true")
    parser.add_argument("--test-num-shards", type=int, default=1, help=argparse.SUPPRESS)
    parser.add_argument("--test-shard-index", type=int, default=0, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.split_scope == "test":
        # This must happen before the test split path is read.
        guard_split_access("test", args.final_test)
    elif args.final_test:
        raise ValueError("--final-test is valid only with --split-scope test")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("Stage A cache requires a PBS GPU compute node")
    index = read_json(ROOT / "analysis/robocerebra_memory_episode_index.json")["episodes"]
    episodes_by_id = {row["trajectory_id"]: row for row in index}
    if args.split_scope == "test":
        full_test_ids = read_json(ROOT / "splits/robocerebra_memory_test.json")
        if len(full_test_ids) != 95 or len(set(full_test_ids)) != 95:
            raise RuntimeError("final-test cache split contract mismatch")
        if args.test_num_shards <= 0 or not 0 <= args.test_shard_index < args.test_num_shards:
            raise RuntimeError("invalid final-test cache shard selection")
        if args.test_num_shards == 1 and (gpu_ids := parallel_gpu_ids()):
            run_parallel_test_cache(args, full_test_ids, episodes_by_id, gpu_ids)
            return
        ids = full_test_ids[args.test_shard_index::args.test_num_shards]
    else:
        train_ids = read_json(ROOT / "splits/robocerebra_memory_train.json")
        val_ids = read_json(ROOT / "splits/robocerebra_memory_val.json")
        test_ids = set(read_json(ROOT / "splits/robocerebra_memory_test.json"))
        ids = train_ids + val_ids
        if len(train_ids) != 734 or len(val_ids) != 85 or len(set(ids)) != 819:
            raise RuntimeError("Stage A train/validation split contract mismatch")
        if set(ids) & test_ids:
            raise RuntimeError("forbidden test contamination in Stage A cache")
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    records: list[dict] = []
    encoder = None
    for number, trajectory_id in enumerate(ids, 1):
        episode = episodes_by_id[trajectory_id]
        target = cache_path(trajectory_id)
        valid, errors = cache_is_valid(target, episode)
        if valid:
            records.append({"trajectory_id": trajectory_id, "status": "reused", "bytes": target.stat().st_size})
            print(f"[{number}/{len(ids)}] REUSE {trajectory_id}", flush=True)
            continue
        if errors != ["missing"]:
            print(f"[{number}/{len(ids)}] REBUILD_INVALID {trajectory_id} {errors}", flush=True)
        video, video_status = ensure_video(episode)
        if encoder is None:
            encoder = FrozenCLIPFeatureEncoder(
                cache_dir=args.model_cache,
                local_files_only=True,
                device=args.device,
            )
            if not all(not parameter.requires_grad for parameter in encoder.parameters()):
                raise RuntimeError("CLIP parameters are not frozen")
        visual_chunks = [
            encoder.encode_images(images).half().cpu()
            for images in decode_batches(video, int(episode["num_frames"]), args.batch_size)
        ]
        visual = torch.cat(visual_chunks)
        with h5py.File(episode["state_source"], "r") as source:
            qpos = torch.from_numpy(np.asarray(source["data/demo_1/states"][:, 1:10], dtype=np.float32))
        texts = [episode["full_instruction"]] + [row["text"] for row in episode["steps"]]
        text_features = encoder.encode_texts(texts).half().cpu()
        payload = {
            "schema_version": "stage-a-v1",
            "trajectory_id": trajectory_id,
            "num_frames": int(episode["num_frames"]),
            "visual_features": visual,
            "robot_qpos": qpos,
            "full_text_feature": text_features[0],
            "step_text_features": text_features[1:],
            "step_boundaries": [(row["start"], row["end"]) for row in episode["steps"]],
            "normalized": True,
            "visual_dtype": str(visual.dtype),
            "image_index_mapping": "image t equals timestep t; extra image T excluded",
            "source_video": str(video),
            "source_video_sha256": sha256(video),
            "download_revision": REVISION,
        }
        errors = validate_cache_payload(payload, episode)
        if errors:
            raise RuntimeError(f"new cache validation failed for {trajectory_id}: {errors}")
        atomic_torch_save(target, payload)
        valid, errors = cache_is_valid(target, episode)
        if not valid:
            raise RuntimeError(f"atomic cache re-open failed for {trajectory_id}: {errors}")
        records.append({
            "trajectory_id": trajectory_id,
            "status": "generated",
            "video": video_status,
            "frames": int(episode["num_frames"]),
            "bytes": target.stat().st_size,
        })
        audit = audit_cache(
            ids, episodes_by_id, time.monotonic() - start, args.device, records,
            split_scope=args.split_scope,
        )
        atomic_json(args.audit, audit)
        print(f"[{number}/{len(ids)}] GENERATED {trajectory_id} frames={episode['num_frames']}", flush=True)
        del visual_chunks, visual, qpos, text_features, payload
        gc.collect()
    audit = audit_cache(
        ids, episodes_by_id, time.monotonic() - start, args.device, records,
        split_scope=args.split_scope,
    )
    if audit["status"] != "PASS":
        audit["status"] = "FAIL"
        atomic_json(args.audit, audit)
        raise RuntimeError("Stage A cache gate failed")
    atomic_json(args.audit, audit)
    print(json.dumps({key: audit[key] for key in ("status", "completed", "frames", "bytes", "runtime_seconds")}, indent=2))


if __name__ == "__main__":
    main()
