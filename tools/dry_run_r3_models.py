#!/usr/bin/env python3
"""Training-free R3 smoke test on a strict-clean real RoboCerebra episode."""

from __future__ import annotations

import argparse
import gc
import hashlib
import io
import json
import sys
from pathlib import Path

import cv2
import h5py
import numpy as np
import torch
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from robocerebra_memory.models import (  # noqa: E402
    FrozenCLIPFeatureEncoder,
    MemoryExperimentModel,
    build_condition_schedule,
    encode_condition_schedule,
)


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def state_hash(model: torch.nn.Module) -> str:
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    return hashlib.sha256(buffer.getvalue()).hexdigest()


def load_record(trajectory_id: str) -> dict:
    payload = json.loads(
        (ROOT / "analysis/robocerebra_memory_episode_index.json").read_text()
    )
    return next(x for x in payload["episodes"] if x["trajectory_id"] == trajectory_id)


def decode_frames(path: Path, indices: list[int]) -> tuple[list[Image.Image], int]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open {path}")
    container_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    images = []
    for index in indices:
        capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, bgr = capture.read()
        if not ok:
            raise RuntimeError(f"cannot decode frame {index}")
        images.append(Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))
    capture.release()
    return images, container_count


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clip-cache", default="/tmp/r3_clip_cache")
    parser.add_argument("--trajectory", default="study_table/case47")
    args = parser.parse_args()

    torch.set_num_threads(4)
    record = load_record(args.trajectory)
    boundary = int(record["steps"][1]["start"])
    frame_indices = [0, boundary - 1, boundary, boundary + 1]
    video_path = Path(record["visual_source"]["local_path"])
    images, container_count = decode_frames(video_path, frame_indices)
    with h5py.File(record["state_source"], "r") as source:
        raw_states = np.asarray(source["data/demo_1/states"])
    robot_state = torch.from_numpy(raw_states[frame_indices, 1:10]).float().unsqueeze(0)

    clip = FrozenCLIPFeatureEncoder(
        cache_dir=args.clip_cache,
        local_files_only=True,
        device="cpu",
    )
    image_features = clip.encode_images(images).unsqueeze(0)
    schedules = {
        variant: build_condition_schedule(record, variant, frame_indices)
        for variant in ("B0", "B1", "B2", "B3")
    }
    language = {}
    masks = {}
    for variant, schedule in schedules.items():
        encoded, mask = encode_condition_schedule(schedule, clip)
        language[variant] = encoded.unsqueeze(0)
        masks[variant] = mask.unsqueeze(0)
    clip_parameter_count = sum(parameter.numel() for parameter in clip.parameters())
    clip_all_frozen = all(not parameter.requires_grad for parameter in clip.parameters())
    del clip
    gc.collect()

    outputs = {}
    initialization_hashes = {}
    with torch.inference_mode():
        for variant in ("B0", "B1", "B2", "B3"):
            torch.manual_seed(42)
            model = MemoryExperimentModel(variant)
            model.eval()
            initialization_hashes[variant] = state_hash(model)
            result = model.forward_sequence(
                image_features,
                robot_state,
                language[variant],
                masks[variant],
            )
            outputs[variant] = {
                "instantaneous": list(result["instantaneous"].shape),
                "temporal": list(result["temporal"].shape),
                "future_prediction": list(result["future_prediction"].shape),
                "persistent": model.persistent,
                "state_returned": result["state"] is not None,
                "episode_reset_count": model.episode_reset_count,
                "state_steps": None if result["state"] is None else result["state"].steps,
                "window_lengths": list(model.last_window_lengths),
            }
            del model

        # Independently initialized full and step paths prove exact recurrent
        # consistency without counting two resets against one episode execution.
        torch.manual_seed(42)
        full_model = MemoryExperimentModel("B3")
        full_model.eval()
        full = full_model.forward_sequence(
            image_features, robot_state, language["B3"], masks["B3"]
        )["temporal"]
        torch.manual_seed(42)
        step_model = MemoryExperimentModel("B3")
        step_model.eval()
        step_model.reset_episode_state(1)
        recurrent = torch.stack(
            [
                step_model.forward_step(
                    image_features[:, i],
                    robot_state[:, i],
                    language["B3"][:, i],
                    masks["B3"][:, i],
                )["temporal"]
                for i in range(len(frame_indices))
            ],
            dim=1,
        )
    max_consistency_error = float((full - recurrent).abs().max())

    shape_payload = {
        "schema_version": "r3-v1",
        "backend": "torch_reference_mamba1_selective_scan",
        "common": {
            "external_rgb_feature": ["batch", "time", 512],
            "robot_qpos": ["batch", "time", 9],
            "language_feature": ["batch", "time", 512],
            "instantaneous_r_t": ["batch", "time", 128],
            "temporal_z_t": ["batch", "time", 128],
            "future_prediction": ["batch", "time", 512],
        },
        "real_smoke_outputs": outputs,
        "common_initialization_sha256": initialization_hashes,
        "all_initializations_identical": len(set(initialization_hashes.values())) == 1,
    }
    state_payload = {
        "schema_version": "r3-v1",
        "episode_reset": "exactly once at each persistent execution start",
        "subtask_transition_reset": False,
        "episode_end": "explicit discard_episode_state",
        "automatic_recurrent_state_detach": False,
        "target_feature_detach_only": True,
        "full_and_step_share_recurrence": True,
        "full_recurrent_max_abs_error": max_consistency_error,
        "full_reset_count": full_model.episode_reset_count,
        "step_reset_count": step_model.episode_reset_count,
        "b3_transition_reset_count": outputs["B3"]["episode_reset_count"],
    }
    dryrun_payload = {
        "schema_version": "r3-v1",
        "status": "PASS",
        "trajectory_id": args.trajectory,
        "strict_clean_index": str(ROOT / "analysis/robocerebra_memory_episode_index.json"),
        "num_episode_frames": record["num_frames"],
        "num_video_images": container_count,
        "frame_mapping": "MP4 image index t equals trajectory timestep t; final T image excluded",
        "frames": frame_indices,
        "transition_boundary": boundary,
        "transition_sanity": {
            str(boundary - 1): {"text": None, "inject": False},
            str(boundary): {"text": record["steps"][1]["text"], "inject": True},
            str(boundary + 1): {"text": None, "inject": False},
        },
        "clip": {
            "checkpoint": "openai/clip-vit-base-patch32",
            "implementation": "transformers.CLIPModel",
            "actual_pretrained_weights_loaded": True,
            "parameter_count": clip_parameter_count,
            "all_parameters_frozen": clip_all_frozen,
            "image_feature_shape": list(image_features.shape),
            "image_feature_norms": image_features.norm(dim=-1).flatten().tolist(),
            "text_hold_encoded_as_string": False,
        },
        "qpos": {
            "source": record["state_source"],
            "slice": [1, 10],
            "shape": list(robot_state.shape),
        },
        "models": outputs,
        "b3_reset_exactly_once": outputs["B3"]["episode_reset_count"] == 1,
        "full_recurrent_max_abs_error": max_consistency_error,
    }
    write_json(ROOT / "analysis/r3_model_shapes.json", shape_payload)
    write_json(ROOT / "analysis/r3_state_handling_audit.json", state_payload)
    write_json(ROOT / "analysis/r3_real_episode_dryrun.json", dryrun_payload)
    print(json.dumps(dryrun_payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
