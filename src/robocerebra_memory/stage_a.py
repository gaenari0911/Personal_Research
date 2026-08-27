"""Shared Stage A contracts for cache-backed representation training."""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Sequence

import torch
from torch import Tensor
from torch.nn import functional as F

from .losses import future_info_nce
from .models import MemoryExperimentModel
from .pilot import collapse_statistics


ROOT = Path(__file__).resolve().parents[2]
VARIANTS = ("B0", "B1", "B2", "B3")
CACHE_ROOT = Path("/ssd1/itaein/datasets/RoboCerebra/r4_clip_cache")
HORIZON = 20


def read_json(path: str | Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def atomic_json(path: str | Path, value: object) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".stage-a-part")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, target)


def atomic_torch_save(path: str | Path, value: object) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".stage-a-part")
    with temporary.open("wb") as stream:
        torch.save(value, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, target)


def state_dict_sha256(state_dict: dict[str, Tensor]) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(state_dict.items()):
        digest.update(name.encode("utf-8"))
        tensor = value.detach().cpu().contiguous()
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def cache_path(trajectory_id: str) -> Path:
    return CACHE_ROOT / f"{trajectory_id.replace('/', '__')}.pt"


def validate_cache_payload(payload: dict, episode: dict) -> list[str]:
    errors = []
    expected_frames = int(episode["num_frames"])
    expected_steps = len(episode["steps"])
    if payload.get("trajectory_id") != episode["trajectory_id"]:
        errors.append("trajectory_id")
    if payload.get("num_frames") != expected_frames:
        errors.append("num_frames")
    visual = payload.get("visual_features")
    qpos = payload.get("robot_qpos")
    full = payload.get("full_text_feature")
    steps = payload.get("step_text_features")
    if not isinstance(visual, Tensor) or tuple(visual.shape) != (expected_frames, 512):
        errors.append("visual_shape")
    elif visual.dtype != torch.float16:
        errors.append("visual_dtype")
    if not isinstance(qpos, Tensor) or tuple(qpos.shape) != (expected_frames, 9):
        errors.append("qpos_shape")
    if not isinstance(full, Tensor) or tuple(full.shape) != (512,):
        errors.append("full_text_shape")
    if not isinstance(steps, Tensor) or tuple(steps.shape) != (expected_steps, 512):
        errors.append("step_text_shape")
    if payload.get("normalized") is not True:
        errors.append("normalization_flag")
    expected_boundaries = [(row["start"], row["end"]) for row in episode["steps"]]
    actual_boundaries = [tuple(row) for row in payload.get("step_boundaries", [])]
    if actual_boundaries != expected_boundaries:
        errors.append("step_boundaries")
    if isinstance(visual, Tensor) and visual.numel() and not torch.isfinite(visual).all():
        errors.append("visual_nonfinite")
    if isinstance(qpos, Tensor) and qpos.numel() and not torch.isfinite(qpos).all():
        errors.append("qpos_nonfinite")
    return errors


def load_cache(trajectory_id: str, episodes_by_id: dict[str, dict]) -> dict:
    path = cache_path(trajectory_id)
    if not path.is_file():
        raise RuntimeError(f"Stage A cache missing: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    errors = validate_cache_payload(payload, episodes_by_id[trajectory_id])
    if errors:
        raise RuntimeError(f"invalid Stage A cache {path}: {errors}")
    return payload


def language_inputs(payload: dict, variant: str, device: torch.device) -> tuple[Tensor, Tensor]:
    length = int(payload["num_frames"])
    full = payload["full_text_feature"].float().to(device)
    steps = payload["step_text_features"].float().to(device)
    language = torch.zeros(1, length, 512, device=device)
    mask = torch.zeros(1, length, dtype=torch.bool, device=device)
    if variant in ("B0", "B1"):
        language[:] = full
        mask[:] = True
    elif variant == "B2":
        for index, (start, end) in enumerate(payload["step_boundaries"]):
            language[:, start:end] = steps[index]
            mask[:, start:end] = True
    elif variant == "B3":
        for index, (start, _end) in enumerate(payload["step_boundaries"]):
            language[:, start] = steps[index]
            mask[:, start] = True
    else:
        raise ValueError(variant)
    return language, mask


def assert_b3_hold_contract(payload: dict, mask: Tensor) -> None:
    expected = torch.zeros(int(payload["num_frames"]), dtype=torch.bool, device=mask.device)
    starts = [int(start) for start, _end in payload["step_boundaries"]]
    expected[starts] = True
    if not torch.equal(mask[0], expected):
        raise RuntimeError("B3 transition/HOLD injection mask mismatch")
    for start in starts[1:]:
        if start > 0 and bool(mask[0, start - 1]):
            raise RuntimeError("B3 boundary-1 must have exact-zero language contribution")
        if not bool(mask[0, start]):
            raise RuntimeError("B3 boundary must inject the new Step")
        if start + 1 < len(expected) and bool(mask[0, start + 1]):
            raise RuntimeError("B3 boundary+1 must have exact-zero language contribution")


def forward_selected(
    model: MemoryExperimentModel,
    payload: dict,
    variant: str,
    anchor_frames: Sequence[int],
    device: torch.device,
) -> tuple[Tensor, Tensor, Tensor]:
    if not anchor_frames:
        raise ValueError("at least one anchor is required")
    if max(anchor_frames) + HORIZON >= int(payload["num_frames"]):
        raise RuntimeError("future target exceeds trajectory")
    visual = payload["visual_features"].float().to(device).unsqueeze(0)
    qpos = payload["robot_qpos"].float().to(device).unsqueeze(0)
    language, mask = language_inputs(payload, variant, device)
    if variant == "B3":
        assert_b3_hold_contract(payload, mask)
    anchors = torch.tensor(anchor_frames, dtype=torch.long, device=device)
    if variant == "B0":
        temporal_parts: dict[int, Tensor] = {}
        instantaneous_parts: dict[int, Tensor] = {}
        by_length: dict[int, list[tuple[int, int]]] = defaultdict(list)
        for output_index, frame in enumerate(anchor_frames):
            start = max(0, frame - model.window_size + 1)
            by_length[frame - start + 1].append((output_index, frame))
        for length, entries in by_length.items():
            indices = torch.tensor(
                [[frame - length + 1 + offset for offset in range(length)] for _, frame in entries],
                device=device,
            )
            instantaneous, tokens = model.encoder(
                visual[0, indices], qpos[0, indices], language[0, indices], mask[0, indices]
            )
            sequence, _ = model.backbone.forward_sequence(tokens)
            for local, (output_index, _frame) in enumerate(entries):
                temporal_parts[output_index] = sequence[local, -1]
                instantaneous_parts[output_index] = instantaneous[local, -1]
        temporal = torch.stack([temporal_parts[index] for index in range(len(anchor_frames))])
        instantaneous = torch.stack([instantaneous_parts[index] for index in range(len(anchor_frames))])
    else:
        instantaneous_all, tokens = model.encoder(visual, qpos, language, mask)
        before = model.episode_reset_count
        state = model.reset_episode_state(1, device=device, dtype=tokens.dtype)
        temporal_all, state = model.backbone.forward_sequence(tokens, state)
        model._episode_state = state
        if model.episode_reset_count != before + 1:
            raise RuntimeError("persistent model must reset exactly once at episode start")
        if state.steps != int(payload["num_frames"]):
            raise RuntimeError("persistent state step count mismatch")
        temporal = temporal_all[0, anchors]
        instantaneous = instantaneous_all[0, anchors]
        model.discard_episode_state()
        if model.episode_state is not None:
            raise RuntimeError("persistent state was not discarded at episode end")
    targets = visual[0, anchors + HORIZON].detach()
    return instantaneous, temporal, targets


def objective_metrics(prediction: Tensor, targets: Tensor, temperature: float = 0.07) -> tuple[Tensor, dict]:
    prediction = F.normalize(prediction, dim=-1)
    targets = F.normalize(targets.detach(), dim=-1)
    loss, _ = future_info_nce(prediction, targets, temperature)
    cosine = prediction @ targets.transpose(0, 1)
    eye = torch.eye(len(prediction), dtype=torch.bool, device=prediction.device)
    metrics = {
        "loss": float(loss.detach()),
        "positive_cosine": float(cosine.diag().mean().detach()),
        "negative_cosine": float(cosine[~eye].mean().detach()),
    }
    return loss, metrics


def parameter_norm(model: torch.nn.Module) -> float:
    total = sum(float((parameter.detach().float() ** 2).sum()) for parameter in model.parameters())
    return math.sqrt(total)


def evaluate(
    model: MemoryExperimentModel,
    variant: str,
    trajectory_ids: Iterable[str],
    anchors_by_id: dict[str, list[dict]],
    episodes_by_id: dict[str, dict],
    device: torch.device,
) -> dict:
    model.eval()
    losses, positives, negatives = [], [], []
    temporal_values, prediction_values = [], []
    episode_count = 0
    with torch.no_grad():
        for trajectory_id in trajectory_ids:
            payload = load_cache(trajectory_id, episodes_by_id)
            frames = [row["frame"] for row in anchors_by_id[trajectory_id]]
            _instantaneous, temporal, targets = forward_selected(
                model, payload, variant, frames, device
            )
            prediction = model.future_head(temporal)
            _loss, metrics = objective_metrics(prediction, targets)
            losses.append(metrics["loss"])
            positives.append(metrics["positive_cosine"])
            negatives.append(metrics["negative_cosine"])
            temporal_values.append(temporal.cpu())
            prediction_values.append(prediction.cpu())
            episode_count += 1
    temporal = torch.cat(temporal_values)
    prediction = torch.cat(prediction_values)
    return {
        "loss": sum(losses) / len(losses),
        "positive_cosine": sum(positives) / len(positives),
        "negative_cosine": sum(negatives) / len(negatives),
        "temporal": collapse_statistics(temporal),
        "future_prediction": collapse_statistics(prediction),
        "episodes": episode_count,
        "anchors": len(temporal),
    }


def build_model_from_common_init(variant: str, common_init: str | Path, device: torch.device) -> tuple[MemoryExperimentModel, str]:
    payload = torch.load(common_init, map_location="cpu", weights_only=False)
    if payload.get("schema_version") != "stage-a-v1" or payload.get("seed") != 42:
        raise RuntimeError("invalid common initialization checkpoint")
    model = MemoryExperimentModel(variant)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    loaded_hash = state_dict_sha256(model.state_dict())
    if loaded_hash != payload["state_dict_sha256"]:
        raise RuntimeError("common initialization hash mismatch")
    model.to(device)
    model.backbone.gradient_checkpointing = variant != "B0"
    return model, loaded_hash
