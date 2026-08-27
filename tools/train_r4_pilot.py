#!/usr/bin/env python3
"""Train the equal-contract B0/B1/B2/B3 R4 representation pilot."""

from __future__ import annotations

import argparse
import copy
import hashlib
import io
import json
import math
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Sequence

import torch
from torch import Tensor
from torch.nn import functional as F


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from robocerebra_memory.losses import future_info_nce  # noqa: E402
from robocerebra_memory.models import MemoryExperimentModel  # noqa: E402
from robocerebra_memory.pilot import collapse_statistics  # noqa: E402
from robocerebra_memory.probes import (  # noqa: E402
    CandidateSet,
    LinearRetrievalProbe,
    multi_positive_probe_loss,
)


CACHE_ROOT = Path("/ssd1/itaein/datasets/RoboCerebra/r4_clip_cache")
VARIANTS = ("B0", "B1", "B2", "B3")


def model_hash(model: torch.nn.Module) -> str:
    stream = io.BytesIO()
    torch.save(model.state_dict(), stream)
    return hashlib.sha256(stream.getvalue()).hexdigest()


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_cache(trajectory_id: str) -> dict:
    path = CACHE_ROOT / f"{trajectory_id.replace('/', '__')}.pt"
    if not path.is_file():
        raise RuntimeError(f"pilot cache disappeared: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload["trajectory_id"] != trajectory_id:
        raise RuntimeError(f"cache identity mismatch: {path}")
    if payload["visual_features"].shape != (payload["num_frames"], 512):
        raise RuntimeError(f"cache visual alignment mismatch: {path}")
    if payload["robot_qpos"].shape != (payload["num_frames"], 9):
        raise RuntimeError(f"cache qpos alignment mismatch: {path}")
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


def forward_selected(
    model: MemoryExperimentModel,
    payload: dict,
    variant: str,
    anchor_frames: Sequence[int],
    device: torch.device,
) -> tuple[Tensor, Tensor, Tensor]:
    visual = payload["visual_features"].float().to(device).unsqueeze(0)
    qpos = payload["robot_qpos"].float().to(device).unsqueeze(0)
    language, mask = language_inputs(payload, variant, device)
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
            batch_visual = visual[0, indices]
            batch_qpos = qpos[0, indices]
            batch_language = language[0, indices]
            batch_mask = mask[0, indices]
            instantaneous, tokens = model.encoder(
                batch_visual, batch_qpos, batch_language, batch_mask
            )
            sequence, _ = model.backbone.forward_sequence(tokens)
            for local, (output_index, _frame) in enumerate(entries):
                temporal_parts[output_index] = sequence[local, -1]
                instantaneous_parts[output_index] = instantaneous[local, -1]
        temporal = torch.stack([temporal_parts[i] for i in range(len(anchor_frames))])
        instantaneous = torch.stack([instantaneous_parts[i] for i in range(len(anchor_frames))])
    else:
        instantaneous_all, tokens = model.encoder(visual, qpos, language, mask)
        before = model.episode_reset_count
        state = model.reset_episode_state(1, device=device, dtype=tokens.dtype)
        temporal_all, state = model.backbone.forward_sequence(tokens, state)
        model._episode_state = state
        if model.episode_reset_count != before + 1 or state.steps != payload["num_frames"]:
            raise RuntimeError("persistent state/reset contract mismatch")
        temporal = temporal_all[0, anchors]
        instantaneous = instantaneous_all[0, anchors]
        model.discard_episode_state()
    prediction = model.future_head(temporal)
    target_indices = anchors + 20
    targets = visual[0, target_indices].detach()
    return instantaneous, temporal, targets


def objective_metrics(prediction: Tensor, targets: Tensor, temperature: float = 0.07):
    prediction = F.normalize(prediction, dim=-1)
    targets = F.normalize(targets.detach(), dim=-1)
    loss, _ = future_info_nce(prediction, targets, temperature)
    cosine = prediction @ targets.transpose(0, 1)
    eye = torch.eye(len(prediction), dtype=torch.bool, device=prediction.device)
    return loss, {
        "positive_cosine": float(cosine.diag().mean().detach()),
        "negative_cosine": float(cosine[~eye].mean().detach()),
    }


def parameter_norm(model: torch.nn.Module) -> float:
    return math.sqrt(sum(float((parameter.detach().float() ** 2).sum()) for parameter in model.parameters()))


def evaluate(
    model: MemoryExperimentModel,
    variant: str,
    ids: list[str],
    anchors: dict,
    device: torch.device,
) -> tuple[dict, dict[str, dict]]:
    model.eval()
    losses, positives, negatives = [], [], []
    all_temporal, all_prediction = [], []
    extracted = {}
    with torch.no_grad():
        for trajectory_id in ids:
            payload = load_cache(trajectory_id)
            rows = anchors[trajectory_id]
            frames = [row["frame"] for row in rows]
            instantaneous, temporal, targets = forward_selected(
                model, payload, variant, frames, device
            )
            prediction = model.future_head(temporal)
            loss, metrics = objective_metrics(prediction, targets)
            losses.append(float(loss))
            positives.append(metrics["positive_cosine"])
            negatives.append(metrics["negative_cosine"])
            all_temporal.append(temporal.cpu())
            all_prediction.append(prediction.cpu())
            extracted[trajectory_id] = {
                "temporal": temporal.cpu(),
                "instantaneous": instantaneous.cpu(),
                "anchors": rows,
                "candidate_features": payload["step_text_features"].float(),
            }
    temporal_values = torch.cat(all_temporal)
    prediction_values = torch.cat(all_prediction)
    return {
        "loss": sum(losses) / len(losses),
        "positive_cosine": sum(positives) / len(positives),
        "negative_cosine": sum(negatives) / len(negatives),
        "temporal": collapse_statistics(temporal_values),
        "future_prediction": collapse_statistics(prediction_values),
        "anchors": len(temporal_values),
    }, extracted


def probe_positive_mask(episode: dict, targets: list[int], device: torch.device) -> Tensor:
    candidates = CandidateSet.from_episode(episode)
    mask = torch.zeros(len(targets), len(candidates.texts), dtype=torch.bool, device=device)
    for row, target in enumerate(targets):
        for index in candidates.positive_indices(target):
            mask[row, index] = True
    return mask


def fit_diagnostic_probe(
    train_data: dict[str, dict],
    val_data: dict[str, dict],
    episodes_by_id: dict[str, dict],
    device: torch.device,
    depth: int,
) -> dict:
    torch.manual_seed(42)
    probe = LinearRetrievalProbe(128, 512).to(device)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=1e-3, weight_decay=0.0)
    train_ids = sorted(train_data)
    losses = []
    for step in range(100):
        trajectory_id = train_ids[step % len(train_ids)]
        data = train_data[trajectory_id]
        target_name = "current_target" if depth == 0 else "previous_1_target"
        valid = [i for i, row in enumerate(data["anchors"]) if row[target_name] >= 0]
        if not valid:
            continue
        representation = data["temporal"][valid].to(device)
        candidates = data["candidate_features"].to(device)
        targets = [data["anchors"][i][target_name] for i in valid]
        scores = probe.scores(representation, candidates)
        mask = probe_positive_mask(episodes_by_id[trajectory_id], targets, device)
        loss = multi_positive_probe_loss(scores, mask)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    probe.eval()
    recalls, reciprocal_ranks = [], []
    with torch.no_grad():
        for trajectory_id, data in val_data.items():
            target_name = "current_target" if depth == 0 else "previous_1_target"
            valid = [i for i, row in enumerate(data["anchors"]) if row[target_name] >= 0]
            if not valid:
                continue
            scores = probe.scores(
                data["temporal"][valid].to(device),
                data["candidate_features"].to(device),
            )
            candidates = CandidateSet.from_episode(episodes_by_id[trajectory_id])
            for row_index, source_index in enumerate(valid):
                positives = candidates.positive_indices(data["anchors"][source_index][target_name])
                best_positive = max(float(scores[row_index, index]) for index in positives)
                rank = 1 + int((scores[row_index] > best_positive).sum())
                recalls.append(float(int(int(scores[row_index].argmax()) in positives)))
                reciprocal_ranks.append(1.0 / rank)
    return {
        "target": "current" if depth == 0 else "previous_1",
        "train_steps": len(losses),
        "final_train_loss": losses[-1] if losses else None,
        "val_samples": len(recalls),
        "val_recall_at_1": sum(recalls) / len(recalls),
        "val_mrr": sum(reciprocal_ranks) / len(reciprocal_ranks),
        "scope": "pilot train/val only; diagnostic, not final evaluation",
    }


def train_variant(
    variant: str,
    train_ids: list[str],
    val_ids: list[str],
    anchors: dict,
    episodes_by_id: dict[str, dict],
    device: torch.device,
    updates: int,
    output_root: Path,
) -> dict:
    torch.manual_seed(42)
    random.seed(42)
    model = MemoryExperimentModel(variant).to(device)
    model.backbone.gradient_checkpointing = variant != "B0"
    initial_hash = model_hash(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    initial_val, _ = evaluate(model, variant, val_ids, anchors["val"], device)
    logs = []
    start_time = time.monotonic()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    order = train_ids.copy()
    random.Random(42).shuffle(order)
    for update in range(updates):
        trajectory_id = order[update % len(order)]
        payload = load_cache(trajectory_id)
        rows = anchors["train"][trajectory_id]
        frames = [row["frame"] for row in rows]
        model.train()
        optimizer.zero_grad(set_to_none=True)
        _instantaneous, temporal, targets = forward_selected(
            model, payload, variant, frames, device
        )
        prediction = model.future_head(temporal)
        loss, metrics = objective_metrics(prediction, targets)
        if not torch.isfinite(loss):
            raise RuntimeError(f"{variant} non-finite loss at update {update}")
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not torch.isfinite(grad_norm):
            raise RuntimeError(f"{variant} non-finite gradient at update {update}")
        optimizer.step()
        logs.append(
            {
                "update": update + 1,
                "trajectory_id": trajectory_id,
                "anchors": len(frames),
                "loss": float(loss.detach()),
                "positive_cosine": metrics["positive_cosine"],
                "negative_cosine": metrics["negative_cosine"],
                "gradient_norm_before_clip": float(grad_norm),
                "parameter_norm": parameter_norm(model),
                "z_norm": float(temporal.detach().norm(dim=-1).mean()),
                "z_mean_per_dimension_std": float(temporal.detach().std(dim=0, unbiased=False).mean()),
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
        )
        print(f"{variant} update {update + 1}/{updates} loss={logs[-1]['loss']:.6f}", flush=True)
    runtime = time.monotonic() - start_time
    final_val, val_data = evaluate(model, variant, val_ids, anchors["val"], device)
    _train_eval, train_data = evaluate(model, variant, train_ids, anchors["train"], device)
    probes = [
        fit_diagnostic_probe(train_data, val_data, episodes_by_id, device, depth)
        for depth in (0, 1)
    ]
    checkpoint_dir = output_root / variant
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / "pilot_final.pt"
    torch.save(
        {
            "schema_version": "r4-v1",
            "variant": variant,
            "updates": updates,
            "seed": 42,
            "model_state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
            "optimizer": "AdamW",
            "learning_rate": 1e-4,
            "weight_decay": 0.01,
        },
        checkpoint_path,
    )
    summary = {
        "variant": variant,
        "initialization_sha256": initial_hash,
        "initial_validation": initial_val,
        "final_validation": final_val,
        "validation_loss_improved": final_val["loss"] < initial_val["loss"],
        "validation_positive_similarity_improved": final_val["positive_cosine"] > initial_val["positive_cosine"],
        "updates": updates,
        "anchors_processed": sum(row["anchors"] for row in logs),
        "final_train_update": logs[-1],
        "mean_train_loss_last_four": sum(row["loss"] for row in logs[-4:]) / min(4, len(logs)),
        "runtime_seconds": runtime,
        "gpu_peak_memory_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
        "execution_mode": "independent_windows" if variant == "B0" else "full_sequence_gradient_checkpointed_no_detach",
        "collapse": final_val["temporal"]["collapsed"] or final_val["future_prediction"]["collapsed"],
        "probes": probes,
        "checkpoint": str(checkpoint_path),
        "logs": logs,
    }
    return summary


def save_retention(
    summary: dict,
    val_id: str,
    episode: dict,
    anchors: dict,
    device: torch.device,
) -> dict:
    checkpoint = torch.load(summary["checkpoint"], map_location="cpu", weights_only=False)
    model = MemoryExperimentModel("B3").to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    payload = load_cache(val_id)
    points = []
    for step_index, (start, end) in enumerate(payload["step_boundaries"]):
        if step_index == 0:
            continue
        for offset in (0, 10, 50, 100):
            frame = start + offset
            if frame < end and frame + 20 < payload["num_frames"]:
                points.append({"step_index": step_index, "offset": offset, "frame": frame})
    with torch.no_grad():
        _instantaneous, temporal, _targets = forward_selected(
            model, payload, "B3", [row["frame"] for row in points], device
        )
    target = ROOT / "analysis/r4_b3_retention_tensors.pt"
    torch.save({"trajectory_id": val_id, "points": points, "z_t": temporal.cpu()}, target)
    for index, point in enumerate(points):
        point["z_norm"] = float(temporal[index].norm())
        if point["offset"] > 0:
            origin_index = next(i for i, row in enumerate(points) if row["step_index"] == point["step_index"] and row["offset"] == 0)
            point["cosine_to_transition"] = float(F.cosine_similarity(temporal[index], temporal[origin_index], dim=0))
    return {"trajectory_id": val_id, "points": points, "tensor_artifact": str(target), "state_continued_without_transition_reset": True}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument("--updates", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, default=ROOT / "analysis/r4_training_summary.json")
    parser.add_argument("--checkpoint-root", type=Path, default=ROOT / "checkpoints/r4")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("R4 training requires a PBS GPU compute node")
    index = load_json(ROOT / "analysis/robocerebra_memory_episode_index.json")["episodes"]
    episodes_by_id = {row["trajectory_id"]: row for row in index}
    train_ids = load_json(ROOT / "splits/robocerebra_r4_pilot_train.json")["trajectory_ids"]
    val_ids = load_json(ROOT / "splits/robocerebra_r4_pilot_val.json")["trajectory_ids"]
    test_ids = set(load_json(ROOT / "splits/robocerebra_memory_test.json"))
    if (set(train_ids) | set(val_ids)) & test_ids:
        raise RuntimeError("test split contamination")
    anchors = load_json(ROOT / "analysis/r4_anchor_manifest.json")["splits"]
    if args.smoke:
        train_ids = train_ids[:1]
        val_ids = val_ids[:1]
        anchors = {
            split: {trajectory_id: rows[:8] for trajectory_id, rows in values.items() if trajectory_id in (train_ids if split == "train" else val_ids)}
            for split, values in anchors.items()
        }
    summaries = {}
    for variant in args.variants:
        summaries[variant] = train_variant(
            variant,
            train_ids,
            val_ids,
            anchors,
            episodes_by_id,
            device,
            args.updates,
            args.checkpoint_root / ("smoke" if args.smoke else "pilot"),
        )
    payload = {
        "schema_version": "r4-v1",
        "status": "PASS",
        "smoke": args.smoke,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "train_trajectories": len(train_ids),
        "val_trajectories": len(val_ids),
        "updates_per_model": args.updates,
        "models": summaries,
        "same_initialization": len({row["initialization_sha256"] for row in summaries.values()}) == 1,
        "test_split_used": False,
        "behavior_cloning": False,
        "action_prediction": False,
    }
    if not args.smoke and "B3" in summaries:
        payload["b3_retention"] = save_retention(
            summaries["B3"], val_ids[0], episodes_by_id[val_ids[0]], anchors["val"], device
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "gpu_name": payload["gpu_name"], "models": {key: {"initial_val": value["initial_validation"]["loss"], "final_val": value["final_validation"]["loss"], "runtime_seconds": value["runtime_seconds"], "peak": value["gpu_peak_memory_bytes"]} for key, value in summaries.items()}}, indent=2))


if __name__ == "__main__":
    main()
