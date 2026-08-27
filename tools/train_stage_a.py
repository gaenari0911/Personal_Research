#!/usr/bin/env python3
"""Independent Stage A smoke or full training process for one variant."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
import traceback
from pathlib import Path

import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from robocerebra_memory.stage_a import (  # noqa: E402
    VARIANTS,
    atomic_json,
    atomic_torch_save,
    build_model_from_common_init,
    evaluate,
    forward_selected,
    load_cache,
    objective_metrics,
    parameter_norm,
    read_json,
    state_dict_sha256,
)


def finite_tree(value) -> bool:
    if isinstance(value, dict):
        return all(finite_tree(item) for item in value.values())
    if isinstance(value, list):
        return all(finite_tree(item) for item in value)
    if isinstance(value, float):
        return math.isfinite(value)
    return True


def optimizer_step(model, optimizer, payload, variant, rows, device) -> dict:
    frames = [row["frame"] for row in rows]
    model.train()
    optimizer.zero_grad(set_to_none=True)
    _instantaneous, temporal, targets = forward_selected(model, payload, variant, frames, device)
    prediction = model.future_head(temporal)
    if prediction.shape != (len(frames), 512) or temporal.shape != (len(frames), 128):
        raise RuntimeError("Stage A output shape contract mismatch")
    loss, metrics = objective_metrics(prediction, targets)
    if not torch.isfinite(loss):
        raise RuntimeError("non-finite InfoNCE loss")
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    if not torch.isfinite(grad_norm):
        raise RuntimeError("non-finite gradient")
    optimizer.step()
    metrics.update({
        "gradient_norm_before_clip": float(grad_norm),
        "parameter_norm": parameter_norm(model),
        "z_norm": float(temporal.detach().norm(dim=-1).mean()),
        "z_mean_per_dimension_std": float(temporal.detach().std(dim=0, unbiased=False).mean()),
        "prediction_mean_per_dimension_std": float(prediction.detach().std(dim=0, unbiased=False).mean()),
        "anchors": len(frames),
    })
    if not finite_tree(metrics):
        raise RuntimeError("non-finite Stage A training metric")
    return metrics


def checkpoint_payload(model, optimizer, variant, config, common_hash, completed_epoch, epoch_position, global_update, best_val) -> dict:
    return {
        "schema_version": "stage-a-v1",
        "variant": variant,
        "model_state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "optimizer_state_dict": optimizer.state_dict(),
        "completed_epoch": completed_epoch,
        "epoch_position": epoch_position,
        "global_update": global_update,
        "best_val": best_val,
        "common_initialization_sha256": common_hash,
        "seed": 42,
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": torch.cuda.get_rng_state_all(),
        "python_rng_state": random.getstate(),
        "config": config,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }


def smoke(args, config, schedule, episodes_by_id, device) -> dict:
    start = time.monotonic()
    torch.manual_seed(42)
    random.seed(42)
    model, common_hash = build_model_from_common_init(args.variant, args.common_init, device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["optimizer"]["learning_rate"]),
        weight_decay=float(config["optimizer"]["weight_decay"]),
    )
    trajectory_id = schedule["smoke"]["trajectory_id"]
    rows = schedule["smoke"]["anchors"]
    payload = load_cache(trajectory_id, episodes_by_id)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    initial_start = time.monotonic()
    initial = evaluate(model, args.variant, [trajectory_id], {trajectory_id: rows}, episodes_by_id, device)
    initial_seconds = time.monotonic() - initial_start
    train_start = time.monotonic()
    update = optimizer_step(model, optimizer, payload, args.variant, rows, device)
    train_seconds = time.monotonic() - train_start
    final_start = time.monotonic()
    final = evaluate(model, args.variant, [trajectory_id], {trajectory_id: rows}, episodes_by_id, device)
    final_seconds = time.monotonic() - final_start
    smoke_checkpoint = ROOT / f"checkpoints/stage_a/smoke/{args.variant}/smoke.pt"
    atomic_torch_save(smoke_checkpoint, {
        "schema_version": "stage-a-smoke-v1",
        "variant": args.variant,
        "common_initialization_sha256": common_hash,
        "model_state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "updates": 1,
        "not_for_full_training": True,
    })
    check = torch.load(smoke_checkpoint, map_location="cpu", weights_only=False)
    if check["variant"] != args.variant or check["updates"] != 1:
        raise RuntimeError("smoke checkpoint save/re-open sanity failed")
    collapsed = final["temporal"]["collapsed"] or final["future_prediction"]["collapsed"]
    result = {
        "schema_version": "stage-a-v1",
        "mode": "smoke",
        "variant": args.variant,
        "status": "PASS",
        "trajectory_id": trajectory_id,
        "sequence_length": int(payload["num_frames"]),
        "anchors": len(rows),
        "updates": 1,
        "initial_validation": initial,
        "train_update": update,
        "final_validation": final,
        "collapse_observed": collapsed,
        "runtime_seconds": time.monotonic() - start,
        "train_update_seconds": train_seconds,
        "validation_seconds": initial_seconds + final_seconds,
        "validation_passes": 2,
        "gpu_peak_memory_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
        "common_initialization_sha256": common_hash,
        "checkpoint": str(smoke_checkpoint),
        "full_training_must_reload_common_init": True,
        "b3_hold_state_sanity": "PASS" if args.variant == "B3" else "NOT_APPLICABLE",
        "state_reset": "episode_start_only" if args.variant != "B0" else "no_persistent_state",
        "test_split_used": False,
        "behavior_cloning": False,
        "action_prediction": False,
        "amp": False,
        "training_dtype": "float32",
    }
    if not finite_tree(result):
        raise RuntimeError("smoke result contains non-finite values")
    return result


def restore_resume(model, optimizer, checkpoint, device):
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    torch.set_rng_state(checkpoint["torch_rng_state"])
    torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state_all"])
    random.setstate(checkpoint["python_rng_state"])
    return checkpoint["completed_epoch"], checkpoint["epoch_position"], checkpoint["global_update"], checkpoint["best_val"]


def full_train(args, config, schedule, episodes_by_id, device) -> dict:
    start = time.monotonic()
    torch.manual_seed(42)
    random.seed(42)
    model, common_hash = build_model_from_common_init(args.variant, args.common_init, device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["optimizer"]["learning_rate"]),
        weight_decay=float(config["optimizer"]["weight_decay"]),
    )
    checkpoint_dir = ROOT / f"checkpoints/stage_a/{args.variant}"
    best_path = checkpoint_dir / "best_val.pt"
    last_path = checkpoint_dir / "last.pt"
    if not args.resume and (best_path.exists() or last_path.exists()):
        raise RuntimeError(f"clean Stage A run refuses to overwrite existing {args.variant} full checkpoint")
    completed_epoch, epoch_position, global_update, best_val = 0, 0, 0, float("inf")
    if args.resume and last_path.is_file():
        saved = torch.load(last_path, map_location=device, weights_only=False)
        if saved["variant"] != args.variant or saved["common_initialization_sha256"] != common_hash:
            raise RuntimeError("resume checkpoint contract mismatch")
        completed_epoch, epoch_position, global_update, best_val = restore_resume(model, optimizer, saved, device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    initial_validation = evaluate(
        model, args.variant, schedule["val_ids"], schedule["anchors"]["val"], episodes_by_id, device
    )
    updates_log, epochs_log = [], []
    checkpoint_interval = int(config["training"]["checkpoint_interval_updates"])
    for epoch in range(completed_epoch + 1, args.epochs + 1):
        order = schedule["epoch_orders"][str(epoch)]
        epoch_losses = []
        start_position = epoch_position if epoch == completed_epoch + 1 else 0
        epoch_start = time.monotonic()
        for position in range(start_position, len(order)):
            trajectory_id = order[position]
            payload = load_cache(trajectory_id, episodes_by_id)
            metrics = optimizer_step(
                model,
                optimizer,
                payload,
                args.variant,
                schedule["anchors"]["train"][trajectory_id],
                device,
            )
            global_update += 1
            row = {
                "epoch": epoch,
                "epoch_position": position + 1,
                "global_update": global_update,
                "trajectory_id": trajectory_id,
                "sequence_length": int(payload["num_frames"]),
                "learning_rate": optimizer.param_groups[0]["lr"],
                **metrics,
            }
            updates_log.append(row)
            epoch_losses.append(metrics["loss"])
            print(
                f"{args.variant} epoch={epoch}/{args.epochs} trajectory={position + 1}/734 "
                f"update={global_update} loss={metrics['loss']:.6f}",
                flush=True,
            )
            if global_update % checkpoint_interval == 0:
                atomic_torch_save(
                    last_path,
                    checkpoint_payload(model, optimizer, args.variant, config, common_hash, epoch - 1, position + 1, global_update, best_val),
                )
        validation_start = time.monotonic()
        validation = evaluate(
            model, args.variant, schedule["val_ids"], schedule["anchors"]["val"], episodes_by_id, device
        )
        validation_seconds = time.monotonic() - validation_start
        collapsed = validation["temporal"]["collapsed"] or validation["future_prediction"]["collapsed"]
        epoch_record = {
            "epoch": epoch,
            "updates": len(order),
            "mean_train_loss": sum(epoch_losses) / len(epoch_losses),
            "final_train_update": updates_log[-1],
            "validation": validation,
            "collapse": collapsed,
            "training_seconds": time.monotonic() - epoch_start - validation_seconds,
            "validation_seconds": validation_seconds,
        }
        epochs_log.append(epoch_record)
        if validation["loss"] < best_val and not collapsed:
            best_val = validation["loss"]
            atomic_torch_save(
                best_path,
                checkpoint_payload(model, optimizer, args.variant, config, common_hash, epoch, 0, global_update, best_val),
            )
        atomic_torch_save(
            last_path,
            checkpoint_payload(model, optimizer, args.variant, config, common_hash, epoch, 0, global_update, best_val),
        )
        epoch_position = 0
        running = {
            "schema_version": "stage-a-v1",
            "mode": "full",
            "variant": args.variant,
            "status": "RUNNING" if epoch < args.epochs else "PASS",
            "epochs_requested": args.epochs,
            "epochs_completed": epoch,
            "global_updates": global_update,
            "initial_validation": initial_validation,
            "epochs": epochs_log,
            "common_initialization_sha256": common_hash,
            "best_checkpoint": str(best_path),
            "last_checkpoint": str(last_path),
            "best_val_loss": best_val,
            "gpu_peak_memory_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
            "runtime_seconds": time.monotonic() - start,
            "execution_mode": "independent_windows" if args.variant == "B0" else "full_sequence_gradient_checkpointed_no_detach",
            "test_split_used": False,
            "behavior_cloning": False,
            "action_prediction": False,
            "amp": False,
        }
        atomic_json(args.output, running)
        if collapsed:
            raise RuntimeError(f"{args.variant} representation collapse at epoch {epoch}")
    result = read_json(args.output)
    if not best_path.is_file() or not last_path.is_file() or not finite_tree(result):
        raise RuntimeError(f"{args.variant} final artifact gate failed")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("smoke", "full"), required=True)
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/stage_a_representation.yaml")
    parser.add_argument("--schedule", type=Path, default=ROOT / "analysis/stage_a_schedule.json")
    parser.add_argument("--common-init", type=Path, default=ROOT / "checkpoints/stage_a/common_init.pt")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    device = torch.device(args.device)
    started = time.monotonic()
    try:
        if device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("Stage A training requires CUDA on a PBS compute node")
        if args.mode == "full" and args.epochs not in (1, 2, 3):
            raise RuntimeError("Stage A full epochs must be 1, 2, or 3")
        config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
        schedule = read_json(args.schedule)
        episodes = read_json(ROOT / config["dataset"]["episode_index"])["episodes"]
        episodes_by_id = {row["trajectory_id"]: row for row in episodes}
        result = (
            smoke(args, config, schedule, episodes_by_id, device)
            if args.mode == "smoke"
            else full_train(args, config, schedule, episodes_by_id, device)
        )
        atomic_json(args.output, result)
        print(json.dumps({"status": result["status"], "variant": args.variant, "runtime_seconds": result["runtime_seconds"]}, indent=2))
    except Exception as error:
        failure = {
            "schema_version": "stage-a-v1",
            "mode": args.mode,
            "variant": args.variant,
            "status": "FAIL",
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
            "cuda_oom": isinstance(error, torch.cuda.OutOfMemoryError) or "out of memory" in str(error).lower(),
            "runtime_seconds": time.monotonic() - started,
            "gpu_peak_memory_bytes": torch.cuda.max_memory_allocated(device) if torch.cuda.is_available() else 0,
            "test_split_used": False,
        }
        atomic_json(args.output, failure)
        print(json.dumps(failure, indent=2), file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
