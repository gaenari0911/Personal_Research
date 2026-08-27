#!/usr/bin/env python3
"""Fit independent Stage B probes on train caches and select on val."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from robocerebra_memory.eval.representation_extractor import (  # noqa: E402
    atomic_json,
    atomic_torch_save,
    checkpoint_identity_matches,
    combined_sampling_sha256,
)
from robocerebra_memory.eval.training import fit_probe_bank, load_shards  # noqa: E402
from robocerebra_memory.stage_a import VARIANTS  # noqa: E402


def selected_probe_is_valid(path: Path, variant: str) -> bool:
    if not path.is_file():
        return False
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        return False
    return bool(
        payload.get("schema_version") == "stage-b-probes-v1"
        and payload.get("variant") == variant
        and payload.get("selection_split") == "val"
        and payload.get("backbone_frozen") is True
        and payload.get("test_split_used") is False
        and isinstance(payload.get("probe_state_dict"), dict)
        and payload.get("probe_state_dict")
    )


def run_parallel_variants(config_path: Path) -> None:
    """Use the allocated eight CPUs as four independent two-thread workers."""
    processes = []
    for variant in VARIANTS:
        env = os.environ.copy()
        for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            env[name] = "2"
        command = [
            sys.executable, str(Path(__file__).resolve()), "--variant", variant,
            "--config", str(config_path), "--parallel-child",
        ]
        print(f"STAGE_B_PROBE_PARALLEL_START {variant}", flush=True)
        processes.append((variant, subprocess.Popen(command, cwd=ROOT, env=env)))
    failures = []
    for variant, process in processes:
        if process.wait() != 0:
            failures.append(variant)
    if failures:
        raise RuntimeError(f"parallel Stage B probe training failed for {','.join(failures)}")
    print("STAGE_B_PROBE_PARALLEL_COMPLETE B0 B1 B2 B3", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/stage_b_memory_eval.yaml")
    parser.add_argument("--train-dir", type=Path)
    parser.add_argument("--val-dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--parallel-child", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    representation_root = ROOT / config["outputs"]["representation_root"] / args.variant
    train_dir = args.train_dir or representation_root / "train"
    val_dir = args.val_dir or representation_root / "val"
    output = args.output or ROOT / config["outputs"]["probe_root"] / args.variant / "selected_probes.pt"
    automatic_parallel = bool(config.get("execution", {}).get("parallel_probe_variants", False))
    automatic_parallel = automatic_parallel and bool(os.environ.get("PBS_JOBID"))
    if automatic_parallel and not args.parallel_child and args.variant == "B0":
        if args.train_dir or args.val_dir or args.output:
            raise RuntimeError("automatic parallel probes do not accept per-variant path overrides")
        run_parallel_variants(args.config)
        return
    if automatic_parallel and not args.parallel_child and selected_probe_is_valid(output, args.variant):
        print(json.dumps({"status": "PASS", "variant": args.variant, "output": str(output), "reused": True}, indent=2))
        return
    train_payloads = load_shards(train_dir, "train", args.variant)
    val_payloads = load_shards(val_dir, "val", args.variant)
    expected = config["dataset"]["expected_trajectories"]
    if len(train_payloads) != int(expected["train"]) or len(val_payloads) != int(expected["val"]):
        raise RuntimeError("Stage B train/val representation trajectory count mismatch")
    train_ids = {payload["trajectory_id"] for payload in train_payloads}
    val_ids = {payload["trajectory_id"] for payload in val_payloads}
    if train_ids & val_ids:
        raise RuntimeError("Stage B train/val representation leakage")
    if not checkpoint_identity_matches(train_payloads[0], val_payloads[0]["checkpoint"]):
        raise RuntimeError("Stage B train/val caches were extracted from different Stage A checkpoints")
    choice = config["probe_training"]
    bank, report = fit_probe_bank(
        train_payloads, val_payloads,
        epochs=int(choice["max_epochs"]), learning_rate=float(choice["learning_rate"]),
        weight_decay=float(choice["weight_decay"]), temperature=float(choice["temperature"]),
        patience=int(choice["early_stopping_patience"]), seed=int(config["seed"]),
    )
    report.update(
        {
            "variant": args.variant,
            "train_trajectories": len(train_payloads),
            "val_trajectories": len(val_payloads),
            "test_split_used": False,
            "stage_a_checkpoint": train_payloads[0]["checkpoint"],
            "train_sampling_sha256": combined_sampling_sha256(
                [(payload["trajectory_id"], payload["sampling_sha256"]) for payload in train_payloads]
            ),
            "val_sampling_sha256": combined_sampling_sha256(
                [(payload["trajectory_id"], payload["sampling_sha256"]) for payload in val_payloads]
            ),
        }
    )
    atomic_torch_save(
        output,
        {
            **report,
            "probe_state_dict": {key: value.detach().cpu() for key, value in bank.state_dict().items()},
        },
    )
    atomic_json(output.with_suffix(".json"), report)
    print(json.dumps({"status": "PASS", "variant": args.variant, "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
