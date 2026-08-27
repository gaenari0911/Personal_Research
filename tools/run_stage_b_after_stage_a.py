#!/usr/bin/env python3
"""Run Stage B selection and exactly-once final evaluation after Stage A.

The validation split is used only to select probes.  Reported metrics and
visualizations are generated from the held-out test95 split exactly once.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from robocerebra_memory.eval.representation_extractor import (  # noqa: E402
    atomic_json,
    load_frozen_stage_a_model,
)
from robocerebra_memory.stage_a import VARIANTS  # noqa: E402


PYTHON = Path(sys.executable)
PIPELINE_REPORT = ROOT / "analysis/stage_b/pipeline_status.json"
REQUIRED_REPORT_FILES = {
    "summary.json",
    "summary.csv",
    "current_retention_by_distance.csv",
    "current_retention_by_transition.csv",
    "memory_depth.csv",
    "sequence_consistency.json",
    "instantaneous_control.json",
}
REQUIRED_VARIANT_FIGURES = {
    "current_retention.svg",
    "transition_robustness.svg",
    "memory_depth.svg",
    "instantaneous_control.svg",
    "sequence_consistency.svg",
    "dashboard.html",
}
REQUIRED_COMPARISON_FIGURES = {
    "retention_distance_comparison.svg",
    "transition_robustness_comparison.svg",
    "memory_depth_comparison.svg",
    "instantaneous_control_comparison.svg",
    "sequence_consistency_comparison.svg",
    "dashboard.html",
}


def _write_report(payload: dict) -> None:
    atomic_json(PIPELINE_REPORT, payload)


def _config(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _validate_stage_a_outputs(config: dict) -> dict:
    """Validate all final Stage A artifacts, with explicit B2/B3 checks."""
    expected_epochs = int(json.loads((ROOT / "analysis/stage_a_compute_plan.json").read_text())["selected_epochs_per_model"])
    models = {}
    common_hashes = set()
    expected_updates = 734 * expected_epochs
    for variant in VARIANTS:
        result_path = ROOT / f"analysis/stage_a_training_{variant}.json"
        best_path = ROOT / f"checkpoints/stage_a/{variant}/best_val.pt"
        last_path = ROOT / f"checkpoints/stage_a/{variant}/last.pt"
        if not result_path.is_file() or not best_path.is_file() or not last_path.is_file():
            raise RuntimeError(f"missing final Stage A artifact for {variant}")
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result.get("status") != "PASS":
            raise RuntimeError(f"Stage A training result is not PASS for {variant}")
        if int(result.get("epochs_completed", -1)) != expected_epochs:
            raise RuntimeError(f"Stage A epoch count mismatch for {variant}")
        if result.get("test_split_used"):
            raise RuntimeError(f"Stage A unexpectedly used test split for {variant}")
        if int(result.get("global_updates", -1)) != expected_updates:
            raise RuntimeError(f"Stage A update count mismatch for {variant}")
        if any(epoch.get("collapse") for epoch in result.get("epochs", [])):
            raise RuntimeError(f"Stage A representation collapse detected for {variant}")
        common_hash = result.get("common_initialization_sha256")
        if not isinstance(common_hash, str) or not common_hash:
            raise RuntimeError(f"Stage A common initialization hash missing for {variant}")
        common_hashes.add(common_hash)
        try:
            _best_model, metadata = load_frozen_stage_a_model(best_path, variant, torch.device("cpu"))
            _last_model, last_metadata = load_frozen_stage_a_model(last_path, variant, torch.device("cpu"))
        except Exception as error:
            raise RuntimeError(f"Stage A evaluation-loader validation failed for {variant}: {error}") from error
        best_payload = torch.load(best_path, map_location="cpu", weights_only=False)
        last_payload = torch.load(last_path, map_location="cpu", weights_only=False)
        best_epoch = int(metadata["completed_epoch"])
        if not 1 <= best_epoch <= expected_epochs or metadata["global_update"] != 734 * best_epoch:
            raise RuntimeError(f"Stage A best checkpoint is incomplete for {variant}")
        if last_metadata["completed_epoch"] != expected_epochs or last_metadata["global_update"] != expected_updates:
            raise RuntimeError(f"Stage A last checkpoint is incomplete for {variant}")
        if (
            best_payload.get("common_initialization_sha256") != common_hash
            or last_payload.get("common_initialization_sha256") != common_hash
        ):
            raise RuntimeError(f"Stage A checkpoint common initialization mismatch for {variant}")
        models[variant] = {
            "status": "PASS",
            "best_checkpoint": str(best_path),
            "last_checkpoint": str(last_path),
            "global_update": metadata["global_update"],
            "completed_epoch": metadata["completed_epoch"],
            "state_dict_sha256": metadata["state_dict_sha256"],
            "last_state_dict_sha256": last_metadata["state_dict_sha256"],
        }
    if len(common_hashes) != 1:
        raise RuntimeError("Stage A variants do not share one common initialization")
    # B2/B3 are called out separately because they are the variants that gate
    # the expensive post-training evaluation in this workflow.
    for variant in ("B2", "B3"):
        if models[variant]["status"] != "PASS":
            raise RuntimeError(f"explicit {variant} final validation failed")
    return {"status": "PASS", "expected_epochs": expected_epochs, "models": models}


def _validate_remaining_walltime(config: dict) -> dict:
    minimum = int(config.get("execution", {}).get("minimum_remaining_walltime_seconds", 0))
    stage_a_config = yaml.safe_load((ROOT / "configs/stage_a_representation.yaml").read_text(encoding="utf-8"))
    requested = float(stage_a_config["budget"]["requested_walltime_seconds"])
    status = json.loads((ROOT / "analysis/stage_a_status.json").read_text(encoding="utf-8"))
    started = float(status.get("job_start_epoch", 0.0))
    if started <= 0:
        raise RuntimeError("Stage A job start time is unavailable for the Stage B walltime gate")
    remaining = max(0.0, requested - (time.time() - started))
    if remaining < minimum:
        raise RuntimeError(
            f"insufficient PBS walltime for Stage B: remaining={remaining:.0f}s minimum={minimum}s"
        )
    return {"status": "PASS", "remaining_seconds": remaining, "minimum_seconds": minimum}


def _visible_gpu_ids() -> list[str]:
    raw = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    ids = [value.strip() for value in raw.split(",") if value.strip()]
    if len(ids) < 2:
        ids = ["0", "1"]
    return ids[:2]


def _run(command: list[str], *, env: dict[str, str] | None = None) -> None:
    printable = " ".join(command)
    print(f"STAGE_B_COMMAND {printable}", flush=True)
    subprocess.run(command, cwd=ROOT, env=env, check=True)


def _run_extract_pair(
    left: str,
    right: str,
    devices: list[str],
    config_path: Path,
    splits: tuple[str, ...] = ("train", "val"),
) -> None:
    """Extract requested splits for two variants in parallel on two GPUs."""
    for split in splits:
        if split not in {"train", "val", "test"}:
            raise ValueError(f"unsupported Stage B extraction split: {split}")
        processes = []
        for variant, device in ((left, devices[0]), (right, devices[1])):
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = device
            command = [
                str(PYTHON), "tools/extract_stage_b_representations.py",
                "--variant", variant, "--split", split,
                "--config", str(config_path), "--device", "cuda",
            ]
            if split == "test":
                command.append("--final-test")
            print(f"STAGE_B_COMMAND {variant} {split} (GPU {device})", flush=True)
            processes.append((variant, subprocess.Popen(command, cwd=ROOT, env=env)))
        failures = []
        for variant, process in processes:
            if process.wait() != 0:
                failures.append(variant)
        if failures:
            raise RuntimeError(f"Stage B {split} extraction failed for {','.join(failures)}")


def _run_probe_training(config_path: Path) -> None:
    """Fit on train and select on val; do not emit validation performance."""
    for variant in VARIANTS:
        _run([str(PYTHON), "tools/train_stage_b_probes.py", "--variant", variant, "--config", str(config_path)])


def _run_test_feature_cache(device: str) -> None:
    """Create the held-out CLIP cache only after all probes are selected."""
    stage_a_config = yaml.safe_load((ROOT / "configs/stage_a_representation.yaml").read_text(encoding="utf-8"))
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = device
    _run(
        [
            str(PYTHON), "tools/cache_stage_a_features.py",
            "--model-cache", str(stage_a_config["features"]["model_cache"]),
            "--batch-size", str(stage_a_config["features"]["image_batch_size"]),
            "--device", "cuda", "--split-scope", "test", "--final-test",
            "--audit", "analysis/stage_b/test_clip_cache_audit.json",
        ],
        env=env,
    )


def _run_final_test_eval(config_path: Path, config: dict) -> None:
    """Evaluate each frozen selected probe on test95 at most once."""
    report_root = ROOT / config["outputs"]["report_root"]
    for variant in VARIANTS:
        sentinel = report_root / "final_test" / variant / "FINAL_TEST_COMPLETED.json"
        if sentinel.is_file():
            print(f"STAGE_B_FINAL_TEST_SKIP {variant} already complete", flush=True)
            continue
        _run([
            str(PYTHON), "tools/eval_stage_b.py", "--variant", variant,
            "--split", "test", "--final-test", "--config", str(config_path),
        ])
    # This also repairs comparison artifacts if all per-variant exactly-once
    # sentinels exist but a previous run stopped while creating comparisons.
    from eval_stage_b import write_comparisons

    representation_root = ROOT / config["outputs"]["representation_root"]
    write_comparisons(report_root, representation_root, "test")


def _validate_stage_b_outputs(config: dict) -> dict:
    report_root = ROOT / config["outputs"]["report_root"]
    final_report_root = report_root / "final_test"
    representation_root = ROOT / config["outputs"]["representation_root"]
    probe_root = ROOT / config["outputs"]["probe_root"]
    expected_counts = config["dataset"]["expected_trajectories"]
    test_cache_audit_path = report_root / "test_clip_cache_audit.json"
    if not test_cache_audit_path.is_file():
        raise RuntimeError("Stage B final-test CLIP cache audit is missing")
    test_cache_audit = json.loads(test_cache_audit_path.read_text(encoding="utf-8"))
    if (
        test_cache_audit.get("status") != "PASS"
        or test_cache_audit.get("split_scope") != "test"
        or int(test_cache_audit.get("completed", -1)) != int(expected_counts["test"])
        or int(test_cache_audit.get("expected_episodes", -1)) != int(expected_counts["test"])
        or test_cache_audit.get("test_split_used") is not True
        or test_cache_audit.get("final_test_gate") is not True
        or test_cache_audit.get("missing")
        or test_cache_audit.get("invalid")
    ):
        raise RuntimeError("Stage B final-test CLIP cache audit contract failed")
    required = {}
    train_sampling_hashes = set()
    val_sampling_hashes = set()
    test_sampling_hashes = set()
    for variant in VARIANTS:
        directory = final_report_root / variant
        files = {path.name for path in directory.iterdir()} if directory.is_dir() else set()
        missing = sorted(REQUIRED_REPORT_FILES - files)
        if missing:
            raise RuntimeError(f"Stage B report files missing for {variant}: {missing}")
        summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
        if summary.get("split") != "test" or summary.get("test_split_evaluated") is not True:
            raise RuntimeError(f"Stage B report split contract failed for {variant}")
        figures = directory / "figures"
        missing_figures = sorted(name for name in REQUIRED_VARIANT_FIGURES if not (figures / name).is_file())
        if missing_figures:
            raise RuntimeError(f"Stage B figure files missing for {variant}: {missing_figures}")
        manifests = {}
        for split in ("train", "val", "test"):
            manifest_path = representation_root / variant / split / "manifest.json"
            if not manifest_path.is_file():
                raise RuntimeError(f"Stage B representation manifest missing for {variant}/{split}")
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if (
                manifest.get("status") != "COMPLETE"
                or manifest.get("variant") != variant
                or manifest.get("split") != split
                or int(manifest.get("trajectory_count", -1)) != int(expected_counts[split])
                or bool(manifest.get("test_split")) != (split == "test")
                or bool(manifest.get("final_test_gate")) != (split == "test")
            ):
                raise RuntimeError(f"Stage B representation manifest contract failed for {variant}/{split}")
            manifests[split] = manifest
        train_sampling_hashes.add(manifests["train"].get("sampling_sha256"))
        val_sampling_hashes.add(manifests["val"].get("sampling_sha256"))
        test_sampling_hashes.add(manifests["test"].get("sampling_sha256"))
        probe_path = probe_root / variant / "selected_probes.pt"
        if not probe_path.is_file():
            raise RuntimeError(f"Stage B selected probe missing for {variant}")
        probe = torch.load(probe_path, map_location="cpu", weights_only=False)
        if (
            probe.get("schema_version") != "stage-b-probes-v1"
            or probe.get("variant") != variant
            or probe.get("selection_split") != "val"
            or not probe.get("backbone_frozen")
            or probe.get("test_split_used")
        ):
            raise RuntimeError(f"Stage B selected probe contract failed for {variant}")
        probe_backbone = probe.get("stage_a_checkpoint", {})
        for split in ("train", "val", "test"):
            if (
                probe_backbone.get("state_dict_sha256")
                != manifests[split].get("checkpoint", {}).get("state_dict_sha256")
            ):
                raise RuntimeError(f"Stage B probe/backbone identity mismatch for {variant}/{split}")
        provenance = summary.get("provenance", {})
        if (
            provenance.get("stage_a_checkpoint", {}).get("state_dict_sha256")
            != manifests["test"].get("checkpoint", {}).get("state_dict_sha256")
        ):
            raise RuntimeError(f"Stage B summary provenance mismatch for {variant}")
        sentinel_path = directory / "FINAL_TEST_COMPLETED.json"
        if not sentinel_path.is_file():
            raise RuntimeError(f"Stage B final-test sentinel missing for {variant}")
        sentinel = json.loads(sentinel_path.read_text(encoding="utf-8"))
        if (
            sentinel.get("status") != "COMPLETE"
            or sentinel.get("variant") != variant
            or sentinel.get("split") != "test"
            or int(sentinel.get("trajectory_count", -1)) != int(expected_counts["test"])
            or sentinel.get("exactly_once_gate") is not True
            or sentinel.get("sampling_sha256") != manifests["test"].get("sampling_sha256")
            or sentinel.get("probe_checkpoint_sha256") != provenance.get("probe_checkpoint_sha256")
        ):
            raise RuntimeError(f"Stage B final-test sentinel contract failed for {variant}")
        required[variant] = {
            "status": "PASS", "report_dir": str(directory), "files": sorted(files),
            "figures": sorted(REQUIRED_VARIANT_FIGURES),
        }
    if len(train_sampling_hashes) != 1 or None in train_sampling_hashes:
        raise RuntimeError("B0/B1/B2/B3 train sampling manifests do not match")
    if len(val_sampling_hashes) != 1 or None in val_sampling_hashes:
        raise RuntimeError("B0/B1/B2/B3 val sampling manifests do not match")
    if len(test_sampling_hashes) != 1 or None in test_sampling_hashes:
        raise RuntimeError("B0/B1/B2/B3 test sampling manifests do not match")
    comparison = final_report_root / "comparison"
    comparison_files = {
        "B0_B1_B2_B3_summary.csv",
        "memory_depth_comparison.csv",
        "retention_distance_comparison.csv",
        "transition_robustness_comparison.csv",
    }
    missing_comparison = sorted(name for name in comparison_files if not (comparison / name).is_file())
    if missing_comparison:
        raise RuntimeError(f"Stage B comparison files missing: {missing_comparison}")
    comparison_figures = comparison / "figures"
    missing_comparison_figures = sorted(
        name for name in REQUIRED_COMPARISON_FIGURES if not (comparison_figures / name).is_file()
    )
    if missing_comparison_figures:
        raise RuntimeError(f"Stage B comparison figures missing: {missing_comparison_figures}")
    return {"status": "PASS", "variants": required, "comparison_dir": str(comparison)}


def run(config_path: Path, plan_only: bool = False) -> dict:
    config = _config(config_path)
    plan = {
        "report_validation_metrics": False,
        "final_test": True,
        "variants": list(VARIANTS),
        "extract_pairs": [["B0", "B1"], ["B2", "B3"]],
        "gpu_ids": _visible_gpu_ids(),
        "commands": [
            "validate completed B2/B3 checkpoints (and all Stage A variants)",
            "extract train+val representations for B0/B1 then B2/B3",
            "train independent probes on train and select on val",
            "after probe selection, create the explicitly gated test95 CLIP cache",
            "freeze the selected probes and extract test95 representations",
            "evaluate test95 exactly once and write final comparison tables/figures",
        ],
        "test_split_used": True,
    }
    if plan_only:
        print(json.dumps(plan, indent=2))
        return plan
    started = time.time()
    payload = {"schema_version": "stage-b-pipeline-v1", "status": "RUNNING", "started_at": time.time(), "plan": plan}
    _write_report(payload)
    try:
        payload["walltime_gate"] = _validate_remaining_walltime(config)
        payload["current_stage"] = "VALIDATE_STAGE_A"
        _write_report(payload)
        payload["stage_a_validation"] = _validate_stage_a_outputs(config)
        devices = plan["gpu_ids"]
        payload["current_stage"] = "EXTRACT_B0_B1"
        _write_report(payload)
        _run_extract_pair("B0", "B1", devices, config_path, ("train", "val"))
        payload["current_stage"] = "EXTRACT_B2_B3"
        _write_report(payload)
        _run_extract_pair("B2", "B3", devices, config_path, ("train", "val"))
        payload["current_stage"] = "TRAIN_AND_SELECT_PROBES"
        _write_report(payload)
        _run_probe_training(config_path)
        payload["current_stage"] = "CACHE_TEST95_FEATURES"
        _write_report(payload)
        _run_test_feature_cache(devices[0])
        payload["current_stage"] = "EXTRACT_TEST_B0_B1"
        _write_report(payload)
        _run_extract_pair("B0", "B1", devices, config_path, ("test",))
        payload["current_stage"] = "EXTRACT_TEST_B2_B3"
        _write_report(payload)
        _run_extract_pair("B2", "B3", devices, config_path, ("test",))
        payload["current_stage"] = "FINAL_TEST_METRICS"
        _write_report(payload)
        _run_final_test_eval(config_path, config)
        payload["current_stage"] = "FINAL_OUTPUT_GATE"
        _write_report(payload)
        payload["stage_b_validation"] = _validate_stage_b_outputs(config)
        payload.update({
            "status": "PASS", "current_stage": "COMPLETE",
            "finished_at": time.time(), "runtime_seconds": time.time() - started,
        })
        _write_report(payload)
        print(json.dumps({"status": "PASS", "output": str(PIPELINE_REPORT)}, indent=2), flush=True)
        return payload
    except Exception as error:
        payload.update({"status": "FAIL", "finished_at": time.time(), "runtime_seconds": time.time() - started, "error": str(error)})
        _write_report(payload)
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/stage_b_memory_eval.yaml")
    parser.add_argument("--plan-only", action="store_true", help="print the one-shot plan without reading GPU data or running commands")
    args = parser.parse_args()
    run(args.config, plan_only=args.plan_only)


if __name__ == "__main__":
    main()
