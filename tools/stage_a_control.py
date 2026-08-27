#!/usr/bin/env python3
"""Machine-readable gates, budget selection, and failure finalization for Stage A."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from robocerebra_memory.stage_a import VARIANTS, atomic_json, read_json  # noqa: E402


STATUS_PATH = ROOT / "analysis/stage_a_status.json"


def config():
    return yaml.safe_load((ROOT / "configs/stage_a_representation.yaml").read_text(encoding="utf-8"))


def stage_b_auto_enabled() -> bool:
    path = ROOT / "configs/stage_b_memory_eval.yaml"
    if not path.is_file() or not os.environ.get("PBS_JOBID"):
        return False
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    configured = bool(payload.get("execution", {}).get("auto_after_stage_a_in_pbs", False))
    override = os.environ.get("STAGE_B_AUTO_RUN")
    return configured if override is None else override == "1"


def current_status() -> dict:
    if STATUS_PATH.is_file():
        return read_json(STATUS_PATH)
    return {
        "schema_version": "stage-a-v1",
        "smoke": {variant: "PENDING" for variant in VARIANTS},
        "cache": "PENDING",
        "training": {variant: "PENDING" for variant in VARIANTS},
        "stage_a": "PENDING",
    }


def save_status(status: dict) -> None:
    status["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    atomic_json(STATUS_PATH, status)


def init_runtime() -> None:
    cfg = config()
    schedule = read_json(ROOT / cfg["dataset"]["schedule"])
    common = torch.load(ROOT / cfg["training"]["common_initialization"], map_location="cpu", weights_only=False)
    if len(schedule["train_ids"]) != 734 or len(schedule["val_ids"]) != 85 or common["seed"] != 42:
        raise RuntimeError("Stage A runtime precheck failed")
    for variant in VARIANTS:
        if (ROOT / f"checkpoints/stage_a/{variant}/best_val.pt").exists() or (ROOT / f"checkpoints/stage_a/{variant}/last.pt").exists():
            raise RuntimeError(f"clean Stage A run refuses existing full checkpoint for {variant}")
    status = {
        "schema_version": "stage-a-v1",
        "job_id": os.environ.get("PBS_JOBID"),
        "execution_host": os.environ.get("HOSTNAME"),
        "job_start_epoch": time.time(),
        "smoke": {variant: "PENDING" for variant in VARIANTS},
        "cache": "PENDING",
        "training": {variant: "PENDING" for variant in VARIANTS},
        "stage_a": "RUNNING",
        "current_stage": "PRECHECK_RUNTIME",
        "test_split_used": False,
        "common_initialization_sha256": common["state_dict_sha256"],
    }
    save_status(status)


def smoke_gate() -> None:
    models = {}
    common_hashes = set()
    for variant in VARIANTS:
        path = ROOT / f"analysis/stage_a_smoke_{variant}.json"
        if not path.is_file():
            raise RuntimeError(f"missing smoke result for {variant}")
        result = read_json(path)
        checkpoint = ROOT / f"checkpoints/stage_a/smoke/{variant}/smoke.pt"
        if result.get("status") != "PASS" or not checkpoint.is_file():
            raise RuntimeError(f"smoke gate failed for {variant}")
        if not math.isfinite(result["train_update"]["loss"]) or not math.isfinite(result["train_update"]["gradient_norm_before_clip"]):
            raise RuntimeError(f"non-finite smoke metrics for {variant}")
        if result.get("cuda_oom", False):
            raise RuntimeError(f"CUDA OOM in smoke {variant}")
        if variant == "B3" and result.get("b3_hold_state_sanity") != "PASS":
            raise RuntimeError("B3 HOLD/state smoke contract failed")
        common_hashes.add(result["common_initialization_sha256"])
        models[variant] = result
    if len(common_hashes) != 1:
        raise RuntimeError("smoke variants did not load the same common initialization")
    summary = {
        "schema_version": "stage-a-v1",
        "status": "PASS",
        "smoke_gate": "PASS",
        "models": models,
        "same_common_initialization": True,
        "test_split_used": False,
    }
    atomic_json(ROOT / "analysis/stage_a_smoke_summary.json", summary)
    status = current_status()
    status["smoke"] = {variant: "PASS" for variant in VARIANTS}
    status["current_stage"] = "SMOKE_GATE_PASS"
    save_status(status)


def cache_gate() -> None:
    audit = read_json(ROOT / "analysis/stage_a_clip_cache_audit.json")
    if (
        audit.get("status") != "PASS"
        or audit.get("completed") != 819
        or audit.get("missing")
        or audit.get("invalid")
        or audit.get("test_split_used")
    ):
        raise RuntimeError("Stage A cache gate failed")
    status = current_status()
    status["cache"] = "PASS"
    status["current_stage"] = "CACHE_GATE_PASS"
    save_status(status)


def record_gpu(name: str, memory_mib: int, count: int) -> None:
    expected_count = int(config()["pbs"]["gpu_count"])
    payload = {
        "schema_version": "stage-a-v1",
        "gpu_name": name,
        "gpu_names": [value for value in name.split("|") if value],
        "gpu_memory_mib": memory_mib,
        "visible_gpu_count": count,
        "expected_gpu_count": expected_count,
        "cuda_available": torch.cuda.is_available(),
        "torch_cuda_version": torch.version.cuda,
        "pbs_job_id": os.environ.get("PBS_JOBID"),
        "execution_host": os.environ.get("HOSTNAME"),
    }
    if not payload["cuda_available"] or count != expected_count or memory_mib < 20000:
        atomic_json(ROOT / "analysis/stage_a_gpu_runtime.json", payload)
        raise RuntimeError("allocated GPU count/visibility/memory does not satisfy the PBS contract")
    atomic_json(ROOT / "analysis/stage_a_gpu_runtime.json", payload)
    status = current_status()
    status["gpu"] = payload
    status["current_stage"] = "GPU_VERIFY_PASS"
    save_status(status)


def decide_budget() -> None:
    cfg = config()
    smoke = read_json(ROOT / "analysis/stage_a_smoke_summary.json")["models"]
    status = current_status()
    elapsed = max(0.0, time.time() - float(status["job_start_epoch"]))
    requested = float(cfg["budget"]["requested_walltime_seconds"])
    remaining = max(0.0, requested - elapsed)
    safety = float(cfg["budget"]["safety_reserve_fraction"])
    extra = float(cfg["budget"]["smoke_scaling_extra_fraction"])
    fixed = float(cfg["budget"]["fixed_overhead_seconds"])
    per_epoch = {}
    for variant in VARIANTS:
        row = smoke[variant]
        train_per_trajectory = float(row["train_update_seconds"])
        validation_per_trajectory = float(row["validation_seconds"]) / float(row["validation_passes"])
        estimate = train_per_trajectory * 734 + validation_per_trajectory * 85 + 120
        per_epoch[variant] = estimate * (1.0 + extra)
    usable = remaining * (1.0 - safety)
    selected = 0
    estimates = {}
    for epochs in (3, 2, 1):
        # B0/B1 and B2/B3 execute as two sequential pairs on two GPUs.
        total = (max(per_epoch["B0"], per_epoch["B1"]) + max(per_epoch["B2"], per_epoch["B3"])) * epochs + fixed
        estimates[str(epochs)] = total
        if not selected and total <= usable:
            selected = epochs
    plan = {
        "schema_version": "stage-a-v1",
        "status": "PASS" if selected else "FAIL",
        "requested_walltime_seconds": requested,
        "elapsed_before_full_training_seconds": elapsed,
        "remaining_walltime_seconds": remaining,
        "safety_reserve_fraction": safety,
        "usable_seconds_after_reserve": usable,
        "per_model_epoch_estimate_seconds": per_epoch,
        "candidate_total_estimates_seconds": estimates,
        "fixed_overhead_seconds": fixed,
        "selected_epochs_per_model": selected,
        "selection_rule": "largest common epoch count in [3,2,1] fitting two-GPU paired execution within remaining walltime with >=25% reserve",
        "parallel_variant_pairs": [["B0", "B1"], ["B2", "B3"]],
        "smoke_scaling_is_conservative": True,
    }
    atomic_json(ROOT / "analysis/stage_a_compute_plan.json", plan)
    if not selected:
        raise RuntimeError("one complete epoch across B0/B1/B2/B3 does not fit Stage A walltime reserve")
    status["selected_epochs_per_model"] = selected
    status["current_stage"] = "FULL_TRAINING_BUDGET_PASS"
    save_status(status)


def print_epochs() -> None:
    plan = read_json(ROOT / "analysis/stage_a_compute_plan.json")
    epochs = int(plan.get("selected_epochs_per_model", 0))
    if epochs not in (1, 2, 3):
        raise RuntimeError("invalid Stage A selected epoch budget")
    print(epochs)


def training_gate() -> None:
    cfg = config()
    plan = read_json(ROOT / "analysis/stage_a_compute_plan.json")
    selected = int(plan["selected_epochs_per_model"])
    models = {}
    common_hashes = set()
    for variant in VARIANTS:
        result_path = ROOT / f"analysis/stage_a_training_{variant}.json"
        best = ROOT / f"checkpoints/stage_a/{variant}/best_val.pt"
        last = ROOT / f"checkpoints/stage_a/{variant}/last.pt"
        if not result_path.is_file() or not best.is_file() or not last.is_file():
            raise RuntimeError(f"missing full Stage A artifacts for {variant}")
        result = read_json(result_path)
        if result.get("status") != "PASS" or result.get("epochs_completed") != selected:
            raise RuntimeError(f"full Stage A training gate failed for {variant}")
        if any(epoch["collapse"] for epoch in result["epochs"]):
            raise RuntimeError(f"representation collapse for {variant}")
        common_hashes.add(result["common_initialization_sha256"])
        models[variant] = result
    if len(common_hashes) != 1:
        raise RuntimeError("full variants did not use the same common initialization")
    common_hash = next(iter(common_hashes))
    fairness = read_json(ROOT / "analysis/stage_a_fairness_audit.json")
    if fairness["common_initialization_sha256"] != common_hash:
        raise RuntimeError("fairness common initialization hash mismatch")
    fairness["status"] = "PASS"
    fairness["selected_epochs_per_model"] = selected
    fairness["same_epoch_budget"] = all(row["epochs_completed"] == selected for row in models.values())
    fairness["checkpoint_independence"] = True
    atomic_json(ROOT / "analysis/stage_a_fairness_audit.json", fairness)
    summary = {
        "schema_version": "stage-a-v1",
        "status": "PASS",
        "stage_a_gate": "PASS",
        "ready_for_probe_test_metric": "YES",
        "models": models,
        "epochs_per_model": selected,
        "compute_plan": plan,
        "test_split_used": False,
    }
    atomic_json(ROOT / "analysis/stage_a_training_summary.json", summary)
    gate = {
        "schema_version": "stage-a-v1",
        "STAGE_A_GATE": "PASS",
        "READY_FOR_PROBE_TEST_METRIC": "YES",
        "SMOKE": "PASS",
        "CACHE": "PASS",
        "B0": "PASS",
        "B1": "PASS",
        "B2": "PASS",
        "B3": "PASS",
        "FAIRNESS": "PASS",
        "TEST_UNUSED": "PASS",
        "last_successfully_completed_stage": "B3_FULL_TRAINING",
    }
    atomic_json(ROOT / "analysis/stage_a_gate.json", gate)
    status = current_status()
    status["training"] = {variant: "PASS" for variant in VARIANTS}
    status["stage_a"] = "PASS"
    status["current_stage"] = "COMPLETE"
    save_status(status)
    write_report(gate, models, plan)

    # The integrated two-GPU PBS job reaches this function only after B2/B3
    # finish.  Start Stage B selection plus exactly-once test95 evaluation in
    # the same allocation so extraction does not need a second reservation.
    # Keep direct/login-node calls side-effect free; they can still use the
    # explicit runner when a GPU allocation is available.
    if stage_b_auto_enabled():
        status = current_status()
        status["stage_b"] = "RUNNING"
        status["current_stage"] = "STAGE_B_PIPELINE"
        save_status(status)
        gate_path = ROOT / "analysis/stage_a_gate.json"
        stage_b_gate = read_json(gate_path)
        stage_b_gate["STAGE_B"] = "RUNNING"
        atomic_json(gate_path, stage_b_gate)
        try:
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "tools/run_stage_b_after_stage_a.py"),
                    "--config",
                    str(ROOT / "configs/stage_b_memory_eval.yaml"),
                ],
                cwd=ROOT,
                check=True,
            )
        except subprocess.CalledProcessError:
            status = current_status()
            status["stage_a"] = "PASS"
            status["stage_b"] = "FAIL"
            status["current_stage"] = "STAGE_B_PIPELINE_FAILED"
            save_status(status)
            stage_b_gate = read_json(gate_path)
            stage_b_gate["STAGE_B"] = "FAIL"
            atomic_json(gate_path, stage_b_gate)
            raise
        status = current_status()
        status["stage_a"] = "PASS"
        status["stage_b"] = "PASS"
        status["current_stage"] = "COMPLETE"
        save_status(status)
        stage_b_gate = read_json(gate_path)
        stage_b_gate["STAGE_B"] = "PASS"
        stage_b_gate["last_successfully_completed_stage"] = "STAGE_B_TEST95_FINAL_EVALUATION"
        atomic_json(gate_path, stage_b_gate)


def consolidate_smoke_failure() -> None:
    models = {}
    for variant in VARIANTS:
        path = ROOT / f"analysis/stage_a_smoke_{variant}.json"
        models[variant] = read_json(path) if path.is_file() else {"variant": variant, "status": "NOT_RUN"}
    atomic_json(ROOT / "analysis/stage_a_smoke_summary.json", {
        "schema_version": "stage-a-v1",
        "status": "FAIL",
        "smoke_gate": "FAIL",
        "models": models,
        "test_split_used": False,
    })


def record_failure(stage: str, exit_code: int) -> None:
    status = current_status()
    # The Stage A gate is committed before the same-allocation Stage B runner
    # starts.  A Stage B failure must not rewrite successful Stage A training as
    # PARTIAL or claim that the backbone is not ready for a retry.
    if (
        stage == "FINAL_GATE"
        and status.get("stage_a") == "PASS"
        and all(status.get("training", {}).get(variant) == "PASS" for variant in VARIANTS)
    ):
        status["stage_b"] = "FAIL"
        status["current_stage"] = "STAGE_B_PIPELINE_FAILED"
        status["failure_exit_code"] = exit_code
        save_status(status)
        gate_path = ROOT / "analysis/stage_a_gate.json"
        gate = read_json(gate_path)
        gate.update(
            {
                "STAGE_A_GATE": "PASS",
                "READY_FOR_PROBE_TEST_METRIC": "YES",
                "STAGE_B": "FAIL",
                "failure_stage": "STAGE_B_PIPELINE",
                "exit_code": exit_code,
            }
        )
        atomic_json(gate_path, gate)
        return
    if stage.startswith("SMOKE_"):
        for variant in stage.removeprefix("SMOKE_").split("_"):
            if variant in VARIANTS:
                result = ROOT / f"analysis/stage_a_smoke_{variant}.json"
                status["smoke"][variant] = read_json(result).get("status", "FAIL") if result.is_file() else "FAIL"
        consolidate_smoke_failure()
    elif stage.startswith("TRAIN_"):
        for variant in stage.removeprefix("TRAIN_").split("_"):
            if variant in VARIANTS:
                result = ROOT / f"analysis/stage_a_training_{variant}.json"
                status["training"][variant] = read_json(result).get("status", "FAIL") if result.is_file() else "FAIL"
    elif stage.startswith("CACHE"):
        status["cache"] = "FAIL"
    passed_full = [variant for variant in VARIANTS if status["training"].get(variant) == "PASS"]
    final = "PARTIAL" if passed_full else "FAIL"
    status["stage_a"] = final
    status["current_stage"] = stage
    status["failure_exit_code"] = exit_code
    save_status(status)
    gate = {
        "schema_version": "stage-a-v1",
        "STAGE_A_GATE": final,
        "READY_FOR_PROBE_TEST_METRIC": "NO",
        "SMOKE": "PASS" if all(value == "PASS" for value in status["smoke"].values()) else "FAIL",
        "CACHE": status.get("cache", "PENDING"),
        **{variant: status["training"].get(variant, "PENDING") for variant in VARIANTS},
        "FAIRNESS": "INCOMPLETE",
        "TEST_UNUSED": "PASS",
        "last_successfully_completed_stage": passed_full[-1] if passed_full else stage,
        "failure_stage": stage,
        "exit_code": exit_code,
    }
    atomic_json(ROOT / "analysis/stage_a_gate.json", gate)
    write_report(gate, {}, read_json(ROOT / "analysis/stage_a_compute_plan.json") if (ROOT / "analysis/stage_a_compute_plan.json").is_file() else {})


def write_report(gate: dict, models: dict, plan: dict) -> None:
    lines = [
        "# STAGE A 완료 보고",
        "",
        f"STAGE_A_GATE: {gate['STAGE_A_GATE']}",
        f"READY_FOR_PROBE_TEST_METRIC: {gate['READY_FOR_PROBE_TEST_METRIC']}",
        "",
        "## Gate",
    ]
    for key in ("SMOKE", "CACHE", "B0", "B1", "B2", "B3", "FAIRNESS", "TEST_UNUSED"):
        lines.append(f"- {key}: {gate.get(key, 'PENDING')}")
    lines.extend(["", "## Training Budget", f"- epochs/model: {plan.get('selected_epochs_per_model', 'NOT_SELECTED')}", ""])
    for variant, result in models.items():
        lines.extend([
            f"## {variant}",
            f"- completed: {result.get('status')}",
            f"- best val loss: {result.get('best_val_loss')}",
            f"- peak VRAM bytes: {result.get('gpu_peak_memory_bytes')}",
            f"- runtime seconds: {result.get('runtime_seconds')}",
            f"- best checkpoint: {result.get('best_checkpoint')}",
            f"- last checkpoint: {result.get('last_checkpoint')}",
            "",
        ])
    lines.extend([
        "## STOP",
        "The Stage A gate performed no Behavior Cloning, action prediction, or test evaluation.",
        "Stage B val selection and test95 final evaluation are tracked separately in analysis/stage_b/pipeline_status.json.",
    ])
    target = ROOT / "analysis/stage_a_completion_report.md"
    temporary = target.with_name(target.name + ".stage-a-part")
    temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    os.replace(temporary, target)


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("init")
    sub.add_parser("smoke-gate")
    sub.add_parser("cache-gate")
    gpu = sub.add_parser("record-gpu")
    gpu.add_argument("--name", required=True)
    gpu.add_argument("--memory-mib", required=True, type=int)
    gpu.add_argument("--count", required=True, type=int)
    sub.add_parser("decide-budget")
    sub.add_parser("print-epochs")
    sub.add_parser("training-gate")
    failure = sub.add_parser("record-failure")
    failure.add_argument("--stage", required=True)
    failure.add_argument("--exit-code", required=True, type=int)
    args = parser.parse_args()
    if args.command == "init":
        init_runtime()
    elif args.command == "smoke-gate":
        smoke_gate()
    elif args.command == "cache-gate":
        cache_gate()
    elif args.command == "record-gpu":
        record_gpu(args.name, args.memory_mib, args.count)
    elif args.command == "decide-budget":
        decide_budget()
    elif args.command == "print-epochs":
        print_epochs()
    elif args.command == "training-gate":
        training_gate()
    else:
        record_failure(args.stage, args.exit_code)


if __name__ == "__main__":
    main()
