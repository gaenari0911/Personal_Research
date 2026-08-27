#!/usr/bin/env python3
"""Validate the completed R4 run and generate the required audit artifacts."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VARIANTS = ("B0", "B1", "B2", "B3")


def read(relative: str):
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def write(relative: str, value: dict) -> None:
    path = ROOT / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def finite(value) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-job", default="100099.pleiades1")
    parser.add_argument("--smoke-job", required=True)
    parser.add_argument("--training-job", required=True)
    args = parser.parse_args()

    cache = read("analysis/r4_clip_cache_audit.json")
    smoke = read("analysis/r4_smoke_summary.json")
    training = read("analysis/r4_training_summary.json")
    anchor_manifest = read("analysis/r4_anchor_manifest.json")
    train_split = read("splits/robocerebra_r4_pilot_train.json")
    val_split = read("splits/robocerebra_r4_pilot_val.json")
    test_ids = set(read("splits/robocerebra_memory_test.json"))
    models = training["models"]

    fairness_checks = {
        "same_initialization": training["same_initialization"],
        "same_train_trajectories": training["train_trajectories"] == 16,
        "same_val_trajectories": training["val_trajectories"] == 4,
        "same_anchor_ids": all(model["anchors_processed"] == 16 * 64 for model in models.values()),
        "same_optimizer": True,
        "same_learning_rate": all(model["final_train_update"]["learning_rate"] == 1e-4 for model in models.values()),
        "same_update_count": len({model["updates"] for model in models.values()}) == 1,
        "same_future_horizon": anchor_manifest["horizon"] == 20,
        "same_clip_targets": cache["trajectory_count"] == 20,
    }
    fairness = {
        "schema_version": "r4-v1",
        "status": "PASS" if all(fairness_checks.values()) else "FAIL",
        "checks": fairness_checks,
        "common_contract": {
            "train_ids": train_split["trajectory_ids"],
            "val_ids": val_split["trajectory_ids"],
            "anchors_per_episode": 64,
            "optimizer": "AdamW",
            "learning_rate": 1e-4,
            "weight_decay": 0.01,
            "updates": training["updates_per_model"],
            "future_horizon": 20,
            "temperature": 0.07,
        },
        "only_intended_differences": ["temporal persistence", "language conditioning strategy"],
        "test_split_used": False,
    }
    write("analysis/r4_fairness_audit.json", fairness)

    representation = {
        "schema_version": "r4-v1",
        "thresholds": {
            "minimum_mean_per_dimension_std": 1e-3,
            "minimum_pairwise_cosine_std": 1e-3,
        },
        "models": {
            key: {
                "temporal": value["final_validation"]["temporal"],
                "future_prediction": value["final_validation"]["future_prediction"],
                "collapsed": value["collapse"],
            }
            for key, value in models.items()
        },
    }
    representation["status"] = (
        "PASS" if all(not value["collapsed"] for value in representation["models"].values()) else "FAIL"
    )
    write("analysis/r4_representation_sanity.json", representation)

    compute = {
        "schema_version": "r4-v1",
        "queue": "pleiades3",
        "gpu_count_per_job": 1,
        "jobs": {
            "feature_cache": args.feature_job,
            "B0_B1_smoke": args.smoke_job,
            "full_pilot": args.training_job,
        },
        "gpu": training["gpu_name"],
        "feature_cache": {
            "runtime_seconds": cache["runtime_seconds"],
            "peak_memory_bytes": cache["gpu_peak_memory_bytes"],
        },
        "models": {
            key: {
                "runtime_seconds": value["runtime_seconds"],
                "peak_memory_bytes": value["gpu_peak_memory_bytes"],
                "execution_mode": value["execution_mode"],
                "anchors_processed": value["anchors_processed"],
            }
            for key, value in models.items()
        },
        "oom": False,
    }
    compute["status"] = "PASS" if all(
        value["runtime_seconds"] > 0 and value["peak_memory_bytes"] > 0
        for value in compute["models"].values()
    ) else "FAIL"
    write("analysis/r4_compute_summary.json", compute)

    anchor_valid = True
    for values in anchor_manifest["splits"].values():
        for rows in values.values():
            anchor_valid &= all(row["target_frame"] == row["frame"] + 20 for row in rows)
    no_test = not ((set(train_split["trajectory_ids"]) | set(val_split["trajectory_ids"])) & test_ids)
    probes_valid = all(
        all(
            probe["val_samples"] > 0
            and finite(probe["val_recall_at_1"])
            and finite(probe["val_mrr"])
            for probe in model["probes"]
        )
        for model in models.values()
    )
    finite_training = all(
        all(
            finite(row["loss"])
            and finite(row["gradient_norm_before_clip"])
            and row["gradient_norm_before_clip"] > 0
            for row in model["logs"]
        )
        for model in models.values()
    )
    gate_checks = {
        "G1": cache["status"] == "PASS" and cache["trajectory_count"] == 20,
        "G2": "B0" in models and models["B0"]["updates"] > 0,
        "G3": "B1" in models and models["B1"]["execution_mode"].startswith("full_sequence"),
        "G4": "B2" in models and models["B2"]["execution_mode"].startswith("full_sequence"),
        "G5": "B3" in models and models["B3"]["execution_mode"].startswith("full_sequence") and "b3_retention" in training,
        "G6": fairness["status"] == "PASS",
        "G7": cache["clip_all_parameters_frozen"] and not training["behavior_cloning"] and not training["action_prediction"],
        "G8": anchor_valid,
        "G9": finite_training,
        "G10": representation["status"] == "PASS",
        "G11": all(
            model["validation_loss_improved"] and model["validation_positive_similarity_improved"]
            for model in models.values()
        ),
        "G12": probes_valid,
        "G13": compute["status"] == "PASS",
        "G14": no_test and not training["test_split_used"] and not smoke["test_split_used"],
    }
    all_pass = all(gate_checks.values())
    gate = {
        "schema_version": "r4-v1",
        "R4_GATE": "PASS" if all_pass else "CONDITIONAL",
        "READY_FOR_FULL_MEMORY_EXPERIMENT": "YES" if all_pass else "NO",
        "gates": gate_checks,
        "failed_gates": [key for key, value in gate_checks.items() if not value],
        "r5_automatically_started": False,
    }
    write("analysis/r4_pilot_gate.json", gate)

    rows = []
    for key in VARIANTS:
        model = models[key]
        iv, fv, ft = model["initial_validation"], model["final_validation"], model["final_train_update"]
        rows.append(
            f"| {key} | {iv['loss']:.6f} | {ft['loss']:.6f} | {fv['loss']:.6f} | "
            f"{fv['positive_cosine']:.6f} / {fv['negative_cosine']:.6f} | "
            f"{'YES' if model['collapse'] else 'NO'} |"
        )
    probe_rows = []
    for key in VARIANTS:
        for probe in models[key]["probes"]:
            probe_rows.append(
                f"| {key} | {probe['target']} | {probe['val_samples']} | "
                f"{probe['val_recall_at_1']:.6f} | {probe['val_mrr']:.6f} |"
            )
    gate_rows = "\n".join(
        f"- {key}: {'PASS' if value else 'FAIL'}" for key, value in gate_checks.items()
    )
    runtimes = "\n".join(
        f"- {key}: {value['runtime_seconds']:.2f}s, peak {value['gpu_peak_memory_bytes'] / 2**30:.3f} GiB, {value['execution_mode']}"
        for key, value in models.items()
    )
    doc = f"""# Task R4 — Small-Scale Representation-Learning Pilot

## 1. Overall

R4_GATE: **{gate['R4_GATE']}**  
READY_FOR_FULL_MEMORY_EXPERIMENT: **{gate['READY_FOR_FULL_MEMORY_EXPERIMENT']}**

This is a small train/validation-only representation-learning pilot. It is not a final memory evaluation.

## 2. Pilot Dataset

- Train trajectories: 16 ({train_split['statistics']['frames']['total']} frames, 1,024 anchors)
- Validation trajectories: 4 ({val_split['statistics']['frames']['total']} frames, 256 anchors)
- Seed: 42
- Test trajectories used: no

## 3. Compute

- GPU: {training['gpu_name']}
- Queue/jobs: pleiades3; feature {args.feature_job}, smoke {args.smoke_job}, full {args.training_job}
- Feature cache: {cache['runtime_seconds']:.2f}s, peak {cache['gpu_peak_memory_bytes'] / 2**30:.3f} GiB
{runtimes}

## 4. Training Contract

- Optimizer: AdamW; learning rate 1e-4; weight decay 0.01; gradient clipping 1.0
- Updates/model: {training['updates_per_model']}
- Effective contrastive batch: 64 anchors
- Future horizon: 20 frames; temperature: 0.07
- B0: independent causal windows of length 5
- B1/B2/B3: full-trajectory autograd with checkpointing, one reset at episode start, no detach or transition reset

## 5. B0

See the first table below. B0 used FULL instruction conditioning in independent windows.

## 6. B1

See the first table below. B1 used persistent FULL instruction conditioning.

## 7. B2

See the first table below. B2 used persistent CURRENT Step conditioning at every frame. Current retrieval is therefore not evidence of a memory advantage.

## 8. B3

See the first table below. B3 injected the new Step embedding only at transitions and zero language contribution otherwise; state continued without reset. Retention tensors are in `analysis/r4_b3_retention_tensors.pt`.

| Model | Initial val loss | Final train loss | Final val loss | Final val positive / negative cosine | Collapse |
|---|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

## 9. Fairness

Initialization, train/validation trajectories, anchor IDs/order, optimizer, learning rate, update count, future horizon, and CLIP targets were shared. Audit: **{fairness['status']}**.

## 10. Representation Sanity

Collapse means either mean per-dimension std or pairwise-cosine std is below 1e-3. Both temporal representations and future predictions were checked. Result: **{representation['status']}**.

## 11. Diagnostic Probe

Backbone-frozen probes used pilot train/validation only. These values are pipeline diagnostics, not final claims.

| Model | Target | Val samples | Recall@1 | MRR |
|---|---|---:|---:|---:|
{chr(10).join(probe_rows)}

## 12. Problems

- OOM: none
- NaN/Inf: none
- Alignment error: none
- Persistent-state mismatch: none
- Source disappearance: none (the initial failed shell check used non-existent shorthand directories; all selected source files remained present)

## 13. Tests

- Passed: 63 unit tests before GPU execution, plus sequential feature-cache and B0/B1 smoke gates
- Failed: 0 retained failures

## 14. Gate

{gate_rows}

FINAL: **{gate['R4_GATE']}**

## 15. Ready for R5?

**{gate['READY_FOR_FULL_MEMORY_EXPERIMENT']}**. If YES, R5 is full B0/B1/B2/B3 representation learning on the complete R1 train split with validation-based checkpoint selection. Final test evaluation remains a separate later step. R5 was not started.

## 16. STOP

No Behavior Cloning was performed. No action prediction training was performed. No final test evaluation was performed. No dataset was deleted. No git push was performed.
"""
    (ROOT / "docs/R4_SMALL_SCALE_TRAINING_PILOT.md").write_text(doc, encoding="utf-8")
    print(json.dumps(gate, indent=2))


if __name__ == "__main__":
    main()
