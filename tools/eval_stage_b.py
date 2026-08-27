#!/usr/bin/env python3
"""CPU-only Stage B retrieval metrics and report generation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from robocerebra_memory.eval.metrics import evaluate_shards, write_evaluation_outputs  # noqa: E402
from robocerebra_memory.eval.probes import ProbeBank  # noqa: E402
from robocerebra_memory.eval.representation_extractor import atomic_json, guard_split_access  # noqa: E402
from robocerebra_memory.eval.training import load_shards  # noqa: E402
from robocerebra_memory.eval.visualization import (  # noqa: E402
    write_comparison_visualizations,
    write_variant_visualizations,
)
from robocerebra_memory.stage_a import VARIANTS  # noqa: E402


def _atomic_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".stage-b-part")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_comparisons(report_root: Path, representation_root: Path, split: str) -> None:
    if split not in {"val", "test"}:
        raise ValueError(f"comparison split must be val or test, got {split}")
    split_report_root = report_root if split == "val" else report_root / "final_test"
    summaries = []
    depth_rows = []
    distance_rows = []
    transition_rows = []
    sampling_hashes = set()
    found_variants = []
    for variant in VARIANTS:
        directory = split_report_root / variant
        summary_path = directory / "summary.json"
        if not summary_path.is_file():
            continue
        found_variants.append(variant)
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != "stage-b-evaluation-v1" or payload.get("split") != split:
            raise RuntimeError(f"invalid Stage B comparison summary: {summary_path}")
        representation_manifest = json.loads(
            (representation_root / variant / split / "manifest.json").read_text(encoding="utf-8")
        )
        sampling_hashes.add(representation_manifest["sampling_sha256"])
        for target, values in payload["metrics"].items():
            summaries.append({"variant": variant, "target": target, **values})
        for filename, destination in (
            ("memory_depth.csv", depth_rows),
            ("current_retention_by_distance.csv", distance_rows),
            ("current_retention_by_transition.csv", transition_rows),
        ):
            with (directory / filename).open(encoding="utf-8", newline="") as stream:
                for row in csv.DictReader(stream):
                    destination.append({"variant": variant, **row})
    if len(found_variants) == len(VARIANTS):
        if len(sampling_hashes) != 1:
            raise RuntimeError("B0/B1/B2/B3 comparison sampling manifests do not match")
        comparison = split_report_root / "comparison"
        _atomic_csv(comparison / "B0_B1_B2_B3_summary.csv", summaries)
        _atomic_csv(comparison / "memory_depth_comparison.csv", depth_rows)
        _atomic_csv(comparison / "retention_distance_comparison.csv", distance_rows)
        _atomic_csv(comparison / "transition_robustness_comparison.csv", transition_rows)
        write_comparison_visualizations(split_report_root)


def run_parallel_final_metrics(config_path: Path, report_root: Path, representation_root: Path) -> None:
    """Evaluate four independent variants concurrently, then merge once."""
    processes = []
    for variant in VARIANTS:
        env = os.environ.copy()
        for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            env[name] = "2"
        command = [
            sys.executable, str(Path(__file__).resolve()), "--variant", variant,
            "--split", "test", "--final-test", "--config", str(config_path),
            "--parallel-child",
        ]
        print(f"FINAL_TEST_METRICS_PARALLEL_START {variant}", flush=True)
        processes.append((variant, subprocess.Popen(command, cwd=ROOT, env=env)))
    failures = []
    for variant, process in processes:
        if process.wait() != 0:
            failures.append(variant)
    if failures:
        raise RuntimeError(f"parallel final-test metrics failed for {','.join(failures)}")
    write_comparisons(report_root, representation_root, "test")
    atomic_json(
        report_root / "final_test/FINAL_TEST_BATCH_COMPLETED.json",
        {"status": "COMPLETE", "variants": list(VARIANTS), "split": "test", "parallel_workers": 4},
    )
    print("FINAL_TEST_METRICS_PARALLEL_COMPLETE B0 B1 B2 B3", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--final-test", action="store_true")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/stage_b_memory_eval.yaml")
    parser.add_argument("--representation-dir", type=Path)
    parser.add_argument("--probe-checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--parallel-child", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    guard_split_access(args.split, args.final_test)
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    representation_root = ROOT / config["outputs"]["representation_root"]
    report_root = ROOT / config["outputs"]["report_root"]
    representation_dir = args.representation_dir or representation_root / args.variant / args.split
    probe_path = args.probe_checkpoint or ROOT / config["outputs"]["probe_root"] / args.variant / "selected_probes.pt"
    output = args.output_dir or (
        report_root / args.variant if args.split == "val" else report_root / "final_test" / args.variant
    )
    # The exactly-once sentinel is canonical and cannot be bypassed with --output-dir.
    completion = report_root / "final_test" / args.variant / "FINAL_TEST_COMPLETED.json"
    batch_completion = report_root / "final_test/FINAL_TEST_BATCH_COMPLETED.json"
    automatic_parallel = bool(config.get("execution", {}).get("parallel_final_metrics", False))
    automatic_parallel = automatic_parallel and bool(os.environ.get("PBS_JOBID"))
    if args.split == "test" and automatic_parallel and not args.parallel_child:
        if args.variant == "B0":
            if args.representation_dir or args.probe_checkpoint or args.output_dir:
                raise RuntimeError("automatic parallel final metrics do not accept path overrides")
            if any((report_root / "final_test" / variant / "FINAL_TEST_COMPLETED.json").exists() for variant in VARIANTS):
                raise RuntimeError("parallel final-test metrics require a clean exactly-once gate")
            run_parallel_final_metrics(args.config, report_root, representation_root)
            return
        if batch_completion.is_file() and completion.is_file():
            print(json.dumps({"status": "PASS", "variant": args.variant, "split": "test", "reused": True}, indent=2))
            return
    if args.split == "test" and completion.exists():
        raise RuntimeError("final test metrics already exist; exactly-once gate refuses a second evaluation")
    payloads = load_shards(representation_dir, args.split, args.variant)
    expected_count = int(config["dataset"]["expected_trajectories"][args.split])
    if len(payloads) != expected_count:
        raise RuntimeError(
            f"Stage B {args.split} representation trajectory count mismatch: expected {expected_count}, found {len(payloads)}"
        )
    checkpoint = torch.load(probe_path, map_location="cpu", weights_only=False)
    if (
        checkpoint.get("schema_version") != "stage-b-probes-v1"
        or checkpoint.get("variant") != args.variant
        or checkpoint.get("selection_split") != "val"
        or not checkpoint.get("backbone_frozen")
    ):
        raise RuntimeError("selected Stage B probe checkpoint contract mismatch")
    representation_checkpoint = payloads[0]["checkpoint"]
    trained_checkpoint = checkpoint.get("stage_a_checkpoint", {})
    if (
        representation_checkpoint.get("state_dict_sha256") != trained_checkpoint.get("state_dict_sha256")
        or representation_checkpoint.get("global_update") != trained_checkpoint.get("global_update")
        or representation_checkpoint.get("completed_epoch") != trained_checkpoint.get("completed_epoch")
    ):
        raise RuntimeError("probe/backbone representation checkpoint identity mismatch")
    probes = ProbeBank(seed=int(checkpoint["seed"]))
    probes.load_state_dict(checkpoint["probe_state_dict"], strict=True)
    probes.eval()
    evaluated = evaluate_shards(
        payloads, probes,
        resamples=int(config["evaluation"]["bootstrap_resamples"]),
        seed=int(config["evaluation"]["bootstrap_seed"]),
    )
    representation_manifest = json.loads((representation_dir / "manifest.json").read_text(encoding="utf-8"))
    provenance = {
        "probe_checkpoint": str(probe_path),
        "probe_checkpoint_sha256": file_sha256(probe_path),
        "stage_a_checkpoint": representation_checkpoint,
        "sampling_sha256": representation_manifest["sampling_sha256"],
    }
    write_evaluation_outputs(output, args.variant, args.split, evaluated, provenance)
    write_variant_visualizations(output, args.variant, evaluated)
    if args.split == "test":
        atomic_json(
            completion,
            {
                "status": "COMPLETE", "variant": args.variant, "split": "test",
                "probe_checkpoint": str(probe_path),
                "probe_checkpoint_sha256": provenance["probe_checkpoint_sha256"],
                "stage_a_checkpoint": representation_checkpoint,
                "sampling_sha256": provenance["sampling_sha256"],
                "trajectory_count": len(payloads),
                "exactly_once_gate": True,
            },
        )
        if not args.parallel_child:
            write_comparisons(report_root, representation_root, "test")
    else:
        write_comparisons(report_root, representation_root, "val")
    print(json.dumps({"status": "PASS", "variant": args.variant, "split": args.split, "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
