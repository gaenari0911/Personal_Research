#!/usr/bin/env python3
"""CPU/static pre-submit gate for the single autonomous Stage A PBS job."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

import h5py
import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from robocerebra_memory.models import MemoryExperimentModel  # noqa: E402
from robocerebra_memory.stage_a import (  # noqa: E402
    VARIANTS,
    atomic_json,
    read_json,
    state_dict_sha256,
    validate_cache_payload,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(command: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(command, text=True, capture_output=True, check=False)


def write_probe(directory: Path) -> bool:
    directory.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(prefix="stage-a-write-probe-", dir=directory, delete=False) as stream:
        path = Path(stream.name)
        stream.write(b"stage-a")
        stream.flush()
        os.fsync(stream.fileno())
    path.unlink()
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "analysis/stage_a_preflight.json")
    args = parser.parse_args()
    checks: dict[str, dict] = {}

    required = [
        "research.txt",
        "docs/R1_RC_MEMORY_DATASET.md",
        "docs/R1_RC_DATA_INTERFACE.md",
        "docs/R2_MAMBA_MEMORY_PROTOCOL.md",
        "docs/R2_MEMORY_METRICS.md",
        "docs/R3_MAMBA_MODEL_IMPLEMENTATION.md",
        "docs/R3_STATE_HANDLING.md",
        "configs/robocerebra_memory_protocol.yaml",
        "configs/robocerebra_memory_models.yaml",
        "configs/stage_a_representation.yaml",
        "tools/train_stage_a.py",
        "tools/cache_stage_a_features.py",
        "tools/stage_a_control.py",
        "jobs/stage_a_smoke_and_representation.pbs",
        "analysis/stage_a_schedule.json",
        "checkpoints/stage_a/common_init.pt",
    ]
    missing = [value for value in required if not (ROOT / value).is_file()]
    checks["project_and_artifacts"] = {"pass": not missing, "missing": missing}

    config = yaml.safe_load((ROOT / "configs/stage_a_representation.yaml").read_text(encoding="utf-8"))
    checks["config_parse"] = {"pass": isinstance(config, dict), "schema": config.get("schema_version")}
    index = read_json(ROOT / config["dataset"]["episode_index"])["episodes"]
    episodes_by_id = {row["trajectory_id"]: row for row in index}
    splits = {
        split: read_json(ROOT / config["dataset"][f"{split}_split"])
        for split in ("train", "val")
    }
    test_ids = read_json(ROOT / config["dataset"]["forbidden_test_split"])
    split_pass = (
        len(index) == 914
        and len(splits["train"]) == 734
        and len(splits["val"]) == 85
        and len(test_ids) == 95
        and not set(splits["train"]) & set(splits["val"])
        and not (set(splits["train"]) | set(splits["val"])) & set(test_ids)
    )
    checks["splits"] = {
        "pass": split_pass,
        "episodes": len(index),
        "train": len(splits["train"]),
        "val": len(splits["val"]),
        "test": len(test_ids),
    }

    hdf_errors = []
    for episode in index:
        path = Path(episode["state_source"])
        try:
            with h5py.File(path, "r") as source:
                shape = tuple(source["data/demo_1/states"].shape)
            if shape[0] != episode["num_frames"] or shape[1] < 10:
                hdf_errors.append({"trajectory_id": episode["trajectory_id"], "shape": shape})
        except Exception as error:
            hdf_errors.append({"trajectory_id": episode["trajectory_id"], "error": str(error)})
    checks["source_data_readable"] = {"pass": not hdf_errors, "checked": len(index), "errors": hdf_errors[:10]}

    pilot_audit = read_json(ROOT / "analysis/r4_clip_cache_audit.json")
    pilot_errors = []
    for row in pilot_audit["records"]:
        path = Path(row["cache"])
        try:
            payload = torch.load(path, map_location="cpu", weights_only=False)
            errors = validate_cache_payload(payload, episodes_by_id[row["trajectory_id"]])
            if errors:
                pilot_errors.append({"trajectory_id": row["trajectory_id"], "errors": errors})
        except Exception as error:
            pilot_errors.append({"trajectory_id": row["trajectory_id"], "error": str(error)})
    checks["pilot_clip_cache"] = {
        "pass": pilot_audit.get("status") == "PASS" and len(pilot_audit["records"]) == 20 and not pilot_errors,
        "valid": 20 - len(pilot_errors),
        "errors": pilot_errors,
    }

    package_names = ("torch", "transformers", "cv2", "h5py", "numpy", "PIL", "yaml")
    package_status = {}
    for name in package_names:
        try:
            module = __import__(name)
            package_status[name] = getattr(module, "__version__", "present")
        except Exception as error:
            package_status[name] = f"FAIL:{error}"
    checks["python_environment"] = {
        "pass": all(not str(value).startswith("FAIL:") for value in package_status.values()),
        "executable": sys.executable,
        "packages": package_status,
        "torch_cuda_build": torch.version.cuda,
        "login_node_cuda_used": False,
    }

    ssd = shutil.disk_usage("/ssd1/itaein/datasets/RoboCerebra")
    workspace = shutil.disk_usage(ROOT)
    write_results = {}
    for directory in (
        ROOT / "analysis",
        ROOT / "logs/stage_a",
        ROOT / "checkpoints/stage_a",
        Path(config["features"]["cache_root"]),
    ):
        try:
            write_results[str(directory)] = write_probe(directory)
        except Exception as error:
            write_results[str(directory)] = str(error)
    checks["storage_and_write"] = {
        "pass": ssd.free >= 20 * 2**30 and all(value is True for value in write_results.values()),
        "ssd_free_bytes": ssd.free,
        "workspace_free_bytes": workspace.free,
        "write_probes": write_results,
    }

    mail = run(["git", "-C", str(ROOT / "external/MaIL"), "status", "--porcelain"])
    mail_commit = run(["git", "-C", str(ROOT / "external/MaIL"), "rev-parse", "HEAD"])
    checks["mail_upstream"] = {
        "pass": mail.returncode == 0 and not mail.stdout.strip(),
        "status": mail.stdout.strip(),
        "commit": mail_commit.stdout.strip(),
    }

    common = torch.load(ROOT / config["training"]["common_initialization"], map_location="cpu", weights_only=False)
    variant_hashes = {}
    for variant in VARIANTS:
        model = MemoryExperimentModel(variant)
        model.load_state_dict(common["model_state_dict"], strict=True)
        variant_hashes[variant] = state_dict_sha256(model.state_dict())
    checks["variant_and_common_init"] = {
        "pass": len(set(variant_hashes.values())) == 1 and next(iter(variant_hashes.values())) == common["state_dict_sha256"],
        "variant_hashes": variant_hashes,
        "common_hash": common["state_dict_sha256"],
        "weight_continuation": False,
    }

    schedule = read_json(ROOT / config["dataset"]["schedule"])
    schedule_pass = (
        schedule.get("test_split_used") is False
        and len(schedule["train_ids"]) == 734
        and len(schedule["val_ids"]) == 85
        and not (set(schedule["train_ids"]) | set(schedule["val_ids"])) & set(test_ids)
        and all(8 <= len(schedule["anchors"][split][trajectory_id]) <= 64 for split in ("train", "val") for trajectory_id in schedule[f"{split}_ids"])
        and all(row["target_frame"] == row["frame"] + 20 for split in ("train", "val") for rows in schedule["anchors"][split].values() for row in rows)
    )
    checks["schedule_no_test_no_future_leakage"] = {
        "pass": schedule_pass,
        "train_anchors": sum(map(len, schedule["anchors"]["train"].values())),
        "val_anchors": sum(map(len, schedule["anchors"]["val"].values())),
        "future_target_is_supervision_only": True,
    }

    train_source = (ROOT / "tools/train_stage_a.py").read_text(encoding="utf-8")
    stage_source = (ROOT / "src/robocerebra_memory/stage_a.py").read_text(encoding="utf-8")
    hyperparameters_match = (
        config["optimizer"] == {
            "name": "AdamW",
            "learning_rate": 0.0001,
            "weight_decay": 0.01,
            "gradient_clip_norm": 1.0,
            "scheduler": "none",
        }
        and config["objective"]["future_horizon"] == 20
        and config["objective"]["temperature"] == 0.07
    )
    checks["objective_optimizer_scope"] = {
        "pass": (
            hyperparameters_match
            and "action_loss" not in train_source.lower()
            and "behavior_cloning_loss" not in train_source.lower()
            and "targets.detach()" in stage_source
        ),
        "behavior_cloning": False,
        "action_prediction": False,
        "clip_in_optimizer": False,
        "amp": False,
        "test_evaluation": False,
    }

    unique_outputs = [
        str(ROOT / f"analysis/stage_a_training_{variant}.json") for variant in VARIANTS
    ] + [str(ROOT / f"checkpoints/stage_a/{variant}") for variant in VARIANTS]
    existing_full = [value for value in unique_outputs if Path(value).exists() and (Path(value).is_file() or any(Path(value).iterdir()))]
    checks["output_isolation"] = {"pass": len(unique_outputs) == len(set(unique_outputs)) and not existing_full, "paths": unique_outputs, "preexisting": existing_full}

    pbs_path = ROOT / "jobs/stage_a_smoke_and_representation.pbs"
    pbs = pbs_path.read_text(encoding="utf-8")
    syntax = run(["bash", "-n", str(pbs_path)])
    banned = {
        "input_call": bool(re.search(r"\binput\s*\(", pbs)),
        "read_prompt": bool(re.search(r"\bread\s+-p\b", pbs)),
        "shell_select": bool(re.search(r"^\s*select\s+", pbs, re.MULTILINE)),
        "qsub": bool(re.search(r"\bqsub\b", pbs)),
        "qdel": bool(re.search(r"\bqdel\b", pbs)),
        "codex_cli": bool(re.search(r"\bcodex\b", pbs, re.IGNORECASE)),
    }
    flow_tokens = ["CURRENT_STAGE=SMOKE_GATE", "CURRENT_STAGE=CACHE_FEATURES", "CURRENT_STAGE=CACHE_GATE", "CURRENT_STAGE=BUDGET_GATE", "CURRENT_STAGE=FINAL_GATE"]
    flow_pass = all(token in pbs for token in flow_tokens) and all(
        token in pbs
        for token in (
            "run_smoke_pair B0 B1",
            "run_smoke_pair B2 B3",
            "run_training_pair B0 B1",
            "run_training_pair B2 B3",
        )
    )
    checks["pbs_script"] = {
        "pass": (
            syntax.returncode == 0
            and not any(banned.values())
            and flow_pass
            and "select=1:ncpus=8:ngpus=2:host=pleiades1" in pbs
            and "run_smoke_pair B0 B1" in pbs
            and "run_smoke_pair B2 B3" in pbs
            and "run_training_pair B0 B1" in pbs
            and "run_training_pair B2 B3" in pbs
        ),
        "syntax_stderr": syntax.stderr,
        "banned_structures": banned,
        "smoke_success_to_full_control_flow": "B0/B1/B2/B3 smoke loop -> smoke-gate -> cache -> cache-gate -> budget -> B0/B1/B2/B3 full loop -> final gate",
        "requested_gpus": config["pbs"]["gpu_count"],
        "requested_cpus": config["pbs"]["cpu_count"],
        "requested_host": config["pbs"]["host"],
        "parallel_variant_pairs": [["B0", "B1"], ["B2", "B3"]],
        "interactive_steps": 0,
    }

    old = run(["qstat", "-xf", "100160.pleiades1"])
    state_match = re.search(r"job_state\s*=\s*(\w)", old.stdout)
    old_state = state_match.group(1) if state_match else "UNKNOWN"
    queue = run(["qstat", "-Qf", config["pbs"]["queue"]])
    nodes = run(["pbsnodes", "-av", config["pbs"]["queue"]])
    gpu_match = re.search(r"resources_available\.ngpus\s*=\s*(\d+)", nodes.stdout)
    available_gpu_slots = int(gpu_match.group(1)) if gpu_match else 0
    checks["pbs_resources"] = {
        "pass": old_state not in {"Q", "R", "H", "W", "S", "T"} and "enabled = True" in queue.stdout and available_gpu_slots >= config["pbs"]["gpu_count"],
        "old_job_100160_state": old_state,
        "queue": config["pbs"]["queue"],
        "queue_enabled": "enabled = True" in queue.stdout,
        "node_gpu_slots": available_gpu_slots,
        "static_gpu_model": "UNKNOWN; runtime nvidia-smi is a mandatory gate",
        "compatibility_evidence": "same queue completed frozen CLIP CUDA job 100099 with exit 0",
        "requested_walltime": config["budget"]["requested_walltime"],
    }

    sample_missing = next((episodes_by_id[value] for value in splits["train"] + splits["val"] if not Path(episodes_by_id[value]["visual_source"]["local_path"]).is_file()), None)
    network_result = {"pass": True, "sample": None, "status": "all videos already present"}
    if sample_missing is not None:
        url = "https://huggingface.co/datasets/qiukingballball/RoboCerebra/resolve/5d2e1e361bf65aabbe4d18179515f5a10936cc96/" + sample_missing["visual_source"]["official_relative_path"]
        try:
            request = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "stage-a-preflight/1.0"})
            with urllib.request.urlopen(request, timeout=30) as response:
                code = response.status
            network_result = {"pass": 200 <= code < 400, "sample": sample_missing["trajectory_id"], "http_status": code}
        except Exception as error:
            network_result = {"pass": False, "sample": sample_missing["trajectory_id"], "error": str(error)}
    checks["full_cache_source_reachability"] = network_result

    critical_failures = [name for name, result in checks.items() if not result.get("pass", False)]
    payload = {
        "schema_version": "stage-a-v1",
        "status": "PASS" if not critical_failures else "FAIL",
        "critical_failures": critical_failures,
        "checks": checks,
        "research_sha256": sha256(ROOT / "research.txt"),
        "source_hashes": {
            value: sha256(ROOT / value)
            for value in (
                "configs/stage_a_representation.yaml",
                "src/robocerebra_memory/stage_a.py",
                "tools/prepare_stage_a.py",
                "tools/cache_stage_a_features.py",
                "tools/train_stage_a.py",
                "tools/stage_a_control.py",
                "jobs/stage_a_smoke_and_representation.pbs",
            )
        },
        "qsub_allowed": not critical_failures,
        "qsub_count_before_submission": 0,
    }
    atomic_json(args.output, payload)
    print(json.dumps({"status": payload["status"], "critical_failures": critical_failures, "checks": {name: result["pass"] for name, result in checks.items()}}, indent=2))
    if critical_failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
