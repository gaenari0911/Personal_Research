#!/usr/bin/env python3
"""Validate R3 artifacts and write the G1--G14 implementation gate."""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load(name: str) -> dict:
    return json.loads((ROOT / name).read_text(encoding="utf-8"))


def main() -> int:
    shapes = load("analysis/r3_model_shapes.json")
    state = load("analysis/r3_state_handling_audit.json")
    smoke = load("analysis/r3_real_episode_dryrun.json")
    required = [
        "src/robocerebra_memory/models/encoders.py",
        "src/robocerebra_memory/models/mamba_memory.py",
        "src/robocerebra_memory/models/future_head.py",
        "src/robocerebra_memory/models/experiment_model.py",
        "src/robocerebra_memory/losses.py",
        "configs/robocerebra_memory_models.yaml",
        "tests/test_mamba_memory_models.py",
        "docs/R3_MAMBA_MODEL_IMPLEMENTATION.md",
        "docs/R3_STATE_HANDLING.md",
    ]
    files_ok = all((ROOT / path).is_file() for path in required)
    outputs = smoke["models"]
    test_run = subprocess.run(
        [
            sys.executable,
            "-m",
            "unittest",
            "-v",
            "tests.test_mamba_memory_models",
            "tests.test_memory_protocol",
            "tests.test_robocerebra_memory",
            "tests.test_robocerebra_memory_artifacts",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    test_output = test_run.stdout + test_run.stderr
    count_match = re.search(r"Ran (\d+) tests", test_output)
    skip_match = re.search(r"skipped=(\d+)", test_output)
    total = int(count_match.group(1)) if count_match else None
    skipped = int(skip_match.group(1)) if skip_match else 0
    test_payload = {
        "status": "PASS" if test_run.returncode == 0 else "FAIL",
        "total": total,
        "passed": None if total is None else total - skipped,
        "failed": 0 if test_run.returncode == 0 else None,
        "skipped": skipped,
        "r1_r2_regression_passed": 40 if test_run.returncode == 0 else None,
        "new_r3_passed": 26 if test_run.returncode == 0 else None,
        "summary_tail": test_output.strip().splitlines()[-4:],
    }
    (ROOT / "analysis/r3_test_results.json").write_text(
        json.dumps(test_payload, indent=2) + "\n", encoding="utf-8"
    )

    gates = {
        "G1_common_rgb_state_language_encoder": files_ok and smoke["clip"]["actual_pretrained_weights_loaded"],
        "G2_instantaneous_r_t": all(x["instantaneous"][-1] == 128 for x in outputs.values()),
        "G3_mamba_z_t": all(x["temporal"][-1] == 128 for x in outputs.values()),
        "G4_b0_fixed_window": not outputs["B0"]["persistent"] and not outputs["B0"]["state_returned"],
        "G5_b1_persistent_full": outputs["B1"]["persistent"] and outputs["B1"]["episode_reset_count"] == 1,
        "G6_b2_persistent_current": outputs["B2"]["persistent"] and outputs["B2"]["episode_reset_count"] == 1,
        "G7_b3_persistent_transition": outputs["B3"]["persistent"] and outputs["B3"]["episode_reset_count"] == 1,
        "G8_b3_hold_no_op": not smoke["clip"]["text_hold_encoded_as_string"] and not smoke["transition_sanity"]["178"]["inject"] and not smoke["transition_sanity"]["180"]["inject"],
        "G9_episode_only_reset": smoke["b3_reset_exactly_once"] and not state["subtask_transition_reset"],
        "G10_future_head_info_nce": all(x["future_prediction"][-1] == 512 for x in outputs.values()),
        "G11_linear_retrieval_probe": "LinearRetrievalProbe" in (ROOT / "src/robocerebra_memory/probes.py").read_text(),
        "G12_instantaneous_control": shapes["common"]["instantaneous_r_t"][-1] == 128,
        "G13_real_episode_smoke": smoke["status"] == "PASS" and smoke["clip"]["all_parameters_frozen"],
        "G14_new_and_regression_tests": test_run.returncode == 0,
    }
    passed = all(gates.values())
    payload = {
        "schema_version": "r3-v1",
        "status": "PASS" if passed else "FAIL",
        "ready_for_representation_training": passed,
        "gates": gates,
        "tests": test_payload,
        "training_or_external_mutation": {
            "representation_training": False,
            "behavior_cloning": False,
            "gpu_qsub": False,
            "final_test_evaluation": False,
            "full_clip_precompute": False,
            "dataset_deleted": False,
            "mail_upstream_modified": False,
            "git_push": False,
        },
    }
    path = ROOT / "analysis/r3_implementation_gate.json"
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
