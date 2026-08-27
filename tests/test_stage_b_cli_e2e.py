import csv
import json
import subprocess
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from robocerebra_memory.eval.representation_extractor import (  # noqa: E402
    SCHEMA,
    atomic_json,
    atomic_torch_save,
    combined_sampling_sha256,
    sample_identity_sha256,
)
from robocerebra_memory.probes import normalize_step_text  # noqa: E402


VARIANTS = ("B0", "B1", "B2", "B3")


def synthetic_payload(variant: str, split: str, trajectory_id: str) -> dict:
    texts = ("S1", "S2", "S3", "S4")
    candidates = torch.zeros(4, 512)
    candidates[:, :4] = torch.eye(4)
    representations = torch.zeros(4, 128)
    representations[:, :4] = torch.eye(4)
    samples = []
    for step in range(4):
        samples.append(
            {
                "trajectory_id": trajectory_id,
                "split": split,
                "frame": step,
                "step_index": step,
                "distance_bin": "0-4",
                "transition_bin": str(step),
                "normalized_step_text": normalize_step_text(texts[step]),
                "steps_since_transition": 0,
                "cumulative_transition_count": step,
                "valid_prev1": step >= 1,
                "valid_prev2": step >= 2,
                "valid_prev3": step >= 3,
                "gt_current": step,
                "gt_prev1": step - 1 if step >= 1 else -1,
                "gt_prev2": step - 2 if step >= 2 else -1,
                "gt_prev3": step - 3 if step >= 3 else -1,
            }
        )
    checkpoint = {
        "path": "synthetic",
        "variant": variant,
        "global_update": 1,
        "completed_epoch": 1,
        "state_dict_sha256": f"synthetic-{variant}",
    }
    payload = {
        "schema_version": SCHEMA,
        "variant": variant,
        "split": split,
        "trajectory_id": trajectory_id,
        "checkpoint": checkpoint,
        "samples": samples,
        "candidate_texts": texts,
        "normalized_candidate_texts": tuple(normalize_step_text(text) for text in texts),
        "candidate_embeddings": candidates,
        "r_t": representations.clone(),
        "z_t": representations.clone(),
        "normalized": True,
    }
    payload["sampling_sha256"] = sample_identity_sha256(trajectory_id, samples)
    return payload


def write_cache(root: Path, variant: str, split: str) -> None:
    directory = root / "representations" / variant / split
    trajectory_id = f"synthetic/{split}"
    payload = synthetic_payload(variant, split, trajectory_id)
    filename = "synthetic.pt"
    atomic_torch_save(directory / filename, payload)
    atomic_json(
        directory / "manifest.json",
        {
            "schema_version": SCHEMA,
            "status": "COMPLETE",
            "variant": variant,
            "split": split,
            "trajectory_count": 1,
            "expected_trajectory_count": 1,
            "shards": [filename],
            "checkpoint": payload["checkpoint"],
            "sampling_sha256": combined_sampling_sha256(
                [(trajectory_id, payload["sampling_sha256"])]
            ),
            "test_split": split == "test",
            "final_test_gate": split == "test",
        },
    )


class StageBCliEndToEndTests(unittest.TestCase):
    def test_all_variants_train_eval_and_comparison_outputs(self):
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            for variant in VARIANTS:
                write_cache(directory, variant, "train")
                write_cache(directory, variant, "val")
            config = {
                "seed": 42,
                "dataset": {"expected_trajectories": {"train": 1, "val": 1, "test": 1}},
                "probe_training": {
                    "learning_rate": 0.01,
                    "weight_decay": 0.0,
                    "temperature": 1.0,
                    "max_epochs": 1,
                    "early_stopping_patience": 1,
                },
                "evaluation": {"bootstrap_resamples": 20, "bootstrap_seed": 42},
                "outputs": {
                    "representation_root": str(directory / "representations"),
                    "probe_root": str(directory / "probes"),
                    "report_root": str(directory / "reports"),
                },
            }
            config_path = directory / "config.yaml"
            atomic_json(config_path, config)
            for variant in VARIANTS:
                trained = subprocess.run(
                    [
                        sys.executable, str(ROOT / "tools/train_stage_b_probes.py"),
                        "--variant", variant, "--config", str(config_path),
                    ],
                    cwd=ROOT, text=True, capture_output=True,
                )
                self.assertEqual(trained.returncode, 0, trained.stderr)
                evaluated = subprocess.run(
                    [
                        sys.executable, str(ROOT / "tools/eval_stage_b.py"),
                        "--variant", variant, "--split", "val", "--config", str(config_path),
                    ],
                    cwd=ROOT, text=True, capture_output=True,
                )
                self.assertEqual(evaluated.returncode, 0, evaluated.stderr)
            required = {
                "summary.json", "summary.csv", "current_retention_by_distance.csv",
                "current_retention_by_transition.csv", "memory_depth.csv",
                "sequence_consistency.json", "instantaneous_control.json",
            }
            for variant in VARIANTS:
                self.assertTrue(required <= {path.name for path in (directory / "reports" / variant).iterdir()})
                figures = directory / "reports" / variant / "figures"
                expected_figures = {
                    "current_retention.svg", "transition_robustness.svg", "memory_depth.svg",
                    "instantaneous_control.svg", "sequence_consistency.svg", "dashboard.html",
                }
                self.assertEqual({path.name for path in figures.iterdir()}, expected_figures)
                for svg in figures.glob("*.svg"):
                    self.assertEqual(ET.parse(svg).getroot().tag, "{http://www.w3.org/2000/svg}svg")
            comparison = directory / "reports" / "comparison"
            expected_comparison = {
                "B0_B1_B2_B3_summary.csv", "memory_depth_comparison.csv",
                "retention_distance_comparison.csv", "transition_robustness_comparison.csv",
            }
            self.assertTrue(expected_comparison <= {path.name for path in comparison.iterdir()})
            comparison_figures = directory / "reports" / "comparison" / "figures"
            expected_comparison_figures = {
                "retention_distance_comparison.svg", "transition_robustness_comparison.svg",
                "memory_depth_comparison.svg", "instantaneous_control_comparison.svg",
                "sequence_consistency_comparison.svg", "dashboard.html",
            }
            self.assertEqual({path.name for path in comparison_figures.iterdir()}, expected_comparison_figures)
            for svg in comparison_figures.glob("*.svg"):
                self.assertEqual(ET.parse(svg).getroot().tag, "{http://www.w3.org/2000/svg}svg")
            with (comparison / "B0_B1_B2_B3_summary.csv").open(encoding="utf-8", newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 16)
            b2 = json.loads((directory / "reports/B2/summary.json").read_text(encoding="utf-8"))
            self.assertIn("oracle-like upper bound", b2["interpretation_warning"])

            # Final reporting uses held-out test95, after probe selection is
            # already complete.  Its comparisons live under final_test only.
            for variant in VARIANTS:
                write_cache(directory, variant, "test")
                evaluated = subprocess.run(
                    [
                        sys.executable, str(ROOT / "tools/eval_stage_b.py"),
                        "--variant", variant, "--split", "test", "--final-test",
                        "--config", str(config_path),
                    ],
                    cwd=ROOT, text=True, capture_output=True,
                )
                self.assertEqual(evaluated.returncode, 0, evaluated.stderr)
            final_root = directory / "reports/final_test"
            for variant in VARIANTS:
                final_summary = json.loads((final_root / variant / "summary.json").read_text(encoding="utf-8"))
                self.assertEqual(final_summary["split"], "test")
                self.assertTrue(final_summary["test_split_evaluated"])
                self.assertTrue((final_root / variant / "FINAL_TEST_COMPLETED.json").is_file())
            self.assertTrue((final_root / "comparison/B0_B1_B2_B3_summary.csv").is_file())
            self.assertTrue((final_root / "comparison/figures/dashboard.html").is_file())


if __name__ == "__main__":
    unittest.main()
