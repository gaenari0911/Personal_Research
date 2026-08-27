import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from robocerebra_memory.eval.representation_extractor import (  # noqa: E402
    atomic_torch_save,
    checkpoint_identity_matches,
    sample_identity_sha256,
    guard_split_access,
    load_split_ids,
    validate_split_count,
    validate_representation_shard,
)


class StageBNoLeakageTests(unittest.TestCase):
    def test_test_path_is_not_read_before_final_test_gate(self):
        path = Path("must_not_be_read.json")
        with mock.patch.object(Path, "read_text", side_effect=AssertionError("test path was read")) as read:
            with self.assertRaises(PermissionError):
                load_split_ids(path, "test", final_test=False)
        read.assert_not_called()

    def test_eval_cli_refuses_test_without_flag_before_cache_access(self):
        completed = subprocess.run(
            [sys.executable, str(ROOT / "tools/eval_stage_b.py"), "--variant", "B0", "--split", "test"],
            cwd=ROOT, text=True, capture_output=True,
        )
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("explicit --final-test", completed.stderr)

    def test_test_clip_cache_refuses_test_without_final_flag(self):
        completed = subprocess.run(
            [
                sys.executable, str(ROOT / "tools/cache_stage_a_features.py"),
                "--model-cache", "must-not-be-read", "--device", "cpu",
                "--split-scope", "test",
            ],
            cwd=ROOT, text=True, capture_output=True,
        )
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("explicit --final-test", completed.stderr)

    def test_final_test_completion_sentinel_blocks_second_evaluation(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            canonical = output / "reports/final_test/B0"
            canonical.mkdir(parents=True)
            (canonical / "FINAL_TEST_COMPLETED.json").write_text("{}", encoding="utf-8")
            config = output / "config.yaml"
            config.write_text(
                json.dumps(
                    {
                        "dataset": {"expected_trajectories": {"test": 1}},
                        "evaluation": {"bootstrap_resamples": 1, "bootstrap_seed": 42},
                        "outputs": {
                            "representation_root": str(output / "representations"),
                            "probe_root": str(output / "probes"),
                            "report_root": str(output / "reports"),
                        },
                    }
                ),
                encoding="utf-8",
            )
            completed = subprocess.run(
                [
                    sys.executable, str(ROOT / "tools/eval_stage_b.py"), "--variant", "B0",
                    "--split", "test", "--final-test", "--output-dir", str(output),
                    "--representation-dir", str(output / "must-not-be-read"),
                    "--config", str(config),
                ],
                cwd=ROOT, text=True, capture_output=True,
            )
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("exactly-once gate", completed.stderr)

    def test_train_and_val_are_allowed_without_final_test(self):
        guard_split_access("train", False)
        guard_split_access("val", False)

    def test_partial_file_is_not_a_valid_final_shard(self):
        payload = {
            "schema_version": "stage-b-representation-v1", "variant": "B0", "split": "train",
            "trajectory_id": "tau", "checkpoint": {
                "variant": "B0", "state_dict_sha256": "abc", "global_update": 1, "completed_epoch": 1,
            },
            "samples": [{
                "trajectory_id": "tau", "split": "train", "frame": 0, "step_index": 0,
                "distance_bin": "0-4", "transition_bin": "0", "normalized_step_text": "s1",
                "steps_since_transition": 0, "cumulative_transition_count": 0,
                "valid_prev1": False, "valid_prev2": False, "valid_prev3": False,
                "gt_current": 0, "gt_prev1": -1, "gt_prev2": -1, "gt_prev3": -1,
            }], "candidate_texts": ("S1",),
            "normalized_candidate_texts": ("s1",), "candidate_embeddings": torch.zeros(1, 512),
            "r_t": torch.zeros(1, 128), "z_t": torch.zeros(1, 128), "normalized": True,
        }
        payload["candidate_embeddings"][0, 0] = 1.0
        payload["sampling_sha256"] = sample_identity_sha256("tau", payload["samples"])
        validate_representation_shard(payload)
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "shard.pt"
            atomic_torch_save(target, payload)
            self.assertTrue(target.is_file())
            self.assertFalse(target.with_name(target.name + ".stage-b-part").exists())

    def test_split_duplicates_and_wrong_count_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "train.json"
            path.write_text('["tau", "tau"]', encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "duplicate"):
                load_split_ids(path, "train")
        with self.assertRaisesRegex(RuntimeError, "count mismatch"):
            validate_split_count(["tau"], "train", 734)

    def test_resume_checkpoint_identity_must_match(self):
        shard = {
            "checkpoint": {
                "variant": "B0", "state_dict_sha256": "old", "global_update": 734,
                "completed_epoch": 1,
            }
        }
        current = {
            "variant": "B0", "state_dict_sha256": "new", "global_update": 734,
            "completed_epoch": 1,
        }
        self.assertFalse(checkpoint_identity_matches(shard, current))


if __name__ == "__main__":
    unittest.main()
