"""Static and unit contracts for the autonomous Stage A job."""

from __future__ import annotations

import json
import re
import sys
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from robocerebra_memory.models import MemoryExperimentModel  # noqa: E402
from robocerebra_memory.stage_a import (  # noqa: E402
    VARIANTS,
    assert_b3_hold_contract,
    language_inputs,
    state_dict_sha256,
    validate_cache_payload,
)


class StageATest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.schedule = json.loads((ROOT / "analysis/stage_a_schedule.json").read_text())
        episodes = json.loads((ROOT / "analysis/robocerebra_memory_episode_index.json").read_text())["episodes"]
        cls.episodes = {row["trajectory_id"]: row for row in episodes}

    def test_schedule_split_and_anchor_contract(self) -> None:
        self.assertEqual(len(self.schedule["train_ids"]), 734)
        self.assertEqual(len(self.schedule["val_ids"]), 85)
        test = set(json.loads((ROOT / "splits/robocerebra_memory_test.json").read_text()))
        self.assertFalse((set(self.schedule["train_ids"]) | set(self.schedule["val_ids"])) & test)
        for split in ("train", "val"):
            for trajectory_id in self.schedule[f"{split}_ids"]:
                rows = self.schedule["anchors"][split][trajectory_id]
                self.assertGreaterEqual(len(rows), 8)
                self.assertLessEqual(len(rows), 64)
                self.assertTrue(all(row["target_frame"] == row["frame"] + 20 for row in rows))
                self.assertTrue(all(row["target_frame"] < self.episodes[trajectory_id]["num_frames"] for row in rows))

    def test_common_initialization_exact_for_all_variants(self) -> None:
        common = torch.load(ROOT / "checkpoints/stage_a/common_init.pt", map_location="cpu", weights_only=False)
        hashes = []
        for variant in VARIANTS:
            model = MemoryExperimentModel(variant)
            model.load_state_dict(common["model_state_dict"], strict=True)
            hashes.append(state_dict_sha256(model.state_dict()))
        self.assertEqual(set(hashes), {common["state_dict_sha256"]})

    def test_b3_hold_mask_exact(self) -> None:
        payload = {
            "num_frames": 8,
            "full_text_feature": torch.randn(512),
            "step_text_features": torch.randn(3, 512),
            "step_boundaries": [(0, 3), (3, 6), (6, 8)],
        }
        language, mask = language_inputs(payload, "B3", torch.device("cpu"))
        assert_b3_hold_contract(payload, mask)
        self.assertEqual(mask[0].nonzero().flatten().tolist(), [0, 3, 6])
        self.assertTrue(torch.equal(language[0, 1], torch.zeros(512)))
        self.assertTrue(torch.equal(language[0, 3], payload["step_text_features"][1]))

    def test_cache_payload_contract(self) -> None:
        episode = {
            "trajectory_id": "scene/case",
            "num_frames": 4,
            "steps": [{"start": 0, "end": 2}, {"start": 2, "end": 4}],
        }
        payload = {
            "trajectory_id": "scene/case",
            "num_frames": 4,
            "visual_features": torch.randn(4, 512).half(),
            "robot_qpos": torch.randn(4, 9),
            "full_text_feature": torch.randn(512).half(),
            "step_text_features": torch.randn(2, 512).half(),
            "step_boundaries": [(0, 2), (2, 4)],
            "normalized": True,
        }
        self.assertEqual(validate_cache_payload(payload, episode), [])
        payload["visual_features"] = torch.randn(5, 512).half()
        self.assertIn("visual_shape", validate_cache_payload(payload, episode))

    def test_pbs_is_unattended_two_gpu_and_has_parallel_variant_pairs(self) -> None:
        script = (ROOT / "jobs/stage_a_smoke_and_representation.pbs").read_text()
        self.assertIn("select=1:ncpus=8:ngpus=2:host=pleiades1", script)
        self.assertIn("run_smoke_pair B0 B1", script)
        self.assertIn("run_smoke_pair B2 B3", script)
        self.assertIn("run_training_pair B0 B1", script)
        self.assertIn("run_training_pair B2 B3", script)
        self.assertNotRegex(script, r"\bqsub\b|\bqdel\b|\binput\s*\(|\bread\s+-p\b")
        self.assertIsNone(re.search(r"^\s*select\s+", script, re.MULTILINE))
        self.assertLess(script.index("CURRENT_STAGE=SMOKE_GATE"), script.index("CURRENT_STAGE=CACHE_FEATURES"))
        self.assertLess(script.index("CURRENT_STAGE=CACHE_GATE"), script.index("CURRENT_STAGE=BUDGET_GATE"))


if __name__ == "__main__":
    unittest.main()
