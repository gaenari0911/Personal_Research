"""Full-metadata smoke tests for the generated R1-RC artifacts."""

from __future__ import annotations

import json
import unittest
from collections import defaultdict
from pathlib import Path

import h5py

from src.robocerebra_memory.interface import Step, validate_boundaries


ROOT = Path(__file__).resolve().parents[1]


class RoboCerebraMemoryArtifactTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        index_path = ROOT / "analysis/robocerebra_memory_episode_index.json"
        if not index_path.is_file():
            raise unittest.SkipTest("R1-RC artifact build has not run")
        cls.payload = json.loads(index_path.read_text(encoding="utf-8"))
        cls.episodes = cls.payload["episodes"]
        cls.by_id = {item["trajectory_id"]: item for item in cls.episodes}
        cls.splits = {
            split: json.loads(
                (ROOT / f"splits/robocerebra_memory_{split}.json").read_text(
                    encoding="utf-8"
                )
            )
            for split in ("train", "val", "test")
        }

    def test_01_every_episode_readable_and_lengths_match(self) -> None:
        for item in self.episodes:
            with h5py.File(item["state_source"], "r") as stream:
                demo = stream["data/demo_1"]
                self.assertEqual(demo["states"].shape[0], item["num_frames"])
                self.assertEqual(demo["actions"].shape[0], item["num_frames"])

    def test_02_all_boundaries_resolve_exactly(self) -> None:
        for item in self.episodes:
            steps = [Step.from_dict(value) for value in item["steps"]]
            result = validate_boundaries(steps, item["num_frames"])
            self.assertTrue(result.valid, (item["trajectory_id"], result.reasons))

    def test_03_every_step_text_present(self) -> None:
        self.assertTrue(self.episodes)
        self.assertTrue(
            all(step["text"].strip() for item in self.episodes for step in item["steps"])
        )

    def test_04_transition_count_is_num_steps_minus_one(self) -> None:
        for item in self.episodes:
            transitions = sum(step["step_index"] > 0 for step in item["steps"])
            self.assertEqual(transitions, item["num_steps"] - 1)

    def test_05_split_disjointness(self) -> None:
        sets = {name: set(values) for name, values in self.splits.items()}
        self.assertFalse(sets["train"] & sets["val"])
        self.assertFalse(sets["train"] & sets["test"])
        self.assertFalse(sets["val"] & sets["test"])
        self.assertEqual(set().union(*sets.values()), set(self.by_id))

    def test_06_no_duplicate_trajectory_across_split(self) -> None:
        owners = defaultdict(list)
        for split, values in self.splits.items():
            for trajectory_id in values:
                owners[trajectory_id].append(split)
        self.assertTrue(all(len(values) == 1 for values in owners.values()))

    def test_07_duplicate_hash_never_crosses_split(self) -> None:
        owners = defaultdict(set)
        for split, values in self.splits.items():
            for trajectory_id in values:
                owners[self.by_id[trajectory_id]["source_sha256"]].add(split)
        self.assertTrue(all(len(values) == 1 for values in owners.values()))


if __name__ == "__main__":
    unittest.main()
