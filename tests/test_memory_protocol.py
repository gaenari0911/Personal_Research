"""Unit and artifact tests for the training-free R2 protocol."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np

from src.robocerebra_memory.metrics import (
    ScoredRetrieval,
    metric_curve,
    retrieval_result,
    trajectory_bootstrap_ci,
    trajectory_macro_metrics,
)
from src.robocerebra_memory.probes import CandidateSet, make_probe_target
from src.robocerebra_memory.sampling import (
    build_balanced_samples,
    distance_bin,
    evenly_spaced_frames,
    transition_bin,
)


ROOT = Path(__file__).resolve().parents[1]


class MemoryProtocolTest(unittest.TestCase):
    def setUp(self) -> None:
        self.episode = {
            "trajectory_id": "scene/case",
            "num_frames": 25,
            "steps": [
                {"step_index": 0, "text": "Pick object", "start": 0, "end": 10},
                {"step_index": 1, "text": "Place object", "start": 10, "end": 25},
            ],
        }

    def test_01_distance_bin_edges(self) -> None:
        expected = {0: "0-4", 4: "0-4", 5: "5-19", 19: "5-19", 20: "20-49", 400: "400+"}
        self.assertEqual({value: distance_bin(value) for value in expected}, expected)

    def test_02_negative_distance_rejected(self) -> None:
        with self.assertRaises(ValueError):
            distance_bin(-1)

    def test_03_transition_bins(self) -> None:
        self.assertEqual([transition_bin(value) for value in (0, 7, 8, 22)], ["0", "7", "8+", "8+"])

    def test_04_evenly_spaced_cap(self) -> None:
        self.assertEqual(evenly_spaced_frames(10, 20, 4), (10, 13, 16, 19))
        self.assertEqual(evenly_spaced_frames(3, 5, 4), (3, 4))

    def test_05_duplicate_text_equivalence(self) -> None:
        episode = dict(self.episode)
        episode["steps"] = [
            {"step_index": 0, "text": "Pick Object", "start": 0, "end": 5},
            {"step_index": 1, "text": "  pick   object ", "start": 5, "end": 10},
        ]
        candidates = CandidateSet.from_episode(episode)
        self.assertEqual(candidates.positive_indices(1), (0, 1))

    def test_06_current_and_previous_targets(self) -> None:
        candidates = CandidateSet.from_episode(self.episode)
        self.assertEqual(make_probe_target(candidates, 12, 1, 0).target_step_index, 1)
        self.assertEqual(make_probe_target(candidates, 12, 1, 1).target_step_index, 0)

    def test_07_missing_previous_target(self) -> None:
        target = make_probe_target(CandidateSet.from_episode(self.episode), 2, 0, 1)
        self.assertFalse(target.eligible)
        self.assertEqual(target.target_step_index, -1)

    def test_08_balanced_sampling_cap_and_labels(self) -> None:
        samples = build_balanced_samples([self.episode], "train", 4)
        self.assertTrue(samples)
        counts = {}
        for item in samples:
            key = (item.step_index, item.distance_bin)
            counts[key] = counts.get(key, 0) + 1
        self.assertTrue(all(value <= 4 for value in counts.values()))
        self.assertTrue(all(item.current_target == item.step_index for item in samples))

    def test_09_perfect_retrieval(self) -> None:
        candidates = np.eye(4)
        self.assertEqual(retrieval_result(candidates[2], candidates, (2,)), (1.0, 1.0, 1))

    def test_10_retrieval_rank(self) -> None:
        candidates = np.eye(3)
        query = np.asarray([1.0, 0.8, 0.0])
        recall, mrr, rank = retrieval_result(query, candidates, (1,))
        self.assertEqual((recall, rank), (0.0, 2))
        self.assertEqual(mrr, 0.5)

    def test_11_trajectory_macro_not_frame_macro(self) -> None:
        values = [
            ScoredRetrieval("a", 1, 1),
            ScoredRetrieval("a", 1, 1),
            ScoredRetrieval("b", 0, 0.5),
        ]
        result = trajectory_macro_metrics(values)
        self.assertEqual(result["recall_at_1"], 0.5)

    def test_12_metric_curve(self) -> None:
        values = [ScoredRetrieval("a", 1, 1, "near"), ScoredRetrieval("a", 0, 0.5, "far")]
        curve = metric_curve(values, ["near", "far"])
        self.assertEqual(curve["near"]["recall_at_1"], 1.0)
        self.assertEqual(curve["far"]["recall_at_1"], 0.0)

    def test_13_bootstrap_is_trajectory_level(self) -> None:
        values = [ScoredRetrieval("a", 1, 1), ScoredRetrieval("b", 0, 0.5)]
        result = trajectory_bootstrap_ci(values, resamples=100, seed=42)
        self.assertEqual(result["bootstrap_unit"], "trajectory")
        self.assertEqual(result["estimate"], 0.5)

    def test_14_dry_run_artifacts_cover_all_splits(self) -> None:
        path = ROOT / "analysis/r2_memory_sampling_statistics.json"
        if not path.is_file():
            raise unittest.SkipTest("R2 dry run has not run")
        payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(payload["all_splits"]["trajectories"], 914)
        self.assertEqual(set(payload["splits"]), {"train", "val", "test"})

    def test_15_dummy_metric_smoke_is_perfect(self) -> None:
        payload = json.loads(
            (ROOT / "analysis/r2_memory_sampling_statistics.json").read_text(encoding="utf-8")
        )
        smoke = payload["dummy_representation_metric_smoke"]
        self.assertEqual((smoke["rank"], smoke["recall_at_1"], smoke["mrr"]), (1, 1.0, 1.0))

    def test_16_split_target_counts_are_nonzero(self) -> None:
        payload = json.loads(
            (ROOT / "analysis/r2_probe_target_statistics.json").read_text(encoding="utf-8")
        )
        for split in ("train", "val", "test"):
            self.assertGreater(payload["splits"][split]["current_balanced_target_count"], 0)
            self.assertGreater(payload["splits"][split]["previous_targets"]["3"]["balanced_target_count"], 0)


if __name__ == "__main__":
    unittest.main()

