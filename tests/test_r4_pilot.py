"""R4 subset, anchor, collapse, and optimized-sequence contract tests."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from robocerebra_memory.models.mamba_memory import MambaMemoryBackbone  # noqa: E402
from robocerebra_memory.pilot import collapse_statistics  # noqa: E402


class R4PilotTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.train = json.loads(
            (ROOT / "splits/robocerebra_r4_pilot_train.json").read_text()
        )["trajectory_ids"]
        cls.val = json.loads(
            (ROOT / "splits/robocerebra_r4_pilot_val.json").read_text()
        )["trajectory_ids"]
        cls.test = set(
            json.loads((ROOT / "splits/robocerebra_memory_test.json").read_text())
        )
        cls.anchors = json.loads(
            (ROOT / "analysis/r4_anchor_manifest.json").read_text()
        )["splits"]
        episodes = json.loads(
            (ROOT / "analysis/robocerebra_memory_episode_index.json").read_text()
        )["episodes"]
        cls.episodes = {row["trajectory_id"]: row for row in episodes}

    def test_subset_counts_disjoint_and_test_free(self) -> None:
        self.assertEqual(len(self.train), 16)
        self.assertEqual(len(self.val), 4)
        self.assertEqual(len(set(self.train)), 16)
        self.assertEqual(len(set(self.val)), 4)
        self.assertFalse(set(self.train) & set(self.val))
        self.assertFalse((set(self.train) | set(self.val)) & self.test)

    def test_each_episode_has_64_valid_future_anchors(self) -> None:
        for split, ids in (("train", self.train), ("val", self.val)):
            self.assertEqual(set(self.anchors[split]), set(ids))
            for trajectory_id in ids:
                rows = self.anchors[split][trajectory_id]
                self.assertEqual(len(rows), 64)
                frames = [row["frame"] for row in rows]
                self.assertEqual(len(frames), len(set(frames)))
                self.assertTrue(
                    all(frame + 20 < self.episodes[trajectory_id]["num_frames"] for frame in frames)
                )

    def test_collapse_detector(self) -> None:
        constant = torch.ones(16, 8)
        varied = torch.eye(8).repeat(2, 1)
        self.assertTrue(collapse_statistics(constant)["collapsed"])
        self.assertFalse(collapse_statistics(varied)["collapsed"])

    def test_vectorized_sequence_matches_recurrence(self) -> None:
        torch.manual_seed(42)
        backbone = MambaMemoryBackbone(
            d_model=8, n_layer=2, d_state=4, d_conv=3, expand=1
        )
        tokens = torch.randn(2, 7, 8)
        sequence, final_state = backbone.forward_sequence(tokens)
        state = backbone.initial_state(2)
        steps = []
        for index in range(tokens.shape[1]):
            output, state = backbone.forward_step(tokens[:, index], state)
            steps.append(output)
        recurrent = torch.stack(steps, dim=1)
        self.assertTrue(torch.allclose(sequence, recurrent, atol=1e-5, rtol=1e-5))
        self.assertEqual(final_state.steps, 7)
        self.assertEqual(state.steps, 7)


if __name__ == "__main__":
    unittest.main()
