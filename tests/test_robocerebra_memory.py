import json
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from src.robocerebra_memory.interface import (
    MemoryEpisode,
    RoboCerebraMemoryDataset,
    Step,
    validate_boundaries,
)


class RoboCerebraMemoryTest(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        self.h5_path = self.root / "demo.hdf5"
        with h5py.File(self.h5_path, "w") as h5:
            demo = h5.create_group("data").create_group("demo_1")
            states = np.zeros((6, 14), dtype=np.float64)
            states[:, 0] = np.arange(6) * 0.05
            states[:, 1:10] = np.arange(54).reshape(6, 9)
            demo.create_dataset("states", data=states)
            demo.create_dataset("actions", data=np.zeros((6, 7)))
        self.steps = [
            {"step_index": 0, "text": "Open cabinet", "start": 0, "end": 2},
            {"step_index": 1, "text": "Pick item", "start": 2, "end": 5},
            {"step_index": 2, "text": "Place item", "start": 5, "end": 6},
        ]
        self.record = {
            "trajectory_id": "scene/case1",
            "full_instruction": "Store the item.",
            "num_frames": 6,
            "steps": self.steps,
            "state_source": str(self.h5_path),
            "action_source": str(self.h5_path),
            "visual_source": None,
        }
        self.episode = MemoryEpisode(self.record)

    def tearDown(self):
        self.episode.close()
        self.tempdir.cleanup()

    def test_01_boundary_half_open_convention(self):
        result = validate_boundaries([Step.from_dict(x) for x in self.steps], 6)
        self.assertTrue(result.valid)

    def test_02_all_frames_assigned_exactly_once(self):
        self.assertEqual([self.episode.get_step_index(t) for t in range(6)], [0, 0, 1, 1, 1, 2])

    def test_03_current_step_lookup(self):
        self.assertEqual(self.episode.get_step_text(2), "Pick item")

    def test_04_transition_event(self):
        self.assertFalse(self.episode.get_analysis_labels(1)["transition_event"])
        self.assertTrue(self.episode.get_analysis_labels(2)["transition_event"])

    def test_05_steps_since_transition_reset(self):
        values = [self.episode.get_analysis_labels(t)["steps_since_transition"] for t in range(6)]
        self.assertEqual(values, [0, 1, 0, 1, 2, 0])

    def test_06_cumulative_transition_count(self):
        values = [self.episode.get_analysis_labels(t)["cumulative_transition_count"] for t in range(6)]
        self.assertEqual(values, [0, 0, 1, 1, 1, 2])

    def test_07_previous_step_labels(self):
        labels = self.episode.get_analysis_labels(5)
        self.assertEqual((labels["previous_1"], labels["previous_2"], labels["previous_3"]), (1, 0, -1))

    def test_08_full_conditioning(self):
        self.assertEqual(self.episode.get_condition(4, "FULL"), "Store the item.")

    def test_09_current_conditioning(self):
        self.assertEqual(self.episode.get_condition(4, "CURRENT"), "Pick item")

    def test_10_hold_conditioning(self):
        self.assertEqual(self.episode.get_condition(0, "HOLD"), "Open cabinet")
        self.assertEqual(self.episode.get_condition(1, "HOLD"), "[HOLD]")
        self.assertEqual(self.episode.get_condition(2, "HOLD"), "Pick item")

    def test_11_hold_does_not_imply_episode_reset(self):
        frame = self.episode.get_frame(2, "HOLD")
        self.assertFalse(frame["episode_start"])
        self.assertTrue(frame["analysis"]["transition_event"])

    def test_12_episode_start_and_end(self):
        self.assertTrue(self.episode.get_frame(0)["episode_start"])
        self.assertTrue(self.episode.get_frame(5)["episode_end"])

    def test_13_invalid_boundary_exclusion(self):
        broken = [Step(0, "x", 0, 3), Step(1, "y", 4, 6)]
        self.assertFalse(validate_boundaries(broken, 6).valid)

    def test_14_lazy_loader_sample(self):
        index = self.root / "index.json"
        index.write_text(json.dumps({"episodes": [self.record]}), encoding="utf-8")
        dataset = RoboCerebraMemoryDataset(index)
        episode = dataset.load_episode("scene/case1")
        self.assertFalse(episode.is_open)
        np.testing.assert_array_equal(episode.get_robot_state(0), np.arange(9))
        self.assertTrue(episode.is_open)
        episode.close()

    def test_15_analysis_separated_from_model_input(self):
        frame = self.episode.get_frame(3)
        self.assertEqual(set(frame["model_input"]), {"observation", "robot_state", "condition"})
        self.assertNotIn("steps_since_transition", frame["model_input"])
        self.assertIn("steps_since_transition", frame["analysis"])

    def test_16_sequential_iterator(self):
        frames = list(self.episode.iter_frames("HOLD"))
        self.assertEqual([x["analysis"]["frame"] for x in frames], list(range(6)))

    def test_17_optional_action(self):
        self.assertNotIn("action", self.episode.get_frame(0)["model_input"])
        self.assertIn("action", self.episode.get_frame(0, include_action=True)["model_input"])


if __name__ == "__main__":
    unittest.main()
