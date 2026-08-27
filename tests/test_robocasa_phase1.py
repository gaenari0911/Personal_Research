import json
import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from robocasa_phase1.interface import (  # noqa: E402
    EXPECTED_ANNOTATION_KEYS,
    EXTERNAL_KEY,
    WRIST_KEY,
    MinMaxActionScaler,
    RoboCasaTrajectoryLoader,
    arm_only_decision,
    build_action_chunk,
    build_observation_indices,
    extract_mail_action,
)


DATASET_ROOT = Path("/ssd1/itaein/datasets/RoboCasa365/WashFruitColander/lerobot")
MANIFEST = ROOT / "analysis/washfruitcolander_arm_only_manifest.json"


class RoboCasaPhase1Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = json.loads(MANIFEST.read_text())
        cls.loader = RoboCasaTrajectoryLoader(DATASET_ROOT, manifest_path=MANIFEST)
        cls.eligible = cls.manifest["episode_ids"]
        cls.first = cls.eligible[0]

    def test_01_action_reorder_contract(self):
        raw = np.arange(12, dtype=np.float64)
        np.testing.assert_array_equal(extract_mail_action(raw), np.arange(5, 12, dtype=np.float32))

    def test_02_12d_to_canonical_7d_extraction(self):
        raw = np.arange(36, dtype=np.float64).reshape(3, 12)
        result = extract_mail_action(raw)
        self.assertEqual(result.shape, (3, 7))
        self.assertEqual(result.dtype, np.float32)
        np.testing.assert_array_equal(result[:, 0], raw[:, 5])
        np.testing.assert_array_equal(result[:, -1], raw[:, 11])

    def test_03_excluded_base_episode_detection(self):
        action = np.zeros((4, 12), dtype=np.float64)
        action[:, 4] = -1.0
        action[2, 1] = 1.0 / 70.0
        decision = arm_only_decision(action)
        self.assertFalse(decision.eligible)
        self.assertEqual(decision.base_nonzero_frames, 1)
        self.assertIn("nonzero_base_action", decision.exclusion_reason)

    def test_04_eligible_episode_detection(self):
        action = np.zeros((4, 12), dtype=np.float64)
        action[:, 4] = -1.0
        decision = arm_only_decision(action)
        self.assertTrue(decision.eligible)
        self.assertEqual(decision.control_mode_values, (-1.0,))
        self.assertEqual(decision.control_mode_change_count, 0)

    def test_05_camera_keys(self):
        self.assertIn(EXTERNAL_KEY, self.loader.info["features"])
        self.assertIn(WRIST_KEY, self.loader.info["features"])
        self.assertTrue(self.loader.video_path(self.first, EXTERNAL_KEY).is_file())
        self.assertTrue(self.loader.video_path(self.first, WRIST_KEY).is_file())

    def test_06_image_shape_color_and_range(self):
        sample = self.loader.get_sample(self.first, 4)
        for key in ("external_rgb", "wrist_rgb"):
            image = sample[key]
            self.assertEqual(image.shape, (5, 3, 128, 128))
            self.assertEqual(image.dtype, np.float32)
            self.assertGreaterEqual(float(image.min()), 0.0)
            self.assertLessEqual(float(image.max()), 1.0)
            self.assertGreater(float(image.std()), 0.01)

    def test_07_action_scaler_roundtrip(self):
        scaler = self.loader.fit_scaler(self.eligible[:3])
        action = self.loader.load_low_dim(self.first)["actions"][:64]
        restored = scaler.inverse_transform(scaler.transform(action))
        np.testing.assert_allclose(restored, action, atol=2e-7, rtol=0.0)
        self.assertTrue(scaler.fitted)

    def test_08_temporal_alignment_shift(self):
        sample = self.loader.get_sample(self.first, 4)
        self.assertEqual(sample["target_action_indices"][0], 4)
        self.assertEqual(sample["observation_indices"][-1], 4)
        self.assertEqual(sample["target_action_indices"][0] - sample["observation_index"], 0)

    def test_09_action_chunk_indices(self):
        actions = np.arange(84, dtype=np.float32).reshape(12, 7)
        chunk, mask, indices = build_action_chunk(actions, 3, 5)
        np.testing.assert_array_equal(indices, [3, 4, 5, 6, 7])
        self.assertTrue(mask.all())
        np.testing.assert_array_equal(chunk, actions[3:8])

    def test_10_end_padding_mask(self):
        actions = np.arange(35, dtype=np.float32).reshape(5, 7)
        chunk, mask, indices = build_action_chunk(actions, 3, 4)
        np.testing.assert_array_equal(mask, [True, True, False, False])
        np.testing.assert_array_equal(indices, [3, 4, -1, -1])
        np.testing.assert_array_equal(chunk[2:], np.zeros((2, 7), dtype=np.float32))

    def test_11_observation_window_no_start_padding(self):
        np.testing.assert_array_equal(build_observation_indices(4, 5), [0, 1, 2, 3, 4])
        with self.assertRaises(IndexError):
            build_observation_indices(3, 5)

    def test_12_episode_boundary_leakage(self):
        trajectory = self.loader.load_trajectory(self.first)
        last_observation = int(trajectory["valid_indices"][-1])
        sample = self.loader.get_sample(self.first, last_observation)
        valid_indices = sample["target_action_indices"][sample["valid_action_mask"]]
        self.assertEqual(valid_indices[-1], trajectory["num_frames"] - 1)
        self.assertTrue(np.all(sample["target_action_indices"][~sample["valid_action_mask"]] == -1))
        self.assertTrue(np.all(sample["observation_indices"] < trajectory["num_frames"]))

    def test_13_missing_updated_annotations_are_explicit(self):
        trajectory = self.loader.load_trajectory(self.first)
        self.assertEqual(
            set(trajectory["missing_expected_annotation_keys"]), set(EXPECTED_ANNOTATION_KEYS)
        )
        self.assertIn("annotation.human.task_description", trajectory["raw_annotations"])

    def test_14_sample_shapes_and_finiteness(self):
        trajectory = self.loader.load_trajectory(self.first)
        middle = int(trajectory["valid_indices"][len(trajectory["valid_indices"]) // 2])
        sample = self.loader.get_sample(self.first, middle)
        self.assertEqual(sample["actions"].shape, (10, 7))
        self.assertEqual(sample["valid_action_mask"].shape, (10,))
        self.assertTrue(np.isfinite(sample["actions"]).all())
        self.assertTrue(np.isfinite(sample["external_rgb"]).all())
        self.assertTrue(np.isfinite(sample["wrist_rgb"]).all())


if __name__ == "__main__":
    unittest.main()
