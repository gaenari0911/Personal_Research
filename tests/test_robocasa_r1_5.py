"""R1.5 tests activated only when the official staging archive is present."""

import json
import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from robocasa_phase1.interface import (  # noqa: E402
    EXPECTED_ANNOTATION_KEYS,
    RoboCasaTrajectoryLoader,
)


STAGING_BASE = Path(
    "/ssd1/itaein/datasets/RoboCasa365/WashFruitColander_latest_staging"
)
STAGING = STAGING_BASE / "lerobot" if (STAGING_BASE / "lerobot").is_dir() else STAGING_BASE
OFFICIAL_AVAILABLE = (STAGING / "meta/info.json").is_file()
MANIFEST = ROOT / "analysis/washfruitcolander_arm_only_manifest.json"
OLD_ROOT = Path("/ssd1/itaein/datasets/RoboCasa365/WashFruitColander/lerobot")


@unittest.skipUnless(OFFICIAL_AVAILABLE, "official annotation endpoint is HTTP 404")
class RoboCasaR15OfficialAnnotationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = json.loads(MANIFEST.read_text())
        cls.old = RoboCasaTrajectoryLoader(OLD_ROOT, manifest_path=MANIFEST)
        cls.new = RoboCasaTrajectoryLoader(STAGING)
        cls.episode_ids = list(cls.new.episode_ids())

    def test_01_annotation_schema_presence(self):
        trajectory = self.new.load_low_dim(self.episode_ids[0])
        self.assertFalse(trajectory["missing_expected_annotation_keys"])
        self.assertTrue(set(EXPECTED_ANNOTATION_KEYS).issubset(trajectory["raw_annotations"]))

    def test_02_annotation_frame_count(self):
        for episode_id in self.episode_ids:
            trajectory = self.new.load_low_dim(episode_id)
            for values in trajectory["raw_annotations"].values():
                self.assertEqual(len(values), trajectory["num_frames"])

    def test_03_episode_mapping(self):
        self.assertEqual(self.episode_ids, list(range(507)))
        for episode_id in self.episode_ids:
            old = self.old.load_low_dim(episode_id)
            new = self.new.load_low_dim(episode_id)
            self.assertEqual(old["num_frames"], new["num_frames"])
            np.testing.assert_array_equal(old["raw_actions"], new["raw_actions"])

    def test_04_arm_only_manifest_mapping(self):
        self.assertEqual(len(self.manifest["episode_ids"]), 489)
        self.assertTrue(set(self.manifest["episode_ids"]).issubset(self.episode_ids))

    def test_05_annotation_loader(self):
        episode_id = self.manifest["episode_ids"][0]
        sample = self.new.get_sample(episode_id, 4)
        self.assertTrue(set(EXPECTED_ANNOTATION_KEYS).issubset(sample["raw_annotations"]))

    def test_06_no_frame_shift(self):
        episode_id = self.manifest["episode_ids"][0]
        sample = self.new.get_sample(episode_id, 4)
        self.assertEqual(sample["observation_index"], 4)
        self.assertEqual(sample["target_action_indices"][0], 4)
        for values in sample["raw_annotations"].values():
            self.assertEqual(len(values), int(sample["valid_action_mask"].sum()))

    def test_07_no_episode_leakage(self):
        episode_id = self.manifest["episode_ids"][0]
        trajectory = self.new.load_low_dim(episode_id)
        sample = self.new.get_sample(episode_id, trajectory["num_frames"] - 1)
        valid = sample["target_action_indices"][sample["valid_action_mask"]]
        self.assertEqual(valid.tolist(), [trajectory["num_frames"] - 1])


if __name__ == "__main__":
    unittest.main()
