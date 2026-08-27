import json
import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from robocasa_phase1.semantic_oracle import (  # noqa: E402
    PREDICATE_VERSION,
    PredicateSnapshot,
    derive_ordered_boundaries,
    label_provenance,
    per_frame_stage_ids,
    predicate_s1_colander_in_sink,
    predicate_s2_all_fruits_in_colander,
    predicate_s3_colander_aligned_with_water_site,
    predicate_s4_full_success,
)


class RoboCasaSemanticOracleTest(unittest.TestCase):
    def test_01_predicate_s1_synthetic_state(self):
        self.assertTrue(predicate_s1_colander_in_sink("left"))
        self.assertFalse(predicate_s1_colander_in_sink("none"))

    def test_02_predicate_s2_synthetic_state(self):
        self.assertTrue(predicate_s2_all_fruits_in_colander([True, True]))
        self.assertFalse(predicate_s2_all_fruits_in_colander([True, False]))
        self.assertFalse(predicate_s2_all_fruits_in_colander([]))

    def test_03_predicate_s3_synthetic_state(self):
        self.assertTrue(
            predicate_s3_colander_aligned_with_water_site(
                [0.02, 0.01, 0.0], [0.0, 0.0, 0.1], 0.1, 0.2
            )
        )
        self.assertFalse(
            predicate_s3_colander_aligned_with_water_site(
                [1.0, 0.0, 0.0], [0.0, 0.0, 0.1], 0.1, 0.2
            )
        )

    def test_04_predicate_s4_synthetic_state(self):
        self.assertTrue(predicate_s4_full_success(True, True, True))
        self.assertFalse(predicate_s4_full_success(True, True, False))

    @staticmethod
    def _valid_timeline():
        return [
            PredicateSnapshot(False, False, False, False),
            PredicateSnapshot(True, False, False, False),
            PredicateSnapshot(False, True, False, False),
            PredicateSnapshot(False, False, True, False),
            PredicateSnapshot(False, False, False, True),
        ]

    def test_05_ordered_state_machine(self):
        boundaries = derive_ordered_boundaries(self._valid_timeline())
        self.assertTrue(boundaries.valid)
        self.assertEqual(
            (boundaries.transition_c1, boundaries.transition_c2, boundaries.transition_c3),
            (1, 2, 3),
        )

    def test_06_no_backward_transition(self):
        timeline = self._valid_timeline() + [PredicateSnapshot(False, False, False, False)]
        boundaries = derive_ordered_boundaries(timeline)
        self.assertTrue(boundaries.valid)
        self.assertEqual(boundaries.terminal_completion, 4)

    def test_07_first_completion_boundary(self):
        timeline = [PredicateSnapshot(True, False, False, False)] + self._valid_timeline()
        boundaries = derive_ordered_boundaries(timeline)
        self.assertEqual(boundaries.transition_c1, 0)

    def test_08_invalid_missing_transition(self):
        timeline = [PredicateSnapshot(True, False, True, True) for _ in range(5)]
        boundaries = derive_ordered_boundaries(timeline)
        self.assertFalse(boundaries.valid)
        self.assertEqual(boundaries.failure_reason, "missing_p2")

    def test_09_boundary_serialization(self):
        payload = derive_ordered_boundaries(self._valid_timeline()).to_dict()
        self.assertEqual(json.loads(json.dumps(payload))["transition_c2"], 2)

    def test_10_per_frame_label_generation(self):
        boundaries = derive_ordered_boundaries(self._valid_timeline())
        np.testing.assert_array_equal(per_frame_stage_ids(boundaries), [0, 0, 1, 2, 3])

    @unittest.skip("SMALL_GATE failed; loader integration is forbidden until real labels exist")
    def test_11_loader_integration(self):
        self.fail("activated only after successful full labeling")

    def test_12_provenance(self):
        provenance = label_provenance()
        self.assertFalse(provenance["official_dataset_annotation"])
        self.assertFalse(provenance["manual_boundary"])
        self.assertEqual(provenance["predicate_version"], PREDICATE_VERSION)


if __name__ == "__main__":
    unittest.main()
