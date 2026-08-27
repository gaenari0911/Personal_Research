import json
import sys
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from libero_phase1.interface import (  # noqa: E402
    HOLD_SYMBOL,
    METHOD_CURRENT,
    METHOD_HOLD,
    METHOD_VANILLA,
    SEMANTIC_FULL,
    SEMANTIC_HOLD,
    SEMANTIC_SUBTASK,
    AnnotationStore,
    OracleTrainingGuardError,
    Phase1TrajectoryInterface,
    build_action_timeline,
)


DATASET_ROOT = Path("/ssd1/itaein/datasets/LIBERO/libero_10")
ANNOTATION_ROOT = ROOT / "annotations/libero10_semantic"
VALIDATION_STATUS = ANNOTATION_ROOT / "validation_status.json"
SPLIT_MANIFEST = ROOT / "splits/libero10_phase1_split.json"


class CommonInterfaceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.store = AnnotationStore(ANNOTATION_ROOT)
        cls.interface = Phase1TrajectoryInterface(
            dataset_root=DATASET_ROOT,
            annotation_root=ANNOTATION_ROOT,
            validation_status_path=VALIDATION_STATUS,
            split_manifest_path=SPLIT_MANIFEST,
            action_horizon=10,
        )

    def test_annotation_coverage_and_hdf5_mapping(self):
        annotations = list(self.store.iter_annotations())
        self.assertEqual(len(annotations), 150)
        self.assertEqual(sum(x["conditioning_eligible"] for x in annotations), 150)
        self.assertEqual(sum(x["needs_review"] for x in annotations), 0)
        seen = set()
        for annotation in annotations:
            key = (annotation["task_id"], annotation["demo_id"])
            self.assertNotIn(key, seen)
            seen.add(key)
            path = DATASET_ROOT / f"{annotation['task_name']}_demo.hdf5"
            self.assertTrue(path.is_file())
            with h5py.File(path, "r") as source:
                self.assertIn(annotation["demo_id"], source["data"])
                self.assertEqual(
                    len(source["data"][annotation["demo_id"]]["actions"]),
                    annotation["trajectory_length"],
                )

    def test_generic_three_subtask_timeline(self):
        annotation = {
            "trajectory_length": 12,
            "alignment": {"observation_semantics": "post_action"},
            "subtasks": [
                {
                    "subtask_id": 0,
                    "instruction": "S1",
                    "action_start": 0,
                    "completion_obs_index": 3,
                    "completion_action_index": 3,
                    "next_subtask_action_start": 4,
                },
                {
                    "subtask_id": 1,
                    "instruction": "S2",
                    "action_start": 4,
                    "completion_obs_index": 7,
                    "completion_action_index": 7,
                    "next_subtask_action_start": 8,
                },
                {
                    "subtask_id": 2,
                    "instruction": "S3",
                    "action_start": 8,
                    "completion_obs_index": None,
                    "completion_action_index": None,
                    "next_subtask_action_start": None,
                },
            ],
        }
        timeline = build_action_timeline(annotation, METHOD_HOLD, "FULL")
        self.assertEqual([x["semantic_input"] for x in timeline if x["is_transition"]], ["S1", "S2", "S3"])
        self.assertEqual(timeline[5]["semantic_input"], HOLD_SYMBOL)
        self.assertEqual(timeline[9]["current_subtask_id"], 2)

    def test_transition_alignment_for_all_methods(self):
        for task_id, demo_id in ((3, "demo_20"), (4, "demo_17"), (9, "demo_9")):
            annotation = self.store.load(task_id, demo_id)
            c = annotation["subtasks"][0]["completion_obs_index"]
            for method in (METHOD_VANILLA, METHOD_CURRENT, METHOD_HOLD):
                trajectory = self.interface.load_trajectory(
                    task_id, demo_id, method, include_observations=False
                )
                timeline = trajectory["action_timeline"]
                if method == METHOD_VANILLA:
                    self.assertTrue(all(timeline[x]["semantic_type"] == SEMANTIC_FULL for x in (c - 1, c, c + 1, c + 2)))
                elif method == METHOD_CURRENT:
                    self.assertEqual([timeline[x]["current_subtask_id"] for x in (c - 1, c, c + 1, c + 2)], [0, 0, 1, 1])
                    self.assertTrue(all(timeline[x]["semantic_type"] == SEMANTIC_SUBTASK for x in (c - 1, c, c + 1, c + 2)))
                else:
                    self.assertEqual(
                        [timeline[x]["semantic_type"] for x in (c - 1, c, c + 1, c + 2)],
                        [SEMANTIC_HOLD, SEMANTIC_HOLD, SEMANTIC_SUBTASK, SEMANTIC_HOLD],
                    )
                policy_row = int(np.flatnonzero(trajectory["target_action_index"] == c + 1)[0])
                self.assertEqual(trajectory["policy_observation_index"][policy_row], c)
                self.assertEqual(trajectory["current_subtask_id"][policy_row], 1)
                self.assertTrue(trajectory["oracle_is_transition"][policy_row])

    def test_episode_start_and_terminal_censoring(self):
        for method in (METHOD_VANILLA, METHOD_CURRENT, METHOD_HOLD):
            trajectory = self.interface.load_trajectory(
                4, "demo_34", method, include_observations=False
            )
            timeline = trajectory["action_timeline"]
            self.assertTrue(timeline[0]["is_transition"])
            self.assertEqual(timeline[0]["steps_since_transition"], 0)
            self.assertEqual(timeline[-1]["current_subtask_id"], 1)
            self.assertTrue(trajectory["is_transition"][0])
            self.assertEqual(trajectory["transition_reason"][0], "policy_sequence_start_replay")
            if method == METHOD_HOLD:
                self.assertEqual(timeline[0]["semantic_type"], SEMANTIC_SUBTASK)
                self.assertEqual(timeline[1]["semantic_type"], SEMANTIC_HOLD)
                self.assertEqual(trajectory["semantic_type"][0], SEMANTIC_SUBTASK)
                self.assertEqual(trajectory["semantic_type"][-1], SEMANTIC_HOLD)
        annotation = self.store.load(4, "demo_34")
        self.assertIsNone(annotation["subtasks"][1]["completion_obs_index"])

    def test_recovery_demos_use_annotation_boundary(self):
        for task_id, demo_id in ((3, "demo_34"), (4, "demo_33"), (9, "demo_3")):
            annotation = self.store.load(task_id, demo_id)
            c = int(annotation["subtasks"][0]["completion_obs_index"])
            trajectory = self.interface.load_trajectory(
                task_id, demo_id, METHOD_HOLD, include_observations=False
            )
            transition_actions = [
                item["action_index"]
                for item in trajectory["action_timeline"]
                if item["is_transition"]
            ]
            self.assertEqual(transition_actions[1], c + 1)
            self.assertEqual(
                annotation["recovery_summary"]["final_selected_boundary"], c
            )
        self.assertNotEqual(
            self.store.load(3, "demo_34")["proxy_comparison"]["proxy_boundary"],
            self.store.load(3, "demo_34")["subtasks"][0]["completion_obs_index"],
        )
        self.assertNotEqual(
            self.store.load(9, "demo_3")["proxy_comparison"]["proxy_boundary"],
            self.store.load(9, "demo_3")["subtasks"][0]["completion_obs_index"],
        )

    def test_annotation_mutation_moves_transition_without_code_change(self):
        original = self.store.load(3, "demo_20")
        original_c = int(original["subtasks"][0]["completion_obs_index"])
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "task_3").mkdir()
            manifest = deepcopy(self.store.manifest)
            manifest["tasks"] = [3]
            manifest["total_annotations"] = 1
            (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            mutated = deepcopy(original)
            new_c = original_c + 5
            mutated["subtasks"][0]["completion_obs_index"] = new_c
            mutated["subtasks"][0]["completion_action_index"] = new_c
            mutated["subtasks"][0]["next_subtask_action_start"] = new_c + 1
            mutated["subtasks"][1]["action_start"] = new_c + 1
            mutated["transition_metadata"]["s1_completion_obs_index"] = new_c
            mutated["transition_metadata"]["s2_action_start"] = new_c + 1
            (root / "task_3/demo_20.json").write_text(
                json.dumps(mutated), encoding="utf-8"
            )
            interface = Phase1TrajectoryInterface(
                dataset_root=DATASET_ROOT,
                annotation_root=root,
                validation_status_path=VALIDATION_STATUS,
                action_horizon=10,
            )
            trajectory = interface.load_trajectory(
                3, "demo_20", METHOD_HOLD, include_observations=False
            )
            transitions = [
                x["action_index"] for x in trajectory["action_timeline"] if x["is_transition"]
            ]
            self.assertEqual(transitions, [0, new_c + 1])

    def test_split_is_trajectory_level_and_leak_free(self):
        manifest = json.loads(SPLIT_MANIFEST.read_text(encoding="utf-8"))
        totals = {"train": 0, "val": 0, "test": 0}
        global_keys = {name: set() for name in totals}
        for task_id in (3, 4, 9):
            task = manifest[f"task_{task_id}"]
            self.assertEqual(len(task["train"]), 40)
            self.assertEqual(len(task["val"]), 5)
            self.assertEqual(len(task["test"]), 5)
            self.assertFalse(set(task["train"]) & set(task["val"]))
            self.assertFalse(set(task["train"]) & set(task["test"]))
            self.assertFalse(set(task["val"]) & set(task["test"]))
            for split in totals:
                totals[split] += len(task[split])
                global_keys[split].update((task_id, demo) for demo in task[split])
        self.assertEqual(totals, {"train": 120, "val": 15, "test": 15})
        self.assertFalse(global_keys["train"] & global_keys["val"])
        self.assertFalse(global_keys["train"] & global_keys["test"])
        self.assertFalse(global_keys["val"] & global_keys["test"])

    def test_action_chunk_mask_and_boundary_crossing(self):
        annotation = self.store.load(3, "demo_20")
        c = int(annotation["subtasks"][0]["completion_obs_index"])
        trajectory = self.interface.load_trajectory(
            3, "demo_20", METHOD_HOLD, include_observations=False
        )
        at_c = int(np.flatnonzero(trajectory["target_action_index"] == c)[0])
        self.assertTrue(trajectory["boundary_crossing_horizon"][at_c])
        self.assertEqual(trajectory["valid_action_mask"][at_c].sum(), 10)
        self.assertEqual(trajectory["boundary_safe_action_mask"][at_c].sum(), 1)
        at_s2 = int(np.flatnonzero(trajectory["target_action_index"] == c + 1)[0])
        self.assertFalse(trajectory["boundary_crossing_horizon"][at_s2])
        self.assertEqual(trajectory["boundary_safe_action_mask"][at_s2].sum(), 10)
        self.assertEqual(trajectory["valid_action_mask"][-1].sum(), 1)
        self.assertEqual(trajectory["boundary_safe_action_mask"][-1].sum(), 1)
        hdf5_path = self.interface._hdf5_path(annotation)
        with h5py.File(hdf5_path, "r") as source:
            actions = np.asarray(source["data/demo_20/actions"])
        np.testing.assert_array_equal(trajectory["action_target"][0, 0], actions[1])
        np.testing.assert_array_equal(trajectory["action_target"][0, 9], actions[10])

    def test_observation_and_target_shapes(self):
        trajectory = self.interface.load_trajectory(
            3, "demo_20", METHOD_CURRENT, include_observations=True
        )
        length = trajectory["trajectory_length"]
        self.assertEqual(trajectory["observation"]["agentview_rgb"].shape, (length - 1, 128, 128, 3))
        self.assertEqual(trajectory["observation"]["eye_in_hand_rgb"].shape, (length - 1, 128, 128, 3))
        self.assertEqual(trajectory["action_target"].shape, (length - 1, 10, 7))
        self.assertEqual(trajectory["target_action_index"][0], 1)
        self.assertEqual(trajectory["target_action_index"][-1], length - 1)
        self.assertEqual(
            trajectory["full_instruction"],
            "put the black bowl in the bottom drawer of the cabinet and close it",
        )

    def test_training_guard(self):
        status = self.interface.validation_status()
        self.assertEqual(status["oracle_status"], "provisional")
        self.assertFalse(status["human_spot_check_completed"])
        self.assertFalse(status["approved_for_model_training"])
        with self.assertRaises(OracleTrainingGuardError):
            self.interface.load_trajectory(
                3,
                "demo_20",
                METHOD_VANILLA,
                purpose="training",
                include_observations=False,
            )
        testing = self.interface.load_trajectory(
            3,
            "demo_20",
            METHOD_VANILLA,
            purpose="training",
            allow_provisional_for_testing=True,
            include_observations=False,
        )
        self.assertEqual(testing["oracle_status"], "provisional")


if __name__ == "__main__":
    unittest.main()
