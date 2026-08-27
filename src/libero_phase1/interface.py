"""Task-4 common trajectory and semantic-conditioning interface.

The HDF5 files store ``obs[t]`` after ``action[t]``.  A causal behavior-cloning
pair is therefore ``obs[t] -> action[t + 1]``.  Annotation action indices remain
the source of truth for semantic stage membership.
"""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

import h5py
import numpy as np


METHOD_VANILLA = "vanilla"
METHOD_CURRENT = "current_subinstruction"
METHOD_HOLD = "hold_transition"
METHODS = (METHOD_VANILLA, METHOD_CURRENT, METHOD_HOLD)

SEMANTIC_FULL = "FULL"
SEMANTIC_SUBTASK = "SUBTASK"
SEMANTIC_HOLD = "HOLD"
HOLD_SYMBOL = "[HOLD]"


class AnnotationContractError(ValueError):
    """Raised when an annotation violates the shared Task-4 contract."""


class OracleTrainingGuardError(RuntimeError):
    """Raised when model training is requested before human approval."""


def _read_json(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _decode_json_attribute(value) -> Dict:
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    return json.loads(value)


def validate_annotation(annotation: Mapping, expected_version: Optional[str] = None) -> None:
    if expected_version and annotation.get("annotation_version") != expected_version:
        raise AnnotationContractError(
            f"annotation version mismatch: {annotation.get('annotation_version')} != {expected_version}"
        )
    length = int(annotation["trajectory_length"])
    if length < 2:
        raise AnnotationContractError("trajectory must contain at least two action rows")
    if annotation.get("alignment", {}).get("observation_semantics") != "post_action":
        raise AnnotationContractError("only post-action HDF5 annotations are supported")

    subtasks = annotation.get("subtasks", [])
    if not subtasks:
        raise AnnotationContractError("annotation has no subtasks")
    starts = [int(item["action_start"]) for item in subtasks]
    ids = [int(item["subtask_id"]) for item in subtasks]
    if starts[0] != 0 or starts != sorted(starts) or len(starts) != len(set(starts)):
        raise AnnotationContractError(f"invalid subtask action starts: {starts}")
    if ids != list(range(len(ids))):
        raise AnnotationContractError(f"subtask ids must be contiguous from zero: {ids}")
    if starts[-1] >= length:
        raise AnnotationContractError(f"subtask start outside trajectory: {starts[-1]} >= {length}")

    for index, subtask in enumerate(subtasks):
        completion_obs = subtask.get("completion_obs_index")
        completion_action = subtask.get("completion_action_index")
        next_start = subtask.get("next_subtask_action_start")
        if completion_obs is not None and not 0 <= int(completion_obs) < length:
            raise AnnotationContractError(f"completion observation outside trajectory: {completion_obs}")
        if completion_obs != completion_action:
            raise AnnotationContractError("completion observation/action indices must match")
        if index + 1 < len(subtasks):
            if completion_obs is None:
                raise AnnotationContractError("non-terminal subtask lacks completion")
            expected_next = int(completion_obs) + 1
            if next_start != expected_next or starts[index + 1] != expected_next:
                raise AnnotationContractError(
                    f"next subtask must start at completion+1: {next_start}, {starts[index + 1]}, {expected_next}"
                )
        elif next_start is not None:
            raise AnnotationContractError("terminal subtask next start must be null")


class AnnotationStore:
    """Dynamically read per-demo annotations without copying boundaries."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.manifest_path = self.root / "manifest.json"
        self.manifest = _read_json(self.manifest_path)
        self.version = self.manifest["annotation_version"]

    def path_for(self, task_id: int, demo_id: str) -> Path:
        return self.root / f"task_{int(task_id)}" / f"{demo_id}.json"

    def load(self, task_id: int, demo_id: str) -> Dict:
        path = self.path_for(task_id, demo_id)
        if not path.is_file():
            raise FileNotFoundError(f"missing annotation: {path}")
        annotation = _read_json(path)
        if int(annotation["task_id"]) != int(task_id) or annotation["demo_id"] != demo_id:
            raise AnnotationContractError(f"annotation identity mismatch in {path}")
        validate_annotation(annotation, self.version)
        return annotation

    def iter_annotations(self, eligible_only: bool = False) -> Iterable[Dict]:
        count = 0
        for task_id in sorted(int(x) for x in self.manifest["tasks"]):
            task_dir = self.root / f"task_{task_id}"
            paths = sorted(
                task_dir.glob("demo_*.json"),
                key=lambda path: int(path.stem.rsplit("_", 1)[-1]),
            )
            for path in paths:
                annotation = self.load(task_id, path.stem)
                count += 1
                if eligible_only and not (
                    annotation.get("conditioning_eligible", False)
                    and not annotation.get("needs_review", False)
                ):
                    continue
                yield annotation
        if count != int(self.manifest["total_annotations"]):
            raise AnnotationContractError(
                f"annotation coverage mismatch: found {count}, manifest says {self.manifest['total_annotations']}"
            )


def _subtask_at_action(subtasks: Sequence[Mapping], action_index: int) -> Mapping:
    current = subtasks[0]
    for subtask in subtasks[1:]:
        if int(subtask["action_start"]) > action_index:
            break
        current = subtask
    return current


def build_action_timeline(
    annotation: Mapping,
    method: str,
    full_instruction: str,
) -> List[Dict]:
    """Build a symbolic semantic record for every raw action index."""
    if method not in METHODS:
        raise ValueError(f"unknown conditioning method {method!r}; choose one of {METHODS}")
    validate_annotation(annotation)
    subtasks = annotation["subtasks"]
    starts = {int(item["action_start"]) for item in subtasks}
    timeline = []
    for action_index in range(int(annotation["trajectory_length"])):
        subtask = _subtask_at_action(subtasks, action_index)
        oracle_transition = action_index in starts
        if method == METHOD_VANILLA:
            semantic_type = SEMANTIC_FULL
            semantic_input = full_instruction
        elif method == METHOD_CURRENT:
            semantic_type = SEMANTIC_SUBTASK
            semantic_input = subtask["instruction"]
        elif oracle_transition:
            semantic_type = SEMANTIC_SUBTASK
            semantic_input = subtask["instruction"]
        else:
            semantic_type = SEMANTIC_HOLD
            semantic_input = HOLD_SYMBOL
        timeline.append(
            {
                "action_index": action_index,
                "current_subtask_id": int(subtask["subtask_id"]),
                "current_subinstruction": subtask["instruction"],
                "semantic_type": semantic_type,
                "semantic_input": semantic_input,
                "is_transition": oracle_transition,
                "steps_since_transition": action_index - int(subtask["action_start"]),
                "time_since_transition_seconds": (
                    action_index - int(subtask["action_start"])
                )
                / 20.0,
            }
        )
    return timeline


def _build_action_chunks(
    actions: np.ndarray,
    target_starts: np.ndarray,
    subtask_starts: Sequence[int],
    horizon: int,
):
    chunks = np.zeros((len(target_starts), horizon, actions.shape[1]), dtype=actions.dtype)
    valid_mask = np.zeros((len(target_starts), horizon), dtype=bool)
    boundary_safe_mask = np.zeros((len(target_starts), horizon), dtype=bool)
    crossing = np.zeros(len(target_starts), dtype=bool)
    later_boundaries = sorted(int(x) for x in subtask_starts if int(x) > 0)
    for row, start_value in enumerate(target_starts):
        start = int(start_value)
        end = min(len(actions), start + horizon)
        count = end - start
        chunks[row, :count] = actions[start:end]
        valid_mask[row, :count] = True
        next_boundary = next((x for x in later_boundaries if x > start), None)
        if next_boundary is None:
            boundary_safe_mask[row, :count] = True
        else:
            safe_count = max(0, min(end, next_boundary) - start)
            boundary_safe_mask[row, :safe_count] = True
            crossing[row] = start < next_boundary < end
    return chunks, valid_mask, boundary_safe_mask, crossing


class Phase1TrajectoryInterface:
    """Read HDF5 trajectories and generate all three Task-4 conditionings."""

    def __init__(
        self,
        dataset_root: Path,
        annotation_root: Path,
        validation_status_path: Path,
        split_manifest_path: Optional[Path] = None,
        action_horizon: int = 10,
    ):
        self.dataset_root = Path(dataset_root)
        self.annotations = AnnotationStore(annotation_root)
        self.validation_status_path = Path(validation_status_path)
        self.split_manifest_path = Path(split_manifest_path) if split_manifest_path else None
        self.action_horizon = int(action_horizon)
        if self.action_horizon < 1:
            raise ValueError("action horizon must be positive")

    def validation_status(self) -> Dict:
        return _read_json(self.validation_status_path)

    def assert_training_allowed(self, allow_provisional_for_testing: bool = False) -> None:
        status = self.validation_status()
        if status.get("approved_for_model_training") is True:
            return
        if allow_provisional_for_testing:
            return
        raise OracleTrainingGuardError(
            "Oracle annotations are PROVISIONAL and not approved for model training; "
            "human spot-check approval is required"
        )

    def _hdf5_path(self, annotation: Mapping) -> Path:
        return self.dataset_root / f"{annotation['task_name']}_demo.hdf5"

    def _check_split(self, task_id: int, demo_id: str, split: Optional[str]) -> None:
        if split is None:
            return
        if self.split_manifest_path is None:
            raise ValueError("split requested but no split manifest was configured")
        manifest = _read_json(self.split_manifest_path)
        members = manifest[f"task_{int(task_id)}"][split]
        if demo_id not in members:
            raise ValueError(f"task_{task_id}/{demo_id} is not in split {split}")

    def load_trajectory(
        self,
        task_id: int,
        demo_id: str,
        method: str,
        *,
        split: Optional[str] = None,
        purpose: str = "inspection",
        allow_provisional_for_testing: bool = False,
        include_observations: bool = True,
    ) -> Dict:
        if purpose not in ("inspection", "testing", "training"):
            raise ValueError("purpose must be inspection, testing, or training")
        if purpose == "training":
            self.assert_training_allowed(allow_provisional_for_testing)

        annotation = self.annotations.load(task_id, demo_id)
        if not annotation.get("conditioning_eligible", False) or annotation.get("needs_review", False):
            raise AnnotationContractError(f"task_{task_id}/{demo_id} is not conditioning eligible")
        self._check_split(task_id, demo_id, split)
        hdf5_path = self._hdf5_path(annotation)
        if not hdf5_path.is_file():
            raise FileNotFoundError(f"missing LIBERO HDF5: {hdf5_path}")

        with h5py.File(hdf5_path, "r") as source:
            demo = source["data"][demo_id]
            actions = np.asarray(demo["actions"])
            length = len(actions)
            if length != int(annotation["trajectory_length"]):
                raise AnnotationContractError(
                    f"trajectory length mismatch for task_{task_id}/{demo_id}: {length} != {annotation['trajectory_length']}"
                )
            for key in ("agentview_rgb", "eye_in_hand_rgb"):
                if len(demo[f"obs/{key}"]) != length:
                    raise AnnotationContractError(f"observation length mismatch for {key}")
            problem_info = _decode_json_attribute(source["data"].attrs["problem_info"])
            full_instruction = problem_info["language_instruction"]
            observations = None
            if include_observations:
                observations = {
                    "agentview_rgb": np.asarray(demo["obs/agentview_rgb"][:-1]),
                    "eye_in_hand_rgb": np.asarray(demo["obs/eye_in_hand_rgb"][:-1]),
                }

        action_timeline = build_action_timeline(annotation, method, full_instruction)
        observation_indices = np.arange(length - 1, dtype=np.int64)
        target_indices = observation_indices + 1
        policy_semantics = [deepcopy(action_timeline[int(index)]) for index in target_indices]
        for record in policy_semantics:
            record["oracle_is_transition"] = record["is_transition"]
            record["is_sequence_start"] = False
            record["transition_reason"] = "oracle_stage_start" if record["is_transition"] else None
        if policy_semantics:
            first = policy_semantics[0]
            first["is_sequence_start"] = True
            first["is_transition"] = True
            first["transition_reason"] = "policy_sequence_start_replay"
            if method == METHOD_HOLD and first["semantic_type"] == SEMANTIC_HOLD:
                first_subtask = annotation["subtasks"][0]
                first["semantic_type"] = SEMANTIC_SUBTASK
                first["semantic_input"] = first_subtask["instruction"]

        chunks, valid_mask, safe_mask, crossing = _build_action_chunks(
            actions,
            target_indices,
            [int(item["action_start"]) for item in annotation["subtasks"]],
            self.action_horizon,
        )
        return {
            "task_id": int(task_id),
            "demo_id": demo_id,
            "trajectory_length": length,
            "policy_length": length - 1,
            "observation": observations,
            "policy_observation_index": observation_indices,
            "target_action_index": target_indices,
            "action_target": chunks,
            "valid_action_mask": valid_mask,
            "boundary_safe_action_mask": safe_mask,
            "boundary_crossing_horizon": crossing,
            "full_instruction": full_instruction,
            "method": method,
            "semantic_type": [item["semantic_type"] for item in policy_semantics],
            "semantic_input": [item["semantic_input"] for item in policy_semantics],
            "current_subtask_id": np.asarray(
                [item["current_subtask_id"] for item in policy_semantics], dtype=np.int64
            ),
            "current_subinstruction": [
                item["current_subinstruction"] for item in policy_semantics
            ],
            "is_transition": np.asarray(
                [item["is_transition"] for item in policy_semantics], dtype=bool
            ),
            "oracle_is_transition": np.asarray(
                [item["oracle_is_transition"] for item in policy_semantics], dtype=bool
            ),
            "transition_reason": [item["transition_reason"] for item in policy_semantics],
            "steps_since_transition": np.asarray(
                [item["steps_since_transition"] for item in policy_semantics], dtype=np.int64
            ),
            "time_since_transition_seconds": np.asarray(
                [item["time_since_transition_seconds"] for item in policy_semantics],
                dtype=np.float64,
            ),
            "conditioning_eligible": bool(annotation["conditioning_eligible"]),
            "transition_confidence": annotation["transition_confidence"],
            "oracle_status": self.validation_status()["oracle_status"],
            "annotation": annotation,
            "action_timeline": action_timeline,
            "alignment": {
                "hdf5_observation": "obs[t] is post-action evidence after action[t]",
                "policy_observation_index": "t",
                "target_action_index": "t+1",
                "semantic_condition_index": "t+1 (target action stage)",
                "dropped_target_action": 0,
            },
        }
