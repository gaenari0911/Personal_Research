"""Leakage-safe, trajectory-sequential RoboCerebra memory interface.

Official ``Step:`` intervals use a half-open ``[start, end)`` convention.  The
loader keeps progress labels outside ``model_input`` and never resets an
episode at a step transition.
"""

from __future__ import annotations

import bisect
import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Callable, Iterator, Mapping, Optional, Sequence, Union

import h5py
import numpy as np


class ConditioningMode(str, Enum):
    FULL = "FULL"
    CURRENT = "CURRENT"
    HOLD = "HOLD"


@dataclass(frozen=True)
class Step:
    step_index: int
    text: str
    start: int
    end: int

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "Step":
        return cls(
            step_index=int(value["step_index"]),
            text=str(value["text"]),
            start=int(value["start"]),
            end=int(value["end"]),
        )


@dataclass(frozen=True)
class BoundaryValidation:
    valid: bool
    first_start_ok: bool
    positive_intervals: bool
    contiguous: bool
    terminal_match: bool
    ordered_indices: bool
    all_frames_assigned_once: bool
    reasons: tuple[str, ...]


def validate_boundaries(steps: Sequence[Step], num_frames: int) -> BoundaryValidation:
    """Validate exact half-open official intervals without correction."""
    reasons = []
    first_start_ok = bool(steps) and steps[0].start == 0
    positive = bool(steps) and all(step.end > step.start for step in steps)
    contiguous = bool(steps) and all(
        left.end == right.start for left, right in zip(steps, steps[1:])
    )
    terminal = bool(steps) and steps[-1].end == num_frames
    ordered = all(step.step_index == index for index, step in enumerate(steps))
    assigned = (
        first_start_ok
        and positive
        and contiguous
        and terminal
        and sum(step.end - step.start for step in steps) == num_frames
    )
    for passed, name in (
        (first_start_ok, "first_start_not_zero"),
        (positive, "nonpositive_interval"),
        (contiguous, "gap_or_overlap"),
        (terminal, "terminal_mismatch"),
        (ordered, "step_order_mismatch"),
        (assigned, "frame_assignment_not_exact"),
    ):
        if not passed:
            reasons.append(name)
    return BoundaryValidation(
        valid=not reasons,
        first_start_ok=first_start_ok,
        positive_intervals=positive,
        contiguous=contiguous,
        terminal_match=terminal,
        ordered_indices=ordered,
        all_frames_assigned_once=assigned,
        reasons=tuple(reasons),
    )


ObservationAdapter = Callable[[str, int], object]


class MemoryEpisode:
    """One lazy HDF5-backed continuous trajectory."""

    def __init__(
        self,
        metadata: Mapping[str, object],
        observation_adapter: Optional[ObservationAdapter] = None,
    ) -> None:
        self.trajectory_id = str(metadata["trajectory_id"])
        self.full_instruction = str(metadata["full_instruction"])
        self.num_frames = int(metadata["num_frames"])
        self.steps = tuple(Step.from_dict(item) for item in metadata["steps"])
        self.state_source = Path(str(metadata["state_source"]))
        self.action_source = Path(str(metadata["action_source"]))
        self.visual_source = metadata.get("visual_source")
        self._observation_adapter = observation_adapter
        self._h5: Optional[h5py.File] = None
        self._demo = None
        validation = validate_boundaries(self.steps, self.num_frames)
        if not validation.valid:
            raise ValueError(
                f"invalid canonical episode {self.trajectory_id}: {validation.reasons}"
            )
        self._ends = tuple(step.end for step in self.steps)

    @property
    def is_open(self) -> bool:
        return self._h5 is not None

    def _ensure_open(self):
        if self._h5 is None:
            self._h5 = h5py.File(self.state_source, "r")
            self._demo = self._h5["data/demo_1"]
            if self._demo["states"].shape[0] != self.num_frames:
                self.close()
                raise ValueError(f"state length changed for {self.trajectory_id}")
            if self._demo["actions"].shape[0] != self.num_frames:
                self.close()
                raise ValueError(f"action length changed for {self.trajectory_id}")
        return self._demo

    def close(self) -> None:
        if self._h5 is not None:
            self._h5.close()
            self._h5 = None
            self._demo = None

    def __enter__(self) -> "MemoryEpisode":
        self._ensure_open()
        return self

    def __exit__(self, *_exc) -> None:
        self.close()

    def __len__(self) -> int:
        return self.num_frames

    def _check_frame(self, frame: int) -> int:
        frame = int(frame)
        if frame < 0 or frame >= self.num_frames:
            raise IndexError(f"frame {frame} outside [0, {self.num_frames})")
        return frame

    def get_step_index(self, frame: int) -> int:
        frame = self._check_frame(frame)
        index = bisect.bisect_right(self._ends, frame)
        if index >= len(self.steps):
            raise RuntimeError(f"unassigned frame {frame} in {self.trajectory_id}")
        step = self.steps[index]
        if not step.start <= frame < step.end:
            raise RuntimeError(f"ambiguous frame {frame} in {self.trajectory_id}")
        return index

    def get_step_text(self, frame: int) -> str:
        return self.steps[self.get_step_index(frame)].text

    def get_condition(
        self, frame: int, mode: Union[ConditioningMode, str]
    ) -> str:
        frame = self._check_frame(frame)
        if not isinstance(mode, ConditioningMode):
            mode = ConditioningMode(str(mode).upper())
        if mode is ConditioningMode.FULL:
            return self.full_instruction
        step_index = self.get_step_index(frame)
        step = self.steps[step_index]
        if mode is ConditioningMode.CURRENT:
            return step.text
        return step.text if frame == step.start else "[HOLD]"

    def get_raw_sim_state(self, frame: int) -> np.ndarray:
        """Return privileged flattened MuJoCo state for auditing, not model input."""
        frame = self._check_frame(frame)
        return np.asarray(self._ensure_open()["states"][frame])

    def get_robot_state(self, frame: int) -> np.ndarray:
        """Return Panda joint qpos (7) plus gripper qpos (2), excluding time/objects."""
        raw = self.get_raw_sim_state(frame)
        if raw.shape[0] < 10:
            raise ValueError(f"state width below 10 for {self.trajectory_id}")
        return np.asarray(raw[1:10], dtype=np.float32)

    def get_action(self, frame: int) -> np.ndarray:
        frame = self._check_frame(frame)
        return np.asarray(self._ensure_open()["actions"][frame], dtype=np.float32)

    def get_observation(self, frame: int):
        frame = self._check_frame(frame)
        if self._observation_adapter is None:
            return None
        return self._observation_adapter(self.trajectory_id, frame)

    def get_analysis_labels(self, frame: int) -> dict:
        frame = self._check_frame(frame)
        step_index = self.get_step_index(frame)
        step = self.steps[step_index]
        result = {
            "trajectory_id": self.trajectory_id,
            "frame": frame,
            "step_index": step_index,
            "step_text": step.text,
            "transition_event": step_index > 0 and frame == step.start,
            "episode_start_event": frame == 0,
            "episode_end_event": frame == self.num_frames - 1,
            "steps_since_transition": frame - step.start,
            "cumulative_transition_count": step_index,
        }
        for depth in range(1, 6):
            previous = step_index - depth
            result[f"previous_{depth}"] = previous if previous >= 0 else -1
        return result

    def get_frame(
        self,
        frame: int,
        mode: Union[ConditioningMode, str] = ConditioningMode.CURRENT,
        include_action: bool = False,
    ) -> dict:
        """Return model inputs and analysis labels in disjoint dictionaries."""
        frame = self._check_frame(frame)
        model_input = {
            "observation": self.get_observation(frame),
            "robot_state": self.get_robot_state(frame),
            "condition": self.get_condition(frame, mode),
        }
        if include_action:
            model_input["action"] = self.get_action(frame)
        return {
            "model_input": model_input,
            "analysis": self.get_analysis_labels(frame),
            "episode_start": frame == 0,
            "episode_end": frame == self.num_frames - 1,
        }

    def iter_frames(
        self,
        mode: Union[ConditioningMode, str] = ConditioningMode.CURRENT,
        include_action: bool = False,
    ) -> Iterator[dict]:
        """Iterate in strict temporal order; timestep shuffling is unsupported."""
        for frame in range(self.num_frames):
            yield self.get_frame(frame, mode=mode, include_action=include_action)


class RoboCerebraMemoryDataset:
    """Canonical index over strict-clean RoboCerebra trajectories."""

    def __init__(
        self,
        episode_index: Union[str, Path],
        observation_adapter: Optional[ObservationAdapter] = None,
    ) -> None:
        self.episode_index = Path(episode_index)
        payload = json.loads(self.episode_index.read_text(encoding="utf-8"))
        records = payload["episodes"] if isinstance(payload, dict) else payload
        self._records = {str(item["trajectory_id"]): item for item in records}
        if len(self._records) != len(records):
            raise ValueError("duplicate trajectory_id in episode index")
        self._observation_adapter = observation_adapter

    def __len__(self) -> int:
        return len(self._records)

    def trajectory_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._records))

    def load_episode(self, trajectory_id: str) -> MemoryEpisode:
        try:
            metadata = self._records[trajectory_id]
        except KeyError as error:
            raise KeyError(f"unknown trajectory_id: {trajectory_id}") from error
        return MemoryEpisode(metadata, self._observation_adapter)
