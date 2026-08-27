"""Lazy RoboCasa365 LeRobot-v2 trajectory interface for MaIL.

The released parquet order is base[0:4], control_mode[4], EEF translation
[5:8], EEF rotation[8:11], gripper[11].  The canonical MaIL action is therefore
``action[5:12]`` and never ``action[:7]``.

RoboCasa's data wrapper inserts the initial pre-action state at the first
interaction, then appends one post-step state per action.  ``collect_demos``
deletes the final extra state, leaving state/image row ``t`` paired causally with
action ``t``.  The original per-frame annotation values are returned at the same
target-action indices without semantic regrouping.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np


EXTERNAL_KEY = "observation.images.robot0_agentview_left"
WRIST_KEY = "observation.images.robot0_eye_in_hand"
EXPECTED_ANNOTATION_KEYS = (
    "subtask_idx",
    "annotation.human.subtask",
    "annotation.human.subtask_name",
    "annotation.human.subtask_stage",
)


def _import_parquet():
    try:
        import pyarrow.parquet as parquet
    except ImportError as exc:  # pragma: no cover - exercised by deployment only
        raise RuntimeError(
            "pyarrow is required to read RoboCasa parquet files; install pyarrow>=17"
        ) from exc
    return parquet


def _column_array(table, name: str, dtype=None) -> np.ndarray:
    if name not in table.column_names:
        raise KeyError(f"missing parquet column: {name}")
    values = table[name].to_pylist()
    return np.asarray(values, dtype=dtype)


def extract_mail_action(action: np.ndarray) -> np.ndarray:
    """Extract canonical [dx,dy,dz,drx,dry,drz,gripper] as float32."""
    action = np.asarray(action)
    if action.shape[-1] != 12:
        raise ValueError(f"expected a 12D LeRobot action, got {action.shape}")
    return np.asarray(action[..., 5:12], dtype=np.float32)


@dataclass(frozen=True)
class ArmOnlyDecision:
    eligible: bool
    base_nonzero_frames: int
    base_nonzero_fraction: float
    base_max_norm: float
    control_mode_values: Tuple[float, ...]
    control_mode_change_count: int
    exclusion_reason: str


def arm_only_decision(action: np.ndarray) -> ArmOnlyDecision:
    """Apply the exact-zero arm-only rule to one released LeRobot episode."""
    action = np.asarray(action, dtype=np.float64)
    if action.ndim != 2 or action.shape[1] != 12 or len(action) == 0:
        raise ValueError(f"expected nonempty [T,12] actions, got {action.shape}")
    base = action[:, 0:4]
    control = action[:, 4]
    nonzero = np.any(base != 0.0, axis=1)
    values = tuple(float(x) for x in np.unique(control))
    changes = int(np.count_nonzero(control[1:] != control[:-1]))
    reasons = []
    if np.any(nonzero):
        reasons.append("nonzero_base_action")
    if values != (-1.0,):
        reasons.append("control_mode_not_constant_arm")
    return ArmOnlyDecision(
        eligible=not reasons,
        base_nonzero_frames=int(nonzero.sum()),
        base_nonzero_fraction=float(nonzero.mean()),
        base_max_norm=float(np.linalg.norm(base, axis=1).max()),
        control_mode_values=values,
        control_mode_change_count=changes,
        exclusion_reason=";".join(reasons),
    )


def build_action_chunk(
    actions: np.ndarray, target_start: int, horizon: int = 10
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create a zero-padded in-episode chunk and its valid source indices."""
    actions = np.asarray(actions)
    if horizon < 1:
        raise ValueError("horizon must be positive")
    if not 0 <= int(target_start) < len(actions):
        raise IndexError(f"target start {target_start} outside episode length {len(actions)}")
    chunk = np.zeros((horizon, actions.shape[1]), dtype=actions.dtype)
    valid = np.zeros(horizon, dtype=bool)
    indices = np.full(horizon, -1, dtype=np.int64)
    end = min(len(actions), int(target_start) + horizon)
    count = end - int(target_start)
    chunk[:count] = actions[int(target_start):end]
    valid[:count] = True
    indices[:count] = np.arange(int(target_start), end, dtype=np.int64)
    return chunk, valid, indices


def build_observation_indices(observation_index: int, obs_seq: int = 5) -> np.ndarray:
    """Return a full, unpadded MaIL window; early incomplete windows are invalid."""
    if obs_seq < 1:
        raise ValueError("obs_seq must be positive")
    observation_index = int(observation_index)
    start = observation_index - obs_seq + 1
    if start < 0:
        raise IndexError("observation window would cross the episode start")
    return np.arange(start, observation_index + 1, dtype=np.int64)


class MinMaxActionScaler:
    """Train-subset-only per-dimension affine scaler to [-1, 1]."""

    def __init__(self):
        self.minimum: Optional[np.ndarray] = None
        self.maximum: Optional[np.ndarray] = None

    @property
    def fitted(self) -> bool:
        return self.minimum is not None and self.maximum is not None

    def fit(self, loader: "RoboCasaTrajectoryLoader", train_episode_ids: Sequence[int]):
        if not train_episode_ids:
            raise ValueError("train_episode_ids must be explicitly nonempty")
        minimum = np.full(7, np.inf, dtype=np.float64)
        maximum = np.full(7, -np.inf, dtype=np.float64)
        for episode_id in train_episode_ids:
            actions = loader.load_low_dim(int(episode_id))["actions"]
            minimum = np.minimum(minimum, actions.min(axis=0))
            maximum = np.maximum(maximum, actions.max(axis=0))
        self.minimum = minimum.astype(np.float32)
        self.maximum = maximum.astype(np.float32)
        return self

    def _scale(self) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("scaler must be fitted on explicit training episodes")
        span = self.maximum - self.minimum
        return np.where(span == 0.0, 1.0, span)

    def transform(self, actions: np.ndarray) -> np.ndarray:
        actions = np.asarray(actions, dtype=np.float32)
        return ((actions - self.minimum) / self._scale() * 2.0 - 1.0).astype(np.float32)

    def inverse_transform(self, normalized: np.ndarray) -> np.ndarray:
        normalized = np.asarray(normalized, dtype=np.float32)
        return (((normalized + 1.0) * 0.5) * self._scale() + self.minimum).astype(np.float32)


class RoboCasaTrajectoryLoader:
    """Episode-local metadata loader plus lazy two-view frame decoder."""

    def __init__(
        self,
        dataset_root: Path,
        *,
        manifest_path: Optional[Path] = None,
        obs_seq: int = 5,
        action_horizon: int = 10,
        image_size: int = 128,
    ):
        self.root = Path(dataset_root)
        self.obs_seq = int(obs_seq)
        self.action_horizon = int(action_horizon)
        self.image_size = int(image_size)
        self.info = json.loads((self.root / "meta/info.json").read_text())
        if EXTERNAL_KEY not in self.info["features"] or WRIST_KEY not in self.info["features"]:
            raise KeyError("required external/wrist camera keys are absent")
        self.eligible_ids = None
        if manifest_path is not None:
            manifest = json.loads(Path(manifest_path).read_text())
            self.eligible_ids = frozenset(int(x) for x in manifest["episode_ids"])

    def episode_ids(self) -> Sequence[int]:
        paths = sorted((self.root / "data").glob("*/episode_*.parquet"))
        return [int(path.stem.rsplit("_", 1)[1]) for path in paths]

    def _assert_allowed(self, episode_id: int) -> None:
        if self.eligible_ids is not None and int(episode_id) not in self.eligible_ids:
            raise ValueError(f"episode {episode_id} is not in the arm-only manifest")

    def parquet_path(self, episode_id: int) -> Path:
        path = self.root / f"data/chunk-{int(episode_id) // 1000:03d}/episode_{int(episode_id):06d}.parquet"
        if not path.is_file():
            raise FileNotFoundError(path)
        return path

    def video_path(self, episode_id: int, key: str) -> Path:
        if key not in (EXTERNAL_KEY, WRIST_KEY):
            raise KeyError(f"unsupported policy camera {key}")
        path = self.root / f"videos/chunk-{int(episode_id) // 1000:03d}/{key}/episode_{int(episode_id):06d}.mp4"
        if not path.is_file():
            raise FileNotFoundError(path)
        return path

    def load_low_dim(self, episode_id: int) -> Dict:
        self._assert_allowed(episode_id)
        parquet = _import_parquet()
        table = parquet.read_table(self.parquet_path(episode_id))
        raw_action = _column_array(table, "action", np.float64)
        state = _column_array(table, "observation.state", np.float64)
        annotations = {
            key: _column_array(table, key)
            for key in table.column_names
            if key.startswith("annotation.") or key == "subtask_idx"
        }
        length = len(raw_action)
        frame_index = _column_array(table, "frame_index", np.int64)
        if not np.array_equal(frame_index, np.arange(length)):
            raise ValueError(f"episode {episode_id}: non-contiguous frame_index")
        valid = np.arange(self.obs_seq - 1, length, dtype=np.int64)
        return {
            "episode_id": int(episode_id),
            "num_frames": length,
            "raw_actions": raw_action,
            "actions": extract_mail_action(raw_action),
            "state": state,
            "timestamps": _column_array(table, "timestamp", np.float32),
            "frame_index": frame_index,
            "episode_index": _column_array(table, "episode_index", np.int64),
            "task_index": _column_array(table, "task_index", np.int64),
            "raw_annotations": annotations,
            "available_annotation_keys": sorted(annotations),
            "missing_expected_annotation_keys": sorted(set(EXPECTED_ANNOTATION_KEYS) - set(annotations)),
            "valid_indices": valid,
            "external_rgb": {"lazy_video_path": str(self.video_path(episode_id, EXTERNAL_KEY))},
            "wrist_rgb": {"lazy_video_path": str(self.video_path(episode_id, WRIST_KEY))},
        }

    def _decode_frames(self, path: Path, indices: Sequence[int]) -> np.ndarray:
        indices = np.asarray(indices, dtype=np.int64)
        if len(indices) == 0 or np.any(indices < 0):
            raise ValueError("frame indices must be nonempty and nonnegative")
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            raise RuntimeError(f"cannot open video: {path}")
        frames = []
        try:
            current = None
            for index in indices:
                index = int(index)
                if current != index:
                    capture.set(cv2.CAP_PROP_POS_FRAMES, index)
                ok, bgr = capture.read()
                if not ok:
                    raise RuntimeError(f"cannot decode frame {index} from {path}")
                current = index + 1
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                rgb = cv2.resize(
                    rgb, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR
                )
                frames.append(rgb)
        finally:
            capture.release()
        return np.stack(frames).astype(np.uint8)

    @staticmethod
    def _mail_image_tensor(frames: np.ndarray) -> np.ndarray:
        """Match MaIL: HWC uint8 -> CHW float32 in [0,1], without extra normalization."""
        return (frames.astype(np.float32) / 255.0).transpose(0, 3, 1, 2)

    def load_trajectory(self, episode_id: int) -> Dict:
        """Return low-dimensional trajectory and lazy RGB descriptors, never full RGB RAM."""
        return self.load_low_dim(episode_id)

    def get_sample(self, episode_id: int, observation_index: int) -> Dict:
        trajectory = self.load_low_dim(episode_id)
        observation_index = int(observation_index)
        if observation_index not in set(trajectory["valid_indices"].tolist()):
            raise IndexError(
                f"invalid policy observation {observation_index}; valid range is "
                f"[{self.obs_seq - 1}, {trajectory['num_frames'] - 1}]"
            )
        obs_indices = build_observation_indices(observation_index, self.obs_seq)
        target_start = observation_index
        chunk, mask, action_indices = build_action_chunk(
            trajectory["actions"], target_start, self.action_horizon
        )
        external = self._decode_frames(self.video_path(episode_id, EXTERNAL_KEY), obs_indices)
        wrist = self._decode_frames(self.video_path(episode_id, WRIST_KEY), obs_indices)
        target_annotations = {
            key: values[action_indices[mask]] for key, values in trajectory["raw_annotations"].items()
        }
        return {
            "episode_id": int(episode_id),
            "observation_index": observation_index,
            "observation_indices": obs_indices,
            "target_action_indices": action_indices,
            "external_rgb": self._mail_image_tensor(external),
            "wrist_rgb": self._mail_image_tensor(wrist),
            "actions": chunk.astype(np.float32),
            "valid_action_mask": mask,
            "raw_annotations": target_annotations,
            "timestamps": trajectory["timestamps"][obs_indices],
        }

    def fit_scaler(self, train_episode_ids: Sequence[int]) -> MinMaxActionScaler:
        return MinMaxActionScaler().fit(self, train_episode_ids)
