# R1 MaIL ↔ RoboCasa WashFruitColander adapter

Status: implementation and smoke tests pass; R1 data gate fails because the
acquired LeRobot v2.1 archive lacks the required updated subtask annotations.

## Canonical action contract

The released 12D LeRobot action is ordered as follows:

| slice | released meaning |
|---|---|
| `[0:4]` | mobile-base command |
| `[4]` | hybrid controller mode |
| `[5:8]` | end-effector translation command |
| `[8:11]` | end-effector rotation command |
| `[11]` | gripper-close command |

The MaIL action is exactly `parquet_action[..., 5:12]`, cast to `float32`:

```text
[dx, dy, dz, d_rx, d_ry, d_rz, gripper_close]
```

Selecting `action[:7]` is prohibited because it mixes four base dimensions,
controller mode, and only two arm dimensions.

The actual `extras/dataset_meta.json` establishes these semantics:

- right-arm controller: `OSC_POSE`;
- `input_type=delta`, reference frame `base`;
- normalized input range `[-1,1]`;
- translation output range `[-0.05,0.05]` metres per controller target;
- rotation output range `[-0.5,0.5]` as the OSC rotation-vector command;
- gripper `+1=close`, `-1=open` in the stored environment convention.

All 489 selected episodes have every base component exactly zero and
`control_mode=-1` at every frame. The 18 excluded episodes have both nonzero
base commands and controller-mode changes.

## Temporal alignment correction

The final policy convention is:

```text
observation[t] -> action[t]
shift = 0
```

This supersedes the earlier provisional `t+1` recommendation. The important
source sequence is:

1. RoboSuite's `DataCollectionWrapper.step` executes the action.
2. On the first interaction, `_on_first_interaction` inserts the episode's
   initial pre-action state before the first post-step state.
3. Every step then appends a post-step state and its action.
4. RoboCasa `collect_demos.py` deletes the final extra state, leaving the
   initial state through the penultimate post-step state aligned with all actions.
5. `convert_hdf5_lerobot.py` renders `states[t]` and writes `actions[t]` in the
   same parquet row.

Across 518,396 state transitions, the next EEF-state delta correlates more with
the same-row translation action than with the next-row action:

| test | same-row | next-row |
|---|---:|---:|
| flattened Pearson | 0.92150 | 0.90663 |
| mean directional cosine | 0.89246 | 0.87342 |

Source and numerical evidence therefore both support shift 0 with HIGH
confidence. Raw annotations, when available, must be read at the same index `t`.

## Camera adapter

```text
agentview_rgb    <- observation.images.robot0_agentview_left
eye_in_hand_rgb  <- observation.images.robot0_eye_in_hand
```

The native videos are H.264/yuv420p RGB, 256×256, 20 Hz. The adapter decodes
with OpenCV, converts BGR to RGB, resizes deterministically to 128×128 using
bilinear interpolation, then matches MaIL's existing path:

```text
HWC uint8 -> CHW float32 / 255
```

No second ImageNet/CLIP normalization is applied in the data adapter.

## Sequence and horizon contract

- `obs_seq=5`: use a complete window `[t-4,...,t]`.
- Start behavior: match the current MaIL `padding_sequence=False`; the first
  valid policy step is `t=4`, rather than repeating or zero-padding observations.
- `action_horizon=10`: target `[action[t],...,action[t+9]]`.
- Episode end: zero-pad the chunk and emit a boolean valid-action mask.
- No action or observation may come from a neighbouring episode.

## Normalization contract

`MinMaxActionScaler.fit(loader, train_episode_ids)` requires an explicit,
nonempty list of training episodes. It computes per-dimension ranges only from
that subset and supports `transform`/`inverse_transform`. R1 used episodes
205/379/189 only for a temporary smoke test; maximum round-trip error was
`8.94e-08`. No final scaler was saved, avoiding future split leakage.

## Lazy loader API

Implementation: `src/robocasa_phase1/interface.py`.

`RoboCasaTrajectoryLoader.load_trajectory(episode_id)` loads one episode's
parquet metadata/action/state and returns lazy video-path descriptors. It never
loads the complete RGB dataset into RAM. `get_sample(episode_id,t)` decodes only
the five requested frames from each selected camera and returns:

```text
external_rgb          [5,3,128,128] float32
wrist_rgb             [5,3,128,128] float32
actions               [10,7] float32
valid_action_mask     [10] bool
observation_indices   [5]
target_action_indices [10], padded slots = -1
raw_annotations       unmodified values at target indices
timestamps            [5]
```

## Annotation blocker

The acquired v2.1 schema contains only
`annotation.human.task_description` and `annotation.human.task_name`. It lacks
`subtask_idx`, `annotation.human.subtask`, `subtask_name`, and `subtask_stage`.
The loader reports those fields as missing instead of fabricating them. R1 does
not group semantic stages and does not generate FULL/CURRENT/HOLD timelines.

## Test result

Fourteen R1 tests pass, covering action reorder/extraction, arm-only selection,
cameras, RGB shape/range, scaler round trip, temporal alignment, horizon indices,
terminal padding, observation windows, episode isolation, annotation absence,
and finite outputs. Together with the ten existing common-interface tests, the
final suite is 24/24 PASS.

Primary source evidence:

- [Official RoboCasa repository](https://github.com/robocasa/robocasa)
- [Official RoboSuite DataCollectionWrapper](https://github.com/ARISE-Initiative/robosuite/blob/master/robosuite/wrappers/data_collection_wrapper.py)
- Local RoboCasa commit `a07e365c958c4216cd6bbd5f30b47f09a65c6f00`.
