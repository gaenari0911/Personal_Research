# VLABench ↔ MaIL compatibility audit

Audit date: 2026-08-24  
Decision: **adapter-compatible at the tensor interface, but benchmark migration rejected**

## Compatibility matrix

| Interface | MaIL baseline | VLABench release | Classification |
|---|---|---|---|
| action width | 7 | 7 policy dimensions | UNCHANGED |
| action meaning | LIBERO-style relative/delta EEF control | absolute robot-base EEF pose target + binary gripper | ADAPTER_ONLY, semantic change |
| actuator width | benchmark adapter owns it | 9 joint/finger targets | ADAPTER_ONLY via official IK wrapper |
| external RGB | 128×128 | `observation.images.image`, 224×224 | ADAPTER_ONLY resize |
| wrist RGB | 128×128 | `observation.images.wrist_image`, 224×224 | ADAPTER_ONLY resize |
| third RGB | absent | `second_image`, 224×224 | omit from policy; optional QA |
| visual encoders | two CLIP encoders, 512D each | two selected RGB streams | UNCHANGED |
| temporal core | Mamba, d_model=128, nominal 16 layers | benchmark-independent | UNCHANGED |
| context/chunk | obs_seq=5, action horizon=10 | 10 Hz data | UNCHANGED; 0.5 s/1.0 s physical spans |
| loss | BC/MSE | continuous pose target except binary gripper | UNCHANGED for baseline comparison |
| language | full natural-language instruction | `task_index` → official task string | UNCHANGED CLIP path |

## Exact action contract

The released `actions` vector is

`[x, y, z, roll, pitch, yaw, gripper]`.

- `xyz`: absolute target in the robot-base-relative Cartesian frame.
- `roll,pitch,yaw`: radians, SciPy extrinsic `xyz` convention.
- `gripper`: `1=open`, `0=closed`.
- Alignment: pre-action `observation[t]` predicts the target `action[t]`.
- Runtime: the official LeRobot wrapper converts this pose to seven arm joint
  targets through IK and appends two finger targets before the dm_control step.

The 7D output width therefore does **not** make the representation equivalent to
the current MaIL/LIBERO action.  The VLABench adapter must normalize and
denormalize absolute targets, then call the official IK conversion.  It must not
integrate model output as a delta.

The release has an additional contract bug: its declared Euler Box is `[-1,1]`,
but sampled official actions span approximately `[-π,π]` for roll/yaw and
`[-π/2,π/2]` for pitch.  Training should derive scaling from the training data,
and rollout code must not clip rotations to the declared Box.

## Observation/action alignment

`VLABench/utils/skill_lib.py::step_trajectory` executes a waypoint and reads the
post-action observation.  Higher-level skill builders prepend the initial
pre-action observation and remove the last post-action observation, producing
equal-length pre-action observations and waypoint targets.  The correct BC
pairing is consequently `obs[t] → action[t]`.

On the first 500 episode parquets, the median absolute XYZ discrepancy was
0.00686 against `obs[t]` and 0.00348 against `obs[t+1]`.  This is expected for a
target action that the next state approaches; it is not evidence for a shifted
training target.

## Camera mapping

Use `observation.images.image` as external view and
`observation.images.wrist_image` as wrist view, resizing 224×224 to 128×128.
Representative two-view H.264 clips show both workspaces continuously.  The
oblique `second_image` improves human diagnosis of occlusion but is not needed
to preserve the two-encoder baseline.

## What remains unchanged

- Two 512D CLIP image features and the current language embedding path.
- Mamba d_model=128 and nominal 16-layer state-space core.
- `obs_seq=5`, `action_horizon=10`, 7D head, BC/MSE objective.

## Adapter-only changes

- Dataset key mapping and 224→128 image preprocessing.
- Absolute-pose per-dimension normalization/denormalization.
- Official robot-frame pose → IK → 9D environment-control conversion.
- Explicit enforcement of `obs[t] → action[t]` and train-split statistics.

## Migration blocker outside the model

The released parquet/video schema stores no stage, skill, boundary, predicate,
episode configuration, or initial physics state.  Thus it cannot supply the
required reproducible semantic boundaries or exact replay, regardless of model
compatibility.  MaIL can technically consume the observations/actions, but the
requested 4–6-stage experiment cannot be reproduced without new instrumented
data generation or substantial annotation.

Primary evidence:

- [VLABench repository](https://github.com/OpenMOSS/VLABench)
- [Official LeRobot VLABench environment](https://huggingface.co/docs/lerobot/vlabench)
- [Official composite dataset](https://huggingface.co/datasets/VLABench/vlabench_composite_ft_lerobot_video)
- Local pinned source: `external/VLABench`, commit
  `cf588fe60c0c7282174fe979f5913170cfe69017`.
