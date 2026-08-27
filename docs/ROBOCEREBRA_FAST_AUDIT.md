# RoboCerebra fast final compatibility audit

Date: 2026-08-24  
Decision: **REJECT**

RoboCerebra publishes real temporal intervals, a full instruction, continuous
state/action trajectories, and external video. The decisive incompatibility is
the annotation level: the intervals are for atomic `pick`, `place`, `open`, and
`pour` steps. The public release has no official grouping of those intervals
into the research-required semantic subgoals. Counting eight or eleven atomic
steps as semantic stages would violate the R0-E definition.

## Provenance and scope

- Code upstream: `https://github.com/qiuboxiang/RoboCerebra.git`
- Audited code commit: `2573426c13dfcd5e7d7831c15587b058aaa1c0c0`
- Commit date: `2026-04-29T20:03:10+08:00`
- Branch: `main`
- Code license: Apache-2.0
- Official data: `qiukingballball/RoboCerebra`, commit
  `5d2e1e361bf65aabbe4d18179515f5a10936cc96`
- Dataset page license: MIT
- Publication: NeurIPS 2025 Datasets & Benchmarks
- Simulator stack: LIBERO, robosuite, MuJoCo, Franka Panda, `OSC_POSE`
- Downloaded for this audit: 22,519,112 bytes. No full dataset was downloaded.

The paper reports 1,000 human-annotated trajectories over 100 task variants and
describes their decomposition as 2 to more than 20 **atomic steps**. It also
states that operators annotated step-level temporal boundaries. The public
training metadata contains 1,000 rows and explicit intervals, so the paper's
atomic-boundary claim is supported by public bytes. It does not establish the
additional semantic grouping required by this research.

## Actual HDF5 schema

Three original training HDF5 files were opened directly:

```text
/
└── data                              group; environment metadata attributes
    └── demo_1                        group; model_file attribute
        ├── actions                   float64 [T, 7]
        └── states                    float64 [T, state_width]
```

There is no image, language, instruction, subtask, stage, boundary, segment, or
annotation dataset inside these HDF5 files. FULL and step intervals live in the
adjacent `task_description.txt`, with literal fields:

```text
Task: <full instruction>
Step: <atomic instruction>
[start_frame, end_frame]
Related Objects: ...
```

This is public-boundary CASE D: an external trajectory-relative annotation.

| sample | actions | states | atomic intervals | final interval matches T | video |
|---|---:|---:|---:|---|---|
| coffee_table/case1 | 5,504×7 | 5,504×84 | 8 | no: 5,540 vs 5,504 | 5,505 frames, 512×512, 60 fps |
| coffee_table/case2 | 3,739×7 | 3,739×116 | 11 | yes | 3,740 frames, 1280×1024, 60 fps |
| coffee_table/case10 | 5,015×7 | 5,015×90 | 11 | yes | 5,016 frames, 512×512, 60 fps |

For all three samples, the first state component is strictly monotonic with a
median increment of 0.05 seconds. Each file has one `demo_1`, the action and
state lengths match, and the corresponding video has `T+1` frames. This supports
a continuous, non-stitched raw trajectory. Case1 nevertheless demonstrates a
real boundary-quality defect at the terminal interval.

## Boundary coverage and quality

All 1,000 public training metadata rows contain step text and numeric intervals.
The metadata audit found:

- 998 start at frame zero;
- 990 have strictly contiguous intervals;
- 979 contain only positive-duration intervals;
- 974 satisfy all three conditions;
- 26 rows have at least one interval anomaly;
- only 2 of the 3 downloaded HDF5 samples have a terminal interval matching the
  HDF5 action length.

The release is therefore `PARTIAL`, rather than a claim that every public
interval is clean. The two clean samples demonstrate that public atomic
boundaries are genuinely actionable; the mismatch and malformed metadata rows
show that validation is still required.

Across all 1,000 rows, the **atomic** step count is min 2, mean 8.934, median 9,
p75 10, p90 12, and max 23. Among the 974 interval-valid rows, annotated
trajectory duration at 20 Hz is median 2,814.5 frames / 140.725 seconds; atomic
interval duration is median 289 frames / 14.45 seconds. These numbers must not
be reported as semantic-stage statistics.

## Atomic steps are not the required semantic stages

The downloaded plans contain instructions such as:

- `Pick up cream cheese from the coffee table`
- `Place down cream cheese into the middle region of short cabinet`
- `Pick up milk from the coffee table`
- `Pour out milk into the red coffee mug`

The paper describes `pick`, `place`, and `pour` as common action primitives and
calls the dataset units atomic steps. A research semantic stage such as "store
cream cheese in the cabinet" would require grouping at least the pick and place
intervals. No group ID, semantic parent ID, parent text, or parent boundary is
present in the HDF5, TXT, train parquet, official converter, or inspected
LeRobot metadata. That grouping cannot be recovered without adding a new rule
or manual annotation.

Consequently:

- ordered official atomic steps: yes;
- ordered official semantic subgoals: no;
- trajectories with verified semantic stage count ≥4: zero;
- large-scale manual/grouping work required for the desired representation: yes.

## Repetition

The 1,000 training rows contain 994 distinct exact FULL instructions. Six FULL
instructions occur twice and all others occur once, so the maximum exact-task
repetition is 2. All 1,000 exact atomic text plans are unique.

If object names are discarded and only the primitive verb sequence is retained,
561 patterns remain and the most frequent pattern occurs 22 times. This is not
a valid semantic plan-family statistic: it collapses tasks merely because they
share a sequence such as `open→pick→place→pick→place→pick→place`. No official
semantic task-family identifier is published. Repetition is therefore below the
10-demo minimum for a defensible exact or semantic plan family.

## Action, alignment, and cameras

The raw action is 7D: Cartesian translation command, axis-angle rotation
command, and one gripper value. The controller is `OSC_POSE`, and the paper
denotes the action as `[Δx, Δθ, ΔGrip]`. The observed raw samples are not
consistently confined to `[-1, 1]`, so MaIL compatibility is `ADAPTER_ONLY`,
requiring scale/convention normalization but no dimensional redesign.

The official converter pairs `orig_states[i]` with `orig_actions[i]`, sets that
state, renders observations, and stores the same-index action. Its audited
alignment is therefore `obs[t] -> action[t]`.

Original HDF5 contains no RGB. Each original case has one public external MP4.
The official converter renders `agentview_rgb` and `eye_in_hand_rgb` at
256×256. The secondary `lerobot/robocerebra_unified` metadata also declares
256×256 external and wrist video at 20 fps plus a 7D action. Thus two-view
availability passes as a conversion capability, not as a property of the raw
HDF5 inspected here.

## Converted release audit

The official conversion script slices every annotated atomic interval and
writes it as a separate HDF5/RLDS episode. The RLDS instruction is derived from
that atomic step's filename. This discards the single continuous FULL trajectory
as the training episode unit.

The secondary LeRobot v3 metadata reports 6,660 episodes, 571,116 frames, and
1,728 task indices. Its `meta/tasks.parquet` contains only `task_index`; it does
not preserve an instruction string or parent FULL/semantic-boundary structure.
Converted data is therefore not a substitute source of truth for this audit.

## Diagnostic timeline and visual review

`robocerebra_sample_timeline.csv` deterministically expands the valid case2
atomic intervals. FULL is constant, CURRENT is the official atomic instruction,
HOLD is emitted after the first frame of each atomic interval, and
`steps_since_transition` / `cumulative_transition_count` are verified. Every row
is marked `OFFICIAL_ATOMIC_STEP_NOT_SEMANTIC_STAGE`; it is not a research-ready
semantic timeline.

Three contact sheets show the published transition frames. They support visual
continuity and expose the annotation as fine-grained motor/action steps. The
case1 sheet is explicitly marked with the terminal-length mismatch.

## Gate

| gate | result | reason |
|---|---|---|
| G1 FULL | PASS | external public FULL field exists |
| G2 ordered semantic subtasks | FAIL | only ordered atomic steps exist |
| G3 temporal boundary | PASS with quality anomalies | public external start/end intervals exist |
| G4 ≥4 semantic stages | FAIL | semantic count is not annotated |
| G5 continuous trajectory | PASS | original HDF5 samples are continuous |
| G6 RGB + action | PASS | external MP4 plus 7D HDF5 action |
| G7 7D compatibility | PASS | `ADAPTER_ONLY` |
| G8 external + wrist | PASS | official converter and LeRobot metadata provide both |
| G9 repetition | FAIL | exact FULL max 2; semantic family absent |
| G10 no manual annotation | FAIL | semantic grouping/boundaries would need creation |

Final decision: **REJECT**. Do not reconstruct semantic boundaries and do not
search another benchmark automatically.

No model was trained, no GPU job was submitted, no existing dataset was deleted,
and no git push was performed.
