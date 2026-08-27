# Automatic Simulator-GT Oracle Annotation (Task 3.1)

## 1. Purpose

Finalize reproducible Oracle S1→S2 transitions for 150 LIBERO-10 demonstrations
without manual frame-by-frame labeling. Only non-terminal ambiguity determines
Phase-1 conditioning eligibility.

## 2. Why Simulator-GT Oracle Is Used

The objective is ground-truth task progress, not learned completion prediction.
Recorded MuJoCo state can be restored exactly and evaluated through the task's
official predicates. RGB is therefore limited to sampled QA.

## 3. Evidence Sources

Evidence priority is recorded simulator state, official LIBERO BDDL predicate,
robosuite bilateral finger contact, named object/articulation state, and finally
RGB review. Gripper commands are release fallback/candidate-search evidence, never
a standalone semantic boundary.

## 4. Dataset Alignment

`action[t] → env.step(action[t]) → obs[t]`. Completion observed at `t` gives
`completion_obs_index = completion_action_index = t`; S2 conditioning starts at
`action[t+1]`.

## 5. Causal Completion Definition

At each timestep, the official S1 relation must be true and a causal release latch
must be true. The latch is set by bilateral contact loss, or by a close-to-open
command when strict bilateral contact was never detected; regrasp resets it. Each
value at `t` uses only state/action evidence through `t`.

## 6. Why Future Stability Cannot Define the Transition

Moving a transition from `t` to `t+4` merely because four future frames confirm it
would leak future information into the conditioning label. The chosen boundary is
the first frame of the final effective candidate run, not its confirmation frame.

## 7. Stability as QA

Five frames (0.25 s at 20 Hz) are inspected after a candidate for confidence and
regression QA. `future_qa.used_to_delay_boundary` is always false. A final run
shorter than five frames is `right_censored`, not failed.

## 8. Recovery / Regression Logic

Every predicate+release run is a causal candidate. A later predicate regression or
regrasp invalidates an earlier candidate; the boundary remains the first frame of
the final effective run. Invalidated runs, reasons, extra gripper cycles, initial
candidate, final boundary, and their frame difference are retained. This use of
future data selects among semantic recovery episodes but never shifts the chosen
episode's start forward for stability confirmation.

## 9. Task-specific Predicates

- Task 3 S1: `In(akita_black_bowl_1, white_cabinet_1_bottom_region) AND released`;
  terminal S2: `Close(white_cabinet_1_bottom_region)`.
- Task 4 S1: `On(porcelain_mug_1, plate_1) AND released`; terminal S2:
  `On(white_yellow_mug_1, plate_2) AND released`.
- Task 9 S1: `In(white_yellow_mug_1, microwave_1_heating_region) AND released`;
  terminal S2: `Close(microwave_1)`.

## 10. Terminal Censoring Policy

Terminal status is `observed` for a five-frame final condition run,
`right_censored` for a shorter final run, and `unresolved` when no valid condition
holds at trajectory end. Task 4 has 37 observed, 9 right-censored, and 4 unresolved
terminal completions. Tasks 3 and 9 have 50 observed each.

## 11. Conditioning Eligibility

Eligibility requires a valid non-terminal S1 boundary only. All 150 demonstrations
are eligible. Terminal right-censoring/unresolved status never changes S1→S2
conditioning eligibility, and no non-terminal case requires human review.

## 12. Automatic Annotation Algorithm

The annotator reads the previously extracted simulator predicate traces, constructs
causal release latches, enumerates causal predicate+release runs, invalidates
regressed/regrasped candidates, selects the start of the final effective S1 run,
classifies terminal status, writes annotations, and runs schema/index/causality QA.

## 13. Annotation Schema

Schema `libero_semantic_subtask_v1` with protocol revision
`task_3_1_causal_boundary` stores alignment, eligibility, ordered subtasks,
completion evidence/status, future QA, recovery events, retention metadata,
gripper-proxy comparison, and review flags. Suggested transition-distance bins are
`0–5`, `6–10`, `11–20`, `21–40`, `41–80`, and `81+`.

## 14. QA

Independent QA passed all 150 annotations: bounds, observation/action equality,
`next_start = completion + 1`, S1/S2 ordering when terminal completion is observed,
terminal next-start null, official predicate/release evidence, retention formula,
aggregate/individual equality, and `used_to_delay_boundary = false`.

## 15. Human Review Policy

Only unresolved non-terminal transitions enter `human_review_queue.csv`; the queue
is empty. Medium-confidence, recovery, retention-outlier, and large-proxy-gap cases
are sampled for QA rather than manual relabeling. The automatic sample contains 21
videos and 42 boundary sheets.

## 16. Coverage

| Task | Total | Eligible | High | Medium | Needs review | Terminal O/RC/U |
|---:|---:|---:|---:|---:|---:|---:|
| 3 | 50 | 50 | 20 | 30 | 0 | 50/0/0 |
| 4 | 50 | 50 | 17 | 33 | 0 | 37/9/4 |
| 9 | 50 | 50 | 30 | 20 | 0 | 50/0/0 |
| **All** | **150** | **150** | **67** | **83** | **0** | **137/9/4** |

## 17. Confidence Statistics

High requires official predicate, bilateral-contact release, stable QA, and no
nearby recovery. Medium remains automatically eligible and covers command-fallback
release, nearby recovery/contact regression, or shortened stability evidence.
There are 67 high, 83 medium, and zero needs-review transitions.

## 18. Recovery Statistics

Non-terminal recovery/candidate-invalidation flags occur in Task 3: 27, Task 4:
26, and Task 9: 15 demonstrations (68 total). Any-stage recovery flags, including
terminal predicate/contact flicker, occur in 27/47/15 demonstrations. The largest
initial-candidate-to-final-boundary gap is 53 frames (Task 3 demo 32). Events are
kept explicitly so downstream work can distinguish contact flicker, predicate
regression, regrasp, and extra-cycle candidates.

## 19. Gripper Proxy Comparison

Across all 150 demonstrations, absolute semantic-Oracle versus preliminary
second-gripper-transition difference is mean 8.33, median 2, p90 8.1, and maximum
142 frames. For non-terminal recovery cases it is mean 14.59, median 1.5, p90 82.9,
and maximum 142. The maximum is Task 9 demo 3; Task 3 demo 34 differs by 110 frames.

## 20. Semantic Retention Statistics

| Task | S1 duration min/mean/median/max | Retention min/mean/median/max | Retention p25/p75 | Median seconds |
|---:|---:|---:|---:|---:|
| 3 | 129/164.20/161.5/216 | 67/84.48/80.5/126 | 76/89.75 | 4.025 |
| 4 | 93/115.14/111/152 | 117/143.04/141.5/195 | 132.25/149.75 | 7.075 |
| 9 | 126/175.22/166.5/278 | 80/129.42/120/235 | 109.5/142.75 | 6.0 |

Retention is `T - next_subtask_action_start`, the number of S2-conditioned actions
from the transition through trajectory end.

## 21. MaIL Context Comparison

MaIL uses `obs_seq=5` (0.25 s). Median retention is 16.1× that context for Task 3,
28.3× for Task 4, and 24.0× for Task 9. Median trajectories are respectively
245.5, 258, and 300.5 steps.

## 22. Known Limitations

Strict bilateral contact can flicker, so fallback/recovery-near cases are medium
confidence. Recovery disambiguation is an offline Oracle operation, not an online
completion predictor. Four Task 4 terminal completions remain unresolved, but they
do not affect the only required S1→S2 transition. No learned vision/VLM signal,
split, model input sequence, training, or rollout was produced.

## 23. Information Passed to Task 4

Read `annotations/libero10_semantic/manifest.json`, select
`conditioning_eligible=true` and `needs_review=false`, keep S1 actions through
completion index `c`, and apply S2 conditioning starting at `c+1`. Terminal status
must not be used to discard an otherwise eligible trajectory. This task does not
itself generate Current-Subinstruction or HOLD sequences.
