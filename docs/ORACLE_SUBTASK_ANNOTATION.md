# Oracle Subtask Annotation (Task 3)

> **Task 3.1 protocol revision (final):** The five-frame future-stability test
> documented below is retained as QA, confidence, and recovery evidence, but it no
> longer delays the causal transition timestep. If the official predicate and
> causal release evidence first hold at `t` and no later regression/recovery
> invalidates that candidate, completion remains `t` and S2 begins at `t+1`.
> Terminal censoring is separated from non-terminal conditioning eligibility.
> The finalized implementation and results are documented in
> `docs/AUTOMATIC_ORACLE_ANNOTATION.md`; this file preserves the Task 3 pilot
> record rather than rewriting its historical decision.

## 1. Purpose

This task investigates trajectory-based semantic completion boundaries for LIBERO-10
Tasks 3, 4, and 9. The intended result is an Oracle transition dataset, but the
full-corpus gate ended in **NEEDS HUMAN REVIEW**; therefore no uncertain 150-demo
annotation set was emitted.

## 2. Dataset Alignment Convention

The dataset convention is `action[t] -> env.step(action[t]) -> obs[t]`. Thus
`obs[t]` is post-action evidence for `action[t]`.

## 3. Definition of Semantic Completion

Placement completion requires the official target relation, release evidence, and
non-immediate regression. Articulation completion requires the official closed
predicate. Gripper command transitions alone are candidate-search signals, never
the semantic label.

## 4. Stable Completion Rule

The pilot rule requires five consecutive frames (0.25 s at 20 Hz) of target
predicate truth and no bilateral grasp contact. A placement additionally rejects
windows with object displacement above 0.01 m per frame. The motion limit is only
a gross-motion safeguard; the official semantic predicate remains primary.

## 5. Recovery / Regression Rule

An early release, predicate truth interval, or gripper cycle is only a candidate.
If the relation regresses or the object is regrasped, completion moves to the first
frame of the final released and stable predicate run. Extra close/open cycles and
earlier predicate runs are retained as recovery evidence.

## 6. Simulator-State Mapping Investigation

The HDF5 `states[t]` vectors were restored read-only with MuJoCo 2.3.7,
robosuite 1.4.0, and the checked-in LIBERO source. Stored XML asset prefixes were
mapped to the local LIBERO and robosuite assets. After XML reset, every tested
simulator flattened-state dimension exactly matched its HDF5 state dimension.
No action was replayed and no policy rollout was performed.

## 7. LIBERO Predicate Availability

- Task 3: official `In(akita_black_bowl_1, white_cabinet_1_bottom_region)` and
  `Close(white_cabinet_1_bottom_region)`.
- Task 9: official `In(white_yellow_mug_1, microwave_1_heating_region)` and
  `Close(microwave_1)`.
- Task 4: official `On(porcelain_mug_1, plate_1)` and
  `On(white_yellow_mug_1, plate_2)`.

LIBERO has no task-level release predicate. Release evidence therefore uses
robosuite's bilateral finger-pad contact check, augmented only when necessary by
gripper opening and object stability. Cabinet close is `qpos > 0`; microwave close
is `qpos > -0.005` in the checked-in object definitions.

## 8. Evidence Priority

The applied order is official LIBERO predicate, named simulator state, derived
simulator geometry/motion, RGB plus gripper/contact evidence, then human review.
The final decision does not promote lower-priority evidence over a conflicting
official predicate or grasp state.

## 9. Task 3 Subtask Definitions

- S1: black bowl is inside the bottom drawer and released.
- S2: bottom drawer satisfies the official closed condition.

## 10. Task 9 Subtask Definitions

- S1: yellow-and-white mug is inside the microwave heating region and released.
- S2: microwave door satisfies the official closed condition.

## 11. Task 4 Subtask Definitions

- S1: porcelain/white mug is on the left plate and released.
- S2: yellow-and-white mug is on the right plate and released.

BDDL object names remove visual identity ambiguity, and all 50 recorded Task 4
demonstrations manipulate the white mug first.

## 12. Annotation Index Convention

If completion is observed at `c`, then
`completion_obs_index = completion_action_index = c` and the next subtask begins
at action `c + 1`. The terminal subtask has `next_subtask_action_start = null`.

## 13. Pilot Annotation

| Task | Demo | T | S1 candidate | S2 candidate |
|---:|---|---:|---:|---:|
| 3 | demo_2 | 199 | 128 | 188 |
| 3 | demo_20 | 247 | 170 | 236 |
| 3 | demo_32 | 317 | 195 | 306 |
| 3 | demo_34 | 287 | 215 | 276 |
| 9 | demo_29 | 224 | 128 | 213 |
| 9 | demo_9 | 301 | 146 | 290 |
| 9 | demo_36 | 449 | 277 | 438 |
| 9 | demo_3 | 415 | 260 | 404 |
| 4 | demo_19 | 216 | 98 | 210 |
| 4 | demo_17 | 259 | 111 | 251 |
| 4 | demo_33 | 331 | 135 | 321 |

These are pilot candidates, not a released Oracle dataset.

## 14. Recovery-case Validation

Task 3 demo 34 rejects its first failed placement/gripper cycle and completes S1
at 215. Task 9 demo 3 rejects preliminary cycles and completes S1 at 260. Task 4
demo 33 rejects a failed second-mug attempt plus transient `On` truth and completes
S2 at 321.

## 15. Cross-task Generalization

The uniform rule validated candidate boundaries for Task 3 50/50 and Task 9
50/50. It validated Task 4 only 37/50. The remaining 13 cases are terminal-censored,
unreleased, or end after predicate regression; consequently the rule does not
generalize to the entire requested corpus without a human-approved contract change.

## 16. Full Annotation Decision

**NEEDS HUMAN REVIEW.** The representative pilot passed, but the full read-only QA
scan overturned the corpus-level gate. Per `research.txt`, no uncertain 150-demo
annotations or aggregate JSONL were created.

## 17. Annotation Schema

The planned schema version is `libero_semantic_subtask_v1`, containing task and
demo identifiers, trajectory length, post-action alignment, ordered subtasks with
action start/completion observation/completion action/next start/predicate/
confidence, recovery metadata, `needs_review`, and notes. It is documented but not
materialized because the gate did not pass.

## 18. QA Rules

Candidate checks include bounds, completion action/observation equality, next start
equal to prior completion plus one, strict boundary ordering, non-overlap,
non-negative durations, terminal null next start, five-frame predicate/release
stability, recovery regression handling, and visual boundary review. The blocking
13 cases failed semantic stability rather than integer-index integrity.

## 19. Confidence Rules

- High: official predicate plus bilateral-contact release evidence and stability.
- Medium: official predicate plus gripper opening and object stability when the
  contact detector never activates.
- Needs review: predicate/release conflict or insufficient recorded horizon.

The validation scan produced candidate evidence counts of Task 3: 47 high/3
medium, Task 9: 47 high/3 medium, and Task 4: 30 high/7 medium/13 needs review.
These are gate diagnostics, not published annotations.

## 20. Annotation Statistics

Final annotation statistics are intentionally unavailable: committed coverage is
0/50 for each task because emitting a partial set could be mistaken for the Oracle
dataset. Diagnostic candidate coverage is 50/50, 50/50, and 37/50 for Tasks 3, 9,
and 4 respectively.

## 21. Recovery Statistics

Final Oracle recovery rates are withheld. The required pilot recoveries (Task 3
demo 34, Task 9 demo 3, Task 4 demo 33) were detected correctly. Corpus-wide
recovery statistics must be recomputed after the human decision because changing
terminal completion policy can change regression classification.

## 22. Preliminary Proxy vs Oracle Comparison

Pilots show why a gripper proxy is insufficient: Task 3 demo 2's stable semantic
candidate precedes the open-command transition by five frames, while recovery demo
34's first release is rejected and the effective completion is 110 frames later.
No final aggregate comparison is reported without a complete Oracle set.

## 23. Oracle Stage Duration Statistics

Not computed as final results because the full Oracle gate did not pass. Pilot
durations can be recovered from Section 13 but must not be treated as corpus
statistics.

## 24. Semantic Retention Gap Statistics

Not computed as final results. The required gap is `S2 completion - (S1
completion + 1) + 1`, equivalent to `S2 completion - S1 completion` actions under
the inclusive stage convention, but applying it to a partial corpus would bias the
reported distribution.

## 25. MaIL obs_seq=5 Comparison

Not computed because final semantic retention gaps are unavailable. After review,
the required comparison is each Oracle S2 duration divided by MaIL's observation
window of five.

## 26. Review Artifacts

- Pilot predicate-aware videos and ±10-frame sheets:
  `analysis/annotation_review/pilot/`.
- Pilot index: `analysis/annotation_review/pilot_manifest.json`.
- Ambiguous terminal sheets: `analysis/annotation_review/ambiguous/task_4/`.
- Ambiguous index: `analysis/annotation_review/ambiguous_manifest.json`.
- Per-state evidence traces: `analysis/semantic_state_traces/`.

## 27. Ambiguous / Needs-review Cases

Task 4 demos 1, 6, 7, 11, 12, 13, 16, 24, 28, 34, 35, 38, and 46 require
review. Full evidence and the three possible policy choices are in
`analysis/annotation_review/NEEDS_HUMAN_REVIEW.md`; exact case facts are in
`analysis/annotation_review/ambiguous_cases.csv`.

## 28. Information Passed to Task 4

No model-pipeline work was performed. If annotation resumes, downstream code must
use schema `libero_semantic_subtask_v1`, post-action completion at `c`, and next
subtask conditioning from `c + 1`. It must reject or explicitly mask unresolved
annotations rather than silently substituting a gripper proxy or `T - 1`.

## 29. Remaining Limitations

The recorded horizon does not contain enough post-release physics for several Task
4 demonstrations, and three trajectories end while the target mug remains in
bilateral finger contact. Human approval is required to exclude cases, weaken the
stability contract, or remove release from Task 4 semantics. No train/validation/
test split, current-subinstruction sequence, HOLD sequence, Mamba model, GPU job,
or training run was created.
