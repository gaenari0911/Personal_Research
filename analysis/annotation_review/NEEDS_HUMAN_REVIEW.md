# NEEDS HUMAN REVIEW — Task 3 Oracle Annotation Gate

> **Superseded by Task 3.1:** this file preserves the original Task 3 gate result.
> Task 3.1 separates causal non-terminal transitions from terminal stability QA.
> All 150 non-terminal boundaries are now conditioning-eligible; the former 13
> Task 4 blockers are classified as 9 terminal `right_censored` and 4 terminal
> `unresolved` cases and do not enter the non-terminal human-review queue. See
> `docs/AUTOMATIC_ORACLE_ANNOTATION.md` and
> `analysis/annotation_review/human_review_queue.csv`.

## Decision

**NEEDS HUMAN REVIEW.** Do not consume the current candidate boundaries as a
150-demonstration Oracle dataset. In accordance with `research.txt`, no files were
created under `annotations/libero10_semantic/` and no aggregate Oracle JSONL was
emitted.

## What passed

- The HDF5 flattened states can be restored read-only with MuJoCo 2.3.7,
  robosuite 1.4.0, and the checked-in LIBERO environment source.
- Restored state dimensions match the HDF5 state dimensions exactly.
- Official BDDL predicates are available: Task 3 `In + Close`, Task 9 `In +
  Close`, and Task 4 `On + On`.
- The required representative and recovery pilots all produce repeatable,
  ordered boundaries under the documented five-frame rule.
- A full read-only validation scan found valid candidates for Task 3 50/50,
  Task 9 50/50, and Task 4 37/50.

## Blocking finding

Task 4 has 13/50 demonstrations for which the required second subtask condition
`yellow-and-white mug on right plate AND released AND stable` cannot be verified
from the recorded horizon with one uniform rule.

- `demo_1`, `demo_34`, and `demo_35`: no second-object close-to-open release
  transition occurs. The official bilateral finger-contact grasp detector remains
  true at the terminal state. Treating official `On` alone as completion would
  silently drop the explicit release requirement.
- `demo_11`: the release command begins only at the final recorded state, after
  the official `On` predicate has regressed to false. There is no post-release
  observation in which both conditions can be verified.
- `demo_6`, `demo_7`, `demo_12`, `demo_13`, `demo_16`, `demo_24`, `demo_28`,
  `demo_38`, and `demo_46`: release and/or the final official `On` run occurs too
  close to the end, leaving only two to four verified stable post-release frames
  (or an equally short final predicate run), below the five-frame / 0.25 s rule.

The dataset's reward/done values are synthetic terminal markers and cannot supply
the missing semantic evidence. Extrapolating unrecorded future physics would not
be trajectory-based Oracle annotation.

## Evidence and reproducibility

- Per-step restored predicate/contact traces:
  `analysis/semantic_state_traces/task_3_all.jsonl`,
  `task_4_all.jsonl`, and `task_9_all.jsonl`.
- Case table: `analysis/annotation_review/ambiguous_cases.csv`.
- Terminal last-21-frame sheets:
  `analysis/annotation_review/ambiguous/task_4/`.
- Machine-readable review index:
  `analysis/annotation_review/ambiguous_manifest.json`.
- Pilot videos and ±10-frame sheets:
  `analysis/annotation_review/pilot/` and `pilot_manifest.json`.

## Human decision required

A reviewer must choose one of the following before Oracle generation resumes:

1. Keep `released + five-frame stable` as the contract and exclude/mark the 13
   demonstrations as unresolved.
2. Approve a task-specific terminal-censoring policy that accepts fewer than five
   observed frames after release, while leaving the three terminal-grasp cases
   unresolved.
3. Redefine Task 4 placement completion to the official BDDL `On` goal without an
   explicit release requirement. This changes the semantic contract requested in
   `research.txt` and is not assumed here.

Until that choice is made, downstream Task 4 interfaces, HOLD/current-subtask
sequences, Mamba implementation, training, and dataset splits must not be created.
