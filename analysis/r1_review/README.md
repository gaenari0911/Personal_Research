# R1 camera review

This directory contains deterministic visual checks for the canonical
WashFruitColander loader. No semantic stage labels were inferred or added.

Each contact sheet contains five consecutive observation frames from one
eligible episode at a representative interior timestep:

- top row: `observation.images.left_rgb` (external view)
- bottom row: `observation.images.robot0_eye_in_hand_rgb` (wrist view)
- columns: chronological order from `t - 4` through `t`

The three sheets cover the shortest, median-length, and longest eligible
episodes. Review checks were camera identity, RGB color order, upright image
orientation, chronological motion, and correspondence between the two views.
All three sheets passed those checks.

These images do **not** validate semantic predicates or subtask stages. The
acquired v2.1 archive has no per-frame `subtask_idx`, `subtask`,
`subtask_name`, or `subtask_stage` columns, which is why R1 G1/G7 remain
failed and R2 is not ready.
