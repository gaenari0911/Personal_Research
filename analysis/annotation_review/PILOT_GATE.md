# Task 3 Pilot Gate

Pilot decision: **PASS**

Final corpus decision: **NEEDS HUMAN REVIEW**. The pilot permitted a full-corpus
read-only validation scan, but that scan found 13 Task 4 trajectories without a
repeatable five-frame released-and-stable terminal completion. See
`analysis/annotation_review/NEEDS_HUMAN_REVIEW.md`. No 150-demo Oracle annotation
set was emitted.

- Alignment: every HDF5 `states[t]` is restored as the post-action `obs[t]`; therefore a completion observed at `c` is assigned to `completion_obs_index = completion_action_index = c`, and the next subtask begins at `c + 1`.
- Official semantic evidence: Task 3 uses LIBERO `In` and `Close`; Task 9 uses `In` and `Close`; Task 4 uses two LIBERO `On` predicates.
- Release evidence: robosuite bilateral finger-pad contact is primary. If that contact detector never activates, gripper opening, target relation, and a five-frame object-stability test are combined; action sign alone is never sufficient.
- Stability: the target predicate, released state, and per-frame object motion no greater than 0.01 m must hold for five consecutive frames (0.25 s at 20 Hz). Terminal articulation predicates must also persist for five frames.
- Recovery: extra close/open command cycles are treated as failed-attempt/regrasp recovery; transient predicate truth followed by false is not accepted, and only the final stable run supplies the boundary.
- Representative outcomes: Task 3 demos 2/20/32 produced ordered boundaries and recovery demo 34 rejected the first failed attempt. Task 9 demos 29/9/36 generalized and recovery demo 3 rejected two preliminary grasp cycles. Task 4 demos 19/17/33 generalized across two object identities; demo 33 rejected the failed second-object attempt and transient `On` flicker.
- Visual review: predicate-aware videos and boundary ±10-frame sheets are under `analysis/annotation_review/pilot/`, indexed by `pilot_manifest.json`.
- Ambiguity assessment: object/target identity is fixed by BDDL names, official predicates are available, state dimensions match after XML restoration, and no pilot requires an arbitrary per-demo exception. Two Task 4 pilots receive medium rather than high confidence because the bilateral-contact grasp detector never activates for the first mug; their release remains supported by gripper opening plus stable official `On`.

The pilot gate permitted the validation scan; it did not survive the full-corpus QA.
