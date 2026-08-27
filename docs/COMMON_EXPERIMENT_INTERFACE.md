# Common Experimental Interface (Task 4)

## 1. Purpose

Task 4 provides one common, symbolic trajectory interface for three future
stateful methods:

1. Stateful Vanilla Mamba
2. Stateful Current-Subinstruction Mamba
3. Stateful HOLD/Transition Mamba

Only the semantic input policy differs. HDF5 loading, Oracle loading,
observation/action alignment, action chunks, masks, splits, and metadata are
shared. This task implements no Mamba model, embedding, training, rollout, or
evaluation.

## 2. Provisional Oracle Status

The Task 3.1 annotations are a **Provisional Oracle**.

```text
oracle_status = provisional
human_spot_check_completed = false
approved_for_model_training = false
```

The authoritative status file is
`annotations/libero10_semantic/validation_status.json`. Task-4 inspection and
unit tests are allowed. Model training raises an explicit error unless a caller
uses the visibly named test-only override. Human approval must update the status
explicitly; Task 4 does not infer approval from an empty review queue.

## 3. Data Sources

- Dataset: `/ssd1/itaein/datasets/LIBERO/libero_10`
- Tasks: 3, 4, and 9; 50 demonstrations per task
- Annotations: `annotations/libero10_semantic/task_<id>/demo_<id>.json`
- Annotation manifest: `annotations/libero10_semantic/manifest.json`
- Split: `splits/libero10_phase1_split.json`
- Full instruction: HDF5 `data.attrs.problem_info.language_instruction`
- Subinstructions: each annotation's ordered `subtasks` array

No 13.7 GB HDF5 data is copied or duplicated. Trajectories and conditioning are
constructed on demand.

## 4. Annotation Source of Truth

`AnnotationStore.load(task_id, demo_id)` reads the corresponding JSON every time.
No demonstration boundary appears in source code, split code, or a derived
conditioning file. The loader validates:

- annotation version;
- task/demo identity;
- post-action alignment declaration;
- non-empty, ordered, generically sized `subtasks`;
- contiguous subtask IDs;
- `completion_obs_index == completion_action_index`;
- `next_subtask_action_start == completion + 1`;
- agreement between one subtask's next start and the following subtask's
  `action_start`;
- terminal `next_subtask_action_start == null`.

The mutation unit test changes a temporary copy of a boundary by +5 and confirms
that Current/HOLD transitions move by +5 without modifying interface code.

## 5. Dataset Alignment

The final LIBERO HDF5 generation loop calls:

```text
obs = env.step(action[j])
append obs
store actions[valid_index]
```

Therefore final HDF5 `obs[t]` is post-action evidence after executing
`action[t]`. It is not a causal input for predicting the same `action[t]`.

The Task 3.1 annotation convention is retained:

```text
completion observed at obs[c]
S1 actions: <= c
S2 actions: >= c + 1
```

## 6. Observation / Action Alignment Audit

### 6.1 LIBERO code evidence

`external/LIBERO/scripts/create_dataset.py`:

- initializes replay from `states[0]` at lines 150–154;
- calls `env.step(action)` before collecting observations at lines 175–177;
- checks the resulting state against `states[j+1]` at lines 179–187;
- appends RGB from that returned observation at lines 214–221;
- subsets `states` and `actions` with the same `valid_index` at lines 225–227.

This also means same-row `states[t]` precedes same-row `action[t]`, while same-row
RGB `obs[t]` follows it. The possible implication for the provisional Task 3.1
state-trace extraction is recorded under Remaining Risks.

### 6.2 Original MaIL behavior

`external/MaIL/dataset/multi_task_dataset_aug.py:198-225` returns identical
`start:end` row slices for the two image streams and actions. Then
`external/MaIL/agents/models/bc/bc_agent.py:193-197` keeps the first five images
but starts action targets at row four. With post-action images, its first target
action has already produced the newest input observation.

Task 4 does not reproduce this mismatch.

### 6.3 Selected causal policy pair

```text
policy observation index: t, t = 0 ... T-2
target action index:       t+1
semantic condition index:  t+1
```

At a boundary:

```text
input  = obs[c]
target = action[c+1]
semantic condition = S2
```

`action[0]` has no stored pre-action RGB observation and is excluded from offline
policy targets. `obs[T-1]` has no following recorded action and is excluded from
offline policy inputs. Thus a T-row demonstration has T-1 causal policy samples.

For HOLD, the first available offline policy sample receives an explicit S1
sequence-start replay because the original S1 event was associated with the
unpaired `action[0]`. `oracle_is_transition` remains distinct from the delivered
`is_transition`, and `transition_reason=policy_sequence_start_replay` records this
case. At online rollout, the true initial pre-action observation will receive S1
normally.

Machine-readable audit: `analysis/observation_action_alignment_audit.json`.

## 7. Semantic Timeline Convention

`build_action_timeline()` constructs a symbolic entry for every original action
index. It iterates arbitrary-length ordered `subtasks`; it is not limited to two
subtasks.

Each record contains:

```text
action_index
current_subtask_id
current_subinstruction
semantic_type
semantic_input
is_transition
steps_since_transition
time_since_transition_seconds
```

At any subtask `action_start`, `is_transition=true` and
`steps_since_transition=0`.

## 8. Full Instruction Source

Vanilla conditioning uses the official HDF5 language instruction in
`problem_info`. It is not reconstructed from a filename and is not independently
hard-coded in the interface.

## 9. Subinstruction Source

Current and HOLD conditioning use the canonical `instruction` value in each
annotation's `subtasks` array. Updating annotation wording or boundaries is
therefore reflected on the next load.

## 10. Vanilla Conditioning

For every raw action timestep:

```text
semantic_type = FULL
semantic_input = official full instruction
```

The action timeline is unchanged at S1→S2 except that the shared stage metadata
(`current_subtask_id`, `is_transition`, retention distance) still changes.

## 11. Current-Subinstruction Conditioning

For every action timestep, the current annotation subtask is returned:

```text
0 ... c     -> S1 instruction
c+1 ... end -> S2 instruction
```

`semantic_type=SUBTASK` at every timestep.

## 12. HOLD/Transition Conditioning

The raw action timeline is:

```text
action 0:   S1, semantic_type=SUBTASK, is_transition=true
1 ... c:    [HOLD]
action c+1: S2, semantic_type=SUBTASK, is_transition=true
c+2 ...:    [HOLD]
```

At the interface level HOLD is the symbolic string `[HOLD]`. No neural embedding
is created in Task 4. A shared learnable HOLD embedding belongs to Task 7.

## 13. Transition Indexing

The annotation defines a completion observation/action index `c` and the next
subtask `action_start=c+1`. Raw semantic timelines use these action indices
directly. Policy samples use target action membership:

```text
obs[c-1] -> action[c]   uses S1
obs[c]   -> action[c+1] uses S2
```

This rule is tested for Task 3 demo 20, Task 4 demo 17, and Task 9 demo 9.

## 14. HOLD Semantics

HOLD means only “no new semantic event.” It is not:

- a copy of the previous subinstruction;
- a state-freeze instruction;
- an instruction to skip the observation;
- a missing semantic slot.

Future stateful models must still process the current observation and update
Mamba state on every HOLD step.

## 15. Trajectory-level Split

`splits/libero10_phase1_split.json` uses seed 42 and eligibility:

```text
conditioning_eligible == true AND needs_review == false
```

Per task:

```text
40 train / 5 validation / 5 test
```

Totals are 120/15/15. Membership is at `(task_id, demo_id)` trajectory level.
Train/validation/test intersections are empty within each task and globally.
The manifest records the creation timestamp, split version, seed, eligibility
rule, annotation version, and annotation-manifest SHA-256.

The builder refuses to overwrite an existing split unless `--force` is explicit.
If human validation later excludes a trajectory, create a new version instead of
silently mutating `libero10_phase1_v1`.

## 16. Split Reproducibility

The artifact builder is:

```text
python3 tools/build_task4_interface_artifacts.py
```

It uses Python `random.Random(42)` over numerically ordered eligible demo IDs.
The checked-in manifest, rather than rerunning a random split during loading, is
the experiment source of truth.

## 17. Action Chunk Interface

For each causal policy sample `obs[t]`, targets begin at `action[t+1]` and extend
for horizon 10:

```text
action_target: [policy_length, 10, 7]
```

Targets preserve the demonstration exactly. No normalization occurs in this data
interface; the future common model adapter must derive train-only normalization
statistics and apply the same transform to all methods.

## 18. Boundary-crossing Chunks

`boundary_crossing_horizon` is true when a later subtask action start lies inside
the current 10-action target chunk. The actions are not changed. This flag makes
cross-boundary cases directly inspectable.

Across each task, an interior two-stage boundary produces nine crossing starts per
demo, or 450 chunks per task. Relative to causal policy decision steps, the rates
are:

- Task 3: 3.6337%
- Task 4: 3.4995%
- Task 9: 2.9640%

## 19. Action Mask

Two masks are returned:

- `valid_action_mask`: false only for target positions padded beyond episode end.
- `boundary_safe_action_mask`: additionally false at and after the next semantic
  transition, implementing the common boundary-safe horizon loss selected in the
  stateful protocol.

For a chunk beginning at the final S1 action `c`, only the first target is
boundary-safe; `action[c+1]` requires the S2 event. For a chunk beginning at
`c+1`, the entire in-bounds remainder is boundary-safe.

## 20. Retention Metadata

For an action in a stage beginning at `b`:

```text
steps_since_transition = action_index - b
time_since_transition_seconds = steps_since_transition / 20
```

The interface revalidates Task 3.1 retention:

```text
retention = T - S2.action_start
```

Mean/median S2 retention is:

- Task 3: 84.48 / 80.5 steps
- Task 4: 143.04 / 141.5 steps
- Task 9: 129.42 / 120 steps

## 21. Human-validation Integration

Human correction must update the per-demo annotation JSON. No interface source,
split loader, or conditioning cache contains copied boundaries. The mutation test
proves that a corrected boundary is applied at the next load.

Because the checked-in split includes all currently eligible trajectories, a
future exclusion requires a new split version. A boundary correction that leaves
eligibility unchanged does not require a split change.

## 22. Training Guard

`Phase1TrajectoryInterface.load_trajectory(..., purpose="training")` reads
`validation_status.json`. With the current status it raises
`OracleTrainingGuardError` and states that human approval is required.

`allow_provisional_for_testing=True` is an explicit test-only escape hatch used by
the guard unit test. Inspection and artifact statistics use
`purpose="inspection"` and do not bypass a training request.

## 23. Unit Tests

Run:

```text
python3 -m unittest discover -s tests -v
```

The tests cover:

- 150/150 annotation coverage and exact HDF5 mapping;
- generic three-subtask timeline generation;
- Vanilla/Current/HOLD boundary alignment;
- policy `obs[c] -> action[c+1]` alignment;
- episode-start S1 event/replay;
- terminal unresolved status not changing S2 conditioning;
- Task 3 demo 34, Task 4 demo 33, and Task 9 demo 3 recovery boundaries;
- temporary annotation mutation by +5;
- 40/5/5 split counts and leakage;
- ten-action padding, valid mask, boundary-safe mask, and crossing flag;
- two-view observation/action shapes and official full instruction;
- provisional training guard.

All 10 test methods pass in the current environment.

## 24. Interface Statistics

Artifacts:

- `analysis/conditioning_interface_stats.json`
- `analysis/conditioning_interface_stats.csv`

Raw action timeline semantic counts:

| Task | Steps | Vanilla FULL | Current SUBTASK | HOLD events | HOLD steps | Mean sparse ratio |
|---:|---:|---:|---:|---:|---:|---:|
| 3 | 12,434 | 12,434 | 12,434 | 100 | 12,334 | 0.8112% |
| 4 | 12,909 | 12,909 | 12,909 | 100 | 12,809 | 0.7802% |
| 9 | 15,232 | 15,232 | 15,232 | 100 | 15,132 | 0.6710% |

Every two-subtask demonstration has exactly two HOLD-method semantic events: S1
at episode start and S2 at the Oracle transition.

## 25. Task 5 Handoff

Primary source:

```text
src/libero_phase1/interface.py
```

`Phase1TrajectoryInterface.load_trajectory()` returns:

- `observation`: synchronized agent-view and eye-in-hand RGB, causal input rows;
- `policy_observation_index` and `target_action_index`;
- ten-step `action_target`;
- `valid_action_mask` and `boundary_safe_action_mask`;
- `boundary_crossing_horizon`;
- official `full_instruction`;
- `semantic_type` and raw `semantic_input`;
- `current_subtask_id` and `current_subinstruction`;
- transition and sequence-start metadata;
- `steps_since_transition` and seconds;
- eligibility, confidence, Oracle status, and original annotation.

Task 5 should add one shared model adapter that converts both RGB views and the
symbolic semantic slot into the protocol's common fused token. It must not create
method-specific dataset loaders. The nominal shared temporal configuration remains
the protocol's 128-D, 16-layer Stateful Mamba with a stateless ten-horizon decoder,
subject to Task-5 environment verification. No such model is implemented here.

## 26. Remaining Risks

1. **Oracle remains provisional.** Human sample review is not complete.
2. **State/RGB evidence offset risk.** Task 3.1 restored `states[t]` and rendered
   review RGB `obs[t]`. LIBERO replay code shows same-row state precedes the action
   while same-row RGB follows it. This can produce a one-row evidence discrepancy.
   Human review should flag it; any corrected JSON is consumed dynamically.
3. **No pre-action RGB for action 0.** Offline causal training necessarily omits
   `action[0]`; rollout still has an initial environment observation.
4. **Action normalization is deferred.** Task 5 must calculate it from the train
   split only and preserve identical treatment across methods.
5. **No CLIP/HOLD embeddings exist yet.** Only symbolic text/type is returned.
6. **No Mamba state API or full-sequence/streaming equivalence test exists yet.**
   Those require Task 5 model implementation and the pinned Mamba environment.
7. **Task 4 does not authorize training.** The guard remains closed.

## Task 4 Gate

The data/interface gate passes: dynamic annotation loading, all conditioning
policies, causal alignment, action chunks/masks, deterministic split, coverage,
mutation behavior, statistics, and the training guard are implemented and tested.

```text
Oracle status: PROVISIONAL
Human approval required before Task 5 training.
```
