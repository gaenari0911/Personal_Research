# RoboCasa365 Arm-Only Multi-Stage Feasibility Audit

Audit date: 2026-08-24  
Decision: **REJECT**  
Qualified primary tasks: **1 / 3 required**

## Executive conclusion

RoboCasa365 does not pass the requested benchmark gate. The official target registry contains 13 composite tasks with an official `num_subtasks` value of 4–6, but only `WashFruitColander` is officially marked as not requiring mobile manipulation and has no navigation label in its released annotation dictionary. Even that task is not strictly arm-only across the complete archive: 18 of 507 episodes switch controller mode and issue nonzero base commands. Filtering those episodes leaves 489 exact-base-zero, no-navigation demonstrations, so the task is `LIKELY_ARM_ONLY`, not `ARM_ONLY_CONFIRMED`.

The three most plausible alternatives do not repair the minimum-three-task gate:

- `WaffleReheat`: 432/500 exact-base-zero episodes, but navigation is an official annotation value in 2 episodes and 68 episodes use base/control.
- `HeatKebabSandwich`: no navigation label in all 503 episodes, but 365 episodes use base/control.
- `StirVegetables`: 89/501 episodes contain official navigation stages and 388 episodes use base/control.

The released data can support a filtered arm-only side study, but not the requested task-level, no-navigation, three-task primary benchmark without changing the benchmark definition.

## 1. Official repository and evidence

The official repository was cloned to `external/robocasa` without modifying it.

| Item | Value |
|---|---|
| Origin | `https://github.com/robocasa/robocasa.git` |
| Branch | `main` |
| Commit | `a07e365c958c4216cd6bbd5f30b47f09a65c6f00` |
| Version | `1.0.1` |
| Commit date | 2026-08-21 |
| Official target set | 18 atomic-seen + 16 composite-seen + 16 composite-unseen |
| Official target repetition | 500 human demonstrations per task |
| Official target composite annotation claim | subtask index, atomic-skill name, stage, and natural-language instruction at every timestep |

Primary local sources:

- `external/robocasa/robocasa/utils/dataset_registry.py`
- `external/robocasa/docs/composite_tasks/task_attributes.json`
- `external/robocasa/docs/datasets/datasets_overview.md`
- `external/robocasa/docs/datasets/using_datasets.md`
- `external/robocasa/robocasa/utils/lerobot_utils.py`
- `external/robocasa/robocasa/scripts/dataset_scripts/convert_hdf5_lerobot.py`
- task class files listed in `analysis/robocasa_target_composite_audit.csv`

The Box links shipped in the official repository were probed with byte-range requests. No complete task archive was downloaded. Low-dimensional parquet blocks and small metadata/video samples were extracted directly from uncompressed tar ranges.

## 2. Exhaustive target-composite audit

The official 4–6-subtask filter yields exactly 13 target tasks. `genuine goals` below groups official pick/place phases into one object-transfer goal and excludes navigation, passive waiting, and `done`. This grouping is an audit interpretation; the official `num_subtasks` value remains separately preserved.

| Task | Split | Official subtasks | Grouped genuine goals | Navigation | Arm verdict | Primary |
|---|---:|---:|---:|---|---|---|
| DeliverStraw | seen | 4 | 3 | REQUIRED | MOBILE_REQUIRED | no |
| GetToastedBread | seen | 4 | 2 | REQUIRED | MOBILE_REQUIRED | no |
| SteamInMicrowave | seen | 6 | 4 | REQUIRED | MOBILE_REQUIRED | no |
| StirVegetables | seen | 4 | 4 | OPTIONAL | HYBRID | no |
| StoreLeftoversInBowl | seen | 5 | 3 | REQUIRED | MOBILE_REQUIRED | no |
| ArrangeBreadBasket | unseen | 5 | 3 | REQUIRED | MOBILE_REQUIRED | no |
| GarnishPancake | unseen | 4 | 2 | REQUIRED | MOBILE_REQUIRED | no |
| GatherTableware | unseen | 4 | 4 | REQUIRED | MOBILE_REQUIRED | no |
| HeatKebabSandwich | unseen | 6 | 6 | OPTIONAL | HYBRID | no |
| MakeIceLemonade | unseen | 5 | 3 | REQUIRED | MOBILE_REQUIRED | no |
| PortionHotDogs | unseen | 4 | 4 | REQUIRED | MOBILE_REQUIRED | no |
| WaffleReheat | unseen | 4 | 4 | OPTIONAL | HYBRID | no |
| WashFruitColander | unseen | 4 | 4 canonical; episode-dependent | NO_NAVIGATION | LIKELY_ARM_ONLY | filtered only |

All 13 tasks have a task-specific archive, 500 official human demonstrations, a continuous task environment, an official full instruction, and a source `_check_success()` predicate. Their full audit rows, sequences, source files, token lengths, archive sizes, and sampled action statistics are in `analysis/robocasa_target_composite_audit.csv`.

### Known-candidate correction

`MealPrepStaging` has four official subtasks but is **pretrain-only**. It is absent from both target composite registries and cannot count toward this target benchmark gate. Its metadata also says `moma_required=Yes`.

## 3. Action schema: two orders must not be confused

The expected 12D order is correct for the environment/original HDF5 representation:

| HDF5/environment index | Meaning |
|---|---|
| `[0:3]` | end-effector translation |
| `[3:6]` | end-effector rotation |
| `[6]` | gripper close |
| `[7:11]` | mobile-base motion |
| `[11]` | control mode |

The released LeRobot parquet deliberately reorders it:

| LeRobot parquet index | Meaning |
|---|---|
| `[0:4]` | mobile-base motion |
| `[4]` | control mode |
| `[5:8]` | end-effector translation |
| `[8:11]` | end-effector rotation |
| `[11]` | gripper close |

Therefore:

- `parquet_action[:7]` is **not** a 7D arm action.
- The raw LeRobot 7D arm target is `parquet_action[5:12]`.
- The official `get_episode_actions()` helper inverse-reorders LeRobot back into HDF5/environment order; only after that inverse reorder is `action[:7]` arm xyz + rotation + gripper.

This is a required data adapter, not an action-head architecture change.

## 4. Actual action audit

Four promising tasks were audited over every low-dimensional episode. The remaining nine have an actual episode-0 sample. `base nz` uses a strict `abs(base)>1e-8` test on at least one base dimension per frame.

| Task | Episodes | Frames | Base nz frames | Episodes using base | Base-free episodes | Navigate episodes | Mean episode |
|---|---:|---:|---:|---:|---:|---:|---:|
| WashFruitColander | 507 | 518,903 | 0.1447% | 18 | 489 | 0 | 51.17 s |
| WaffleReheat | 500 | 420,489 | 0.7232% | 68 | 432 | 2 | 42.05 s |
| HeatKebabSandwich | 503 | 652,820 | 1.8388% | 365 | 138 | 0 | 64.89 s |
| StirVegetables | 501 | 409,202 | 5.2097% | 388 | 113 | 89 | 40.84 s |

The low frame fraction for `HeatKebabSandwich` is misleading: short base bursts occur in 72.6% of episodes. This is why frame-level near-zero statistics alone cannot establish task-level arm-only feasibility.

For `WashFruitColander`, the full 518,903×12 action tensor gives:

- arm 7D min: `[-1, -1, -1, -1, -1, -1, -1]`
- arm 7D max: `[1, 1, 1, 1, 1, 1, 1]`
- arm mean: `[0.04630, -0.02778, -0.01625, 0.01010, -0.00142, -0.00682, -0.22253]`
- arm std: `[0.31715, 0.46711, 0.35002, 0.09395, 0.10734, 0.13866, 0.97493]`
- base mean absolute value: `0.00021042`
- base mean norm: `0.00074065`
- base max absolute value: `1.0`
- base p95 norm: `0.0`
- control values: `{-1: 518,056 frames, +1: 847 frames}`
- control changes: `35` across `18` episodes

Full vectors, max norms, percentiles, and per-episode rows are preserved in `analysis/robocasa_arm_action_audit.json`.

### Episode-0 evidence for the other nine tasks

| Task | Base nz frames | Base mean norm | Control changes |
|---|---:|---:|---:|
| DeliverStraw | 25.21% | 0.2370 | 3 |
| GetToastedBread | 25.42% | 0.2095 | 2 |
| SteamInMicrowave | 16.97% | 0.1523 | 2 |
| StoreLeftoversInBowl | 24.94% | 0.2389 | 4 |
| ArrangeBreadBasket | 11.76% | 0.0933 | 2 |
| GarnishPancake | 30.54% | 0.2702 | 5 |
| GatherTableware | 46.86% | 0.3968 | 4 |
| MakeIceLemonade | 18.72% | 0.1387 | 8 |
| PortionHotDogs | 13.78% | 0.1009 | 6 |

These single episodes are not used to claim a population statistic; they corroborate the official `moma_required=Yes` flag and the fixture geometry in task source.

## 5. Per-frame annotation verification

Every one of the 13 episode-0 parquet files contains these actual columns:

- `annotation.human.task_description`
- `annotation.human.task_name`
- `annotation.human.subtask`
- `annotation.human.subtask_name`
- `annotation.human.subtask_stage`
- `subtask_idx`

The parquet values are integer IDs. The strings live in `lerobot/meta/tasks.jsonl`; absence of literal strings in a parquet viewer is expected.

The important semantic finding is that the official fields do not form one pure semantic-goal index:

- `stage` values observed: `pick`, `place`, `execute`, `navigate`, `done`.
- `subtask_idx` increments for each current instruction segment, including separate pick/place phases and `done`.
- The same atomic skill can span two indices, such as `PickPlaceCounterToMicrowave` with one `pick` and one `place` segment.
- Official `num_subtasks` is therefore not equal to the number of `subtask_idx` runs. Example: `WaffleReheat` is officially 4-subtask but has 6 normal runs including pick/place split and done; rare navigation gives up to 8 runs.

For a semantic-memory experiment, a deterministic adapter would have to group consecutive official phases into semantic goals while retaining the original fields. This audit did not create or overwrite annotations.

### Decoded representative timeline: WaffleReheat episode 0

| Frames | Seconds | Stage | Atomic skill | Current instruction |
|---|---:|---|---|---|
| 0–175 | 0.00–8.75 | execute | OpenMicrowave | open the microwave |
| 176–350 | 8.80–17.50 | pick | PickPlaceCounterToMicrowave | pick up the bowl with the waffle from the counter |
| 351–464 | 17.55–23.20 | place | PickPlaceCounterToMicrowave | place the bowl with the waffle in the microwave |
| 465–588 | 23.25–29.40 | execute | CloseMicrowave | close the microwave door |
| 589–674 | 29.45–33.70 | execute | TurnOnMicrowave | turn on the microwave |
| 675–690 | 33.75–34.50 | done | done | task complete |

Decoded episode-0 timelines for all four detailed candidates are in `analysis/robocasa_semantic_stage_audit.json`.

## 6. Representative candidates

### A. WashFruitColander

- Official split/subtasks: composite-unseen, 4.
- Canonical goals: put colander in sink; put fruit in colander; turn spout when needed; turn on faucet.
- Actual variation: 1–3 fruits and optional spout adjustment produce 6–11 phase-level runs including done.
- Navigation: none in the released dictionary or all 507 timelines.
- Action: 489 exact-base-zero episodes; 18 use base/control briefly.
- Long horizon: median 50.1 s; p95 70.88 s; maximum 98.4 s.
- Annotation retention interval: median 6.55 s; p95 12.1 s; maximum 20.75 s.
- Verdict: `LIKELY_ARM_ONLY`, filtered only. A fixed-base rollout across target layouts is still needed for confirmation.

### B. WaffleReheat

- Official split/subtasks: composite-unseen, 4 genuine goals.
- Navigation: optional, not absent. Two episodes have 310 `navigate` frames total.
- Action: 432 exact-base-zero episodes; 68 use base/control.
- Long horizon: median 41.33 s; p95 55.82 s; maximum 64.65 s.
- Annotation retention interval: median 7.05 s; p95 13.55 s; maximum 22.3 s.
- Visual note: the open microwave door heavily occludes the left external view; wrist view is important.
- Verdict: `HYBRID`; excluded by the task-level no-navigation gate.

### C. HeatKebabSandwich

- Official split/subtasks: composite-unseen, 6 genuine goals.
- Timeline: rack out; baguette in; kebab in; rack in; door close; timer on.
- Navigation labels: none in all 503 episodes.
- Action: only 138 exact-base-zero episodes; 365 use base/control, although the bursts are short.
- Long horizon: median 64.5 s; p95 83.22 s; maximum 107.25 s.
- Annotation retention interval: median 6.75 s; p95 15.34 s; maximum 33.5 s.
- Verdict: `HYBRID`; source geometry and the majority of demonstrations do not justify fixed-base rollout.

### D. StirVegetables

- Official split/subtasks: composite-seen, 4 grouped goals.
- Navigation: explicit `NavigateKitchen` in 89 episodes and 17,499 frames.
- Action: 113 exact-base-zero episodes; 388 use base/control.
- Long horizon: median 38.95 s; p95 59.15 s; maximum 86.4 s.
- Annotation retention interval: median 4.95 s; p95 12.72 s; maximum 37.7 s during stirring.
- Verdict: `HYBRID`; excluded.

## 7. Transition naturalness and human review burden

One official left-view video was byte-range extracted for each detailed candidate, and a 21-frame contact sheet was generated under `analysis/robocasa_review/`.

Visual inspection found continuous episodes without stage resets:

- `WashFruitColander`: colander transfer, successive fruit transfers, and faucet operation form one continuous sink workflow.
- `WaffleReheat`: door opening, bowl insertion, door closing, and button actuation are continuous; the door causes temporary camera occlusion.
- `StirVegetables`: two ingredient transfers, spatula retrieval, and stirring are continuous.
- `HeatKebabSandwich`: rack manipulation, two placements, door close, and timer actuation are continuous.

The contact sheets establish continuity and gross sequence only. Boundary review should use the MP4 around the annotation change frames, not a 21-frame montage. A reviewer should check:

1. whether the previous goal is visibly complete at the labeled boundary;
2. whether object contact/fixture state agrees with `pick`, `place`, or `execute`;
3. whether any base burst is a true navigation segment or an unlabeled reach adjustment;
4. whether `done` begins only after success;
5. whether door/rack occlusion makes the external view insufficient.

All audited episodes have exactly one `next.done=True` at the final row and at least one positive reward, but annotation-boundary generation source is not included in the repository, so exact boundary provenance cannot be reconstructed from code alone.

## 8. Full instruction and CLIP limit

The exact MaIL tokenizer, Hugging Face `openai/clip-vit-base-patch32`, was loaded from its official tokenizer files with special tokens enabled. The model limit is 77 tokens.

- All 13 canonical task instructions are below the limit: 15–41 tokens.
- `WashFruitColander` actual 157 variants: 32–36 tokens.
- `StirVegetables` actual 238 variants: 26–28 tokens.
- `WaffleReheat`: 25 tokens.
- `HeatKebabSandwich`: 32 tokens.

No truncation is required for the audited full instructions.

## 9. Observation/action timing

The official source confirms a nontrivial alignment convention:

1. `collect_demos.py` says states are recorded **after** playing the action and removes the final extra state.
2. `convert_hdf5_lerobot.py` reloads `states[t]`, renders observations, and stores `actions[t]` in the same LeRobot row.
3. Reward is explicitly computed at the resulting state `s'`.

Thus row `t` pairs an action with its post-action image/state, not an unambiguous pre-action observation for that same action. For behavior cloning, the safest adapter candidate is `obs[t] -> action[t+1]` with the final row dropped, followed by replay validation. The annotation stream must be shifted consistently. This is a data-interface correction and must be identical for all methods.

The actual timestamps are 20 Hz with a mean delta of approximately 0.0500000 s. MaIL's observation window 5 spans 0.25 s; action horizon 10 spans 0.50 s. The audited episode medians are roughly 39–65 s, so persistent-state evaluation remains physically meaningful.

## 10. Storage and download policy

The 13 official 4–6-subtask archives sum to 26,415,523,840 bytes (26.42 GB decimal), so the requested 10 GB stop rule prohibited an exhaustive archive download.

RoboCasa supports task-specific and split-specific download. Archive sizes for a hypothetical three-task set `WashFruitColander + WaffleReheat + HeatKebabSandwich` total 6.08 GB, but no selected-task migration download was performed because the gate is REJECT.

Existing data was preserved:

- `/ssd1/itaein/datasets/LIBERO/libero_10`: present, not deleted.
- `/ssd1/itaein/datasets/CALVIN/debug`: present, not deleted.

## 11. Final gate

| Requirement | Result |
|---|---|
| At least 3 tasks | **FAIL: 1 filtered primary candidate** |
| Each 4–6 genuine semantic goals | FAIL across the no-navigation pool |
| Continuous episode | pass |
| Repeated demonstrations | pass |
| Official per-frame annotation | pass, but mixed phase/semantic granularity |
| No navigation | fail |
| Arm-only/effectively 7D | fail at task level |
| No control-mode complication | fail |
| Preserve MaIL 7D head | pass with action-order adapter |
| Fair common two-view setup | pass |
| Official full instruction | pass |
| Rollout success predicate | pass |
| Alignment understood | pass, adapter still required |

**FINAL ROBOCASA DECISION: REJECT.**

Do not migrate, do not download the selected archives, and do not delete LIBERO or CALVIN debug data.
