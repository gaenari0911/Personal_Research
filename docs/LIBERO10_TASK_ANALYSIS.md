# Task 2 — LIBERO-10 Task and Temporal-Horizon Analysis

This document records Task 2 only. It inventories the official LIBERO-10 tasks, screens them for Phase-1 suitability, and evaluates whether MaIL's finite execution context can represent event-driven subtask transitions fairly. No dataset was downloaded, no source or configuration was changed, no annotation was created, and no model was trained or executed.

## 1. Goal of Task 2

The Phase-1 question is whether ordered sub-instructions and explicit transition/`[HOLD]` inputs improve Mamba temporal representations and action prediction. Task 1 established that released MaIL does not maintain a trajectory-persistent Mamba state: it uses a five-observation sliding history plus learned action-query tokens and executes an action chunk.

Task 2 therefore has two linked goals:

1. Identify LIBERO-10 tasks with clear, annotatable, ordered semantic subgoals.
2. Determine which temporal execution scheme can test HOLD/Transition fairly when semantic stages may exceed MaIL's five-timestep observation context.

All semantic stages below are preliminary hypotheses derived from official language, initial-state, and goal predicates. They are not Oracle annotations.

## 2. LIBERO-10 Task Inventory

The exact order comes from `external/LIBERO/libero/libero/benchmark/libero_suite_task_map.py`, selected by the default task order `[0,1,...,9]` in `benchmark/__init__.py`. Natural language and symbolic goals were cross-checked against each BDDL file under `libero/libero/bddl_files/libero_10/`.

| ID | Official task name / instruction | Relevant objects | Target | Preliminary high-level stages | BDDL source |
|---:|---|---|---|---|---|
| 0 | `LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket` — “put both the alphabet soup and the tomato sauce in the basket” | alphabet soup, tomato sauce; multiple distractor groceries | basket contain region | place soup in basket → place tomato sauce in basket, with order to be verified from demonstrations | `LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket.bddl` |
| 1 | `LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket` — “put both the cream cheese box and the butter in the basket” | cream cheese, butter; multiple distractor groceries | basket contain region | place cream cheese in basket → place butter in basket, order TBD | `LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket.bddl` |
| 2 | `KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it` — “turn on the stove and put the moka pot on it” | stove, moka pot; frying-pan distractor | stove cook region and stove switch state | turn on stove → place moka pot on stove; BDDL is a conjunction and does not itself enforce execution order | `KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it.bddl` |
| 3 | `KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it` — “put the black bowl in the bottom drawer of the cabinet and close it” | black bowl, cabinet; wine bottle/rack distractors | initially open bottom drawer | place bowl inside open bottom drawer → close bottom drawer | `KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it.bddl` |
| 4 | `LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate` — “put the white mug on the left plate and put the yellow and white mug on the right plate” | white mug, yellow-white mug; red mug distractor; two plates | white mug→left plate, yellow-white mug→right plate | place white mug on left plate → place yellow-white mug on right plate, order TBD | `LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate.bddl` |
| 5 | `STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy` — “pick up the book and place it in the back compartment of the caddy” | book, caddy; mug distractor | caddy back compartment | reach/grasp book → transport → place in back compartment | `STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy.bddl` |
| 6 | `LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate` — “put the white mug on the plate and put the chocolate pudding to the right of the plate” | white mug, pudding, plate; red mug distractor | plate and table region right of plate | place white mug on plate → place pudding right of plate, order TBD | `LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate.bddl` |
| 7 | `LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket` — “put both the alphabet soup and the cream cheese box in the basket” | soup, cream cheese; tomato/ketchup distractors | basket contain region | place soup in basket → place cream cheese in basket, order TBD | `LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket.bddl` |
| 8 | `KITCHEN_SCENE8_put_both_moka_pots_on_the_stove` — “put both moka pots on the stove” | two visually similar moka pots, stove | common stove cook region; stove initially on | place one moka pot → place the other moka pot; demonstrated identity/order TBD | `KITCHEN_SCENE8_put_both_moka_pots_on_the_stove.bddl` |
| 9 | `KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it` — “put the yellow and white mug in the microwave and close it” | yellow-white mug, microwave; white mug distractor | initially open microwave heating region | place mug inside microwave → close microwave | `KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it.bddl` |

The BDDL goals specify final logical conjunctions, not demonstration ordering or frame-level completion events. Except where a close operation logically must follow object insertion, stage order must be verified from demonstrations.

## 3. Dataset Availability

**LIBERO-10 demonstrations: NOT AVAILABLE — DATASET REQUIRED.**

Controlled read-only checks covered:

- MaIL's configured author paths:
  - `/hkfs/work/workspace/scratch/ll6323-david_dataset/data/libero`
  - `/home/temp_store/wang`
- The research root and external clones.
- User-home/project paths to depth five for names containing `libero` or `dataset`.
- Relevant environment variables and home symlinks.
- Clearly mounted shared NFS roots `/data1` through `/data7`, searched to depth five with a 45-second bound.

No LIBERO HDF5 file or dataset directory was found. The NFS search timed out without a match; this is not proof that no inaccessible or more deeply nested cluster copy exists. `locate` was unavailable because its database could not be read. No unbounded filesystem search was performed.

- Dataset path: **NOT AVAILABLE**
- Dataset total size: **NOT AVAILABLE**
- Usable for metadata/frame inspection: **No**
- Download: deliberately not attempted because `/home` is 99% used and Task 2 prohibits an unsolicited large download.

The 742 MB `external/LIBERO` checkout contains benchmark code/assets/BDDL, not demonstration HDF5 data.

## 4. Dataset Structure

### Verified from MaIL and official LIBERO code

MaIL expects one robomimic-style `.hdf5` file per task and reads each demonstration's:

- `num_samples`
- `actions`
- `obs/agentview_rgb`
- `obs/eye_in_hand_rgb`

The configured action dimension is 7. Official LIBERO uses the `OSC_POSE` controller, a default simulation control frequency of 20 Hz, and a common benchmark demonstration-path convention for every suite.

### Not verified without demonstrations

**NOT AVAILABLE — DATASET REQUIRED:**

- Actual LIBERO-10 HDF5 keys and metadata on this system
- Actual action shapes and controller `env_args`
- Whether stored samples occur at every 20 Hz control step or were downsampled
- Demonstration/filter counts
- RGB frame counts and exact stored resolutions
- Dataset size and schema equality across all ten files

The simulator's 20 Hz default must not be reported as the demonstrated dataset sampling rate until HDF5 metadata or collection code/data confirms it.

## 5. Per-task Statistics

No trajectory-length or demonstration-count value is inferred from task names, BDDL files, simulator horizon, MaIL's `max_len_data=520`, or `num_data=10`. The latter is only MaIL's cap on demos loaded per file, not the dataset count.

| ID | Task shorthand | Demonstrations | Min | Mean | Median | Max | Std. dev. | Action dim | Frame count |
|---:|---|---|---|---|---|---|---|---:|---|
| 0 | soup + tomato → basket | DATASET REQUIRED | — | — | — | — | — | expected 7; unverified from data | DATASET REQUIRED |
| 1 | cream cheese + butter → basket | DATASET REQUIRED | — | — | — | — | — | expected 7; unverified from data | DATASET REQUIRED |
| 2 | stove on + moka pot → stove | DATASET REQUIRED | — | — | — | — | — | expected 7; unverified from data | DATASET REQUIRED |
| 3 | bowl → drawer + close | DATASET REQUIRED | — | — | — | — | — | expected 7; unverified from data | DATASET REQUIRED |
| 4 | two mugs → two plates | DATASET REQUIRED | — | — | — | — | — | expected 7; unverified from data | DATASET REQUIRED |
| 5 | book → caddy compartment | DATASET REQUIRED | — | — | — | — | — | expected 7; unverified from data | DATASET REQUIRED |
| 6 | mug → plate + pudding → right | DATASET REQUIRED | — | — | — | — | — | expected 7; unverified from data | DATASET REQUIRED |
| 7 | soup + cream cheese → basket | DATASET REQUIRED | — | — | — | — | — | expected 7; unverified from data | DATASET REQUIRED |
| 8 | two moka pots → stove | DATASET REQUIRED | — | — | — | — | — | expected 7; unverified from data | DATASET REQUIRED |
| 9 | mug → microwave + close | DATASET REQUIRED | — | — | — | — | — | expected 7; unverified from data | DATASET REQUIRED |

## 6. Long-Horizon Suitability Criteria

Ratings below concern task definitions, not observed trajectory quality. Evaluation feasibility is uniformly “medium”: MaIL can isolate training through a one-file/subset data directory, but released rollout code ignores `task_id` and evaluates all 10 tasks until a small task-selection interface is added.

| ID | Ordered multi-stage behavior | Stage ambiguity / memory value | Repeated or similar actions | Annotation feasibility | Preliminary suitability |
|---:|---|---|---|---|---|
| 0 | High: two object placements | High with many distractors and completed-object state | High: two pick/place cycles to same target | Medium-high; object entry events should be visible, order TBD | High |
| 1 | High: two object placements | High with many distractors | High | Medium-high; order TBD | High |
| 2 | High if demonstrations follow instruction order | Medium-high: switch completion must be remembered during pot manipulation | Low repetition but strong action-mode change | Medium; switch boundary may need state/contact rule | Medium-high, data-dependent |
| 3 | High and causally ordered: insert before close | High: after insertion, policy must switch from object to drawer | Low repetition, strong mode change | High; `In` then `Close` predicates are conceptually clear | Very high |
| 4 | High: two identity-target bindings | Very high: similar mug/plate manipulation with left/right role binding | Very high | High if placement completion is visually clear; order TBD | Very high |
| 5 | Primarily one manipulation goal | Low-medium; task phase often visible from book pose | Low | High, but decomposition risks becoming motor primitives rather than semantic subtasks | Low for the core hypothesis |
| 6 | High: two distinct object/location goals | High: after first placement the next object/target must be selected | High at pick/place level, distinct targets | High; goal predicates are distinct, order TBD | Very high |
| 7 | High: two object placements | High with distractors | High | Medium-high; order TBD | High |
| 8 | High: two placements | High: objects are same type and target is shared | Very high | Medium: object identity and demonstrated ordering may be inconsistent | High but annotation-risky |
| 9 | High and causally ordered: insert before close | High: must retain insertion completion then manipulate door | Low repetition, strong mode change | High; `In` then `Close` are clear | Very high |

## 7. Candidate Task Ranking

### 1. Task 4 — two mugs to distinct plates

- **Why suitable:** combines two explicit semantic goals, repeated pick/place motions, distractor mug, and identity-to-left/right target binding. Progress memory can distinguish which nearly similar manipulation remains.
- **Potential stages:** place white mug on left plate → place yellow-white mug on right plate; lower-level reach/grasp/transport phases should remain within these semantic subtasks unless Task 3 evidence demands otherwise.
- **Potential ambiguity:** BDDL does not enforce which placement is demonstrated first; finished versus pending object identity may be crucial.
- **Annotation difficulty:** low-to-medium once videos are available; placement events should be observable, order consistency is unknown.
- **Known statistics:** none.
- **Unknown:** demonstrations, durations, failure/retry behavior, actual order distribution.

### 2. Task 3 — black bowl into bottom drawer, then close

- **Why suitable:** the insertion and closure are distinct, causally ordered semantic stages with different action regimes and clear symbolic predicates.
- **Potential stages:** place bowl into already open bottom drawer → close drawer.
- **Potential ambiguity:** after the bowl is no longer salient, progress memory may help determine that closure is next.
- **Annotation difficulty:** low in principle; exact “in” completion and onset of closing still need an operational rule.
- **Known statistics:** none.
- **Unknown:** durations, retries, whether demonstrations insert/release/regrasp the drawer consistently.

### 3. Task 9 — mug into microwave, then close

- **Why suitable:** closely parallels Task 3 in a different articulated-object setting, enabling a clear insertion→closure transition.
- **Potential stages:** place yellow-white mug in open microwave → close microwave.
- **Potential ambiguity:** the policy must change from mug manipulation to door manipulation after insertion.
- **Annotation difficulty:** low-to-medium; microwave-door motion and mug containment should be visible.
- **Known statistics:** none.
- **Unknown:** durations, exact door interaction, demonstration consistency.

### 4. Task 6 — mug on plate, pudding right of plate

- **Why suitable:** two distinguishable objects and target relations, repeated manipulation structure, and an explicit change of target semantics (`On` versus relative `right-of`).
- **Potential stages:** place white mug on plate → place pudding in the defined right-side table region.
- **Potential ambiguity:** BDDL final goals do not enforce order; relative placement completion may be less visually crisp than containment.
- **Annotation difficulty:** medium.
- **Known statistics:** none.
- **Unknown:** order consistency and spatial tolerance in successful demonstrations.

### 5. Task 0 — soup and tomato sauce into basket

- **Why suitable:** repeated object-to-common-receptacle operations among many distractors make completed-object identity and progress relevant.
- **Potential stages:** place soup → place tomato sauce, or the demonstrated reverse order.
- **Potential ambiguity:** strong visual/distractor ambiguity; both actions share the same target and motor pattern.
- **Annotation difficulty:** medium; basket-entry events are clear, but object order and any recovery behavior are unknown.
- **Known statistics:** none.
- **Unknown:** all trajectory statistics and order consistency.

Tasks 1 and 7 are close alternatives to Task 0. Task 8 is scientifically interesting for repeated, visually similar goals but is held below the top group until demonstrations establish consistent object identity/order. Task 2 is a useful action-mode-change alternative but its BDDL conjunction alone does not prove the stove-first demonstration order. Task 5 is a reasonable simple sanity task, not a strong primary test of trajectory-specific semantic progress.

## 8. Representative Trajectory Inspection

**NOT AVAILABLE — DATASET REQUIRED.**

No representative trajectory, contact sheet, or sampled frame sequence was inspected because no LIBERO-10 HDF5 demonstration was found. Consequently:

- No observed action ordering is reported.
- No retries, pauses, or recovery phases are characterized.
- No visual boundary feasibility claim is treated as confirmed.
- No duration in frames/timesteps is estimated.

When data becomes available, the Task-2 evidence gap should be closed before Task 3 by sampling approximately 12–20 evenly spaced frames from 2–3 demonstrations for each top candidate. This is an inspection requirement, not an annotation performed here.

## 9. Preliminary Semantic Stages

These coarse stages are grounded in language and BDDL predicates, not frame inspection. They intentionally avoid assigning boundaries.

### Task 4 — two mugs to distinct plates

- S1 candidate: manipulate the white mug until it is on the left plate.
- S2 candidate: manipulate the yellow-white mug until it is on the right plate.
- Open issue: swap S1/S2 if demonstrations consistently use the reverse order; if order varies, per-trajectory decomposition must follow the demonstrated order rather than impose one globally.

### Task 3 — bowl into drawer and close

- S1 candidate: manipulate the bowl until it is inside the initially open bottom drawer.
- S2 candidate: manipulate the drawer until it is closed.

### Task 9 — mug into microwave and close

- S1 candidate: manipulate the yellow-white mug until it is inside the initially open microwave.
- S2 candidate: manipulate the microwave door until it is closed.

### Task 6 — mug and pudding placements

- S1 candidate: place the white mug on the plate.
- S2 candidate: place the chocolate pudding to the right of the plate.
- Open issue: demonstrated order is not specified by the conjunctive goal.

### Task 0 — two groceries into basket

- S1 candidate: place the first demonstrated target grocery into the basket.
- S2 candidate: place the second target grocery into the basket.
- Open issue: map S1/S2 to named objects per trajectory only after order inspection.

Reach, grasp, transport, and release are possible motor phases inside each semantic subtask, not automatically separate language subtasks. Task 3 must lock the semantic granularity before annotation.

## 10. MaIL Temporal Context Review

For the inspected LIBERO-10 decoder-only BC configuration:

- Observation context: `obs_seq=5`.
- Action horizon: 10 in `benchmark_libero10.yaml`.
- Training source window: `5 + 10 - 1 = 14` timesteps.
- Model input at training: five observation embeddings plus ten learned action-query embeddings; the language variant adds one fixed task prefix.
- Previous actions: absent.
- Persistent Mamba state/cache: absent.
- Training context: independently sampled fixed windows.
- Rollout context: a Python deque of at most five images, recomputed from scratch at each model call.
- Replanning: only after the current 10-action prediction chunk has been consumed.

Although Mamba is a state-space architecture, released MaIL uses it as a finite-sequence model. Its effective observable trajectory history is local and bounded by five image observations, not by Mamba's theoretical recurrent capacity.

## 11. Temporal Horizon Analysis

### Quantitative comparison

- Trajectory length / 5: **NOT AVAILABLE — DATASET REQUIRED**.
- Typical semantic-stage duration / 5: **NOT AVAILABLE — DATASET REQUIRED**.
- Transition interval / 5: **NOT AVAILABLE — DATASET REQUIRED**.
- Frequency of stages longer than 5: **NOT AVAILABLE — DATASET REQUIRED**.

No numeric duration claim can be made from task names or BDDL.

### Structural comparison

Let a semantic stage begin at transition timestep `b`, have duration `d`, and let the input contain the current and previous four context values. Its transition/subinstruction is visible for `t=b,...,b+4`. If `d>5`, the number of timesteps before the next transition at which the window can contain only `[HOLD]` values is:

```text
max(d - 5, 0)
```

The corresponding within-stage fraction is `max(d-5,0)/d`. This is an exact property of a five-step sliding window, not an estimate of LIBERO durations.

A longer fixed context `L` simply changes this quantity to `max(d-L,0)`. It preserves transition visibility only if `L` is at least the maximum relevant inter-transition duration, which is unknown without data and annotations.

### Action-chunk interaction

MaIL predicts and executes ten actions before replanning. If an Oracle semantic transition occurs after the model call but inside that ten-step execution chunk, the new context cannot affect the remaining actions in the chunk. This issue is independent of the five-step visibility problem and is especially important for Boundary Action Error. A Phase-1 temporal design must either process semantic inputs at action-step resolution or explicitly prevent action chunks from crossing Oracle transitions. The choice is deferred; no action horizon was changed in Task 2.

## 12. HOLD Visibility Problem

The problem is structural and confirmed for any semantic stage longer than five timesteps:

```text
transition at b:  [S_k]
b+1 ... b+4:      windows can still contain S_k
b+5 onward:       S_k is outside the window
current context:  [HOLD, HOLD, HOLD, HOLD, HOLD]
```

With no carried Mamba state or external task identity, `[HOLD]` at `b+5` does not specify what is being held. Current images might sometimes reveal the task stage, but relying on them collapses the proposed explicit event-memory mechanism back toward implicit visual progress inference.

How often this occurs in LIBERO-10 is **NOT AVAILABLE — DATASET REQUIRED**. The multi-object and insert-then-close BDDL goals establish multiple semantic transitions but not their duration. The risk is therefore logically real and task-frequency unknown, rather than numerically demonstrated.

## 13. Fairness Analysis

With a finite five-step context far from a transition:

```text
Vanilla:                FULL  FULL  FULL  FULL  FULL
Current Subinstruction: S2    S2    S2    S2    S2
HOLD/Transition:        HOLD  HOLD  HOLD  HOLD  HOLD
```

- **Vanilla** continuously receives global task identity but not explicit current progress.
- **Current Subinstruction** continuously receives both task identity and current semantic stage.
- **HOLD/Transition** loses explicit identity after the transition event exits the window.

This does not isolate “repeated semantic conditioning versus remembered event conditioning.” It instead compares an always-observable current stage with an unavailable event. Method C can be disadvantaged by an execution artifact rather than by a failure of the proposed memory representation.

Increasing only Method C's window or giving only Method C an external current-task variable would also be unfair. The temporal execution scheme, observation history, action-step cadence, Mamba architecture, and state-reset protocol must be identical across A/B/C. Only semantic input values may differ.

For a stateful comparison, all methods should carry the same Mamba state. Method A may receive full instruction inputs under its defined schedule, Method B current-subinstruction inputs at every action step, and Method C transition events/`[HOLD]`; each sees identical observations and resets at the same episode boundaries.

## 14. Temporal Architecture Options

### Option A — Keep released finite window

**Advantages**

- Smallest change and closest to released MaIL.
- Fixed-window batching and action-query code remain intact.
- Lowest memory and implementation risk.

**Disadvantages**

- Transition identity is guaranteed to disappear after five steps.
- HOLD semantics are only valid for stages no longer than the window.
- Method B retains current-stage information while Method C may see only HOLD.
- Ten-action open-loop chunks can cross semantic boundaries without an update.

**Assessment:** unsuitable for the primary HOLD-memory claim unless data later proves every relevant stage fits inside five steps, which is currently unverified and cannot be assumed.

### Option B — Use a longer fixed window

**Advantages**

- Preserves MaIL's batched finite-sequence structure.
- Mamba scales more favorably with sequence length than attention-based backbones.
- Could cover observed transition intervals after statistics are available.

**Disadvantages**

- A finite `L` is only a bound; any longer stage recreates HOLD-only windows.
- Suitable `L` cannot be selected without trajectory/boundary statistics.
- Visual encoding memory grows roughly with the number of frames; cached/precomputed features may be needed.
- Training cost, padding, and position-embedding limits change.
- It does not independently solve action chunks crossing boundaries.

**Assessment:** useful fallback or controlled ablation, but not a principled guarantee of event persistence.

### Option C — Stateful/persistent Mamba

**Advantages**

- Matches the intended semantics: a transition input updates state and later HOLD inputs preserve/update that state with observations.
- Event influence is not bounded by an arbitrary observation window.
- A/B/C can use one common stateful execution and reset protocol.
- Makes the linear probe claim about trajectory-specific temporal state conceptually direct.

**Disadvantages**

- Released MaIL does not use its available inference cache in the BC policy.
- Training must preserve episode order and define state detachment/truncated backpropagation rather than independently shuffling windows.
- Learned action-query tokens and ten-action chunk execution complicate cache semantics and boundary-time updates.
- Requires careful episode reset, batching of unequal trajectories, and leakage prevention.
- Modification level is moderate-to-large and baseline reproduction must be protected.

**Assessment:** best scientific match to the stated hypothesis and the only internal-memory option that preserves events for arbitrary stage lengths.

### Option D — Explicit external task state

**Advantages**

- Simple and guarantees that current subtask identity never disappears.
- Can retain much of released MaIL's finite-window execution.
- Easy to inspect and debug.

**Disadvantages**

- If only Method C receives the retained identity, comparison is unfair.
- If the identity is fed every step, Method C becomes functionally similar to Current-Subinstruction rather than event-driven HOLD.
- The claimed memory may reside in an oracle external variable rather than the learned temporal representation.
- Weakens novelty and the interpretation of probe results.

**Assessment:** useful engineering control/debug baseline, not the recommended primary realization of the research hypothesis.

## 15. Decision Table

| Option | Long-range memory | HOLD meaning preserved | MaIL modification | Fair comparison | Alignment with research idea |
|---|---|---|---|---|---|
| Finite window | No; five observations only | No after `b+4` for stages `d>5` | Minimal | Poor for B↔C when events expire | Low |
| Longer window | Bounded by chosen `L` | Only when `d≤L` | Small-to-moderate; higher visual/sequence cost | Fair if identical for A/B/C, but still duration-dependent | Medium |
| Stateful Mamba | Yes, episode-persistent in principle | Yes, through learned internal state | Moderate-to-large; ordered training/cache/reset/chunk changes | High if all methods share execution/state rules | Very high |
| External task state | Yes, but outside Mamba | Yes explicitly | Small-to-moderate | Low if privileged; comparison collapses toward B if repeated for all | Low-to-medium |

## 16. Recommended Temporal Strategy

```text
Recommended: C — Stateful / Persistent Mamba

Reason:
The HOLD/Transition claim is an event-memory claim. A five-step or any insufficient
fixed window can remove the only token that identifies the held subtask. Persistent
internal state preserves the intended distinction between repeating S_k and emitting
S_k once followed by HOLD, without granting Method C an oracle external state.

Main evidence:
Released MaIL carries no state across windows; transition visibility ends exactly
after five steps, and 10-action chunks can delay new semantic inputs. Official
LIBERO-10 definitions contain multiple semantic goals, but their durations remain
unknown because demonstrations are unavailable.

Required modification level:
Moderate-to-large. All A/B/C methods need the same ordered/stateful execution,
episode reset and state-detachment protocol, and action-step-aware handling of the
current action chunk. Mamba architecture/hyperparameters should otherwise remain
matched.

Main risk:
Changing released MaIL from shuffled fixed windows/action-query chunks to stateful
execution can introduce implementation differences or make baseline reproduction
harder. The state/cache semantics and training protocol must be validated before
attributing any result to instruction conditioning.
```

Option B should be retained as a lower-risk fallback/ablation after actual stage-duration statistics are known. Option A should not be the primary HOLD test. Option D may be used as a diagnostic upper-bound/control, not as Ours.

This recommendation is a temporal-design checkpoint, not implementation authorization.

## 17. Recommended Candidate Tasks

Provisional candidates for Task 3 review, in order:

1. **Task 4 — two mugs to distinct plates:** strongest combination of repeated actions, object/target role binding, and progress ambiguity.
2. **Task 3 — bowl into drawer then close:** clear causal insertion→closure transition and easy symbolic semantics.
3. **Task 9 — mug into microwave then close:** clear causal transition in a second articulated setting.
4. **Task 6 — mug on plate then pudding right:** distinct goals and action repetition, with order to verify.
5. **Task 0 — soup and tomato sauce into basket:** repeated same-target manipulation with distractors, with order to verify.

The first three are recommended for initial representative-trajectory inspection once data is available. Final selection must remain provisional until demonstration statistics and order consistency are observed.

## 18. Remaining Unknowns

- LIBERO-10 dataset location or acquisition plan.
- Demonstration count and trajectory-length distribution for every task.
- Stored observation/action frequency and action controller metadata.
- Actual order used for conjunctive goals.
- Typical semantic-stage duration and frequency of `d>5` stages.
- Boundary visibility, retries, pauses, and recovery behavior in demonstrations.
- Whether every trajectory within a task follows the same coarse decomposition.
- Suitable longer-window `L` for an Option-B fallback.
- Stateful Mamba cache/training design compatible with MaIL's learned action queries.
- Whether action horizon must be reduced, replanned per step, or clipped at Oracle boundaries.
- Annotation granularity and operational completion predicates, deferred to Task 3.
- Boundary metric window `K`, deferred until sampling and boundary statistics exist.

## 19. Information Required for Task 3

Task 3 must not begin until the temporal-design checkpoint is reviewed and enough demonstration access exists to validate the provisional task choices. Required inputs are:

| Information | Current status |
|---|---|
| Candidate ranking | Available: Tasks 4, 3, 9, 6, 0 |
| Exact instructions and BDDL predicates | Available |
| Provisional semantic stages | Available, not annotated |
| Demonstration availability | Missing |
| Trajectory-length statistics | Missing — dataset required |
| Action/sample frequency | Simulator default 20 Hz known; stored rate missing |
| Candidate annotation unit | Prefer action timestep/frame index because MaIL HDF5 aligns images and actions; verify one-to-one alignment from data |
| Typical stage duration | Missing — representative trajectories required |
| Relation to MaIL context | Exact formula available; empirical `d/5` missing |
| Recommended temporal execution | Stateful/persistent Mamba, pending human checkpoint |

Before annotation, inspect 2–3 demonstrations per leading candidate with sparse 12–20-frame contact sheets, confirm task order, then define completion predicates and boundary indexing. That future work belongs to Task 3 or a data-availability follow-up, not this Task 2 execution.
