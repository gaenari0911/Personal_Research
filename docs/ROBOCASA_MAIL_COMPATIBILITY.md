# RoboCasa365 ↔ MaIL Compatibility Audit

Audit date: 2026-08-24  
Compatibility if an eligible arm-only task suite existed: **MOSTLY**  
Benchmark migration decision: **NOT AUTHORIZED (RoboCasa gate REJECT)**

## 1. Compatibility summary

RoboCasa can be adapted to MaIL without replacing MaIL's action head, visual-encoder philosophy, language encoder, Mamba core, or MSE objective. The incompatibility is primarily benchmark/task selection, not tensor plumbing: fewer than three target tasks satisfy the requested no-navigation and effectively-7D gate.

The required adapter is still nontrivial and must be treated as common infrastructure shared by every experimental method.

| Component | MaIL baseline | RoboCasa released data | Compatibility |
|---|---|---|---|
| Action head | 7D xyz + rotation + gripper | 12D hybrid action | compatible only after correct reorder/filter |
| Observation views | external agent view + eye-in-hand | left, right, eye-in-hand | compatible with left + eye-in-hand |
| Image size | 128×128 | 256×256 | deterministic resize |
| FPS | benchmark-dependent | 20 Hz | compatible; physical horizons must be reported |
| Language | frozen/precomputed CLIP ViT-B/32 512D | official full instruction | compatible, all audited instructions <77 tokens |
| Temporal input | obs sequence 5 | continuous 20 Hz trajectories | compatible |
| Action horizon | 10 in the locked protocol | future 7D actions obtainable | compatible after alignment fix |
| Backbone | decoder-only Mamba, nominal 128D/16 layers | dataset-agnostic | preserve |
| Objective | mean MSE on normalized actions | continuous Cartesian actions | preserve |
| Semantic conditioning | requires time-aligned context adapter | official per-frame IDs/strings | usable after deterministic semantic grouping |

## 2. Action adapter

### Original environment/HDF5 order

```text
[0:3]   end-effector translation
[3:6]   end-effector rotation
[6]     gripper
[7:11]  mobile base
[11]    control mode
```

### Released LeRobot order

```text
[0:4]   mobile base
[4]     control mode
[5:8]   end-effector translation
[8:11]  end-effector rotation
[11]    gripper
```

The MaIL target from raw parquet must be:

```python
mail_action_7d = parquet_action[..., 5:12]
```

Alternatively, use RoboCasa's official inverse reorder to HDF5 order and then select `[..., :7]`. Selecting raw parquet `[..., :7]` is an invalid mixture of base, control mode, and two arm dimensions.

The adapter must also reject or explicitly filter any episode containing:

- nonzero `base_motion`;
- a control-mode change or `control_mode=+1`;
- a `navigate` stage;
- a task/source configuration requiring a new base pose for reachability.

Filtering demonstrates data feasibility but does not prove fixed-base rollout feasibility. `WashFruitColander` therefore remains `LIKELY_ARM_ONLY` until target-layout rollouts succeed with base fixed.

## 3. Observation adapter

Use two views to preserve MaIL identity:

```text
agentview_rgb    <- observation.images.robot0_agentview_left
eye_in_hand_rgb  <- observation.images.robot0_eye_in_hand
```

Both RoboCasa views are 256×256 H.264/YUV420P at 20 Hz. Decode, convert to RGB, and deterministically resize to 128×128 before the existing independent visual encoders.

Do not add `robot0_agentview_right` to the primary comparison. A third encoder/view increases observation information and parameter/compute capacity, so it would no longer be the same MaIL visual setup. The right view can be retained for human review or a separately declared camera ablation.

The wrist view is important: the external left view is heavily occluded by the microwave door in `WaffleReheat`, and small rack/object contacts are difficult to see in `HeatKebabSandwich`.

## 4. Language adapter

The exact MaIL tokenizer `openai/clip-vit-base-patch32` has a 77-token limit including special tokens. Actual/canonical audited instructions occupy 15–41 tokens, so the official full instruction fits without truncation.

Preserve MaIL's language philosophy:

- frozen CLIP text representation, 512D;
- one shared tokenizer/encoder for M0–M3;
- identical full instruction for all methods where the protocol requires it;
- no learned task-ID replacement.

For M2 and M3, current subinstructions must be derived from the official annotation dictionary. Do not substitute hand-written labels.

## 5. Semantic event adapter

The released columns are genuinely frame-aligned, but their granularity is mixed:

```text
subtask_idx
annotation.human.subtask          # current natural-language phase
annotation.human.subtask_name     # atomic skill
annotation.human.subtask_stage    # pick/place/execute/navigate/done
```

`subtask_idx` cannot directly serve as a genuine semantic-goal index because a single transfer is often split into separate pick and place segments. A common deterministic event adapter would need to:

1. decode integer IDs through `meta/tasks.jsonl`;
2. preserve every official raw field;
3. merge phase runs that belong to one semantic object-transfer goal;
4. exclude `done` and navigation from genuine manipulation-stage counts;
5. emit the same boundary series to M1, M2, and M3 masking/evaluation code;
6. record both raw and grouped boundaries for audit.

This is more than a tokenizer change but less than a policy architecture change. Because the benchmark gate is already REJECT, no production semantic adapter was implemented here.

## 6. Observation/action alignment

RoboCasa collection and conversion source indicates that stored state/image row `t` is post-action for `action[t]`. Directly training `obs[t] -> action[t]` would use an observation containing the effect of its target action.

A prospective adapter must choose and validate one convention, most plausibly:

```text
input:  stored observation/state at row t
target: arm action from row t+1
drop:   final row
shift:  semantic context consistently with the chosen control-time meaning
```

The initial pre-action observation is not clearly preserved by the converted rows, so this cannot be dismissed as a naming issue. Validate with official simulator replay before any training.

The reward/done convention is internally consistent with post-action state: all audited episodes have exactly one final `done` and a positive success reward.

## 7. Preserve MaIL identity

If a valid task suite were available, the following should remain unchanged:

- 7D Cartesian action head;
- two independent RGB encoders and their fusion philosophy;
- CLIP ViT-B/32 language representation;
- decoder-only Mamba core;
- nominal `d_model=128` and published 16-layer setting once locked;
- observation sequence length 5;
- action horizon 10;
- mean MSE over normalized 7D actions;
- shared normalization, optimizer, training budget, and rollout controller across M0–M3.

Allowed common data/interface changes:

- LeRobot key mapping and action reorder;
- fixed episode filter derived from official base/control/navigation fields;
- deterministic 256→128 resize;
- alignment shift established before training;
- deterministic grouping of official phase labels into semantic events;
- padding/masks for full continuous episodes.

Not allowed as a quiet compatibility fix:

- changing the action head to 12D;
- predicting base/control only for some methods;
- adding a third camera to one method;
- replacing CLIP with task IDs;
- adding a navigation policy or hierarchical planner;
- authoring new semantic annotations;
- changing Mamba architecture per method.

## 8. M0–M3 conceptual timelines

No embeddings or policies were implemented. Under the existing protocol, a valid grouped semantic stream would be consumed as follows:

- **M0 — MaIL Fixed Window:** full official task instruction; five-frame finite window; 10-action head.
- **M1 — Stateful Vanilla:** same visual/action/language adapters; persistent Mamba state; no current-stage signal.
- **M2 — Stateful Current-Subinstruction:** current grouped official subinstruction at every control step.
- **M3 — Stateful HOLD/Transition:** first grouped subinstruction at episode start, new one only at an audited semantic boundary, otherwise HOLD.

All methods must use the same action-order adapter, alignment shift, camera mapping, episode filter, normalization, and horizon mask. Otherwise differences would not isolate semantic-progress memory.

## 9. Physical horizons and retention suitability

RoboCasa is 20 Hz:

- observation window 5 = 0.25 s;
- prediction horizon 10 = 0.50 s;
- `WashFruitColander` median episode = 50.1 s;
- `WaffleReheat` median episode = 41.33 s;
- `HeatKebabSandwich` median episode = 64.5 s;
- `StirVegetables` median episode = 38.95 s.

Their official annotation runs have median durations around 4.95–7.05 s and maxima of 20.75–37.7 s. These are long relative to a five-frame window and would be suitable for persistent semantic-memory measurement if the action/navigation gate passed.

## 10. Final compatibility verdict

Tensor and architecture compatibility is **MOSTLY**: MaIL's 7D head, two-view encoder, CLIP language path, Mamba core, horizon, and MSE can be preserved with shared data adapters.

Benchmark compatibility is **NO** for the requested experiment: the official target set does not provide three task-level `NO_NAVIGATION` and effectively arm-only 4–6-stage candidates. Migration, training, and selected-task archive download must not proceed under the current gate.
