# Task 2.5 — Stateful / Persistent Mamba Experimental Protocol

Status: **protocol decision only; no implementation, data acquisition, annotation, training, or rollout was performed.**

This document is the hand-off contract for Tasks 3–8. It distinguishes the published finite-window MaIL reference from the stateful controlled experiment and fixes the common temporal protocol before any method-specific code is written.

## 1. Purpose

The purpose of Task 2.5 is to define how a MaIL-derived Mamba policy can carry causal state across an entire demonstration while comparing three semantic-conditioning methods fairly:

1. Stateful Vanilla: full task instruction at every control step.
2. Stateful Current-Subinstruction: current subinstruction at every control step.
3. Stateful HOLD/Transition: a new subinstruction only at its transition, and HOLD otherwise.

The selected protocol prioritizes a clean test of semantic retention over exact reproduction of MaIL's published inference loop. It does not claim that using Mamba by itself is novel.

## 2. Research Question

The primary question is:

> When the temporal backbone carries a causal recurrent state through the observation trajectory, can sparse, explicit subtask-transition events maintain task progress and action quality over long gaps without repeating the current subtask label at every step?

The controlled contrast is:

- Stateful Vanilla → Stateful Current-Subinstruction: benefit of explicit decomposition/stage identity.
- Stateful Current-Subinstruction → Stateful HOLD/Transition: cost or benefit of replacing dense stage identity with sparse transition events and learned retention.

All three methods must share the temporal backbone, visual encoder, semantic projection/fusion, action decoder, loss masks, optimizer, data split, update cadence, state reset, and evaluation.

## 3. Why Finite-Window MaIL Is Insufficient

The inspected decoder-only BC baseline uses five observation frames, predicts ten actions, keeps rollout images in a bounded Python deque, and executes the predicted chunk before replanning. It never passes recurrent cache state to Mamba. After five HOLD steps, the transition subinstruction that established the current stage can leave the window entirely.

Consequently, a finite-window Method C could receive only `[HOLD, HOLD, HOLD, HOLD, HOLD]`. Failure would then conflate “Mamba cannot retain the transition” with “the transition was deleted before Mamba could see it.” Extending the window would test a larger fixed buffer, not episode-persistent state.

Published finite-window MaIL remains an optional reference result. It is not one of the three core stateful ablations.

## 4. MaIL/Mamba Stateful API Inspection

### Local MaIL contract

The inspected revision is MaIL commit `a8012a0018ce2e5e26adff3bb3336190be2595ea`.

- `external/MaIL/agents/models/oc_ddpm/mamba.py` imports `Mamba` and `Block` from `mamba_ssm.modules.mamba_simple`.
- `create_block()` supplies a unique `layer_idx` to every Mamba layer.
- `MixerModel.allocate_inference_cache()` returns a dictionary mapping every layer index to that layer's allocated cache.
- `MixerModel.forward(hidden_states, inference_params=None, cond=None)` forwards the same `inference_params` through all blocks.
- The BC wrapper never creates or passes `InferenceParams`; cache allocation exists but is not integrated into BC training or rollout.
- `Enc_only.forward()` forms observation embeddings, appends ten learned action-query embeddings, sends the whole sequence through the encoder, and applies the action head to query-position outputs.
- Its learned finite `pos_emb` is sized for the fixed window. It is not an episode-length streaming position scheme.
- MaIL's README requests `mamba-ssm==1.2.0.post1`; it is not installed in the inspected environments.

### Upstream v1.2.0 behavior

The official upstream v1.2.0 source is the closest source tag to MaIL's `1.2.0.post1` package pin and matches MaIL's private import contract:

- `Mamba.forward()` accepts `[B, L, D]` and an optional `InferenceParams`.
- At offset zero, a multi-token causal scan can initialize/update cache with the last convolution and SSM states.
- When `seqlen_offset > 0`, `forward()` calls `step()`; `step()` asserts that exactly one token is supplied.
- `InferenceParams.key_value_memory_dict[layer_idx]` holds the layer cache. `seqlen_offset` is bookkeeping that the caller must advance.
- The cache tensors are mutated in place with `copy_()` or the selective-state update kernel.
- `InferenceParams.reset()` resets offsets/length bookkeeping but does **not** visibly zero `key_value_memory_dict`; a safe episode reset must recreate the object/cache or explicitly zero every tensor.

Primary upstream sources:

- [Mamba v1.2.0 recurrent/cache implementation](https://github.com/state-spaces/mamba/blob/v1.2.0/mamba_ssm/modules/mamba_simple.py)
- [Mamba v1.2.0 `InferenceParams`](https://github.com/state-spaces/mamba/blob/v1.2.0/mamba_ssm/utils/generation.py)
- [Mamba v1.2.0 release](https://github.com/state-spaces/mamba/releases/tag/v1.2.0)

### Feasibility conclusion

Stateful inference is feasible at the Mamba-block level. Stateful BC execution is **not** available unchanged: MaIL does not own cache lifecycle, advance offsets, isolate batch slots, reset episodes, or expose a differentiable state-carry training path.

The upstream cache is explicitly an inference/generation facility and uses in-place mutation. It must not be assumed to preserve an autograd graph across training segments. Phase-1 training should use causal sequence scans; cached one-token stepping is the inference path. Exact equivalence must be tested after the pinned environment exists.

## 5. Definition of Persistent State

Persistent state is not one LSTM-like tensor. It is the collection of recurrent cache tensors for every layer of the persistent temporal encoder, plus minimal sequence bookkeeping:

```text
PersistentState = {
  layer_idx: (conv_state, ssm_state),
  ...
  seqlen_offset / valid length metadata
}
```

For Mamba v1.2.0, layer `l` has:

```text
conv_state[l]: [B, d_model * expand, d_conv]
ssm_state[l]:  [B, d_model * expand, d_state]
```

With the inspected MaIL dimensions `d_model=128`, `expand=2`, `d_conv=4`, and `d_state=16`, the expected per-layer shapes are:

```text
conv_state[l]: [B, 256, 4]
ssm_state[l]:  [B, 256, 16]
```

The published script override indicates 16 decoder-only Mamba layers, while the benchmark YAML default is 6. Therefore the state is a 16-layer collection if the published override is locked; the final layer count remains a Task-4 configuration lock, not a reason to change the state definition.

The block residual passed between layers inside one forward call is ephemeral and is not an additional cross-timestep state object.

Episode reset method: allocate a new zero-initialized state/cache object, or zero both tensors at every layer and reset all offsets. Reusing `InferenceParams.reset()` alone is insufficient unless its cache tensors are also proven zeroed.

Detach location: in a future functional training-state implementation, detach **both** `conv_state` and `ssm_state` for every layer at a TBPTT boundary. Do not detach only the exposed `h_t` while leaving the recurrent graph attached.

## 6. Candidate Stateful Execution Strategies

| Strategy | Hypothesis alignment | Memory/efficiency | Transition precision | Library fit | Decision |
|---|---|---|---|---|---|
| 1. One observation per recurrent update | Excellent | Efficient at inference; many calls | Exact control-step timing | Matches `step()` after initialization | **Selected inference unit** |
| 2. Small chunk per update | Good only with per-token semantic alignment | Better kernel use | Boundaries inside chunks need careful alignment | Multi-token offset stepping is not the cached API | Not selected for inference |
| 3. Full trajectory causal training, cached inference | Excellent | Highest visual-training memory; efficient Mamba scan | Exact per-token alignment | Uses optimized training scan and official inference cache | **Selected default training/inference pairing** |
| 4. Ordered truncated segments with carried state | Good | Bounded memory | Exact if masks/ordering are correct | Requires a functional/detachable state path not supplied by MaIL | Fallback after profiling |

The execution unit is one environment/control timestep. Training may process many such units in one causal scan for efficiency; that does not change their semantic meaning.

## 7. Training Sequence Strategy

Default Phase-1 strategy:

1. Treat one demonstration as one ordered causal sequence.
2. At sequence index `t`, construct exactly one recurrent input token from `observation_t` and `semantic_t`.
3. Run the persistent temporal Mamba over the whole valid sequence in chronological order.
4. Produce a fresh action prediction from each `h_t` without placing action-query tokens in temporal memory.
5. Apply action/validity masks near episode end and at subtask boundaries as defined in Section 13.

No timestep-level shuffle is allowed. Whole trajectories may be shuffled between optimizer batches/epochs because every trajectory begins from zero state.

If whole-trajectory training does not fit the actual GPU, switch to contiguous ordered segments. Segment `j+1` must consume the final recurrent state value from segment `j`; segment order cannot be randomly permuted.

## 8. BPTT / TBPTT Analysis

### Full-trajectory BPTT

This is the default because it directly lets a late action loss train retention of an early transition. Mamba's linear-time causal scan makes the temporal backbone more favorable than attention for long sequences, although the two-view visual encoder may still dominate activation memory.

### Truncated BPTT

TBPTT is the memory fallback. It carries state values across adjacent segments but detaches the graph every `K` valid timesteps. `K` must be selected only after measuring trajectory lengths, stage-duration/gap distributions, and GPU memory.

Use overlapping or randomized truncation boundaries across epochs if TBPTT is required, so transitions are not always placed immediately before a detach. Include a small full-BPTT diagnostic subset when feasible.

### Detached carry without cross-segment gradient

This preserves forward memory values but gives no direct credit from a late loss to a transition before the most recent detach. It is equivalent to TBPTT with the segment length as the credit horizon, not to full long-range learning.

### Sparse-HOLD limitation

If `S1` occurs at `t=10`, the decisive action loss is at `t=70`, and `K=16`, that loss cannot backpropagate directly to the transition input. Local recurrent training may still learn generally stable dynamics, but retention for 60 steps is not guaranteed. Phase-1 adds no auxiliary state loss; therefore this limitation must be reported, and full BPTT is preferred unless measured resource limits force TBPTT.

## 9. State Detach Policy

Default full-trajectory policy:

- Do not detach within a valid episode.
- Release the graph after the trajectory loss/backward pass.
- The next demonstration starts from a new zero state.

TBPTT fallback:

- Detach all per-layer convolution and SSM states together after every `K` **valid** timesteps.
- Carry their numerical values into the next adjacent segment.
- Never detach at semantic transitions; doing so would systematically weaken the signal being studied.
- Never count padding as part of `K`.
- Do not use the inference cache's in-place tensors as a differentiable cross-segment state without an explicit autograd test and functional-state design.

## 10. Episode Reset Protocol

Reset means zero/new recurrent tensors **and** zero sequence bookkeeping.

| Boundary | Rule |
|---|---|
| Environment reset/new rollout | Reset before consuming the first observation |
| Demonstration boundary | Always reset; never carry across demos, even for the same task |
| Task change | Reset because it also begins a new episode/demo |
| Ordinary optimizer batch boundary | Reset only because default batches contain complete trajectories; batch boundaries inside a TBPTT episode carry state |
| Padding start | Stop updating and stop computing loss for that slot; padding is not a semantic/state transition |
| Parallel batch slot finishes | Reset only that slot before assigning another episode |
| Train → validation/test | Destroy all train states and create fresh zero states |

At `t=0`, Method C emits the first subinstruction as an initial transition event, not HOLD.

## 11. Observation Context Protocol

Selected context: current synchronized observation only, consisting of the same two RGB views and preprocessing used by the common baseline, plus the persistent recurrent state.

```text
(agent-view_t, eye-in-hand_t) -> shared visual encoder -> visual feature_t
visual feature_t + semantic feature_t -> one 128-D temporal token x_t
```

The former five-frame deque is removed from the stateful core methods. Feeding overlapping five-frame windows into a carried cache would reinsert observations multiple times and make “one recurrent step” ambiguous. It also increases computation and couples state behavior to arbitrary window overlap.

| Context option | Baseline compatibility | Redundancy/memory | State meaning | Decision |
|---|---|---|---|---|
| Current observation + persistent state | Removes the deque but keeps the same current sensor modalities | Lowest; every frame enters once | One update equals one control step | **Selected** |
| Last five observations + persistent state | Superficially retains `obs_seq=5` | Overlapping frames enter up to five times | Cache position no longer maps cleanly to environment time | Reject |

No finite absolute `pos_emb` is applied to the episode temporal stream. Mamba's causal recurrence carries order; the learned ten horizon-query embeddings remain in the stateless action decoder.

## 12. Previous Action Decision

Selected: do **not** add `action_{t-1}` in Phase-1.

Reasons:

- Inspected MaIL does not consume previous actions.
- Adding it expands the baseline change and introduces action scaling/alignment/reset questions unrelated to the semantic-retention hypothesis.
- The same observation stream already reflects action consequences, although it may be partially observable.

The recurrent state does not literally contain executed-action inputs under this decision; it can only infer action effects from observation history. A previous-action ablation is a future extension if dataset/controller inspection shows that current images are insufficient.

| Option | Effect | Decision |
|---|---|---|
| A. Keep MaIL input contract: no previous action | Isolates persistent visual/semantic memory | **Selected** |
| B. Add previous action to all three methods | Potentially reduces partial observability but adds a new modality | Future ablation |
| C. Assume state indirectly contains action information | Accept only in the limited sense that later observations encode consequences; do not claim executed actions were inputs | Clarification, not a separate implementation |

## 13. Action Chunk / Replanning Analysis

| Protocol | Transition delay | Baseline preservation | Suitability |
|---|---:|---|---|
| A. Predict 10, execute 10 | Up to 9 steps | Closest to published MaIL | Reject for core stateful test |
| B. Predict 10, execute 1 | None beyond one control cycle | Preserves ten-step output head | **Selected** |
| C. Horizon 1, execute 1 | None | Removes chunk head/auxiliary future targets | Not selected |
| D. Execute chunk, interrupt at oracle transition | No post-boundary delay, but requires privileged online interruption | Hybrid behavior and variable cadence | Not selected |

At every control step, the policy predicts ten actions but executes only horizon index 0. It observes again, updates temporal state once, and replans. All three methods use this receding-horizon protocol.

Training produces ten targets at each valid `t`, but future target index `j` is masked if it crosses the episode end or the first annotated subtask boundary after `t`. This **boundary-safe horizon mask** prevents Method C from being penalized for predicting actions belonging to a semantic transition it has not yet received. The same mask is applied to A, B, and C. If Task 3 shows boundary uncertainty too high for this rule, the conservative fallback is first-action-only loss for all methods; unmasked cross-boundary chunk loss is not the preferred core protocol.

Published MaIL's open-loop execute-10 result, if reproduced, must be labeled separately.

## 14. Semantic Conditioning Protocol

Every control timestep has exactly one semantic input slot. All methods use the same embedding source, dimension, learned projection, fusion, dropout, and parameter count.

Recommended common interface:

```text
o_t: two-view visual feature, 256-D -> W_o -> 128-D
s_t: text/subinstruction/HOLD source, 512-D -> W_s -> 128-D
x_t = common_norm(W_o(o_t) + W_s(s_t))
```

The exact normalization placement is an implementation detail to lock in Task 4, but it must be shared and fixed before any results. Semantic information is fused into the observation-time token rather than inserted as additional persistent tokens. Thus each environment step advances Mamba by exactly one token for every method.

## 15. Vanilla Baseline Definition

Core Method A uses **V1: full instruction at every timestep**:

```text
semantic_t = full task instruction embedding, for every valid t
```

It uses the same persistent temporal encoder and receding-horizon control as B and C. It must be named **Stateful Vanilla**, not published MaIL.

V2—full instruction at `t=0`, then HOLD—is reserved as a sparse-frequency control and is not added to the initial core run matrix.

## 16. Current-Subinstruction Definition

Method B receives the oracle current subinstruction at every valid step:

```text
semantic_t = embedding(S_k) while t belongs to stage k
```

Boundary alignment follows Section 17. This is the dense upper-control condition for explicit stage identity. It does not freeze or manually overwrite recurrent state.

## 17. HOLD/Transition Definition

Method C receives:

```text
semantic_t = embedding(S_k), if t is the first action step of stage k
semantic_t = HOLD, otherwise
```

The transition is applied **before** processing `observation_t`, updating the recurrent state, and predicting `action_t`. Equivalently, the boundary means “stage k is active for the action at t.” Task 3 annotations must encode this incoming-stage convention explicitly.

At episode start, `S_1` is supplied at `t=0`. HOLD means “no new semantic transition”; it does **not** freeze Mamba. Visual/physical state updates every timestep.

## 18. HOLD Representation Candidates

| Candidate | Benefit | Problem | Decision |
|---|---|---|---|
| Learnable HOLD embedding | Same slot/shape; can learn a neutral symbol | Adds a parameter and may develop task-correlated behavior | **Selected, instantiated for all methods** |
| Zero embedding | No parameter | Different norm/distribution; projection bias may make it non-neutral | Reject |
| Special learned language token | Natural text-token framing | Functionally similar to learned embedding; tokenizer/model coupling | Acceptable implementation-equivalent alternative |
| Reuse previous semantic embedding | Smooth input | Repeats current subtask and collapses toward Method B | Reject |
| No semantic token | Literal absence | Changes token count/update cadence/architecture | Reject |

Use one shared learned 512-D HOLD vector passed through the same `W_s`. Instantiate it in the common model for A/B/C so parameter counts are identical, even though core A/B never select it. Initialization and optimization are shared.

## 19. Fairness Analysis

The practical core set is intentionally not matched for semantic information frequency:

- A vs B holds frequency dense and tests full-task versus decomposed current-stage semantics.
- B vs C changes dense repetition to sparse transitions and directly tests whether persistent state can retain stage identity.
- A vs C combines decomposition and sparsity and should not be interpreted as isolating either alone.

Providing A's full instruction every step while C receives sparse transitions is fair for the **practical conditioning** question, because that asymmetry is the treatment. It is not a pure token-frequency control. If core results are promising or ambiguous, the predeclared sparse-controlled comparison is:

```text
Stateful Vanilla-Sparse: full instruction at t=0, then HOLD
Stateful HOLD/Transition: S_k at each transition, then HOLD
```

This asks whether intermediate semantic refresh at meaningful boundaries helps when non-transition frequency is matched. It is not part of the initial three-run set.

All core methods must share seeds, trajectories, annotation version, batch order, model initialization scheme, backbone size, semantic slot, HOLD parameter existence, loss masks, number of optimizer updates, and rollout protocol.

## 20. One-State vs Fast/Slow-State Analysis

Selected Phase-1 design: one Mamba recurrent state jointly carries visual history, action-relevant dynamics, and semantic progress.

An explicit fast/slow design could update a fast visuomotor state every step and a slow semantic state only at transitions. It might improve retention and interpretability, but it adds architectural novelty, method-specific gates, more state-reset logic, and confounds whether ordinary Mamba state was sufficient.

Fast/slow state is a future extension after the single-state hypothesis is evaluated. HOLD never gates the single state off.

## 21. Batch / DataLoader Requirements

### Selected default: whole-trajectory batches

Each dataset item is one complete demonstration with:

```text
episode_id
ordered observations/actions
semantic sequence
transition flags
valid length / valid mask
horizon target mask
task ID and split ID
```

Batch similar lengths when practical. Padding is allowed after the valid suffix; padded outputs and losses are ignored and no subsequent episode may be concatenated behind padding in the same state stream. Because the slot ends at padding, state changes caused by padded tokens are discarded before any reuse.

### Fallback: ordered segment batches / parallel slots

If whole trajectories do not fit, each segment must carry `episode_id`, monotonically increasing `segment_id`, `is_first`, `is_last`, and `valid_mask`. A state table is keyed by stable episode/slot identity. Only `is_first` initializes zero; only `is_last` deletes state. Random segment shuffling is forbidden.

Parallel trajectory slots are efficient but more error-prone: when one episode ends, only that slot is reset. Unit tests must use deliberately unequal lengths to catch whole-batch resets or leakage.

| Loader option | Complexity | GPU efficiency | State correctness | TBPTT fit | Decision |
|---|---|---|---|---|---|
| A. Whole-trajectory batch + padded suffix | Lowest | Padding overhead; length bucketing helps | Easiest: every item starts at zero and ends permanently | Full BPTT; no cross-batch state table | **Default** |
| B. Ordered segment batch keyed by episode/segment | Medium | Bounded memory and regular shapes | Correct only with strict monotonic ordering and state table | Excellent | **TBPTT fallback** |
| C. Multiple asynchronous trajectory slots | Highest | Best utilization for variable lengths | Most reset/leakage risk | Good | Later optimization after B is verified |

## 22. Train/Validation State Leakage Prevention

- Split by entire demonstration/trajectory, never by timestep/window.
- No episode ID may appear in more than one split.
- Prefer task-stratified trajectory splits with fixed manifests and content hashes.
- Create a fresh model execution state for every validation/test episode.
- Destroy training state tables before evaluation; never reuse an `InferenceParams` instance across splits.
- Validation ordering may be deterministic, but state must reset per demo regardless of adjacent task identity.
- Probe training also uses trajectory-level splits independent of the policy-training evaluation split definition.

MaIL's released same-directory train/validation defaults are not acceptable evidence of held-out performance.

## 23. Probe Representation

Primary `h_t`:

> The final-normalized output of the persistent temporal encoder after consuming the current fused observation-semantic token and before the stateless action decoder.

This `[B, 128]` representation is causally available at `t` and directly conditions all action predictions. It is not a custom feature created for probing.

Optional secondary representation:

> The first fresh action-query representation immediately before the shared action head.

Do not use a flattened raw multi-layer cache as the primary probe; its enormous layer-specific feature space would not correspond to one representation consumed by the policy and would make capacity control difficult.

Probe protocol: freeze the trained policy; fit the same regularized linear probe and hyperparameter selection procedure for each method using trajectory-level probe splits. Report balanced/macro accuracy as well as overall accuracy because stage durations can be imbalanced.

## 24. Semantic Retention Metrics

Primary new diagnostic: **Subtask Retention Curve**.

- x-axis: valid control steps since the last oracle subtask transition.
- y-axis: balanced linear-probe accuracy for current subtask ID from primary `h_t`.
- Predeclared candidate bins: `0–5`, `6–10`, `11–20`, `21–40`, `40+`; freeze exact bins after dataset statistics and before model evaluation.

Method B is given the current label every step, so high probe accuracy is expected and is an upper-control, not evidence of retention. Method C's accuracy decay with transition distance is the main retention result.

Also report:

- **Long-gap Action Error:** normalized and, if action semantics permit, denormalized error binned by transition distance.
- Overall action prediction error and rollout success.
- Per-task/per-stage curves and sample counts per bin.
- Performance immediately before and after transitions.

Choose `K`/bins from training-set stage-duration quantiles without looking at test outcomes. Confidence intervals must respect trajectory clustering rather than treating frames as independent.

## 25. Train/Inference Consistency

Training and inference use the same causal recurrence and one fused observation-semantic token per control step.

```text
training:  [x_0, ..., x_T] -> one causal scan -> [h_0, ..., h_T]
inference: x_t + cache_{t-1} -> h_t + cache_t
```

No future observation or future semantic transition is an input. Future actions are targets only. The action decoder is fresh/stateless at every `t`; horizon index 0 is executed.

Before experiments, Task 4/5 must numerically compare full-sequence and cached stepwise outputs in evaluation mode from zero state, including multiple batch sizes and transitions. Define a tolerance appropriate to dtype/kernel. Also compare gradients for the training path; do not infer training support from inference output equivalence.

Dropout and other stochastic layers must be disabled for equivalence tests. Streaming inference must advance the offset exactly once per environment step and not once per action query.

## 26. MaIL Action-Query Cache Issue

The issue **exists** under a naive persistent conversion.

Current MaIL sends five observation tokens followed by ten learned query tokens through one Mamba. If that whole call is cached repeatedly, the state history becomes:

```text
obs_t, q_0, ..., q_9, obs_{t+1}, q_0, ..., q_9, ...
```

Those queries are hypothetical positions used to decode future actions; they are not observations or executed actions. Persisting them contaminates temporal state with repeated unexecuted plans, advances the cache offset by eleven tokens per control step, and makes state semantics depend on action-horizon length. It also conflicts with v1.2.0 cached stepping, which accepts one token after the initial offset.

Therefore action queries must never be committed to the persistent temporal cache. Only the fused observation-semantic token advances that cache.

## 27. Persistent Encoder vs Persistent Entire Policy

| Design | Modification | State meaning | Query contamination | Decision |
|---|---|---|---|---|
| A. Persist entire MaIL token sequence | Superficially smaller | Mixes observations, semantics, and hypothetical query tokens | Severe unless complex rollback/copy logic is added | Reject |
| B. Persistent temporal encoder + stateless action decoder | Larger but localized interface change | Cache contains only realized observation/semantic history | None | **Selected** |

Recommended action decoder:

1. Keep ten learned horizon-query embeddings.
2. Combine each fresh query with the same current `h_t` through one shared lightweight MLP/action head.
3. Return ten 7-D actions.
4. Discard all decoder/query activations after the call; only temporal encoder state persists.

This is not an exact reproduction of MaIL's one-stack query processing. It is the cleanest controlled architecture for the hypothesis. A separately reproduced finite-window MaIL number may be reported as a reference, not silently relabeled as the stateful baseline.

## 28. Final Decision Matrix

| Design Question | Selected Protocol | Alternative | Reason |
|---|---|---|---|
| Stateful execution unit | One observation/control timestep per state update | Multi-step chunk | Exact transition timing; matches cached `step()` |
| Persistent state representation | Per-layer `(conv_state, ssm_state)` plus offset metadata | Exposed hidden token only | These are the actual upstream recurrent cache tensors |
| Observation context | Current two-view observation only | Five-frame window + state | Avoid duplicate reprocessing and overlap ambiguity |
| Action horizon/execution | Predict 10, execute 1; boundary-safe target mask | Execute 10 or horizon 1 | Immediate state/transition update while retaining horizon head |
| Training trajectory ordering | Whole ordered demonstration | Shuffled windows | Required for causal persistent state |
| BPTT strategy | Full-trajectory BPTT by default | Ordered TBPTT fallback | Preserves direct sparse-transition credit |
| State detach rule | No intra-episode detach; fallback detach every measured `K` valid steps | Detach every step/transition | Preserve long-range graph; bound memory only when necessary |
| Episode reset | New/zero all layer states and offsets per demo | `InferenceParams.reset()` alone | Prevent leakage; upstream reset does not visibly zero cache dict |
| Previous action input | None | Add `a_{t-1}` to all methods | Preserve input contract and isolate semantic hypothesis |
| Vanilla language protocol | Full instruction every step (V1) | Instruction at `t=0` then HOLD (V2) | Practical dense baseline; V2 retained as sparse control |
| Current-subinstruction protocol | Current `S_k` every step | Transition-only | Dense stage-identity upper-control |
| HOLD representation | Shared learnable HOLD vector through common projection | Zero, no token, reuse previous | Same slot/distribution; avoids collapsing to B |
| Transition timing | Incoming `S_k` before state update/action at boundary `t` | Apply after action or at replan only | No oracle-event delay; unambiguous annotation |
| Probe representation | Final temporal encoder `h_t` before action decoder | Raw cache or query state | Causal representation actually used for action prediction |
| Retention metric | Probe/action curves vs steps since transition | Overall frame accuracy only | Directly measures semantic memory decay |
| Persistent architecture | Persistent encoder + stateless query decoder | Persist whole MaIL sequence | Prevent query pollution |
| Batch strategy | Whole-trajectory padded batch | Ordered segment/parallel state table | Simplest correct full-BPTT implementation |
| Checkpoint recurrent state | Do not save transient state; resume at episode boundary | Save mid-episode caches | Recurrent state is sample execution state, not model parameter |
| State timescales | Single shared Mamba state | Explicit fast/slow states | Minimum architecture needed for Phase-1 |

## 29. Recommended Minimal Phase-1 Protocol

```text
Observation input:
  Current agent-view + eye-in-hand RGB at control step t; no sliding window.

Temporal input token:
  Shared additive fusion of projected visual feature and one projected semantic slot.

Temporal unit:
  One valid environment timestep.

Persistent state:
  All temporal-encoder layer convolution and SSM states; cache contains no action queries.

Temporal architecture:
  Single persistent Mamba encoder, nominally the published 16-layer/128-D setting
  subject to Task-4 config lock.

Action prediction:
  Fresh stateless ten-query shared decoder/head from h_t -> [10, 7].

Execution interval:
  Replan each step; execute only action index 0.

Action training mask:
  Mask episode overflow and targets at/after the next semantic boundary.

Vanilla semantic input:
  Full instruction every timestep.

Subinstruction semantic input:
  Oracle current subinstruction every timestep.

HOLD semantic input:
  New S_k at the first action step of stage k; shared learned HOLD otherwise.

Episode reset:
  Fresh zero state and offset at every demonstration/rollout.

Training ordering:
  Whole ordered trajectories; shuffle only complete trajectories.

Gradient policy:
  Full-trajectory BPTT; ordered TBPTT with measured K only if memory profiling requires it.

Primary h_t:
  Final-normalized temporal encoder output for current step, before action decoding.
```

## 30. Implementation Risks

1. **Inference-only cache:** `InferenceParams` and in-place cache updates are not a promised differentiable TBPTT interface. A functional state path may be necessary if fallback TBPTT is used.
2. **Exact package mismatch:** MaIL requests `1.2.0.post1`, while static upstream verification used the official `v1.2.0` tag. Inspect the installed wheel source after environment creation and lock its hash.
3. **Sequence/step equivalence:** fused and selective kernels may differ numerically; equivalence must be tested in the chosen dtype.
4. **Cache reset:** offset reset alone may retain tensor contents. Add explicit per-layer zero/new-cache tests.
5. **Action-query contamination:** naive reuse of current `Enc_only` is invalid for persistent cache.
6. **Finite position embeddings:** current `pos_emb` cannot index arbitrary episode streams; remove it from the temporal encoder rather than silently wrapping positions.
7. **Language repetition:** dense A/B repeatedly inject semantic vectors and can dominate the state. This is part of the practical contrast but warrants the predeclared sparse control.
8. **Boundary-safe horizon masking:** annotation error can incorrectly remove action targets. Preserve unmasked targets and masks for audit.
9. **Visual activation memory:** full trajectories with two ResNets may force TBPTT even though Mamba is linear-time.
10. **Variable lengths:** padding masks do not automatically prevent recurrent updates. Padded suffix state must be discarded and never reused.
11. **Batch-slot leakage:** partial slot resets can mix demonstrations if episode IDs are mishandled.
12. **Sparse credit under TBPTT:** long-gap losses cannot train transitions beyond `K`; report this if fallback is used.
13. **Transition off-by-one:** annotation, target action, and simulator control indices may use different conventions.
14. **Oracle availability:** online rollout needs a reproducible oracle transition source without peeking at future observations/actions.
15. **Probe leakage:** B directly receives the target stage label; overall probe accuracy cannot support a retention claim.
16. **Architecture comparability:** stateful core results are controlled among A/B/C, not exact published-MaIL reproduction.
17. **Checkpoint resume:** default MaIL saves only model weights; a robust optimizer/RNG resume format is still needed.

Checkpoint rule: standard checkpoints save model parameters, optimizer/scheduler/scaler, epoch/update, RNG states, split/config/annotation hashes, and sampler position only at an episode boundary. They do **not** save transient recurrent cache. Exact mid-episode resume is out of Phase-1; if later added, it must save cache tensors together with episode ID and exact timestep.

## 31. Required Information From Dataset

The following values must be measured before Task 4 freezes resource-dependent numbers:

- Per-task and per-demonstration trajectory lengths: min/median/quantiles/max.
- Number of demonstrations and valid trajectory-level split sizes.
- Stored sampling/control frequency and whether frames/actions were subsampled.
- Exact action semantics, dimension ordering, scaling, and whether actions are deltas or absolute commands.
- Observation/action timestamp alignment and whether `action_t` follows `observation_t`.
- Controller metadata and termination/no-op tails.
- Exact two RGB keys, shapes, and missing/corrupt frame incidence.
- Oracle stage count, boundary indices, stage-duration and inter-transition-gap distributions.
- Boundary ambiguity/inter-annotator or rule consistency.
- Fraction of ten-step targets that cross a boundary or episode end.
- Memory footprint for full-trajectory two-view visual training on the allocated GPU.
- Class balance for subtask probe labels and transition-distance bins.

These determine the final retention bins, TBPTT `K` if needed, batching buckets, and whether the boundary-safe horizon mask leaves adequate supervision.

## 32. Information Passed to Task 3/4

### Task 3 annotation contract

- Annotate complete demonstrations, not windows.
- For each valid action index `t`, emit `stage_id_t`, `subinstruction_t`, and `is_transition_t`.
- `is_transition_t=True` means the new stage is active **for action `t`**.
- The first valid timestep is a transition to `S_1`.
- Preserve uncertainty/notes; do not force a boundary without evidence.
- Derive `steps_since_transition_t` and the boundary-safe ten-horizon mask reproducibly.
- Version and hash the annotation manifest.

### Task 4 interface contract

- One observation-semantic recurrent token per valid timestep.
- One shared conditioning projection/fusion module for A/B/C.
- One persistent temporal encoder; one fresh stateless ten-query action decoder.
- State API must expose `init_state`, `forward_sequence`, `forward_step`, `detach_state`, and selective batch-slot reset semantics without relying on hidden globals.
- Training whole-sequence scan and cached inference step must have an automated equivalence test.
- Dataset returns whole ordered trajectories first; segment mode is a fallback, not an independent shuffled-window loader.
- Model optionally returns primary `h_t` without changing normal actions or adding probe-only features.
- All state is cleared at demo/split boundaries.
- Standard checkpoints resume only at episode boundaries and exclude recurrent cache.
- Configuration must label `published_mail_reference` separately from `stateful_vanilla`.

## 33. Remaining TBD

The following are intentionally unresolved until dataset/environment inspection, but their selection rules are fixed:

- Exact core Mamba layer count: nominal published override 16 versus YAML default 6; lock one shared value in Task 4.
- Whether full-trajectory BPTT fits; if not, choose `K` from stage-gap statistics and GPU profiling, not convenience.
- Exact trajectory batch size and length buckets.
- Final transition-distance bins/K after training-only statistics.
- Exact normalization location in the shared additive fusion.
- Exact stateless action-decoder MLP width/depth; it must be fixed and identical across methods.
- Installed `mamba-ssm==1.2.0.post1` source hash and exact cache/autograd behavior.
- Numeric sequence-vs-step equivalence tolerance for the selected dtype/kernel.
- Oracle transition delivery mechanism during simulator rollout.
- Whether annotation uncertainty requires first-action-only loss instead of boundary-safe horizon masking.
- Whether a sparse Vanilla V2 control is justified after the initial core results; it is predeclared but not automatically added.

No Task 3 work, model implementation, environment creation, dataset download, annotation, training, GPU submission, rollout, or evaluation was performed in Task 2.5.
