# Phase-1 Experiment Plan: Explicit Task-Progress Conditioning for Long-Horizon Manipulation

## 1. Research Motivation

Long-horizon robot manipulation requires a policy to represent not only observation/action history, but also where the trajectory currently lies in a semantically ordered task. A generic temporal model conditioned on the full natural-language instruction must infer this progress implicitly.

This study tests whether progress can be represented more clearly by decomposing the full instruction into ordered sub-instructions and explicitly signaling semantic subtask transitions. The temporal backbone for the first experiment is a Mamba-family selective state-space model. Mamba is a controlled backbone choice, not the research novelty. The contribution under test is the combination of instruction decomposition and an explicit subtask transition/`[HOLD]` mechanism.

## 2. Primary Research Question

With the same Mamba temporal backbone, does providing ordered sub-instructions and explicit transition/`[HOLD]` signals improve temporal context representations and action prediction for long-horizon manipulation compared with conditioning on the full instruction and relying on the model to infer task progress implicitly?

Two effects must be isolated:

- **Q1 — Instruction decomposition:** Does explicitly conditioning on the current sub-instruction outperform repeatedly conditioning on the full instruction?
- **Q2 — Event-driven task progress:** Does emitting a new sub-instruction only at a subtask transition and `[HOLD]` otherwise outperform repeating the current sub-instruction at every timestep?

## 3. Hypotheses

- **H1 — Decomposition hypothesis:** Current-subinstruction conditioning will make the task phase less ambiguous than full-instruction conditioning and may reduce action prediction error, especially in long multi-stage trajectories.
- **H2 — Transition/`[HOLD]` hypothesis:** Event-driven semantic updates will produce a temporal representation that tracks task progress more explicitly than repeated current-subinstruction conditioning.
- **H3 — Boundary hypothesis:** Any benefit from explicit progress conditioning will be especially visible near subtask transitions, where the required action regime changes.
- **H4 — Representation hypothesis:** A linear probe trained on frozen temporal representations will predict the current subtask more accurately for subtask-aware methods, particularly HOLD/Transition Mamba, than for Vanilla Mamba.

These are hypotheses, not expected-result constraints. A null or negative result is valid evidence.

## 4. Compared Methods

All methods use the same observation, previous-action, encoder, temporal-backbone, and action-head design wherever the inspected baseline permits. The intended independent variable is how instruction/task-progress information is injected.

### Method A — Vanilla Mamba

- Inputs include `observation_t`, `previous_action_t-1`, and the full instruction.
- The full instruction is provided at every timestep.
- Task progress must be inferred implicitly from trajectory history.

### Method B — Current-Subinstruction Mamba

- Uses the current ordered sub-instruction instead of the full instruction.
- The same current sub-instruction is provided at every timestep until the next subtask starts.
- This comparison against Method A isolates the effect of instruction decomposition.

### Method C — HOLD/Transition Mamba (Ours)

- Emits the relevant sub-instruction when a new subtask begins.
- Emits `[HOLD]` at all other timesteps in that subtask.
- `[HOLD]` means that no new semantic task transition occurred. It does **not** freeze the Mamba state; observations/actions may continue to update the temporal state.
- Its comparison against Method B isolates the effect of event-driven transition signaling.

The exact token or embedding representation for `[HOLD]` is intentionally deferred.

## 5. Oracle Boundary Assumption

Phase 1 uses ground-truth or manually annotated oracle subtask boundaries. Each trajectory will eventually provide, at minimum, the current subtask ID, current sub-instruction, and whether a transition occurs at each timestep.

This separates two questions that must not be confounded in the first experiment:

1. Is subtask-aware temporal representation useful?
2. Can subtask completion be predicted accurately?

Only the first question is in Phase 1. A learned subtask-completion head is considered only after evidence is obtained under oracle boundaries.

## 6. Evaluation Metrics

### 6.1 Overall Action Prediction Error

Measure the discrepancy between predicted and ground-truth actions over held-out trajectories. The metric and action-loss definition will inherit the selected baseline repository's convention after inspection rather than introducing an arbitrary new definition.

- **Exact action loss/metric:** TBD after Task 1 baseline inspection.

### 6.2 Subtask Boundary Action Error

For every transition timestep `b`, separately aggregate action prediction error over `[b-K, b+K]`. This evaluates whether explicit task-progress information reduces ambiguity as the action regime changes.

- **Boundary window `K`:** TBD after Task 2 inspection of trajectory FPS, action frequency, and subtask-duration statistics.
- Boundary clipping, overlap handling, and aggregation details must be defined reproducibly in Task 8.

### 6.3 Subtask State Probe Accuracy

Freeze the trained policy/Mamba and train a simple linear classifier from temporal representation `h_t` to the current oracle subtask ID. This measures how explicitly the representation encodes trajectory-specific task progress. Probe data splitting and fitting must avoid leakage from policy training/evaluation trajectories.

### 6.4 Optional Rollout Success Rate

Report task success rate if the selected benchmark/simulator supports stable evaluation rollouts. Simulator setup must not delay the three primary metrics above.

## 7. Fair-Comparison Rules

Wherever technically possible, keep the following identical across all three methods:

- Visual and language encoders
- Mamba implementation, architecture, hidden size, and layer count
- Action head and action representation
- Observation and previous-action inputs
- Dataset and train/validation/test splits
- Optimizer, learning rate, batch size, training steps/epochs, and random seeds
- Data augmentation, preprocessing, normalization, and evaluation protocol
- Checkpoint selection and reporting procedure

Only task-progress/instruction conditioning should vary. Any unavoidable architectural or parameter-count difference must be measured and reported. Method B and Method C should be especially closely matched so that their comparison isolates repeated versus event-driven semantic conditioning. No method-specific tuning or metric changes may be introduced merely to improve reported results.

## 8. Components Excluded From Phase 1

The following are explicitly out of scope:

- World models, world loss, and future latent/image/video prediction
- Learnable world/special tokens
- Teacher VLA, student VLA, teacher-student distillation
- Context or action distillation
- Learned subtask-completion head

These components may only be considered after the Phase-1 mechanism has been evaluated in isolation.

## 9. Full Task 0–15 Roadmap

The tasks are sequential and must not be reordered or started implicitly.

0. **Research hypothesis and scope:** Freeze the questions, hypotheses, methods, metrics, exclusions, roadmap, success criteria, and deferred decisions in this document.
1. **Repository/baseline inspection and reproducibility:** Identify actual repositories, training entry points, loaders, model/evaluation/configuration paths, and verify the unmodified baseline minimally. Do not implement Ours.
2. **Dataset and long-horizon task analysis:** Inspect instructions, demonstrations, trajectory/action/observation formats and lengths, and semantic subtask structure; nominate 2–3 candidate tasks.
3. **Subtask decomposition and oracle annotation:** Define ordered sub-instructions and a per-timestep annotation format containing subtask ID, current sub-instruction, and transition flag. Do not train a completion model.
4. **Common experimental data/conditioning interface:** Expose observation, previous action, full instruction, current sub-instruction, subtask ID, transition flag, and ground-truth action through one shared pipeline.
5. **Vanilla Mamba baseline:** Establish full-instruction conditioning with working training, evaluation, and checkpoints.
6. **Current-Subinstruction Mamba:** Add repeated current-subinstruction conditioning while preserving the shared architecture and pipeline.
7. **HOLD/Transition Mamba:** Add transition-time sub-instructions and non-transition `[HOLD]`, matched as closely as possible to Method B.
8. **Evaluation metrics:** Implement reproducible overall error, boundary error, and linear-probe accuracy, plus rollout success if feasible; save machine-readable results.
9. **Tiny-set sanity/overfit test:** Validate ordering, shapes, state handling, boundary indexing, action normalization, language/`[HOLD]` conditioning, and loss reduction. Do not proceed if this fails.
10. **One-task pilot:** Compare all three methods on one representative long-horizon task, initially with one seed if needed.
11. **Multi-task/multi-seed core experiment:** Only after a mandatory human checkpoint following Task 10, expand to the selected 2–3 tasks and multiple seeds under identical conditions.
12. **Automatic result aggregation:** Record method, task, seed, errors, probe accuracy, optional success rate, training time, and parameter count; produce per-task, mean, standard-deviation, and overall summaries.
13. **Result analysis:** Separately analyze A→B decomposition effects, B→C transition/`[HOLD]` effects, and representation-probe evidence; report negative findings and explanations without post-hoc architecture inflation.
14. **Learned subtask-completion head:** Only if justified by Phase-1 evidence, replace oracle transitions and compare oracle versus predicted transitions.
15. **World-model/teacher-student extension:** Only after the temporal-context mechanism has supporting evidence, consider future-latent supervision, special/context tokens, teacher policies, and context/action distillation.

## 10. Success Criteria for Phase 1

Phase 1 is successfully completed when:

1. Vanilla, Current-Subinstruction, and HOLD/Transition methods run through the same experimental pipeline.
2. All three are compared fairly using oracle subtask boundaries.
3. Overall Action Prediction Error is obtained.
4. Subtask Boundary Action Error is obtained.
5. Subtask State Probe Accuracy is obtained with the policy/Mamba frozen.
6. At least one one-task pilot experiment is completed.
7. Results are expanded to multiple tasks and seeds if the pilot and resources permit.
8. The decomposition effect (A→B) and HOLD/Transition effect (B→C) can be interpreted separately, whether results improve, remain unchanged, or degrade.

The success criterion is a valid, reproducible test of the hypotheses—not that Method C must outperform the baselines.

## 11. Open Questions / Decisions Deferred to Later Tasks

| Decision | Status | Resolution task |
|---|---|---|
| Baseline repository and exact code revision | TBD | Task 1 |
| Dataset/benchmark | TBD | Tasks 1–2 |
| Mamba implementation and integration point | TBD | Task 1 |
| Training, configuration, and evaluation entry points | TBD | Task 1 |
| Exact visual/language encoders and action head | TBD | Task 1 |
| Existing action loss and primary action-error definition | Inherit baseline; TBD | Task 1 |
| Candidate tasks and final 2–3 task selection | TBD | Task 2 |
| Observation/action formats, rates, normalization, and trajectory statistics | TBD | Task 2 |
| Ordered sub-instruction decomposition policy | TBD | Task 3 |
| Oracle boundary source and annotation schema | TBD | Task 3 |
| Transition-frame indexing convention | TBD | Tasks 3 and 8 |
| `[HOLD]` token/embedding representation and injection mechanism | TBD | Tasks 4 and 7 |
| Boundary window `K` | TBD from trajectory statistics | Tasks 2 and 8 |
| Linear-probe representation layer, split, balancing, and fitting protocol | TBD | Task 8 |
| Stable rollout availability and success definition | TBD | Tasks 1–2 and 8 |
| Seeds, number of runs, compute budget, and stopping/checkpoint rule | TBD | Tasks 1, 10, and 11 |
| Parameter-count parity handling | TBD after model inspection | Tasks 1 and 4–7 |

## Task-Gating Rules

- Re-read this plan before each task.
- Inspect real repository code before making structural assumptions.
- Preserve the working baseline and make only the minimum changes required for the active task.
- Do not advance to the next task without an explicit instruction; Task 10→11 additionally requires a human checkpoint.
- Run proportionate validation where possible and clearly distinguish executed checks from planned checks.
- Keep failed and negative results, and do not change metrics or tune methods asymmetrically to favor an outcome.

