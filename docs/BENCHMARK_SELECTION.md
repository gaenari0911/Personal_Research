# Main Long-Horizon Benchmark Selection

Audit date: 2026-08-24  
Decision state: recommendation complete; human approval required before migration

## Executive decision

**Primary recommendation: CALVIN.**

CALVIN is the only audited candidate that combines an official five-language-goal, no-environment-reset rollout with MaIL's exact 7D Cartesian action width and an exact two-RGB-view sensor option. It does not natively provide a single compound instruction or five-stage BC demonstrations. Those are real limitations, but they can be kept separate as an explicitly custom prompting/data condition without introducing mobile-base control, a hybrid control-mode output, a new rotation representation, or a new semantic ontology into the core comparison.

Final ranking:

1. **CALVIN — 86/100**
2. **RoboCasa365 — 79/100**
3. **FurnitureBench — 69/100**

This is a benchmark-selection audit only. No candidate dataset was downloaded, no LIBERO artifact was deleted, and no model or training job was created.

## 1. Research Requirements

The primary benchmark must expose at least four, preferably five or six, language-level semantic goals in one continuous manipulation rollout. A semantic stage must be independently describable as a goal; `reach/grasp/lift/move/release` decompositions do not qualify merely to inflate a stage count.

The ordering of requirements used in this audit is:

1. Four to six genuine semantic stages.
2. One continuous environment episode with at least three transitions.
3. Compatibility with MaIL's 7D Cartesian manipulation control.
4. Official semantic labels or reliable official success boundaries.
5. Language compatibility.
6. Reproducible rollout evaluation.
7. Storage and engineering convenience.

The intended controlled comparison contains a full-instruction condition, current-stage conditioning, and sparse transition conditioning. Therefore full-task language and stage boundaries are useful, but they must not be gained at the cost of turning the experiment into a mobile-navigation or contact-rich assembly study.

## 2. Why LIBERO Is Insufficient as Primary Benchmark

The completed local LIBERO audit found useful long trajectories, but the three strongest selected LIBERO-10 tasks each reduce to **two semantic stages and one transition** under the defensible language-level ontology. The median trajectories were 245.5, 258.0, and 300.5 steps, while the post-transition retention intervals were 80.5, 141.5, and 120.0 steps respectively. They are temporally long enough, but do not exercise repeated semantic state updates across three or more transitions.

Evidence is preserved in:

- [`analysis/oracle_experiment_motivation.csv`](../analysis/oracle_experiment_motivation.csv)
- [`analysis/oracle_annotation_summary.csv`](../analysis/oracle_annotation_summary.csv)
- [`docs/ORACLE_SUBTASK_ANNOTATION.md`](ORACLE_SUBTASK_ANNOTATION.md)

LIBERO remains useful as a two-stage diagnostic and regression benchmark. It is insufficient as the main benchmark for repeated 4–6-stage progress tracking.

## 3. MaIL Baseline Constraints

The local MaIL source, rather than prior notes, was treated as authoritative.

| Property | Verified MaIL setting | Source |
|---|---|---|
| Action | 7D | [`benchmark_libero10.yaml`](../external/MaIL/config/benchmark_libero10.yaml) |
| Views | `agentview_rgb`, `eye_in_hand_rgb` | [`goal_bc_mamba_dec.yaml`](../external/MaIL/config/agents/goal_bc_mamba_dec.yaml) |
| Resolution | 128×128 per view | same config |
| Language | per-task 512D embedding, linearly projected and prepended | [`goal_bc_agent.py`](../external/MaIL/agents/models/bc/goal_bc_agent.py) |
| Backbone | Mamba `MixerModel`, `d_model=128` | `goal_bc_mamba_dec.yaml` |
| Observation window | 5 frames | `benchmark_libero10.yaml` |
| Action horizon | 10 train / 10 inference | `benchmark_libero10.yaml` |
| Objective | BC, MSE action prediction | `goal_bc_agent.py` |

The loader reads the two RGB streams, the action tensor, and one task embedding for every demonstration. Training retains the first five observations in the sampled window and predicts ten action rows beginning at observation index four.

An important qualification is that released MaIL uses Mamba over finite windows. It does **not** already retain recurrent state across an entire episode. Episode-persistent state is the proposed research intervention, not a property that can be attributed to the released baseline.

## 4. RoboCasa Audit

### Official long-horizon structure

RoboCasa365 v1.0 was released on 2026-02-18 and v1.0.1 updated rollout horizons on 2026-05-12. Its official target-task table includes genuine long composite tasks:

| Task | Official subtasks | Horizon | Navigation listed | Language-level sequence |
|---|---:|---:|---|---|
| `SteamInMicrowave` | 6 | 45 s | yes | vegetable to bowl; bowl to microwave; close door; press start, with navigation annotations |
| `StoreLeftoversInBowl` | 5 | 58 s | yes | chicken to bowl; vegetable to bowl; bowl to fridge, with navigation annotations |
| `ArrangeBreadBasket` | 5 | 83 s | yes | open cabinet; bread to basket; basket to dining counter, with navigation annotations |
| `WaffleReheat` | 4 | 86 s | yes | open microwave; insert bowl; close door; turn on, with navigation annotations |
| `WashFruitColander` | 4 | 60 s | **no** | colander to sink; fruit to colander; faucet/water operation |

These values come from the [official target dataset overview](https://robocasa.ai/docs/build/html/datasets/datasets_overview.html), not from task-name inference. The official task code further confirms that `SteamInMicrowave` is one uninterrupted environment with four terminal semantic predicates, while `WashFruitColander` initializes the robot at the sink and keeps its objects on the adjacent counter ([Steam source](https://github.com/robocasa/robocasa/blob/main/robocasa/environments/kitchen/composite/steaming_food/steam_in_microwave.py), [Wash source](https://github.com/robocasa/robocasa/blob/main/robocasa/environments/kitchen/composite/washing_fruits_and_vegetables/wash_fruit_colander.py)).

`PackIdenticalLunches` is also semantically useful: its source defines four food placements into two nearby containers. However, the official target table expands the demonstration to 15 annotated subtasks and lists Navigation, so it is not evidence for a clean fixed-base four-stage benchmark ([source](https://github.com/robocasa/robocasa/blob/main/robocasa/environments/kitchen/composite/packing_lunches/pack_identical_lunches.py)).

### Per-frame semantic annotations

On 2026-07-07 the RoboCasa team announced that **target composite task datasets** had been updated so that every timestep contains:

- subtask index;
- atomic-skill name;
- stage, such as pick/place/navigate;
- natural-language subtask instruction.

The [official release page](https://robocasa.ai/) and [dataset overview](https://robocasa.ai/docs/build/html/datasets/datasets_overview.html) both make the all-target-composite and per-timestep scope explicit. The update occurred after v1.0.1, but the announcement does not give it a separate semantic-annotation version number. It is therefore recorded as the “2026-07-07 target-composite update,” not invented as v1.0.2.

The current official public documentation describes the four aligned semantic fields but does not publish their exact serialized LeRobot column names. Those names require inspecting `meta/info.json`/Parquet schema after selection. This uncertainty concerns field spelling, not whether per-frame annotations exist.

### Action dimensions: nominal, effective, rollout-required

The official action conversion source splits a 12D vector as follows ([`env_utils.py`](https://github.com/robocasa/robocasa/blob/main/robocasa/utils/env_utils.py)):

| Slice | Width | Meaning |
|---|---:|---|
| `[0:3]` | 3 | end-effector position |
| `[3:6]` | 3 | end-effector rotation |
| `[6:7]` | 1 | gripper close |
| `[7:11]` | 4 | base motion |
| `[11:12]` | 1 | control mode |

Therefore:

- **Nominal dataset action dimension: 12.**
- **Effective arm-manipulation dimension: 7 only when base motion is unused and control mode remains arm-active.**
- **Dimensions required by the standard rollout interface: 12/dictionary fields.** A 7D policy still needs a wrapper that reconstructs constant base and mode fields; training-time projection does not prove that the environment accepts a 7D vector.

The official RoboCasa365 paper reports 220 mobile-manipulation tasks and 145 tasks that can be performed without mobility. That establishes that fixed-base-like tasks exist in the overall suite, but not that three annotated target composites with four to six stages are arm-only. Among the inspected target candidates, only `WashFruitColander` omits Navigation. The hard three-task pure-manipulation gate is therefore **not passed** at audit time.

### Cameras, language, data, and evaluation

Official LeRobot datasets contain left third-person, right third-person, and eye-in-hand video streams. The proposed MaIL adapter would use right third-person plus eye-in-hand and discard the left view. This is **our modification**, although it preserves a fair common two-view input.

Target data contains 500 human demonstrations per task across 50 tasks and 193 hours. The full release contains more than 2,200 hours. Task-specific downloads are officially supported using `--tasks`; official documentation does not publish byte size per task or for the entire release. Environment assets alone are approximately 10 GB. No bytes were downloaded for this audit.

RoboCasa provides simulator construction from dataset metadata, task horizons, success checks, and official evaluation scripts. It is exceptionally strong in full-instruction and semantic annotation quality.

### Official versus our modification

**OFFICIAL**

- Continuous composite demonstration and full-task language.
- Per-frame semantic fields on all target composites.
- Three RGB cameras.
- 12D hybrid mobile-manipulation control.
- Task-specific download and simulator success evaluation.

**OUR PROPOSED MODIFICATION**

- Select right third-person plus eye-in-hand and resize to 128×128.
- Project to 7D only after verifying `base_motion≈0` and constant arm-active `control_mode` across every selected episode.
- Otherwise change MaIL's output head to 12D and accept mobile-control confounding.

### MaIL fairness verdict

**QUESTIONABLE — substantial modification** for the strongest mobile composite tasks.  
**MOSTLY — minor architecture adaptation** only for a later verified arm-only subset.

Because the audit found only one officially no-navigation 4–6-subtask target task, RoboCasa365 is ranked second rather than first.

## 5. CALVIN Audit

### Official action, sensors, and data

The official dataset defines both absolute and relative 7D Cartesian actions. The recommended `rel_actions` field is:

```text
relative TCP xyz (3)
+ relative Euler xyz (3)
+ binary gripper (1)
= 7
```

Relative position and rotation are normalized/clipped according to the documented scaling, and the environment accepts the same seven-value representation. Nominal, effective, and rollout-required dimensions are all 7. CALVIN runs continuous control at 30 Hz ([dataset README](https://github.com/mees/calvin/blob/main/dataset/README.md), [main repository](https://github.com/mees/calvin)).

The two native RGB views are:

- `rgb_static`: 200×200×3;
- `rgb_gripper`: 84×84×3.

Both can be resized to 128×128 without changing sensor count or camera role. Proprioception is available but optional; an image-only MaIL comparison can omit it symmetrically.

Every environment contributes six hours of teleoperated play, 24 hours total. Data is stored as one `.npz` interaction timestep with images, actions, robot state, and scene state. `auto_lang_ann.npy` stores raw language, task ID, embedding, and `[start,end]` sequence indices for annotated segments.

### Official five-task evaluation continuity

The official LH-MTLC evaluator generates 1,000 valid chains, each of length five. It creates one initial environment state, calls `env.reset(...)` once, and then iterates over five tasks. There is no environment reset between tasks. A task oracle detects completion and the next language instruction is supplied. Each stage has a 360-step cap, or 12 seconds at 30 Hz ([evaluator](https://github.com/mees/calvin/blob/main/calvin_models/calvin_agent/evaluation/evaluate_policy.py), [sequence generator](https://github.com/mees/calvin/blob/main/calvin_models/calvin_agent/evaluation/multistep_sequences.py)).

This answers the required questions directly:

| Question | Finding |
|---|---|
| One continuous environment trajectory? | **Yes.** |
| Environment reset between tasks? | **No.** |
| Continuous training interactions? | **Yes,** as play data; **no**, not as packaged five-stage compound demonstrations. |
| Semantic completion/boundary? | Official task-oracle success at rollout; offline language segments have start/end indices. |
| Next instruction externally provided? | **Yes.** The evaluator switches instruction after success. |

There is one further critical fact: the official evaluator calls `model.reset()` at the beginning of **every subtask**. Thus the environment is continuous, but official policy hidden state is not. A persistent-Mamba experiment must remove or bypass that call. This is a small evaluator modification, but must be disclosed.

### Compound full-instruction feasibility

**Official CALVIN protocol**

```text
instruction S1 → oracle success → instruction S2 → ... → instruction S5
```

The environment remains continuous, but stage identity is externally revealed and the baseline policy is reset at every boundary.

**Our proposed controlled protocol**

```text
FULL = "S1, then S2, then S3, then S4, then S5"
M0/M1: FULL throughout
M2: current S_k
M3: S_k at transition, HOLD otherwise
```

Evaluation can be modified without collecting demonstrations: retain the same environment, official initial state, task transition validity, task oracle, and action interface; change only prompt scheduling and policy-state reset. However, the official dataset does not package aligned action demonstrations for the same five-task compound chains. Supervised BC for `FULL` therefore requires one of:

1. mining and validating suitable consecutive language segments from continuous play;
2. relabelling valid continuous play intervals as compound chains; or
3. collecting simulator rollouts for fixed valid chains.

The first two possibilities must be measured after dataset inspection; availability must not be assumed. A pure evaluation-only FULL prompt is technically possible but would test zero-shot composition rather than matched BC learning.

This changes the language/prompt protocol moderately, but leaves the task dynamics, action space, transition validity, success oracle, and continuous evaluation intact. The study must report “CALVIN-based controlled protocol,” not imply that compound instructions are official CALVIN.

### Temporal and semantic memory suitability

- Five stages create four within-episode transitions.
- Each stage can last up to 360 control steps; actual average/median stage durations require data or rollout logs.
- Valid chains are state-dependent, so earlier object/fixture changes determine which later goals remain possible.
- Lift→place chains create explicit object binding across a boundary.
- The official external instruction switch weakens the need to infer stage from a full instruction, but keeping Mamba state across the switch gives a clean test of retained progress without navigation.

### Storage and evaluation practicality

Official packages are 166 GB for D→D, 517 GB for ABC→D, and 656 GB for ABCD→D. A 1.3 GB debug set is useful only for adapter smoke tests. It is not a sufficient scientific training pilot. There is no official task-only package.

The official simulator, task oracle, 1,000 seeded chains, success-at-depth rates, and average completed chain length provide a reproducible evaluation. CALVIN's public leaderboard reports success after 1–5 consecutive instructions and average length.

### Official versus our modification

**OFFICIAL**

- Five valid language tasks in one no-reset environment chain.
- External instruction switch after task-oracle success.
- Per-subtask `model.reset()`.
- 7D Cartesian control, two RGB cameras, 30 Hz.
- Play data with language segment indices.

**OUR PROPOSED MODIFICATION**

- Keep Mamba/policy state across stage boundaries.
- Freeze three valid five-stage chains for the first pilot.
- Concatenate official utterances for a disclosed compound FULL condition.
- Mine/relabel or collect matched compound demonstrations only if required after the first data audit.

### MaIL fairness verdict

**YES — architecture-preserving adapter only.**

Dataset keys, image resize, CLIP text embedding, and evaluator interface change; the 7D output width, Cartesian control family, two-view encoder, Mamba backbone, fixed observation window, action horizon, and BC objective can remain.

## 6. FurnitureBench Audit

FurnitureBench is genuinely long-horizon. Official full-assembly demonstrations are continuous, and the environment supports skill ranges using `from_skill`/`to_skill`; the documented range exposes the first five skills. Full assembly evaluates completed phases such as grasping, placing, inserting, and screwing. These phases are semantically more meaningful than raw reach/move primitives, although exact phase count varies by furniture.

The [official dataset documentation](https://clvrai.github.io/furniture-bench/docs/tutorials/dataset.html) gives:

- 5,100 successful teleoperation demonstrations;
- 219.6 hours;
- wrist and front RGB images at 224×224;
- 8D actions;
- per-timestep reward and skill-completion flags.

The action is:

```text
delta EE position xyz (3)
+ delta quaternion xyzw (4)
+ gripper (1)
= 8
```

Nominal, effective, and rollout-required widths are all 8. Control runs at 10 Hz. MaIL needs a 7→8 output-head change and a 3D-rotation→quaternion target change. The two-view camera setup is otherwise an excellent match ([official environment guide](https://clvrai.github.io/furniture-bench/docs/tutorials/furniture_bench.html)).

The decisive weakness is language: neither natural-language full-task instructions nor natural-language subtask instructions are present in the official dataset schema. Skill completion pulses and part-assembly rewards can define boundaries, but converting them to named semantic intervals and authoring language is our ontology. This is substantially less clean for a language-conditioned semantic-memory paper.

FurnitureSim provides a reproducible simulator and matching wrist/front image interface, but relies on the older Isaac Gym stack and GPU rendering. Official automated assembly scripts currently cover one-leg, cabinet, lamp, and round-table. Evaluation is reproducible, but less lightweight than CALVIN.

Raw `.pkl` storage totals 1,179 GB: 457 GB low randomness, 499 GB medium, and 223 GB high. Downloads can be restricted by furniture and randomness. The smallest published bucket is 11 GB, before relying on an unspecified compression ratio.

**OFFICIAL:** continuous assembly, two views, 8D actions, skill flags, part rewards, real and simulation evaluation.  
**OUR MODIFICATION:** semantic stage ontology, all language labels, interval conversion, 8D quaternion MaIL head.

### MaIL fairness verdict

**QUESTIONABLE — substantial modification.** The backbone can remain, but both action semantics and the language-conditioned protocol become project-defined.

## 7. Additional Candidate Audit

No fourth candidate was added to the scored ranking. A new candidate was allowed only if it simultaneously offered continuous offline 4+ stage manipulation, language or official semantic structure, and reproducible evaluation while improving on the primary candidates.

The common alternatives do not change the decision: benchmarks with language but no official multi-stage boundaries would lose RoboCasa's main advantage, while scripted pick/place benchmarks with five actions do not match MaIL's continuous Cartesian control as closely as CALVIN. Adding a weaker fourth row would not increase decision confidence. This section records a negative result, not a claim that no other long-horizon benchmark exists.

## 8. Long-Horizon Comparison

| Property | RoboCasa365 | CALVIN | FurnitureBench |
|---|---|---|---|
| Genuine semantic depth | 4–6 official target examples, sometimes more annotated subtasks | exactly 5 official language tasks per chain | 5+ assembly phases, task dependent |
| Continuous environment | yes | yes | yes |
| Within-chain transitions | 3–5+ | exactly 4 | typically 4+ |
| Stage duration | official task horizons 45–86 s for inspected examples; per-stage stats require data | up to 360 steps/12 s per stage; median requires rollout/data | long episodes: published averages 374–2,282 steps by task/randomness; per-stage median requires data |
| Full instruction | official | not official | absent |
| Main confound | mobile navigation/control mode | external prompt switching | contact-rich assembly and custom language |

RoboCasa is strongest on full-instruction semantic richness. CALVIN is strongest on clean repeated state transitions under MaIL-compatible control. FurnitureBench is strongest on very long continuous contact-rich behavior, but the experiment would no longer have official language semantics.

## 9. Semantic Annotation Comparison

| Benchmark | Official stage signal | Time alignment | Natural-language subtask | Custom work |
|---|---|---|---|---|
| RoboCasa365 | subtask index, atomic skill, pick/place/navigate stage | every timestep | yes | exact field-name/schema inspection only |
| CALVIN | task oracle at rollout; language segment `[start,end]` offline | event/segment aligned | yes, one task at a time | compose chain-level labels; preserve hidden state |
| FurnitureBench | skill-completion pulse and assembly reward | per timestep | no | define stage intervals, names, and all language |

## 10. Action-Space Comparison

| Benchmark | Nominal | Effective manipulation | Rollout required | Representation | Decision |
|---|---:|---:|---:|---|---|
| MaIL reference | 7 | 7 | 7 | xyz + 3 rotation + gripper | reference |
| RoboCasa365 | 12 | 7 only in verified arm-only intervals | 12/dict | arm axis-angle + gripper + base4 + mode1 | major penalty until base audit passes |
| CALVIN | 7 | 7 | 7 | relative xyz + Euler xyz + gripper | exact width/family match |
| FurnitureBench | 8 | 8 | 8 | delta xyz + quaternion + gripper | head and rotation-target change |

## 11. Camera Comparison

| Benchmark | Official image-policy views | Native resolution | MaIL change |
|---|---|---|---|
| RoboCasa365 | left, right, eye-in-hand | 256×256 dataset videos | choose right + eye-in-hand; drop one; resize |
| CALVIN | static, gripper | 200×200 and 84×84 | rename and resize only |
| FurnitureBench | wrist, front | 224×224 | rename and resize only |

## 12. Language Comparison

RoboCasa provides the ideal language package: one full composite instruction plus aligned subtask instructions. CALVIN provides multiple paraphrases and task-level segment language but changes the goal externally at success; a compound full instruction is ours. FurnitureBench provides no official language in its trajectory schema.

For all candidates, retaining MaIL's CLIP-style task embedding is possible. The important distinction is provenance of the text and whether the action data was collected under the same instruction granularity.

## 13. Evaluation Comparison

- **CALVIN:** strongest controlled protocol—official simulator, task oracle, 1,000 five-task chains, success-at-depth 1–5, average completed length. Required custom change: retain policy state and optionally alter prompt schedule.
- **RoboCasa365:** official simulator, task-specific success predicates, registered target splits, rollout scripts, and published multi-/foundation-task evaluation. Mobile control remains part of most desired composites.
- **FurnitureBench:** real benchmark plus FurnitureSim, full-assembly and single-skill modes. Reproducible but simulator dependencies and assembly physics add engineering and experimental variance.

## 14. Storage / Compute Comparison

| Benchmark | Published full size | Small package | Task subset | Practical first step after approval |
|---|---|---|---|---|
| CALVIN | 166/517/656 GB by split | 1.3 GB debug | no | debug adapter smoke test, then D→D 166 GB |
| RoboCasa365 | 2,200+ hours; official byte count not published | metadata/source requires no data | yes | query one task manifest/size, then download one target task only |
| FurnitureBench | 1,179 GB raw total | smallest published task/randomness bucket 11 GB | yes | one furniture/randomness bucket |

CALVIN's 166 GB minimum real training split is its main practicality cost. It remains manageable on `/ssd1` if capacity is checked before the next task. No capacity assumption is made here.

## 15. Baseline Adaptation Cost

Scale: 5 nearly unchanged; 4 dataset/evaluator adapter; 3 input/output head change; 2 substantial policy change; 1 effectively a new baseline.

| Component | RoboCasa365 | CALVIN | FurnitureBench |
|---|---:|---:|---:|
| Action compatibility | 2 | 5 | 3 |
| Camera compatibility | 4 | 5 | 5 |
| Language compatibility | 5 | 4 | 1 |
| Temporal compatibility | 5 | 4 | 5 |
| Evaluation compatibility | 4 | 4 | 3 |

CALVIN's temporal/evaluation score is four rather than five because official code resets model state and supplies each stage prompt. RoboCasa's action score is two because its long annotated target tasks mostly list Navigation. FurnitureBench's language score is one because the complete language protocol must be authored.

## 16. 100-point Scoring

| Category | Max | CALVIN | RoboCasa365 | FurnitureBench |
|---|---:|---:|---:|---:|
| A. Long-horizon semantic structure | 30 | 27 | 29 | 26 |
| B. MaIL compatibility | 25 | 25 | 12 | 15 |
| C. Semantic annotation quality | 15 | 9 | 15 | 7 |
| D. Evaluation quality | 10 | 10 | 10 | 8 |
| E. Dataset practicality | 10 | 6 | 9 | 5 |
| F. Experimental cleanliness | 10 | 9 | 4 | 8 |
| **Total** | **100** | **86** | **79** | **69** |

Scoring rationale:

- RoboCasa wins A and C, but loses 13 MaIL-compatibility points and six cleanliness points because base motion and a hybrid control mode are part of most desired annotated tasks.
- CALVIN loses three A points because the evaluator supplies stage prompts and the offline data does not package compound five-stage demonstrations. It loses six practicality points because the minimum scientific split is 166 GB. It nevertheless wins because its control and sensor interface preserve the experiment.
- FurnitureBench has excellent duration and continuity. Missing language, 8D quaternion control, large storage, and simulator stack cost prevent it from being the primary language-conditioned benchmark.

The machine-readable scores are in [`benchmark_comparison.csv`](../analysis/benchmark_comparison.csv) and [`benchmark_compatibility.json`](../analysis/benchmark_compatibility.json).

## 17. Final Ranking

### Rank 1 — CALVIN

Best isolation of persistent semantic memory from action/control confounds. It gives four transitions in one environment episode while keeping MaIL's 7D and two-view structure.

### Rank 2 — RoboCasa365

Best annotations and full-instruction data, but the inspected target set supplies only one confirmed no-navigation 4–6-subtask task. It becomes Rank 1 only if a later action-statistics pilot identifies three suitable arm-only target composites.

### Rank 3 — FurnitureBench

Excellent long continuous assembly and useful skill flags, but official language is absent and MaIL must move to an 8D quaternion action target.

## 18. Primary Recommendation

Use **CALVIN D→D** as the first main benchmark and report the protocol as a **CALVIN-based persistent-chain experiment**.

The core experiment should first keep the official external stage instruction switch and compare:

1. MaIL finite-window baseline with the official per-stage policy reset;
2. the same policy interface with Mamba state retained across all five stages;
3. current-stage and sparse-transition conditioning under the same official oracle boundaries.

This is the cleanest answer to whether retained state helps across repeated semantic transitions. The single compound FULL condition should be a second, explicitly custom axis after determining whether matched continuous chain demonstrations can be mined or must be generated.

Biggest advantage: exact 7D/two-view control compatibility with four no-reset transitions.  
Biggest compromise: full-chain language and full-chain BC demonstrations are not official; the evaluator normally reveals each stage and resets the model.

MaIL remains a valid baseline: **YES — architecture-preserving adapter only.**

## 19. Candidate Tasks

CALVIN's official benchmark uses five-task chains rather than named composite tasks. The first pilot should freeze three valid chains, all with five genuine language-level goals. Exact initial-state/chain pairs must be passed through the official `check_sequence` validator after environment setup; the strings below are proposed fixed subsets, not claimed to be three named official composite tasks.

### Candidate 1 — controlled fixture chain

Stages: 5; expected difficulty: easy/controlled.

```text
turn on LED
→ move slider left
→ open drawer
→ rotate red block right
→ lift blue block from table
```

This chain exercises four progress updates without carrying an object across a stage boundary. It is the cleanest state-retention sanity check.

### Candidate 2 — drawer binding chain

Stages: 5; expected difficulty: medium.

```text
open drawer
→ lift blue block from table
→ place the grasped block in the drawer
→ move slider left
→ turn on LED
```

Stages two and three share object identity and require the drawer state established in stage one. It directly tests binding plus progress memory.

### Candidate 3 — delayed object dependency chain

Stages: 5; expected difficulty: hard.

```text
turn on lightbulb
→ move slider right
→ open drawer
→ lift pink block from table
→ place the grasped block in the drawer
```

The object-dependent pair occurs after three completed fixture goals, making it the best delayed-progress condition.

All action/camera compatibility is high: 7D relative control, static plus gripper RGB, and official oracle boundaries. Compound sentences made from these stages are **our** prompt condition. Details are in [`benchmark_candidate_tasks.csv`](../analysis/benchmark_candidate_tasks.csv).

## 20. Decision Tree

```text
Does RoboCasa365 provide three verified target composites with
4–6 genuine stages, per-frame labels, and base_motion≈0 throughout?
        ├── YES → use RoboCasa365 arm-only subset.
        └── NO  → current audit result
                    ↓
Does CALVIN provide a no-environment-reset five-stage chain with
7D Cartesian action, two RGB views, official success boundaries,
and a reasonable persistent-state evaluator modification?
        ├── YES → select CALVIN.  ← current decision
        └── NO
                    ↓
Can FurnitureBench's custom language ontology and 8D quaternion
head be accepted as part of the research protocol?
        ├── YES → use FurnitureBench.
        └── NO  → stop and audit another benchmark.
```

## 21. Risks

1. **Compound instruction/data mismatch.** CALVIN's official data trains task-level language segments, not the proposed five-clause FULL instruction. Do not compare FULL BC until matched targets are obtained.
2. **Policy reset semantics.** Removing `model.reset()` changes official evaluator behavior. Preserve an official-reset baseline and label the persistent version clearly.
3. **Boundary leakage.** If the task oracle switches prompts, transition timing is externally known. Separate “memory across transitions” from “internally discover transition under FULL.”
4. **Chain sampling.** The three proposed chains are valid transition templates, but their exact initial-state pairs must be validated and frozen. Do not silently replace failed chains.
5. **Dataset age/bugs.** CALVIN's repository documents historical annotation/scene fixes. The next task must use current checksums and corrected packages.
6. **Actual duration distribution.** Per-stage mean, median, and maximum duration are not available from metadata alone and require the selected data/rollout audit.
7. **Storage.** D→D is 166 GB before derived caches. Check `/ssd1` free space and reserve conversion overhead before download.

## 22. Next Migration Step

Only after human approval:

1. Check `/ssd1` capacity and download the 1.3 GB CALVIN debug set for a schema/evaluator smoke test.
2. Validate the three candidate chains using the official state-transition checker.
3. Confirm image/action alignment, language segment indices, corrected dataset version, and task-oracle behavior.
4. If the smoke gate passes, download the 166 GB D→D split to a new benchmark-specific path.
5. Implement a dataset/evaluator adapter while leaving the upstream MaIL repository intact.
6. Preserve the official-reset evaluation and add a separately named persistent-state evaluation.
7. Determine whether valid compound demonstrations can be mined before authorizing any new simulator data generation.
8. Only then, and only with separate approval, remove the exact raw LIBERO path `/ssd1/itaein/datasets/LIBERO/libero_10` while preserving all LIBERO annotations, analysis, docs, code, and MaIL source.

## Sources

Only official project sites, repositories, documentation, and papers were used for technical claims.

- CALVIN: [project/repository](https://github.com/mees/calvin), [dataset format](https://github.com/mees/calvin/blob/main/dataset/README.md), [official evaluator](https://github.com/mees/calvin/blob/main/calvin_models/calvin_agent/evaluation/evaluate_policy.py), [chain generator](https://github.com/mees/calvin/blob/main/calvin_models/calvin_agent/evaluation/multistep_sequences.py), [project site/leaderboard](https://calvin.cs.uni-freiburg.de/), [paper](https://arxiv.org/abs/2112.03227).
- RoboCasa365: [release page](https://robocasa.ai/), [dataset overview](https://robocasa.ai/docs/build/html/datasets/datasets_overview.html), [dataset use/format](https://github.com/robocasa/robocasa/blob/main/docs/datasets/using_datasets.md), [action conversion](https://github.com/robocasa/robocasa/blob/main/robocasa/utils/env_utils.py), [paper](https://robocasa.ai/assets/robocasa365_iclr26.pdf).
- FurnitureBench: [project site](https://clvrai.github.io/furniture-bench/), [dataset format and size](https://clvrai.github.io/furniture-bench/docs/tutorials/dataset.html), [real environment API](https://clvrai.github.io/furniture-bench/docs/tutorials/furniture_bench.html), [FurnitureSim API](https://clvrai.github.io/furniture-bench/docs/tutorials/furniture_sim.html), [paper](https://arxiv.org/abs/2305.12821).

## Stop attestation

- No large dataset was downloaded.
- No LIBERO data was deleted.
- No model was implemented or trained.
- No GPU job was submitted.
- No git push was performed.
- Waiting for human benchmark selection.
