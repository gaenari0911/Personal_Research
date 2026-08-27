# Task R0-D — Final VLABench 7D Multi-Stage Feasibility Audit

Audit date: 2026-08-24  
Final decision: **REJECT**

## 1. VLABench Repository

- path: `/home/itaein/Personal_Research/external/VLABench`
- upstream: `https://github.com/OpenMOSS/VLABench.git`
- branch: `main`
- commit: `cf588fe60c0c7282174fe979f5913170cfe69017`
- version: package `0.1`; commit date 2025-11-10; setup declares Python `>3.8`
  and README recommends Python 3.10; pinned requirements include MuJoCo 3.2.2,
  MuJoCo-MJX 3.2.2, dm_control 1.0.22, h5py 3.11.0, NumPy 1.25.0,
  Gym 0.26.2, Gymnasium 0.29.1, and LeRobot commit `6674e368…`

The clone is clean.  Evidence is pinned to the commit above, the official
[LeRobot environment page](https://huggingface.co/docs/lerobot/vlabench), and the
[official composite dataset](https://huggingface.co/datasets/VLABench/vlabench_composite_ft_lerobot_video).

## 2. Composite Task Audit

- total composite tasks: 22 on the official LeRobot task surface; 36 exact
  upstream composite registrations; 41 names in their union because five
  LeRobot names are non-matching aliases
- 4-stage: 6 (`cluster_billiards`, `cluster_book`, `cluster_drink`,
  `cluster_toy`, `heat_food`, `take_chemistry_experiment`)
- 5-stage: 0
- 6-stage: 0
- qualified 4–6 stage: **0**

The CSV audits all 41 union names; source-name aliases are retained separately
so the API mismatch is visible rather than silently normalized. The four
surface cluster tasks are four semantic transfers, but the transfers are
order-insensitive and have LOW dependency.  `heat_food` and
`take_chemistry_experiment` have strong ordered semantics but are absent from
the released composite dataset.  Five LeRobot task names also fail exact
upstream registration (`hammer_nail`, `play_poker`, `rearrange_book`,
`rearrange_chemistry_tube`, `use_seesaw_complex`), and several registered tasks
have no implemented expert sequence.

## 3. Semantic Stage Definition

- official source: task success conditions and ordered predicates in the pinned
  task implementations
- expert skill source: `get_expert_skill_sequence()` and
  `scripts/trajectory_generation.py`
- semantic grouping rule: merge reach/grasp/lift/move/release for one object
  transfer into one language-describable world-state transition; retain opening,
  closing, activation, pouring, and returning as separate stages; exclude wait,
  reset, camera motion, and raw waypoints

This rule prevents counting motor primitives as fake semantic stages.

## 4. Demonstration Repetition

- dataset: `VLABench/vlabench_composite_ft_lerobot_video`
- total episodes: 5,977; total frames: 2,539,771; 10 Hz; 20 GB
- task-specific repetitions: the generator targets roughly 500 successful
  episodes per source task.  A metadata-only stratified audit of the first 500
  episode parquets found 43–51 instances for each ranked cluster family.  The
  released repository does not publish an exact post-filter count by source
  family, so no stronger exact per-task claim is made.
- CALVIN singleton issue present?: no; these are repeated demonstrations, not a
  single long language chain

The 167 `task_index` strings collapse to 12 source families in the actual
release, largely because painting and math arguments create many textual
variants.  The published total should not be confused with 167 distinct
composite task implementations.

## 5. Semantic Boundary Feasibility

- official per-frame boundary: **absent**
- expert skill boundary: ordered in source during generation, but neither skill
  index nor transition frame is serialized
- simulator predicate: available in source for some tasks (notably `heat_food`
  and chemistry), but per-frame simulator state and episode configuration are
  absent from the released data
- manual labeling requirement: cluster boundaries would require manual labels
  or a new offline reconstruction/generation pipeline; gripper transitions are
  incomplete and not semantically authoritative
- verdict: **FAIL — no candidate has released level 1–3 boundaries**

If trusted boundaries existed, CURRENT and HOLD timelines would be a
deterministic transformation of the ordered stage IDs. They do not currently:
creating CURRENT/HOLD from gripper toggles would promote a level-4 heuristic to
ground truth and would invalidate the experiment.

Schema inspection found only images, EEF/joint observations, action,
`metadata.filepath`, timestamp/frame/episode/index, and `task_index`.  There is
no predicate value hidden in the 21-frame images; the overlaid statement that a
boundary is unavailable is an audit warning, not a field from the dataset.

## 6. Action Schema

- dataset action dimension: 7
- environment action dimension: 9 after the wrapper
- translation: absolute target `[x,y,z]`
- rotation: absolute extrinsic Euler `xyz`, radians
- gripper: binary, `1=open`, `0=closed`
- absolute/relative/delta: absolute EEF target; the wrapper's nominal
  `delta_eef` mode still follows the absolute conversion path
- coordinate frame: Cartesian position relative to the robot base

The raw dm_control task receives seven arm joint-position targets plus two
finger targets.  The 7D policy interface is converted to that 9D actuator vector
with the official inverse-kinematics path.

## 7. Action Representation Consistency

- dataset representation: 7D absolute robot-frame EEF target
- environment representation: 9D joint/finger position target
- official transform: Euler → rotation/quaternion → IK arm joints; binary
  gripper → two finger targets
- replay result: transform verified in source; exact released-episode replay is
  not reproducible because initial physics/config/assets are not in the dataset
- known mismatch: declared Euler action bounds are `[-1,1]`, while sampled
  official data reaches roll/yaw ±π and pitch ±π/2
- final verdict: **ADAPTER_REQUIRED / EXACT REPLAY BLOCKED**

## 8. MaIL Action Compatibility

- MaIL: 7D LIBERO-style relative/delta EEF action
- VLABench: 7D absolute robot-frame pose target plus binary gripper
- output head change: none; remains 7D
- adapter required: yes—train-split normalization, output denormalization,
  absolute-target interpretation, and official IK wrapper
- verdict: width-compatible but **not representation-identical**

Do not integrate the VLABench output as a delta and do not clip its Euler
components to the wrapper's incorrect declared Box.

## 9. Observation / Action Alignment

- observation index: pre-action `observation[t]`
- target action: waypoint target `action[t]`
- source evidence: skill builders prepend the initial observation and remove the
  final post-action observation around `SkillLib.step_trajectory`
- replay evidence: numeric sample is consistent with a target—the action is
  closer to `obs[t+1]` than `obs[t]`; exact environment replay remains blocked

Sample medians: XYZ 0.00686 versus current and 0.00348 versus next; Euler
0.00649 versus current and 0.00538 versus next.

## 10. Camera

- available: `observation.images.image`, `second_image`, `wrist_image`; all
  224×224, AV1/yuv420p, 10 Hz
- recommended 2-view: `image` (broad external/front) + `wrist_image`
- resize: 224×224 → 128×128
- 3-view needed: no for the ranked cluster tasks; retain `second_image` only for
  optional QA
- MaIL impact: two CLIP encoders remain unchanged

## 11. Candidate A

- task: `cluster_billiards`
- full instruction: “Cluster the objects into two classes.”
- semantic stages:
  - S1: place billiards ball A in its category container
  - S2: place billiards ball B in its category container
  - S3: place billiards ball C in the other category container
  - S4: place billiards ball D in the other category container
- stages: 4
- demos: 48 in the audited first-500 sample; release design is roughly 500 per
  source family, exact final family count unpublished
- episode duration: representative episode 0, 610 frames / 61.0 s
- dependency: LOW; placements share a final goal but do not causally require the
  preceding placement
- boundary source: level 5 only in released data; gripper proxy is not accepted
- action compatibility: 7D head retained, absolute-action adapter required

## 12. Candidate B

- task: `cluster_drink`
- full instruction: “Cluster the objects into two classes.”
- semantic stages:
  - S1: place drink A in its category container
  - S2: place drink B in its category container
  - S3: place drink C in the other category container
  - S4: place drink D in the other category container
- stages: 4
- demos: 51 in the audited first-500 sample
- episode duration: representative episode 8, 545 frames / 54.5 s
- dependency: LOW
- boundary source: level 5; only three close-to-reopen spans are visible before
  terminal state, demonstrating why gripper-only segmentation is unreliable
- action compatibility: 7D head retained, absolute-action adapter required

## 13. Candidate C

- task: `cluster_toy`
- full instruction: “Cluster the objects into two classes.”
- semantic stages:
  - S1: place toy A in its category container
  - S2: place toy B in its category container
  - S3: place toy C in the other category container
  - S4: place toy D in the other category container
- stages: 4
- demos: 43 in the audited first-500 sample
- episode duration: representative episode 30, 634 frames / 63.4 s
- dependency: LOW
- boundary source: level 5; only three reopenings before termination in the
  representative trajectory
- action compatibility: 7D head retained, absolute-action adapter required

## 14. Long-Horizon Statistics

- stage durations: episode 0 gripper-proxy transfer spans are 13.9, 18.6, 13.8,
  14.4 s; episode 8 has observable spans 17.1, 12.8, 13.0 s; episode 30 has
  15.9, 16.7, 14.4 s
- transitions: four semantic transfers intended, but transition-only heuristics
  miss the terminal transfer in two of three reviewed episodes
- `steps_since_transition` potential: computable only after a trusted boundary
  source is added; currently unavailable without level-4/5 inference
- `cumulative_transition_count`: same limitation; gripper toggles must not be
  substituted for semantic completion

At 10 Hz, MaIL's 5-frame observation window covers 0.5 s and its 10-action chunk
covers 1.0 s—much shorter than a typical transfer stage.

Temporal horizon is therefore adequate for testing `steps_since_transition`,
and the intended four transfers would provide three to four increments of
`cumulative_transition_count`. Mamba suitability fails at the supervision
layer, not because episodes are too short: neither quantity is reproducibly
computable from the release.

## 15. Human Review

- videos generated: three H.264 two-view MP4s and three 21-frame PNG sheets in
  `analysis/vlabench_review/`
- stage continuity: all reviewed episodes are visually continuous with no
  environment reset or cut between transfers
- expected final review workload: six artifacts for decision QA; a production
  boundary dataset would still require episode-scale annotation or new
  instrumented generation and is therefore outside the allowed workload

## 16. Language

- FULL source: official per-episode `task_index` mapped to the full task string
- subinstruction source: no official natural-language per-stage field; one could
  generate deterministic object/container templates only from expert arguments
- CLIP token count: ranked full instruction is 9 tokens including start/end
  tokens with the installed OpenAI CLIP tokenizer
- artificial text required: no for FULL; **yes for stage-level subinstructions**

## 17. MaIL Preservation

- UNCHANGED: 7D output width, two 512D CLIP visual encoders, language path,
  Mamba d_model=128/nominal 16 layers, obs_seq=5, horizon=10, BC/MSE
- ADAPTER_ONLY: camera key mapping/resize, absolute-action statistics,
  denormalization, robot-frame pose/IK rollout, explicit alignment
- MODIFIED: action meaning changes from relative/delta to absolute target; no
  core neural architecture change
- overall baseline identity: model identity is substantially preserved, but the
  experiment's required semantic-stage supervision is not supplied

| component | locked value | status |
|---|---:|---|
| action head | 7 | UNCHANGED width / ADAPTER_ONLY meaning |
| vision encoders | two CLIP encoders, 512D each | UNCHANGED |
| language encoder/path | CLIP 512D task embedding | UNCHANGED |
| Mamba core | decoder-only selective SSM | UNCHANGED |
| d_model | 128 | UNCHANGED |
| layers | 16 published override | UNCHANGED |
| d_state | 16 | UNCHANGED |
| d_conv | 4 | UNCHANGED |
| expand | 2 | UNCHANGED |
| obs_seq | 5 | UNCHANGED |
| action horizon | 10 | UNCHANGED |
| loss | BC/MSE | UNCHANGED |

## 18. Dataset Download Strategy

- selected-task download possible: only through custom episode-file filtering
  after scanning parquet metadata; the official repository is pooled and has no
  simple task-specific package flag
- expected download: 20 GB for the full composite LeRobot dataset; simulator
  asset size is not published
- target path: `/ssd1/itaein/datasets/VLABench/`
- full dataset needed?: no download is justified after REJECT; metadata and the
  three tiny representative derivatives are sufficient for this audit

## 19. Storage / Cleanup Preparation

- `/ssd1` free: 513,727,045,632 bytes (478.46 GiB)
- LIBERO raw: 13,730,613,000 bytes at `LIBERO/libero_10`; preserved
- CALVIN debug: 2,615,026,055 bytes at `CALVIN/debug`; preserved
- expected VLABench: 20 GB dataset plus an unpublished asset footprint
- migration safe: capacity is sufficient, but migration is **not authorized or
  recommended** under the rejected gate

## 20. Generated Files

- docs: `docs/VLABENCH_FINAL_AUDIT.md`,
  `docs/VLABENCH_MAIL_COMPATIBILITY.md`
- analysis: `analysis/vlabench_composite_task_audit.csv`,
  `vlabench_semantic_stage_audit.json`, `vlabench_action_semantics_audit.json`,
  `vlabench_replay_audit.json`, `vlabench_camera_audit.json`,
  `vlabench_candidate_rankings.csv`, `vlabench_migration_gate.json`
- review: three H.264 MP4s, three 21-frame PNGs, and README under
  `analysis/vlabench_review/`

## 21. FINAL DECISION

**REJECT**

Reason: VLABench is technically consumable by a 7D MaIL head, has repeated
continuous demonstrations and rollout success conditions, but it does not
provide three data-backed, genuinely dependent 4–6-stage tasks with
reproducible level 1–3 boundaries.  The released data has no boundary/predicate
trace, the only data-backed four-stage candidates are low-dependency cluster
placements, and the stronger ordered tasks are absent from the release.  Exact
episode replay is also impossible from the published fields.

Full-task rollout success is available through `should_terminate_episode`.
Training/rollout camera and action semantics can be matched with the official
wrapper, but stage/transition completion cannot: training has no boundary trace
and rollout exposes only task-specific final conditions or coarse progress for
most candidates. This is a direct training/evaluation consistency failure for
the requested semantic-retention protocol.

## 22. If PASS

Not applicable.  No VLABench download, deletion, environment setup, adapter
implementation, smoke test, training, or GPU submission is authorized by this
audit.

## 23. If REJECT

**DO NOT SEARCH ANOTHER BENCHMARK.**

Recommend exactly one option: **C. mixed benchmark design**.

Keep LIBERO as the stable 7D MaIL training/regression baseline, and use the
already-audited RoboCasa `WashFruitColander` subset as the multi-stage
evaluation/pilot component.  This preserves the current baseline while placing
semantic transition analysis in a task where state-based stage reconstruction
has already been demonstrated, without opening another benchmark search.

## 24. STOP

No large dataset was downloaded.  
No LIBERO raw data was deleted.  
No CALVIN debug data was deleted.  
No model was trained.  
No GPU job was submitted.  
Waiting for human decision.
