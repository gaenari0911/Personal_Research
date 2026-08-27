# CALVIN Small Validation Before Final Migration

Audit date: 2026-08-24 (Asia/Seoul)  
Final decision: **REJECT CALVIN as the primary benchmark for the matched supervised 4–6-stage experiment**

## 1. Executive decision

The R0-A recommendation does not survive the small-data training-chain gate. CALVIN remains an excellent **evaluation benchmark** and is a near match to MaIL's action and camera interface, but its D→D play annotations do not supply repeated, matched five-stage supervised demonstrations for the proposed `FULL / current-stage / HOLD-transition` comparison.

The critical observation is based on the full D→D language metadata, not just the 1.3 GB debug sample:

- after merging overlapping annotations of the same task into one semantic event, the defensible non-overlap, at-most-one-second rule yields **9 five-stage training chains, 9 unique patterns, and 0 five-stage validation chains**;
- no five-stage pattern repeats even twice under that rule;
- the relaxed same-recording-episode rule yields 3,135 five-event training windows, but 3,133 unique patterns and unlabeled gaps up to 901 frames (30.0 s);
- turning those gaps into HOLD/masked-loss supervision would introduce a new, unvalidated training protocol central to the claim.

Gate E (training supervision) therefore fails. The result is **REJECT**, not PASS or CONDITIONAL. The next audit target is RoboCasa365's arm-only target-composite subset; no RoboCasa data was downloaded.

## 2. Work performed and safety boundary

Completed:

- cloned the official CALVIN repository and its submodules;
- downloaded only the official debug dataset;
- verified its official SHA-256 and extracted it;
- inspected every debug NPZ row and all debug language annotations;
- fetched only the ZIP64 central directory and seven metadata entries from the full D→D archive using HTTP byte ranges;
- verified every extracted metadata entry against its ZIP CRC32;
- audited MaIL compatibility, observation/action alignment, language-chain scale, compound CLIP token counts, evaluator resets, and storage;
- generated five review segments as 21-frame contact sheets and H.264 MP4s.

Not performed:

- no full D→D, ABC→D, or ABCD→D download;
- no model training, GPU job, scheduler submission, or simulator rollout;
- no Mamba implementation and no MaIL/CALVIN upstream source edit;
- no LIBERO deletion or relocation;
- no Git push.

## 3. Official repository provenance

| Field | Value |
|---|---|
| Repository | `https://github.com/mees/calvin.git` |
| Local path | `/home/itaein/Personal_Research/external/calvin` |
| Branch | `main` |
| Commit | `fa03f01f19c65920e18cf37398a9ce859274af76` |
| Commit date | `2025-09-08T10:51:51+02:00` |
| Commit subject | `add FLOWER paper` |
| Package version | `Calvin 0.0.1` |
| `calvin_env` submodule | `1431a46bd36bde5903fb6345e68b5ccc30def666` |
| Inspection date | 2026-08-24 |

The repository has no release tag at the checked-out HEAD, so the exact commit is the reproducibility anchor. The package version comes from [`calvin_agent/__init__.py`](../external/calvin/calvin_models/calvin_agent/__init__.py).

## 4. Runtime and environment assessment

The official quick start requests Python 3.8. The checked source pins or names:

- `torch==1.13.1`;
- `pytorch-lightning==1.8.6`;
- `hydra-core==1.1.1`;
- `setuptools==57.5.0`;
- CALVIN environment dependencies including PyBullet, NumPy quaternion, OpenCV, SciPy, and Gym.

None of the existing Conda environments contains a complete CALVIN + MaIL runtime. Reusing `base`, `py3.8.12`, or an EgoDex environment would mix incompatible old robotics dependencies with unrelated packages. The recommendation is a separate **`calvin-mail`** environment, initially based on Python 3.8 and the official pins, followed by an explicit compatibility test with MaIL. It was not created during this audit.

## 5. Downloaded dataset and integrity

| Item | Value |
|---|---|
| Official URL | `http://calvin.cs.uni-freiburg.de/dataset/calvin_debug_dataset.zip` |
| Archive path | `/ssd1/itaein/datasets/CALVIN/debug/calvin_debug_dataset.zip` |
| Extracted root | `/ssd1/itaein/datasets/CALVIN/debug/calvin_debug_dataset` |
| Archive bytes | 1,299,150,917 |
| Official SHA-256 | `c66d09147e2c806b244f18ea7d61e388d4dac11f828929779437f728d03e1204` |
| Observed SHA-256 | same |
| Checksum result | **PASS** |
| Current debug tree | 2,615,026,055 bytes, including archive, extraction, and metadata-only copy |

The official download sizes and checksum procedure are documented in [`dataset/README.md`](../external/calvin/dataset/README.md:5).

## 6. Full D→D metadata-only acquisition

The server supports byte ranges. The 177,379,436,142-byte D→D ZIP was **not** downloaded. Its 72,747,586-byte ZIP64 central directory was inspected, and only these entries were extracted:

| Split | Entry | Raw bytes | CRC32 |
|---|---|---:|---|
| training | `ep_start_end_ids.npy` | 624 | `ab2a4aca` |
| training | `ep_lens.npy` | 376 | `30f4e134` |
| training | `scene_info.npy` | 408 | `8ec10886` |
| training | `lang_annotations/auto_lang_ann.npy` | 8,108,180 | `c39066c2` |
| validation | `ep_start_end_ids.npy` | 192 | `70990dfc` |
| validation | `ep_lens.npy` | 160 | `80bc28d9` |
| validation | `lang_annotations/auto_lang_ann.npy` | 1,617,294 | `8efe0c8d` |

Each entry was raw-deflate decompressed and checked against its central-directory size and CRC32. The result is stored under `/ssd1/itaein/datasets/CALVIN/debug/metadata_only`. This is sufficient for the full annotation-chain audit, but not for training or full-frame visual verification.

## 7. Actual debug split structure

| Property | Training | Validation |
|---|---:|---:|
| NPZ frame files | 2,771 | 1,675 |
| Frame index range | 358482–361252 | 553567–555241 |
| Missing frame indices | 0 | 0 |
| Physical episodes | 1 | 1 |
| `ep_start_end_ids.npy` | `[358482, 361252]` | `[553567, 555241]` |
| Language segments | 9 | 8 |
| Distinct task IDs | 7 | 6 |
| Default embedding | `[N,1,384]`, float32 | `[N,1,384]`, float32 |

The episode bounds are inclusive: `end - start + 1 == ep_lens`. There is no explicit timestamp field. The checked Hydra config declares `control_freq: 30`, so consecutive file indices are interpreted as 30 Hz ticks.

## 8. Actual NPZ schema

Every inspected debug frame has the same keys:

| Key | Actual shape | Actual dtype | Use in proposed adapter |
|---|---:|---|---|
| `actions` | `[7]` | float64 | audit/reference only |
| `rel_actions` | `[7]` | float64 | policy target, cast to float32 |
| `robot_obs` | `[15]` | float64 | alignment/debug metadata |
| `scene_obs` | `[24]` | float64 | continuity/debug metadata |
| `rgb_static` | `[200,200,3]` | uint8 | map to `agentview_rgb` |
| `rgb_gripper` | `[84,84,3]` | uint8 | map to `eye_in_hand_rgb` |
| `rgb_tactile` | `[160,120,6]` | uint8 | omit |
| `depth_static` | `[200,200]` | float32 | omit |
| `depth_gripper` | `[84,84]` | float32 | omit |
| `depth_tactile` | `[160,120,2]` | float32 | omit |

The official README describes action and state arrays as float32, but the actual 2022 debug package stores them as float64. This is a harmless adapter cast, but it is recorded rather than silently assuming the documentation dtype.

## 9. Action semantics and empirical distribution

Official source constructs `actions[t]` from the desired TCP state in the next recorded frame, then computes:

```text
rel_xyz = clip(actions[:3] - robot_obs[:3], ±0.02) / 0.02
rel_euler = clip(wrap(actions[3:6] - robot_obs[3:6]), ±0.05) / 0.05
gripper = actions[6] in {-1,+1}
```

See [`datarenderer.py`](../external/calvin/calvin_env/calvin_env/datarenderer.py:244) and [`utils.py`](../external/calvin/calvin_env/calvin_env/utils/utils.py:160). Recomputing this formula over all 4,446 debug rows produced a maximum absolute error of exactly **0.0**.

Empirical `rel_actions` summary:

| Split | Motion-dimension mean range | Motion-dimension std range | Gripper open `+1` | Gripper close `-1` | Gripper switches |
|---|---:|---:|---:|---:|---:|
| training | -0.00272 to 0.00059 | 0.1397 to 0.2822 | 59.40% | 40.60% | 66 |
| validation | -0.00384 to 0.00802 | 0.1672 to 0.2895 | 58.45% | 41.55% | 42 |

Motion saturation is low. The largest per-dimension rate of `abs(action) >= 0.999` is 0.108% in training and 1.313% in validation. Both gripper values occur in both splits.

## 10. Observation/action alignment

The source-level rendering loop explicitly says “action is robot state of next frame,” buffers the current images/state, and saves them with the following target state. The causal policy pair is therefore:

```text
input:  rgb_static[t], rgb_gripper[t]
target: rel_actions[t]
shift:  0
```

No arbitrary one-step shift is needed. `rel_actions[t]` is exactly derived from `robot_obs[t]` and `actions[t]`. `actions[t][:6]` and observed `robot_obs[t+1][:6]` differ by at most 0.00385 in the debug data because the stored desired target and realized next state are not mathematically identical; that does not change the target alignment.

For MaIL's five-observation/ten-action window, observations `t-4...t` predict action targets starting at row `t`, exactly as its current slicing does. This differs from the earlier LIBERO post-action-image caveat and must not reuse LIBERO's +1 shift.

## 11. MaIL compatibility verdict

Overall compatibility: **NEAR**.

| Interface | Verdict | Required adaptation |
|---|---|---|
| Action width | EXACT | none; both are 7D |
| Action semantics | NEAR | use CALVIN `rel_actions`, cast float64→float32, fit MaIL's z-score scaler on CALVIN train |
| Camera count/roles | EXACT | static + wrist/gripper |
| Camera tensors | NEAR | RGB-preserving resize to 128×128, HWC→CHW, `/255` |
| Language | NEAR | re-encode raw text with CLIP ViT-B/32; default CALVIN embeddings are 384D MiniLM, while MaIL expects 512D |
| Temporal step counts | EXACT | retain 5 observations and 10 target actions |
| Physical time | NEAR | CALVIN runs faster, so the same step counts cover less real time |

MaIL's checked configuration uses `action_dim=7`, `obs_seq=5`, and `train_action_seq=10` in [`benchmark_libero10.yaml`](../external/MaIL/config/benchmark_libero10.yaml:170); its two 128×128 RGB roles are declared in [`goal_bc_mamba_dec.yaml`](../external/MaIL/config/agents/goal_bc_mamba_dec.yaml:40).

## 12. Camera resize and qualitative audit

Selected adapter:

```text
rgb_static  [T,200,200,3] uint8 -> bilinear resize -> [T,3,128,128] float32 -> agentview_rgb
rgb_gripper [T, 84, 84,3] uint8 -> bilinear resize -> [T,3,128,128] float32 -> eye_in_hand_rgb
```

No crop is used. The source arrays and generated montages confirm RGB channel order: red/pink/blue blocks retain their expected colors. The static view provides the whole work surface and fixture state; the gripper view provides close contact/object evidence but can temporarily lose the target.

Representative review files are in [`analysis/calvin_review`](../analysis/calvin_review). The MP4s are H.264, `yuv420p`, and fast-start encoded for VS Code/browser compatibility; the JPEG contact sheets remain the fallback when video playback is unavailable.

Human review should focus on:

| Segment | Primary evidence to inspect |
|---|---|
| `turn_on_lightbulb [553636,553700]` | robot contacts the switch; yellow bulb changes from off to visibly on in `rgb_static` |
| `lift_blue_block_slider [553691,553755]` | blue block leaves the slider surface and becomes held; check both views |
| `lift_red_block_table [554046,554110]` | red block leaves table contact and follows the gripper |
| `place_in_slider [554110,554145]` | held block enters the slider and the gripper releases |
| `push_pink_block_right [359714,359757]` | pink block translates right while remaining supported rather than being lifted |

## 13. Language annotations

Actual fields are:

```text
language.ann   raw utterance list
language.task  task-ID list
language.emb   [N,1,384] float32 default MiniLM embedding
info.indx      [start,end] integer intervals
info.episodes  empty list in inspected files
```

Representative debug examples:

| Task | Utterance | `[start,end]` | Inclusive frames | Seconds |
|---|---|---:|---:|---:|
| `turn_on_lightbulb` | “turn on the light bulb” | `[553636,553700]` | 65 | 2.167 |
| `lift_blue_block_slider` | “in the slider pick up the blue block” | `[553693,553757]` | 65 | 2.167 |
| `lift_red_block_table` | “lift the red block from the table” | `[554046,554110]` | 65 | 2.167 |
| `place_in_slider` | “put it in the slider” | `[554110,554145]` | 36 | 1.200 |
| `push_pink_block_right` | “sweep the pink block to the right” | `[359714,359757]` | 44 | 1.467 |

The full D→D metadata contains 5,124 train and 1,011 validation annotations across all 34 task IDs.

## 14. Continuity and physical reset boundaries

`ep_start_end_ids.npy`, not a filename gap or language gap, is treated as the physical episode/reset source. Official rendering code derives these inclusive bounds from recorded `done` events in [`datarenderer.py`](../external/calvin/calvin_env/calvin_env/datarenderer.py:110).

Full D→D metadata:

| Split | Physical episodes | Episode length min / median / max |
|---|---:|---:|
| training | 31 | 1,675 / 18,374 / 30,839 frames |
| validation | 4 | 16,136 / 22,601.5 / 37,683 frames |

Language annotations can overlap, touch, or be separated by unlabeled play while still belonging to one physical episode. Those cases are not interchangeable and are separated in the chain audit.

## 15. Training-chain result

The full analysis is in [`CALVIN_TRAINING_CHAIN_AUDIT.md`](CALVIN_TRAINING_CHAIN_AUDIT.md). The decisive counts are:

| Chain definition | Train 5-stage occurrences / patterns | Validation 5-stage occurrences / patterns | Interpretation |
|---|---:|---:|---|
| A: `next.start <= prev.end+1` | 7 / 7 | 1 / 1 | overlaps allowed; not reliable sequential supervision |
| B: non-overlap, gap ≤30 frames | **9 / 9** | **0 / 0** | most defensible short-gap evidence; far too small |
| C: same physical episode | 3,135 / 3,133 | 590 / 590 | large count, but almost every pattern is a singleton with unlabeled gaps |

Under the requested scale rubric, every five-stage pattern is **very weak** (`<20`), and no pattern reaches exploratory scale.

## 16. Full-play training alternative and HOLD risk

A C-based curriculum could retain full play observations and define five consecutive annotated events as a compound target. It would have to specify all of the following:

1. what semantic input is active after one offline annotation ends and before the next begins;
2. whether BC loss is computed on unlabeled-gap actions;
3. whether the recurrent state updates during masked intervals;
4. how overlapping annotations are resolved;
5. how train/validation splits avoid leakage among heavily overlapping chain windows;
6. how offline `[start,end]` boundaries are calibrated against first task-oracle success.

For Method C, emitting HOLD through a gap is syntactically possible. For the current-stage method, a gap has no official “current subinstruction.” Masking its loss while still advancing state is a new training contract; dropping its observations destroys the claimed episode continuity. This is not a minor adapter detail.

## 17. Compound instruction feasibility

Official utterances were deterministically combined by stripping terminal periods, joining stages with `; then `, and adding one final period. The exact MaIL encoder tokenizer, `openai/clip-vit-base-patch32`, reports a 77-token limit including start/end tokens.

Across 7,573 four- and five-stage candidate occurrences:

| Statistic | Tokens |
|---|---:|
| minimum | 24 |
| mean | 44.81 |
| median | 45 |
| p75 / p90 / p95 | 50 / 54 / 56 |
| maximum | 68 |
| over 77 | **0** |

For the nine B-definition training five-stage chains, the range is 44–55 tokens with mean/median 49. Compound length is therefore not the blocker.

## 18. Evaluator reset and success boundary

Official evaluation is **CASE B**:

- `env.reset(...)` occurs once before the five-task sequence at [`evaluate_policy.py:129`](../external/calvin/calvin_models/calvin_agent/evaluation/evaluate_policy.py:129);
- no environment reset occurs in the subtask loop;
- `model.reset()` occurs inside every subtask rollout at [`evaluate_policy.py:157`](../external/calvin/calvin_models/calvin_agent/evaluation/evaluate_policy.py:157);
- success is checked after every action with the task oracle at [`evaluate_policy.py:160`](../external/calvin/calvin_models/calvin_agent/evaluation/evaluate_policy.py:160).

For a persistent-state comparison, move policy reset next to the one sequence-level environment reset and remove/bypass the per-subtask reset. Preserve the task oracle, initial states, task order, and 360-step stage cap. This is an evaluator modification and must be reported as such.

Offline `[start,end]` and online oracle success are not identical boundaries. The offline end is a mined clip boundary. Online evaluation switches immediately at the first detected success. An offline transition label therefore needs a measured calibration offset before it can be treated as the same transition target.

## 19. Temporal comparison with LIBERO

| Quantity | Steps | CALVIN 30 Hz | LIBERO 20 Hz |
|---|---:|---:|---:|
| Observation window | 5 | 0.167 s | 0.250 s |
| Action horizon | 10 | 0.333 s | 0.500 s |
| CALVIN stage cap | 360 | 12.0 s | — |

Keeping MaIL's step counts is the cleanest architecture-preserving comparison, but the physical context and prediction horizon are two-thirds as long in CALVIN. Equalizing wall-clock duration would require roughly 8 observations and 15 actions and would no longer preserve the released MaIL sequence lengths.

## 20. Human verification burden

| Scope | Estimate | Purpose |
|---|---:|---|
| Five supplied debug segments | 15–25 min | confirm RGB, task visibility, and broad boundary plausibility |
| All B-definition 4/5-stage train candidates (40 windows) | 45–75 min | verify short-gap chronological chains and reject annotation collisions |
| All 3,259 deduplicated train events | roughly 45–80 h at 50–90 s/event | exhaustive event/boundary validation |
| Stratified 5% event audit (about 163 events) | 2.5–4 h | estimate annotation precision, not certify every chain |

Human review cannot create repeated chain patterns that are absent from the metadata. It can only validate the few mined candidates or quantify annotation error.

## 21. Storage audit

Observed after debug extraction:

| Item | Bytes |
|---|---:|
| `/ssd1` available | 513,727,045,632 |
| Existing LIBERO-10 | 13,730,613,000 |
| Current CALVIN debug tree | 2,615,026,055 |
| Full D→D archive | 177,379,436,142 |
| Full D→D extracted total from ZIP central directory | 178,284,517,323 |
| Archive + extraction | 355,663,953,465 |
| Estimated remaining after both | 158,063,092,167 |

The D→D archive plus extraction fits. A second full-size converted copy while retaining the ZIP does not fit with a comfortable safety margin. If CALVIN were pursued for another purpose, use a streaming adapter or an in-RAM cache. Removing or relocating the verified archive would require separate explicit approval.

Recommended future full-data root would be `/ssd1/itaein/datasets/CALVIN/task_D_D`. No symlink is necessary because `/ssd1` is the intended storage filesystem.

## 22. Adapter contract

If CALVIN is retained as a secondary evaluation benchmark, the minimum contract is:

```text
input keys:
  rgb_static   -> agentview_rgb
  rgb_gripper  -> eye_in_hand_rgb
target key:
  rel_actions

image transform:
  RGB HWC uint8 -> bilinear 128x128 -> CHW float32 / 255
action transform:
  float64 -> float32 -> training-split z-score
alignment:
  obs[t] -> rel_actions[t] (zero shift)
language:
  raw text -> openai/clip-vit-base-patch32 -> 512D
episode boundary:
  ep_start_end_ids.npy only
```

## 23. Acceptance gates

| Gate | Result | Evidence |
|---|---|---|
| A. Small download/schema | PASS | official checksum matched; all NPZ rows load |
| B. Action/camera compatibility | PASS | 7D action, two RGB views, deterministic NEAR adapter |
| C. Language/continuity | PASS | raw text, task IDs, intervals, and physical episode bounds exist |
| D. Long-horizon structure | PASS with qualification | same-episode five-event windows exist, but contain gaps and singleton patterns |
| **E. Training supervision** | **FAIL — critical** | only 9 unique B-definition train five-stage chains; 0 validation; no repeated five-stage pattern |
| F. Evaluation reproducibility | PASS | official 1,000×5 evaluator and task oracle; reset edit localized |
| G. Storage | PASS with constraint | archive + extraction fit; avoid a second full copy |

## 24. Final verdict and next benchmark

**REJECT CALVIN as the primary benchmark for this experiment.**

This is not a rejection of CALVIN generally. It is a rejection of using CALVIN to claim a matched supervised comparison across full compound, current-stage, and transition/HOLD conditioning without adding a new data-construction assumption. The action/camera interface and evaluator remain suitable for a secondary compositional-evaluation study.

Next recommendation: audit **RoboCasa365 arm-only target composite tasks**, metadata first. Before any download, require at least three official 4–6-stage target tasks whose action records show `base_motion=0` and constant arm-active `control_mode`. If that gate fails, return to benchmark search rather than silently accepting mobile-control confounding.

## 25. Generated evidence

- [`analysis/calvin_schema_audit.json`](../analysis/calvin_schema_audit.json)
- [`analysis/calvin_action_camera_compatibility.json`](../analysis/calvin_action_camera_compatibility.json)
- [`analysis/calvin_evaluator_reset_audit.json`](../analysis/calvin_evaluator_reset_audit.json)
- [`analysis/calvin_language_segment_gaps.csv`](../analysis/calvin_language_segment_gaps.csv)
- [`analysis/calvin_chain_patterns.csv`](../analysis/calvin_chain_patterns.csv)
- [`analysis/calvin_chain_statistics.json`](../analysis/calvin_chain_statistics.json)
- [`analysis/calvin_compound_instruction_tokens.csv`](../analysis/calvin_compound_instruction_tokens.csv)
- [`analysis/calvin_migration_gate.json`](../analysis/calvin_migration_gate.json)
- [`analysis/calvin_review/review_manifest.csv`](../analysis/calvin_review/review_manifest.csv)
- [`analysis/run_calvin_small_validation.py`](../analysis/run_calvin_small_validation.py)

