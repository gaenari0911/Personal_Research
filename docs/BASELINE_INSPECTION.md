# Task 1 — MaIL Baseline Inspection and Minimal Reproducibility

This document records read-only inspection results for Task 1. No MaIL or LIBERO source, model, dataset, configuration, loss, or simulation code was modified. No dataset was downloaded and no training or rollout was run.

## 1. Baseline Selection

The primary upstream baseline is **ALRhub/MaIL**, using its decoder-only Mamba behavior-cloning path.

Two closely related paths matter:

- `bc_mamba_dec`: the simplest decoder-only Mamba BC implementation, but it has **no language input**.
- `goal_bc_mamba_dec`: the same general BC/Mamba structure with a precomputed CLIP task embedding prepended to the visual sequence. This is the more relevant Phase-1 starting point because the experiment requires instruction conditioning.

The architecture can support the intended comparison while leaving the Mamba backbone unchanged, but the released code is not ready to run the three methods as-is. Language must be made timestep-aligned, several upstream path/config defects must be repaired, and task-subset evaluation needs a small selection interface. The overall feasibility conclusion is **B: feasible with small-to-moderate data/config/policy-interface changes, not usable unchanged**.

## 2. Exact Repository Revision

### MaIL

- Repository: `https://github.com/ALRhub/MaIL.git`
- Local path: `/home/itaein/Personal_Research/external/MaIL`
- Origin fetch/push: `https://github.com/ALRhub/MaIL.git`
- Branch: `main`
- HEAD: `a8012a0018ce2e5e26adff3bb3336190be2595ea`
- Commit date: `2025-02-17T11:02:54+01:00`
- Commit subject: `Update LICENSE`
- Status after inspection: clean, `main...origin/main`

### LIBERO reference implementation

- Repository: `https://github.com/Lifelong-Robot-Learning/LIBERO.git`
- Local path: `/home/itaein/Personal_Research/external/LIBERO`
- Origin fetch/push: `https://github.com/Lifelong-Robot-Learning/LIBERO.git`
- Branch: `master`
- HEAD: `8f1084e3132a39270c3a13ebe37270a43ece2a01`
- Commit date: `2025-03-15T20:13:56+08:00`
- Commit subject: `Add support for dataset download from huggingface`
- Status after inspection: clean, `master...origin/master`

Both clones retain their independent upstream `.git` directories. The research root itself is not a valid Git repository: its `.git/` directory is empty and `git status` fails. It was not initialized.

## 3. Host / Runtime Environment

| Item | Observed value |
|---|---|
| Host | `pleiades1` |
| Scheduler evidence | PBS `qsub`/`qstat` available; no jobs listed; no active `PBS_*` allocation variables |
| OS | Rocky Linux 8.8 (Green Obsidian) |
| Kernel | `4.18.0-477.21.1.el8_8.x86_64` |
| CPU | 2 × Intel Xeon Gold 6342, 48 physical cores total |
| RAM | 1.0 TiB total, about 857 GiB available at inspection |
| GPU | Not accessible on the current host; `nvidia-smi` cannot communicate with the driver |
| NVIDIA driver | Not observable because `nvidia-smi` fails |
| System CUDA toolkit | CUDA 11.3, `nvcc` 11.3.58 |
| Active Python | 3.9.7 in Conda `base` |
| Conda | 4.10.3 |
| Compiler | GCC/G++ 8.5.0 |
| Disk | 7.0 TiB filesystem, about 139–140 GiB available, 99% used after clones |

The host appears to be a PBS cluster head/login environment rather than an active compute allocation. Irrespective of site policy, GPU work is not currently possible because the driver is unavailable. No GPU training or job submission was attempted.

Existing environments do not provide a complete MaIL runtime:

| Environment | Python | Torch | Torch CUDA | Relevant observation |
|---|---:|---:|---:|---|
| `base` | 3.9.7 | absent | — | `h5py` only among core requirements |
| `py3.8.12` | 3.8.12 | 1.11.0 | 11.3 | no Hydra, Mamba, robomimic, robosuite, or LIBERO |
| `egodex-box` | 3.10.20 | 2.5.1+cu124 | 12.4 | no MaIL stack |
| `sam2-egodex` | 3.10.20 | 2.5.1+cu121 | 12.1 | Hydra 1.3.5/OmegaConf 2.3.1 present; no WandB/Mamba/robotics stack |
| `unidepth-egodex` | 3.11.15 | 2.4.1+cu121 | 12.1 | no Hydra/Mamba/robotics stack |
| `unified-object-annotation` | 3.10.20 | 2.5.1+cu124 | 12.4 | no MaIL stack |

All tested Torch installations reported `torch.cuda.is_available() == False`.

## 4. Repository Structure

Important MaIL areas and their actual roles are:

| Path | Role |
|---|---|
| `run_benchmark.py` | Hydra/WandB training and rollout entry point used by scripts |
| `train.py` | Alternate entry point; performs one direct `train_vision_agent()` pass followed by simulation |
| `config/benchmark_libero10.yaml` | LIBERO-10 data, sequence, optimization, and rollout configuration |
| `config/benchmark_libero_goal.yaml` | Language/task-embedding dataset and simulator configuration |
| `config/benchmark_libero_new.yaml` | Non-language Spatial/Object configuration |
| `config/agents/bc_mamba_dec.yaml` | Non-language decoder-only Mamba BC object graph |
| `config/agents/goal_bc_mamba_dec.yaml` | CLIP-task-token decoder-only Mamba BC object graph |
| `dataset/multi_task_dataset_aug.py` | Multi-file LIBERO HDF5 window loader without language |
| `dataset/multi_task_dataset_goal.py` | Same loader with one precomputed task embedding per trajectory |
| `agents/base_agent.py` | Dataset/model instantiation, DataLoaders, action scaler, parameter logging |
| `agents/models/bc/bc_agent.py` | Non-language BC policy, training step, prediction, checkpoint methods |
| `agents/models/bc/goal_bc_agent.py` | Language-prefix BC variant |
| `agents/models/bc/transformer.py` | Despite the filename, contains the `Enc_only` wrapper used around Mamba |
| `agents/models/bc/goal_transformer.py` | Language variant of that wrapper |
| `agents/models/oc_ddpm/mamba.py` | Vendored `MixerModel` stack built from `mamba_ssm` blocks |
| `agents/module/vision/` | ResNet18/SpatialSoftmax two-view visual encoder |
| `agents/utils.py` | Z-score action scaling and inverse scaling |
| `simulation/benchmark_sim_new.py` | Non-language LIBERO rollout and success logging |
| `simulation/benchmark_sim_goal.py` | Language-token LIBERO rollout |
| `get_task_embeddings.py` | Offline CLIP/BERT/GPT-2/RoBERTa task-embedding generator; released files use CLIP |
| `task_embeddings/*.pkl` | Precomputed suite-level task embeddings |
| `script/` | Hydra multirun commands; several names are stale, as detailed below |

## 5. Training Entry Point

The reproduced script path is intended to be:

```text
script/bc/libero_10_bc_mamba_dec.sh
  -> python run_benchmark.py --config-name=benchmark_libero10 --multirun ...
  -> Hydra composes benchmark + agent config
  -> hydra.utils.instantiate(cfg.agents)
  -> hydra.utils.instantiate(cfg.simulation)
  -> repeated agent.train_vision_agent()
  -> simulator rollout at epochs 39, 49, and 59
  -> last_bc.pth
```

Important release defects:

1. `run_benchmark.py` declares a default `benchmark_libero.yaml`, which does not exist. The scripts avoid this only by supplying `--config-name`.
2. Official BC scripts override `agents=bc_mamba`, but only `bc_mamba_dec.yaml` and `bc_mamba_encdec.yaml` exist. Exact script config composition therefore fails. The language scripts similarly use missing names such as `goal_bc_mamba` instead of `goal_bc_mamba_dec`.
3. `wandb.entity` and `wandb.project` are mandatory placeholders in configs, and `run_benchmark.py` uses online WandB.
4. The language dataset contains a machine-specific absolute task-embedding path.

A corrected **config-only composition** with `agents=bc_mamba_dec` succeeded. No corrected training command was executed.

## 6. End-to-End Training Data Flow

For the decoder-only BC example with `obs_seq=5` and `train_action_seq=10`:

```text
benchmark_libero10.yaml + bc_mamba_dec.yaml
  -> MultiTaskDataset scans every *.hdf5 in dataset_path
  -> one valid sliding window of length 14 (= 5 + 10 - 1)
  -> DataLoader batch
  -> images: first 5 timesteps
  -> targets: actions from window index 4 through 13 (10 actions)
  -> two independent ResNet18 + SpatialSoftmax encoders
  -> concatenate two 128-D features -> observation token 256-D
  -> Linear(256, 128) + learned position embedding
  -> append 10 learned action-query embeddings, each 128-D
  -> 16-layer Mamba when the published shell override is honored
  -> select the 10 action-query outputs
  -> Linear(128, 7)
  -> mean MSE against z-score-normalized target actions
  -> backward -> Adam step
```

Concrete training tensor roles are:

| Stage | Tensor role and shape |
|---|---|
| Dataset images | two tensors `[B, 14, 3, 128, 128]` |
| Dataset actions | `[B, 14, 7]` |
| Policy images | two tensors `[B, 5, 3, 128, 128]` |
| BC targets | `[B, 10, 7]` |
| Visual tokens | `[B, 5, 256]` |
| Embedded observation tokens | `[B, 5, 128]` |
| Learned action queries | `[B, 10, 128]` |
| Mamba input/output | `[B, 15, 128]` |
| Predicted action chunk | `[B, 10, 7]` |

For `goal_bc_mamba_dec`, one `[B, 1, 512]` CLIP task embedding is projected to `[B, 1, 256]` and prepended to the five visual tokens. The resulting Mamba sequence is `[B, 16, 128]`; predicted actions remain `[B, 10, 7]`.

`BaseAgent` constructs shuffled train and non-shuffled validation DataLoaders and derives action-scaling statistics from all valid train actions. `BC_Agent.train_step()` performs the forward pass, `torch.nn.functional.mse_loss`, zero-grad, backward, and optimizer step.

## 7. Dataset Structure

MaIL expects a directory containing one LIBERO/robomimic-style HDF5 file per task. It scans all filenames ending in `.hdf5`; the scan is not sorted and has no task allowlist.

For each selected `demo_i`, the code actually reads:

- `demo.attrs["num_samples"]`
- `demo["actions"]`
- `demo["obs"]["agentview_rgb"]`
- `demo["obs"]["eye_in_hand_rgb"]`

It constructs an internal valid-length mask, but does not read dataset `dones`, `rewards`, `states`, proprioception, or timestep fields even though several are listed under `dataset_keys` in YAML. Those YAML keys are not used by this loader.

Actions and masks are padded in memory to `max_len_data=520`; images remain per-trajectory arrays. Sliding windows are generated only within valid trajectory lengths, so training windows themselves contain no padded timesteps. With no HDF5 filter configured, all demo keys are used, limited to the first `num_data=10` sorted demo indices per file.

The language loader obtains the task name by removing `_demo` from each HDF5 basename and looks it up in a suite-level embedding dictionary.

No `.hdf5`/`.h5` dataset was present under the research root, so no sample was loaded. Actual per-suite schema equivalence remains unverified by data, although official LIBERO exposes every suite through the same `get_task_demonstration()` convention and MaIL uses the same loader for suite overrides.

## 8. Observation Representation

The active BC path uses only two normalized RGB streams:

- `agentview_rgb`
- `eye_in_hand_rgb`

Dataset images are converted from `THWC` uint-like arrays to float `TCHW` in `[0,1]`. Each camera has an independent non-pretrained robomimic `VisualCore` with `ResNet18Conv`, SpatialSoftmax (`32` keypoints), and a 128-D feature output. BatchNorm layers are replaced with GroupNorm and ImageNet normalization is applied. The two outputs are concatenated to 256 dimensions per observation timestep.

Although low-dimensional keys (`ee_ori`, `ee_pos`, `gripper_states`, `joint_states`) appear in YAML, the inspected BC policy does not load or encode them.

## 9. Action Representation

- Configured dimension: `7`.
- Dataset source: raw HDF5 `actions` array.
- Controller evidence: LIBERO `ControlEnv` defaults to `OSC_POSE`; MaIL/LIBERO simulation uses a 7-vector and explicitly treats the final element as gripper command (`-1` opens it in the dummy action).
- Exact semantic order of the first six dimensions is not documented by MaIL. It should be verified from dataset `env_args` and the pinned robosuite controller before final metric reporting. It is consistent with an OSC pose command plus gripper, but that is not treated here as a fully verified dataset fact.
- Absolute versus delta: not explicitly asserted by MaIL; TBD from actual dataset metadata/controller configuration.
- Normalization: per-dimension z-score using the mean/std of all valid train actions, with `1e-12` epsilon.
- Prediction: an action chunk, not a single action. The LIBERO-10 config uses 10 targets; other configs commonly use 5.
- Target alignment: target chunk begins at the last observation in the input window (`actions[:, obs_seq_len-1:]`).
- Loss weighting: none; all batch, time, and action dimensions receive the default mean MSE weighting.

The policy does **not** consume previous actions. Adding previous-action history would be a separate design change and is not part of the inspected baseline.

## 10. Language Conditioning

The standard `bc_mamba_dec` path is not language-conditioned. The `goal_bc_mamba_dec` path implements language as follows:

1. `get_task_embeddings.py` reads each official LIBERO task's raw language string.
2. It uses `openai/clip-vit-base-patch32` text features.
3. It stores one float32 vector of shape `[1,512]` per task in a suite pickle.
4. The released pickles contain 10 tasks for Spatial/Object/Goal/LIBERO-10 and 90 for LIBERO-90.
5. `MultiTaskDataset.__getitem__()` returns the same task vector for every window from that task.
6. `goal_bc_agent.BC_Policy.linear` learns a `512 -> 256` projection.
7. That one projected vector is prepended to the sequence of 256-D visual tokens.
8. The combined tokens pass through the shared `256 -> 128` token projection and Mamba.

The CLIP encoder is absent from training and therefore frozen in practice; only the 512-to-256 projection is learned. Raw text is not consumed at training or rollout time. The embedding is one prefix token per sampled window, not repeated at every timestep and not injected through cross-attention.

Timestep-specific conditioning is technically feasible without modifying Mamba itself, but not through the current one-prefix-token interface unchanged. The dataset must return time-aligned context embeddings and the policy must either fuse one context token with each observation token or define an explicit interleaved sequence. The former is likely the smaller, fairer shared change for all three methods; the final choice is deferred to Task 4/7.

## 11. Mamba Architecture

- Type: decoder-only/unidirectional selective state-space stack used as an encoder over observation/language tokens plus learned action-query tokens.
- Library: upstream README requires `mamba-ssm==1.2.0.post1`; it is not pinned in `requirements.txt`.
- Wrapper: `agents.models.bc.transformer.Enc_only` (or the nearly identical goal variant).
- Backbone: `agents.models.oc_ddpm.mamba.MixerModel` using `mamba_ssm.modules.mamba_simple.Mamba` and `Block`.
- `d_model`: 128.
- Published BC shell override: 16 Mamba layers. Benchmark YAML defaults to 6.
- `d_state`: 16.
- `d_conv`: 4.
- Expansion: 2.
- Normalization: standard LayerNorm (`rms_norm=False`) with final `norm_f`; fused add/norm disabled.
- Residual: the `mamba_ssm` prenorm residual `Block`; `residual_in_fp32=False`.
- Positional information: the outer `Enc_only` adds a learned `pos_emb` to observation tokens. Query embeddings are learned by action-horizon position. Mamba itself receives ordered continuous embeddings rather than token IDs.
- Input/output shape: `[batch, sequence, 128]` to `[batch, sequence, 128]`.
- Action representation actually used: Mamba outputs at learned action-query positions, passed to `Linear(128,7)`.
- Inference state: although `MixerModel` implements cache allocation, the BC policy does not use it. It recomputes a bounded sliding observation history and an action-query sequence.

The Mamba receives neither the full trajectory nor a persistent recurrent cache. Training uses fixed windows; rollout retains at most `obs_seq_len` images in Python deques and replans only after executing the prior predicted chunk.

### Candidate probe representation

The primary `h_t` candidate is the final-normalized `encoder_output` at the last observation token, immediately before action-query positions. It is the clearest per-timestep temporal-context representation. A secondary diagnostic candidate is the first action-query representation immediately before `action_pred`, because it directly drives action prediction and causally follows all current context tokens. Token indices must be derived from explicit metadata once language becomes timestep-aligned rather than hardcoded.

## 12. Action Loss

`BC_Agent.train_step()` uses:

```text
F.mse_loss(predicted_action_chunk, normalized_target_action_chunk)
```

This is default mean reduction over batch, action-horizon, and seven action dimensions. There is no dimension weighting, mask weighting, auxiliary loss, or separate gripper loss. This normalized-space MSE is the inherited baseline definition for Phase-1 Overall Action Prediction Error unless a later evaluation task explicitly records an additional denormalized metric.

## 13. Evaluation Pipeline

### Offline

`BC_Agent.evaluate()` computes MSE, but the released `eval_vision_agent()` path is not operational for the inspected two-view dataset:

- The non-language dataset returns four elements, while evaluation unpacks three.
- Evaluation does not mirror the training crop/alignment for two image streams and target actions.
- The goal variant has the same unpack mismatch.

Therefore no trustworthy held-out offline metric exists as released. Train and validation configs also point to the same directory with both filter keys unset, so even after repairing the code they are not held-out splits by default.

### Simulator rollout

`run_benchmark.py` calls `MultiTaskSim.test_agent()` at epochs 39, 49, and 59. Each task uses official BDDL and fixed init states. A rollout is successful when the environment reward equals `1`; LIBERO's base domain defines this sparse reward as task success. Per-task success is the mean over `num_episode=20` init-state rollouts and overall success is the mean across suite tasks.

Seeds are set for Torch, CUDA when available, NumPy, Python `random`, and the environment. The rollout code indexes the first `num_episode` task init states deterministically.

No rollout was attempted because LIBERO is not installed, datasets are absent, and no GPU/driver is available on the current host.

## 14. Checkpoint / Logging

- Logging: WandB. `run_benchmark.py` uses online mode; `train.py` uses offline mode.
- Hydra changes into a timestamped run/sweep directory; `BaseAgent.working_dir` is that current directory.
- Final checkpoint: `last_bc.pth`, a model `state_dict` only.
- Nominal best checkpoint: `eval_best_bc.pth`, selected by validation MSE in `BC_Agent.train_agent()`.
- Actual benchmark runner: bypasses `train_agent()` and calls `train_vision_agent()` directly, so it does not create the nominal best checkpoint.
- Load: `load_pretrained_model(path, sv_name)` loads only model weights.
- Resume: no optimizer, scheduler, epoch, RNG, or full training-state resume implementation was found.

## 15. LIBERO Support

Code/config evidence supports:

| Suite | Evidence in MaIL | Status |
|---|---|---|
| LIBERO-Spatial | BC/DDPM scripts override `task_suite=libero_spatial`; embedding pickle present | Supported by common loader/config path |
| LIBERO-Object | `benchmark_libero_new.yaml` default and scripts; embedding pickle present | Supported |
| LIBERO-Goal | `benchmark_libero_goal.yaml`, scripts, embedding pickle | Supported |
| LIBERO-10 (long-horizon suite) | `benchmark_libero10.yaml`, dedicated scripts, embedding pickle | Directly targeted |
| LIBERO-90 | dedicated scripts through goal config, embedding pickle, simulator 90-task branch | Targeted, but expensive and not Phase-1 candidate selection |

There is no official benchmark key named `libero_long` in the pinned LIBERO code. The long-horizon ten-task suite is `libero_10` / `LIBERO_10`.

## 16. LIBERO-Long Compatibility

Technical compatibility with LIBERO-10 is strong but not unchanged-run ready:

- MaIL includes a dedicated `benchmark_libero10.yaml` and LIBERO-10 BC/Mamba scripts.
- Official LIBERO exposes LIBERO-10 through the same `Benchmark` API and common demonstration path convention as the other suites.
- MaIL's generic HDF5 loader does not encode suite-specific schemas; it reads the same two RGB keys and actions for all files.
- A bundled `libero_10.pkl` contains 10 CLIP embeddings of shape `[1,512]`.
- The language config can be overridden to `task_suite=libero_10`, as demonstrated by MaIL ablation scripts.

However, the released shell agent aliases are invalid, dataset paths point to the authors' cluster, the language loader uses an author-specific absolute embedding path, dependencies/data are absent, and rollout forces all suite tasks. The conclusion is therefore **B: usable with small config/dataset-path fixes plus a small task-selection change; no large Mamba restructuring is required**.

Actual HDF5 equality across suites and trajectory statistics are deferred until data inspection in Task 2.

## 17. Single-task / Task-subset Feasibility

Training does **not** inherently require all ten tasks. `MultiTaskDataset` simply reads every `.hdf5` in `data_directory`. A directory containing only one task file or an explicit subset would train on that subset; the language loader looks up embeddings only for files it encounters. This can be exposed cleanly through a task/file allowlist or a subset data directory.

The released code does not provide such an allowlist. More importantly, simulator evaluation is hardcoded to all 10 tasks (or all 90):

- `benchmark_sim_new.MultiTaskSim.test_agent()` sets `num_tasks` to 10/90 and builds `contexts = arange(num_tasks)`.
- `benchmark_sim_goal.MultiTaskSim.test_agent()` does the same.
- Both constructors accept `task_id`, but `test_agent()` never uses it.

Thus:

- Single-task/subset **training**: possible through dataset-directory selection; a small explicit allowlist is preferable for reproducibility.
- Single-task/subset **rollout**: requires a small simulator/config modification to honor task IDs.
- Training all 10 tasks simultaneously: not structurally required.

## 18. Candidate Modification Points for Phase-1

These are inspection findings only; no implementation decision or source change was made.

### A. Full instruction

- File/class/function: `dataset/multi_task_dataset_goal.py` — `MultiTaskDataset.__getitem__()`.
- Current role: returns one fixed precomputed 512-D CLIP task vector for every window.
- File/class/function: `agents/models/bc/goal_bc_agent.py` — `BC_Policy.forward()`.
- Current role: projects the vector to 256-D and prepends it to visual tokens.
- Candidate: retain this pipeline as Method A after replacing the absolute embedding path with a config-controlled path and defining consistent time/token metadata.
- Risk: the current prefix is one token per window, not “full instruction at every timestep”; the shared Phase-1 conditioning interface must define whether every method uses aligned fusion or a common token convention.

### B. Current subinstruction / timestep conditioning

- Dataset candidate: extend the language dataset sample to return a context sequence aligned with the image/action window.
- Policy candidate: `goal_bc_agent.BC_Policy.forward()` immediately after visual encoding and before the `self.model(obs)` call.
- Wrapper candidate: `goal_transformer.Enc_only.forward()` before `input_seq` is passed to `self.encoder`.
- Preferred minimal direction to evaluate later: project each context embedding with the same learned projection and fuse it with its corresponding visual token; use the identical interface for all methods.
- Risks: off-by-one alignment between the 14-step source window, five observation steps, and ten target actions; changed token counts; parameter-count differences if concatenation adds a new projection.

### C. HOLD/Transition

- Candidate path: use the same timestep-context field and the same projection/fusion module as Method B; only its values differ between transition and non-transition timesteps.
- Candidate location: the dataset/conditioning interface should select subinstruction versus HOLD before `BC_Policy.forward()` fuses context with visual tokens.
- Risks: choosing a HOLD representation prematurely, accidentally freezing temporal state, and introducing method-specific parameters. The exact token/embedding construction remains deferred to Task 7.

### D. Hidden representation extraction

- File/class/function: `agents/models/bc/transformer.py` or `goal_transformer.py` — `Enc_only.forward()`.
- Current value: `encoder_output = self.encoder(input_seq)`, the final-normalized Mamba output before `action_pred`.
- Primary candidate `h_t`: last observation-token representation.
- Secondary candidate: first action-query representation before the linear action head.
- Required later change: optional feature return or a controlled forward hook, with the normal policy output unchanged.
- Risks: language-prefix and variable-history offsets; accidental graph retention/memory growth; probing action-query tokens could measure head-oriented information rather than a strictly observation-time state.

## 19. Reproducibility Check Results

| Check | Result | Evidence / reason |
|---|---|---|
| Repository clone/revision | **PASS** | Clean MaIL and LIBERO clones with exact hashes recorded |
| Static Python syntax | **PASS** | AST parsing passed for all 56 MaIL Python files |
| MaIL import in active base | **FAIL** | Active Python has no Torch/Hydra/WandB/Mamba stack |
| MaIL import in closest existing env | **FAIL** | `sam2-egodex` reaches `train.py` but fails first at missing `wandb`; direct Mamba import fails at missing `mamba_ssm` |
| Exact published script config | **FAIL** | Hydra cannot find `agents/bc_mamba` |
| Corrected Hydra config composition | **PASS** | `benchmark_libero10 + agents=bc_mamba_dec` resolves to `BC_Agent`, `MixerModel`, window 14, obs 5, action horizon 10, action dim 7 |
| Model instantiate | **FAIL** | No existing environment provides `mamba_ssm`; full policy also needs robomimic/torchvision stack |
| Dataset sample | **NOT TESTED** | No HDF5 dataset present; no dataset was downloaded |
| Single-batch forward | **NOT TESTED** | Model cannot instantiate and data is absent |
| Single-batch loss | **NOT TESTED** | Forward/data prerequisites unavailable |
| Simulator rollout | **NOT TESTED** | LIBERO not installed, dependencies/data absent, GPU driver unavailable |

The Task-1 success requirement is met at the inspection level, not at runnable-baseline level: configuration can be composed after correcting stale names, but executable reproduction remains blocked by an isolated environment, data paths/data, upstream defects, and appropriate compute allocation.

## 20. Environment / Dependency Issues

Upstream requirements and current state:

- MaIL recommends Python 3.8, PyTorch CUDA 12.1 wheels, and `mamba-ssm==1.2.0.post1`.
- MaIL pins Hydra 1.2.0, NumPy 1.22.4, WandB 0.13.1, Transformers 4.21.1, robomimic 0.3.0, and several 2022-era packages.
- Pinned LIBERO requirements use robomimic 0.2.0 and robosuite 1.4.0, while MaIL requests robomimic 0.3.0 and leaves robosuite unpinned. This is a concrete dependency-resolution risk.
- The system toolkit is CUDA 11.3, while the README's suggested Torch wheel is CUDA 12.1. Existing cu121/cu124 Torch environments still cannot access a driver on this host.
- Mamba 1.2.0 uses compiled CUDA/Triton extensions and private module paths used by MaIL's vendored wrapper; version substitution is risky.
- The research filesystem is already 99% used. A new CUDA/PyTorch/Mamba/robotics environment would plausibly consume several GB (rough estimate 5–10+ GB including package caches/build artifacts); an exact solver plan was not produced.
- `conda run` itself attempted to create temporary files inside read-only existing environments under the managed sandbox; direct environment Python executables were used for read-only checks instead.

Because no existing environment is suitable, GPU is unavailable, dependency versions conflict, and disk usage is high, creating `personal-research-mail` during Task 1 was judged unsafe and low-value. No package was installed and no environment was created or modified.

## 21. Remaining TBD

- Resolve and lock a compatible Python/Torch/CUDA/Mamba/LIBERO/robomimic/robosuite environment on an allocated compute node.
- Decide whether to patch the pinned upstream revision externally or maintain a minimal research overlay for stale Hydra aliases, data paths, and task selection.
- Locate or acquire LIBERO datasets; verify actual HDF5 keys, controller metadata, action semantics, trajectory lengths, and suite consistency.
- Define a real train/validation/test split; released defaults reuse the same data.
- Repair/replace offline evaluation before treating MSE as a held-out result.
- Confirm whether `obs_seq=5`, action horizon 10, and 16 Mamba layers are the Phase-1 common settings after compute/data inspection.
- Decide a common timestep-conditioning fusion interface in Task 4; do not choose the HOLD representation yet.
- Define task-subset rollout configuration while keeping upstream source intact if possible.
- Decide whether the primary probe uses last-observation or first-action-query representations, then lock it before results.

## 22. Recommendation for Task 2

Task 2 should analyze **LIBERO-10** as the technical long-horizon suite; this is the official name corresponding to the requested LIBERO-Long concept. It should not yet select tasks based only on their names. Once data is available, inspect every candidate's:

- actual demonstration count and trajectory-length distribution;
- two required RGB keys and seven-dimensional action metadata;
- control frequency (official environment default is 20 Hz) versus stored demonstration sampling;
- semantic multi-stage structure and annotatable transition evidence;
- compatibility with `max_len_data=520`, the fixed-window loader, and action horizon;
- availability of enough trajectories for leakage-free policy/probe splits.

For Phase 1, use the language-enabled decoder-only BC architecture as the likely base, but first preserve a corrected non-language `bc_mamba_dec` config composition as a structural sanity reference. Task 2 should only inspect/select data and tasks; it should not implement conditioning, annotations, HOLD, metrics, or training.
