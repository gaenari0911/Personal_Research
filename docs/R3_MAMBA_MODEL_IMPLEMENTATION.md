# R3 Mamba Memory Model Implementation

## Scope and result

R3 implements one executable PyTorch architecture class for B0/B1/B2/B3. It
does not implement an action head and did not run representation training,
probe fitting, BC, or final-test evaluation. The four variants differ only in
language scheduling and persistent-state execution.

The architecture follows the dimensions in MaIL's
`agents/models/oc_ddpm/mamba.py::MixerModel` and `config/local_train.yaml`:
`d_model=128`, 16 Mamba-1 layers, `d_state=16`, causal convolution width 4,
and expansion 2. MaIL itself is unchanged.

## Common encoder

The frozen feature measurement is OpenAI CLIP ViT-B/32 through
`transformers.CLIPModel`.

- external RGB -> frozen, L2-normalized CLIP image feature, 512D
- exactly `states[t,1:10]` -> `Linear(9,64)` -> GELU
- concatenated 576D observation/state -> `Linear(576,128)` -> LayerNorm = `r_t`
- frozen CLIP text feature 512D -> trainable `Linear(512,128)`
- `x_t = LayerNorm(r_t + mask * language_projection)`

`r_t` is returned as `instantaneous` before any language is added. The exact
9D-width check rejects a flattened privileged MuJoCo state. Previous action,
time, progress, Step ID, and boundary labels are absent from the model API.

The CLIP wrapper has a `get(namespace,key)` / `put(namespace,key,value)` cache
contract and a process-local implementation. R4 may substitute an on-disk
cache without changing the encoder. R3 encoded only four real images rather
than precomputing the dataset.

## HOLD means no event

`build_condition_schedule(..., "B3")` emits a Step string only at its official
half-open interval start and emits `None` elsewhere. `encode_condition_schedule`
never sends those `None` positions to CLIP; it creates exact zero feature rows
and a false injection mask. Thus the literal string `[HOLD]` is never a natural
language input. RGB, qpos, causal convolution, and selective-SSM state still
update at every HOLD timestep.

## Temporal backbone and probe point

`MambaMemoryBackbone` is a differentiable PyTorch reference implementation of
the Mamba-1 input projection, causal depthwise convolution, input-selective
`dt/B/C`, SSM recurrence, gating, residual stack, and final LayerNorm. It uses
one timestep kernel for both sequence and recurrent execution. This makes CPU
tests possible without the optional fused CUDA extension. Its backend name is
`torch_reference_mamba1_selective_scan`.

This is not a byte-for-byte copy of MaIL's optimized `mamba_ssm` kernel and the
checkpoint layouts are not claimed to be interchangeable. R4 can add the
official optimized kernel behind the existing interface after a parity test.
The architecture dimensions and semantic state contract are fixed now.

The primary `z_t` is the final LayerNorm output after the current token and
before the future head. Raw layer convolution/SSM caches are implementation
state and are not probe inputs.

The upstream Mamba project documents the Mamba block and selective SSM as its
core interfaces: <https://github.com/state-spaces/mamba/blob/main/README.md>.
The frozen encoder choice follows OpenAI's released ViT-B/32 CLIP model:
<https://github.com/openai/CLIP/blob/main/model-card.md>.

## Variant execution

| Variant | Language | Temporal execution |
|---|---|---|
| B0 | full instruction every token | independently recompute unpadded `[max(0,t-4),t]`; never retain a Mamba cache |
| B1 | full instruction every token | reset once at episode start; persist through all Step transitions |
| B2 | current official Step every token | same persistence; oracle current-conditioning upper bound |
| B3 | Step only at its start, zero otherwise | same persistence; HOLD continues observation and Mamba updates |

All four variants use `MemoryExperimentModel`. Resetting a B0 model is an
error, and stepping a persistent model before an explicit reset is an error.
With seed 42 the complete initial state-dict SHA-256 is identical across all
four instantiated variants.

## Future objective and probes

The future head is `Linear(128,512)` followed by L2 normalization. The reusable
loss uses one-way in-batch InfoNCE at temperature 0.07. The target is a frozen
CLIP feature at `t+20`; the loss explicitly detaches it. No future image feature
appears in a model-forward argument.

`LinearRetrievalProbe` is bias-free `Linear(128,512)` plus normalization. It can
take temporal `z_t` or instantaneous `r_t`. Candidate Step features come from
the same frozen CLIP text encoder. Existing current/previous-k target builders
are reused, and the training loss accepts every casing/whitespace-equivalent
duplicate Step as a positive.

## Real smoke and tests

The real smoke uses strict-clean `study_table/case47` (`T=258`, MP4 images
`T+1=259`) at image/timestep indices 0, 178, 179, and 180. It loads qpos 9D,
actual pretrained CLIP image/text features, and all four 16-layer models.
Every variant returned `[1,4,128]` instantaneous/temporal features and
`[1,4,512]` future predictions. The B3 transition at 179 injected the new Step;
178 and 180 were exact language no-ops. Full and recurrent B3 output differed
by at most `9.5367431640625e-7`.

The selected images are a sparse interface smoke, not a trajectory-level
metric or a substitute for sequential training/evaluation.

Run the focused tests:

```bash
PYTHONPATH=.:src /home/itaein/.conda/envs/egodex-box/bin/python \
  -m unittest -v tests.test_mamba_memory_models
```

Run the real smoke after the official checkpoint is cached:

```bash
PYTHONPATH=.:src /home/itaein/.conda/envs/egodex-box/bin/python \
  tools/dry_run_r3_models.py --clip-cache /tmp/r3_clip_cache
```

Artifacts are `analysis/r3_model_shapes.json`,
`analysis/r3_state_handling_audit.json`, and
`analysis/r3_real_episode_dryrun.json`.

The final gate reruns 26 R3 tests plus the 40 directly relevant R1/R2
RoboCerebra regression tests: 66/66 pass with no skip or failure. An earlier
repository-wide run also completed 109 tests with 101 passes, no failure, and
eight intentional pre-existing skips.
