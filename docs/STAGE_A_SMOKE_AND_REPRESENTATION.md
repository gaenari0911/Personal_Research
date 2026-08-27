# Stage A — Autonomous Smoke and Representation Learning

Stage A runs one unattended PBS job pinned to `pleiades1`, with two CUDA GPUs
and eight CPU cores. It performs B0/B1/B2/B3
smoke tests, validates or completes the frozen CLIP cache for the 734-train and
85-validation trajectories, chooses one common epoch budget from smoke timing
with a 25% walltime reserve, and then trains B0/B1 and B2/B3 in two sequential
parallel pairs. Every variant remains an independent Python process pinned to
one logical CUDA device. Cache construction uses CUDA device 0.

## Fixed scientific contract

- Input: frozen CLIP ViT-B/32 external-RGB feature, Panda qpos 9D, and the
  registered language condition.
- Objective: predict the frozen normalized CLIP feature at `t+20` with one-way
  in-batch InfoNCE at temperature 0.07.
- The deterministic R2 balanced grid supplies up to 64 anchors per trajectory
  (minimum eight required); the same exact anchor IDs are shared by all variants.
- Optimizer: AdamW, learning rate `1e-4`, weight decay `0.01`, gradient clipping
  at `1.0`, and no scheduler.
- Precision: float32 training without AMP. Cached CLIP features are normalized
  float16 and are cast to float32 when loaded.
- B0 uses independent causal windows of at most five timesteps. B1/B2/B3 use
  full-trajectory causal processing with layer checkpointing and no automatic
  recurrent-state detach or subtask-transition reset.
- Test 95, Behavior Cloning, action prediction, probes, retention curves, and
  final metrics are outside Stage A.

## Fair initialization

`checkpoints/stage_a/common_init.pt` is created once at seed 42 before
submission. Every smoke and full process constructs a fresh model, loads this
exact checkpoint, and creates a fresh optimizer. Trained weights never flow
from one variant to the next. Smoke checkpoints are explicitly excluded from
full training.

## Automatic gates

The job stops non-zero without interaction if runtime precheck, GPU visibility,
any smoke, the 819-episode cache, the conservative walltime calculation, any
full training process, representation sanity, checkpoint existence, or final
fairness fails. Completed cache files and checkpoints are retained.

The cache is trajectory-atomic and resume-safe: valid files are reused, new
files are written to a temporary path and re-opened after atomic rename, and an
MP4 must contain exactly `T+1` images while only `0..T-1` are encoded.

## Main outputs

- Status: `analysis/stage_a_status.json`
- Smoke gate: `analysis/stage_a_smoke_summary.json`
- Cache audit: `analysis/stage_a_clip_cache_audit.json`
- Budget: `analysis/stage_a_compute_plan.json`
- Per-model training: `analysis/stage_a_training_B0.json` through `B3.json`
- Final gate: `analysis/stage_a_gate.json`
- Checkpoints: `checkpoints/stage_a/<variant>/best_val.pt` and `last.pt`
- Master log: `logs/stage_a/integrated.log`

Stage B is never started automatically.
