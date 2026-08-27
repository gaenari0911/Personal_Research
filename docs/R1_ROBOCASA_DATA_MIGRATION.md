# R1 RoboCasa WashFruitColander data migration

Audit date: 2026-08-24  
Final result: **R1_DATA_GATE = FAIL; cleanup skipped**

## Acquisition and provenance

The official RoboCasa registry at commit
`a07e365c958c4216cd6bbd5f30b47f09a65c6f00` identifies:

```text
target/composite/WashFruitColander/20250811/lerobot.tar
Box share: https://utexas.box.com/s/8omrgv8mqo2p5s1hqjmas67p30etsiic
```

Both the share URL and the direct static URL returned HTTP 404 on the audit date,
and `git ls-remote` confirmed the local commit is still official `main`. Following
the automatic recovery policy, the identical RoboCasa path was acquired from
NVIDIA's verified
[PhysicalAI Robotics Manipulation Kitchen Demos](https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-Manipulation-Kitchen-Demos)
distribution—not a personal mirror. Because the prompt strictly requires an
official current RoboCasa source, and this alternate archive predates the 2026
annotation update, the acquisition cannot satisfy G1 even though its contents
are otherwise intact.

| item | value |
|---|---|
| target root | `/ssd1/itaein/datasets/RoboCasa365/WashFruitColander` |
| archive | `lerobot.tar` |
| archive bytes | 1,977,016,320 |
| expected/provided SHA-256 | `4905fff0dfe1c16c9bbab51d44cda0e89a8e15d042420c04da94dc1d2bf4fd0c` |
| actual SHA-256 | same |
| tar integrity | PASS |
| extracted bytes | 1,976,216,669 |
| extracted files | 3,557 |
| resume/retry | aria2, 16 ranges, resumable |

No whole RoboCasa365 dataset was downloaded.

## Actual schema

The extracted dataset is LeRobot `v2.1`:

- 507 episode parquet files;
- 518,903 frames at 20 Hz;
- one contiguous `frame_index` stream per episode;
- 12D action and 16D observation state;
- 1,521 H.264 videos: left, right, and eye-in-hand for every episode;
- 507 episode metadata records and 157 full-instruction variants;
- replay extras including per-episode state, XML, and episode metadata.

All parquet files and all video headers were opened. Each video's frame count,
256×256 geometry, and 20 Hz rate match its parquet. No corrupt episode, NaN,
missing camera, or frame-count mismatch was found.

The critical mismatch is annotation version. Actual parquet contains only:

```text
annotation.human.task_description
annotation.human.task_name
```

It does **not** contain the repository's post-2026 advertised fields:

```text
subtask_idx
annotation.human.subtask
annotation.human.subtask_name
annotation.human.subtask_stage
```

No annotation was invented, modified, or imported from a third-party conversion.

## Full episode action audit

The exact selection rule is:

1. all four released base components equal exactly `0.0` at every frame;
2. control mode equals exactly `-1.0` for every frame and never changes;
3. all three videos exist and match the parquet length;
4. official task metadata declares `moma_required=No`.

An arbitrary tolerance was unnecessary: serialized stationary base values are
exact zero, while nonzero values form discrete controller commands. Results:

| result | episodes |
|---|---:|
| total | 507 |
| eligible exact arm-only | 489 |
| excluded base/control | 18 |
| corrupt/other | 0 |

Excluded episode IDs:

```text
12, 18, 37, 181, 190, 226, 242, 256, 265, 278, 324, 331,
387, 392, 419, 425, 499, 500
```

Every excluded episode has both nonzero base commands and a non-default/control
mode change. The actual 489/507 result exactly matches the previous audit.

## Action and alignment findings

- Canonical arm extraction: released `action[5:12]`, `float32`.
- Actual arm range: all seven dimensions span `[-1,1]` across the archive.
- Controller: normalized, base-frame, delta `OSC_POSE`; translation scale 0.05 m,
  rotation-vector scale 0.5, gripper `+1=close/-1=open`.
- Correct policy alignment: `obs[t] -> action[t]`, shift 0, HIGH confidence.

The alignment correction follows the initial-state insertion in RoboSuite's
collection wrapper and is corroborated by all 518,396 numerical state
transitions. Details are in `analysis/robocasa_alignment_audit.json`.

## Loader and review

The lazy loader implements two-view 256→128 bilinear RGB conversion, complete
five-observation windows, 10-action chunks, terminal padding masks, explicit
episode-local indexing, and train-ID-only scaler fitting.

Short/median/long eligible episodes were tested:

| selection | episode | frames | sample outputs |
|---|---:|---:|---|
| short | 205 | 531 | RGB `[5,3,128,128]`, action `[10,7]` |
| median | 379 | 1,002 | same |
| long | 189 | 1,968 | same |

The three contact sheets show correctly oriented, natural-colour external and
wrist frames with synchronized progression. All outputs are finite.

## HARD GATE

| gate | result | evidence |
|---|---|---|
| G1 dataset | **FAIL** | archive readable but not the current annotation-bearing official release; official Box URL is 404 |
| G2 ≥400 eligible | PASS | 489 |
| G3 action ordering | PASS | `[5:12]` verified against modality/source |
| G4 base/control | PASS | selected episodes exact-zero/default-only |
| G5 cameras | PASS | 1,521/1,521 metadata-consistent; sampled decode PASS |
| G6 alignment | PASS | HIGH, `t→t`, source plus 518,396 transitions |
| G7 loader | **FAIL** | tensor/metadata loader works, but required raw subtask streams are absent |
| G8 horizon | PASS | 10 actions plus valid mask |
| G9 boundary safety | PASS | no cross-episode read |
| G10 tests | PASS | R1 14/14; combined 24/24 |
| G11 corruption | PASS | zero corrupt/NaN/video mismatch |

Final: **FAIL**. G1 and G7 are critical and cannot be repaired without the
updated official archive. Dropping the annotation requirement or copying labels
from an unofficial conversion would violate the task.

## Cleanup decision

Cleanup was not executed. Safety preflight established both requested targets
are real directories, not symlinks, but the failed data gate overrides path
safety:

| path | bytes | files | result |
|---|---:|---:|---|
| `/ssd1/itaein/datasets/LIBERO/libero_10` | 13,730,613,000 | 10 | PRESERVED |
| `/ssd1/itaein/datasets/CALVIN/debug` | 2,615,026,055 | 4,476 | PRESERVED |

Free space at the cleanup decision was 509,766,660,096 bytes. No bytes were
freed by cleanup.

## Environment and prohibited work

- Audit interpreter: Python 3.9.7, NumPy 1.20.3, pandas 1.3.4,
  PyArrow 17.0.0, OpenCV 4.10.0.
- Dataset environment metadata: RoboCasa 0.5.1, RoboSuite 1.5.2, MuJoCo 3.3.1.
- LeRobot runtime was not required; the archive declares codebase v2.1.
- No PyTorch/model environment was needed for loader validation.
- No model training, Mamba implementation, semantic grouping, timeline creation,
  GPU job, git reset, git push, or unapproved deletion occurred.

## R2 readiness

**NO.** R2 requires official raw subtask annotations. The current data can support
arm-action and camera experiments, but cannot reproducibly build the requested
deterministic four-stage and FULL/CURRENT/HOLD timelines.
