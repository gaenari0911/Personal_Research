# R1.5 RoboCasa365 annotation recovery

Audit date: 2026-08-24  
Result: **R1_5_GATE = CONDITIONAL**

## Outcome

The official 2026-07-07 annotation update is real, but its public data endpoint
is currently broken. RoboCasa's latest official `main` announces per-frame
subtask index, atomic-skill name, stage, and natural-language instruction for
target composite datasets. The latest repository and official documentation
still route WashFruitColander to a single Box share, and both that share and the
direct tar URL return HTTP 404. No unofficial annotation source was used.

This matches the task's narrow CONDITIONAL rule: the release and exact official
URL are confirmed, while maintainer-controlled delivery prevents obtaining the
bytes. It is not reported as PASS because no parquet schema, episode identity,
or frame-level annotation alignment could be inspected.

## Repository and registry audit

| property | value |
|---|---|
| official remote | `https://github.com/robocasa/robocasa.git` |
| local/upstream main | `a07e365c958c4216cd6bbd5f30b47f09a65c6f00` |
| worktree | clean |
| 7/7 announcement commit | `f1e7fbee9879359ea5f3aa9c7c20a7bbcc706728` |
| task/split/source | `WashFruitColander / target / human` |
| registered path | `v1.0/target/composite/WashFruitColander/20250811/lerobot` |
| horizon | 3150 |
| archive key | `target/composite/WashFruitColander/20250811/lerobot.tar` |
| expected bytes/checksum | not published |

`get_ds_meta` derives the path above from `dataset_registry.py`. The download
script does not query a second service: it reads `box_links_ds.json`, changes
`https://utexas.box.com/s/<id>` into
`https://utexas.box.com/shared/static/<id>.tar`, downloads that one tar, and
extracts it.

The WashFruitColander Box entry was last changed on 2026-02-24. The 2026-07-07
annotation commit changed documentation only and did not update the registry or
Box ID. The current official `main`, the 365 release branch, and the horizon
update branch all resolve the task to the same version date and share ID.

## Endpoint diagnosis

| endpoint | status on 2026-08-24 |
|---|---:|
| WashFruitColander Box share | HTTP 404 |
| WashFruitColander direct `.tar` | HTTP 404 |
| WaffleReheat peer share | HTTP 404 |
| WashLettuce peer share | HTTP 404 |
| StoreLeftoversInBowl peer share | HTTP 404 |

The peer failures show this is not merely a mistyped WashFruitColander URL. The
official Hugging Face organization publishes only `robocasa-assets`; it has no
RoboCasa365 trajectory dataset or annotation delta. The official website and
documentation point back to the same GitHub registry/download script.

## Minimal download decision

No annotation-only, one-parquet, metadata-only, or alternate official endpoint
is published. Range extraction cannot be tested against a 404 response. A full
download was therefore not attempted, and
`/ssd1/itaein/datasets/RoboCasa365/WashFruitColander_latest_staging` was not
created. The existing dataset was not overwritten.

Personal Hugging Face conversions and other unofficial datasets appeared in
search, but were explicitly excluded. NVIDIA remains only the provenance of the
old R1 reference archive, not a source of latest annotations.

## Previous dataset and reconciliation baseline

The preserved v2.1 dataset remains 507 episodes and 518,903 frames. It has only
`annotation.human.task_description` and `annotation.human.task_name`; the four
new fields are absent.

`washfruitcolander_episode_mapping.csv` records all 507 episode lengths and two
deterministic fingerprints:

1. SHA-256 of the exact float64 action array;
2. SHA-256 of action, contiguous frame index, and float32 timestamp bytes.

Every row is marked BLOCKED rather than guessed. When official bytes return,
mapping must compare these fingerprints, frame counts, task metadata, and video
duration before accepting episode indices. The existing 489 arm-only IDs are
all preserved, but zero can yet be called annotation-matched.

## Schema, timelines, and visualization

Remote parquet/schema metadata is not exposed by the dead Box endpoint. The
required fields' dtype, shape, examples, and unique counts are therefore
recorded as unknown—not inferred from an unofficial conversion.

`washfruitcolander_annotation_runs.csv` contains only its schema header because
there are no official annotation rows to summarize. No overlay was generated.
This avoids fabricated timelines and manual semantic reconstruction.

## Loader and tests

`src/robocasa_phase1/interface.py` was intentionally not changed. Task R1.5
allows the loader extension only after official acquisition and exact
episode/frame alignment. A seven-test R1.5 suite was added and is automatically
skipped until the official staging dataset exists. It covers the requested
schema, frame count, episode mapping, manifest mapping, loader, same-row shift,
and episode-isolation gates once bytes become available.

All existing R1/common-interface tests remain 24/24 PASS. The seven new
official-data tests are 0 failed / 7 skipped due solely to the missing staging
archive.

## Gate and next action

G1 is CONDITIONAL and G2–G6 are blocked. G7 regression is PASS. R2 is not ready.
The only valid unblocker is restoration of the official Box share or publication
of an official RoboCasa annotation archive/delta. When that occurs, download it
to the staging path, run the prepared seven tests, reconcile all fingerprints,
and only then enable raw annotation fields in the loader.

No model was trained and no LIBERO, CALVIN, or RoboCasa dataset was deleted.
