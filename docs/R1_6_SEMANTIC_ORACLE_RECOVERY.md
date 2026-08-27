# R1.6 semantic-oracle recovery

Date: 2026-08-24  
Result: **R1_6_GATE = FAIL; READY_FOR_R2 = NO**

## Source and history audit

The current official RoboCasa `main` is clean and synchronized at
`a07e365c958c4216cd6bbd5f30b47f09a65c6f00`. The exact task implementation is
`robocasa/environments/kitchen/composite/washing_fruits_and_vegetables/wash_fruit_colander.py`.
Relevant official helpers are `object_utils.py::check_obj_in_receptacle`,
`sink.py::get_obj_basin_loc`, `sink.py::get_handle_state`, and
`sink.py::check_obj_under_water`.

Searching current source and all fetched branches/history for `subtask_idx`,
`subtask_name`, `subtask_stage`, and `annotation.human.subtask` found no
generation implementation. The July 7 commits add announcement/documentation
only. This is annotation-generation audit CASE C: no reusable generator is
published, so the simulator-predicate route was examined.

## Reconstruction data

Each episode has `states.npz`, compressed full `model.xml`, and `ep_meta.json`.
For all five fixed samples, the flattened state width exactly equals
`1 + nq + nv` derived from the XML joint list. Thus the archive does retain the
full MuJoCo kinematic state.

The ordinary parquet `observation.state` is insufficient: its 16 dimensions are
base position/quaternion, relative EEF position/quaternion, and two gripper
positions. It contains no object pose, contact, sink joint, or water-site state.

The saved XML is not self-contained. It references hundreds of absolute mesh and
texture paths from the collection machine. Standard robot and Objaverse assets
could in principle be restored, but every sample uses a Lightwheel colander.
Those assets are absent locally and absent from the official RoboCasa Hugging
Face asset repository; the official Lightwheel Box link also returns HTTP 404.
Without the colander collision geometry, the exact official fruit-colander
contact predicate cannot be evaluated.

An isolated MuJoCo installation was attempted in `/tmp`; sandbox network access
was unavailable. This is not the decisive blocker—the missing official
Lightwheel collision assets would still prevent exact model reconstruction.

## Small validation

Fixed selection with seed 42:

| role | episode | frames | full-state layout | water first on | spout >0.1 rad change |
|---|---:|---:|---|---:|---|
| short | 205 | 531 | PASS | 515 | no |
| median | 379 | 1,002 | PASS | 986 | no |
| long | 189 | 1,968 | PASS | 1,952 | yes |
| random A | 341 | 972 | PASS | 956 | no |
| random B | 60 | 991 | PASS | 975 | no |

Initial exact reconstruction: 0/5. Action replay: 0/5. Exact predicate
timelines: 0/5. Final simulator success checks: 0/5. The diagnostic files expose
only exact handle/spout qpos and mark P1–P4 unknown. They are not labels.

SMALL_GATE = FAIL because G2–G4 cannot pass. Per the task, full 489 processing
was not attempted and no threshold tuning, gripper proxy, video inference, VLM,
or manual annotation was substituted.

## Outputs and tests

`semantic_oracle.py` implements only source-backed Boolean reductions, monotonic
first-completion state-machine logic, serialization, and provenance. It does not
extract missing contacts or create real labels. Eleven synthetic unit tests pass;
loader integration is skipped because no full labeling passed. Existing R1 and
common-interface tests remain 24/24 PASS.

`washfruitcolander_semantic_boundaries.csv` contains its header only. The stats
record 489 episodes as unprocessed, not invalid, so no misleading 0% acceptance
rate is reported. No per-frame semantic label directory or boundary review
montage was created.

## Recommendation

Recommendation A: wait for restoration of the official RoboCasa annotation/data
and Lightwheel asset endpoints. The current dataset remains suitable for the
already validated action/camera baseline, but R2 semantic timelines must not be
built from guessed boundaries.

No model was trained and no data was deleted.
