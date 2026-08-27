# VLABench representative review set

Each H.264 MP4 concatenates the official `observation.images.image` (left) and
`observation.images.wrist_image` (right) streams at the released 10 Hz.  Each
PNG samples 21 evenly spaced frames from the same episode.  The overlay says
`official semantic boundary unavailable` deliberately: the released parquet
schema has no stage, skill, predicate, or boundary column.

| candidate | episode | frames | duration | what to inspect |
|---|---:|---:|---:|---|
| cluster_billiards | 0 | 610 | 61.0 s | four object transfers, no reset/cut, visibility in front+wrist views |
| cluster_drink | 8 | 545 | 54.5 s | four intended transfers; terminal gripper state makes transition-only segmentation incomplete |
| cluster_toy | 30 | 634 | 63.4 s | four intended transfers; object identity/category ambiguity and occlusion |

Do not treat a gripper transition as a semantic label.  Episode 0 has four
close-to-reopen spans (139/186/138/144 frames), while episodes 8 and 30 expose
only three reopenings before termination.  This is useful as a QA proxy, but it
does not meet boundary level 1-3.
