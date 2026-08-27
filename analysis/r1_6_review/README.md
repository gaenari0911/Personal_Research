# R1.6 review status

No semantic-boundary montage or overlay was generated because the five-episode
SMALL_GATE failed before any valid boundary existed. Creating a transition
visualization at guessed frames would turn a reconstruction failure into a
fabricated label.

The per-frame diagnostic CSVs in `analysis/r1_6_small_predicate_timelines/`
contain only source-exact handle/spout joint values recovered from the flattened
simulator states. `P1` through `P4` and semantic stage are explicitly marked
`UNKNOWN` / `UNAVAILABLE`.

Existing R1 camera sheets were preserved but were not reused to infer labels.
