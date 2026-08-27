# CALVIN Training Chain Audit

Audit date: 2026-08-24  
Source: official CALVIN D→D `auto_lang_ann.npy` and physical episode metadata  
Training-supervision verdict: **FAIL**

## 1. Question answered

This audit asks a narrower question than “does CALVIN evaluate five tasks without an environment reset?” It asks whether the released offline D→D play data provides enough **actual continuous, language-annotated 4–5-stage supervision** to train the proposed methods fairly:

- full compound instruction throughout;
- current subinstruction;
- transition instruction followed by HOLD.

The answer is no under a defensible short-gap definition. Relaxing continuity to an entire physical recording episode creates thousands of windows but introduces unlabeled gaps and almost no repeated compound patterns.

## 2. Data source and coverage

The full 177 GB archive was not downloaded. The official server's ZIP64 directory was range-read, then only the relevant metadata entries were extracted and CRC32-verified.

| Split | Raw annotations | Task IDs | Physical recording episodes | Default embedding |
|---|---:|---:|---:|---|
| training | 5,124 | 34 | 31 | `[5124,1,384]` float32 |
| validation | 1,011 | 34 | 4 | `[1011,1,384]` float32 |

Every annotation interval maps to exactly one `ep_start_end_ids.npy` interval. Physical episode bounds, rather than language gaps, are the reset source of truth.

## 3. Interval convention

This audit treats `info.indx` as inclusive `[start,end]`, matching the requested convention and the dataset loader's usable final index behavior. Duration is:

```text
duration_frames = end - start + 1
duration_seconds = duration_frames / 30
```

Raw annotations often overlap because the automatic annotator can produce several sliding clips and utterance variants for the same underlying task execution. Counting each row as a new stage would strongly inflate chain length.

## 4. Event deduplication

Before chain construction, annotations are grouped by `(physical_episode, task_id)` and merged when their intervals overlap or touch. Each merged event preserves:

- minimum start and maximum end;
- all source annotation IDs;
- all raw utterance variants;
- source annotation count;
- one deterministic representative utterance, chosen lexicographically.

This reduces:

| Split | Raw annotations | Deduplicated semantic events | Raw/event ratio |
|---|---:|---:|---:|
| training | 5,124 | 3,259 | 1.57× |
| validation | 1,011 | 606 | 1.67× |

Different task IDs that overlap are not merged. They remain visible as an ambiguity/collision and are handled explicitly by Definition A.

## 5. Gap definition

For chronological consecutive events in the same physical episode:

```text
gap = next.start - prev.end - 1
```

- `gap < 0`: annotations overlap;
- `gap = 0`: intervals touch exactly;
- `gap > 0`: unlabeled frames occur between events.

Reset boundaries are emitted in the gap CSV with no numeric gap and never form a chain.

## 6. Gap statistics

Deduplicated-event gaps:

| Split | N | Min | Mean | Median | p75 | p90 | p95 | Max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| training | 3,228 | -68 | 92.14 | 51 | 138 | 252 | 348.3 | 901 |
| validation | 602 | -73 | 97.81 | 48.5 | 138 | 302.1 | 386.8 | 921 |

Positive-only gaps:

| Split | N | Min | Mean | Median | p75 | p90 | p95 | Max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| training | 2,608 | 1 | 117.80 | 80 | 162.25 | 282 | 375 | 901 |
| validation | 493 | 1 | 123.37 | 74 | 181 | 334.8 | 400.2 | 921 |

At 30 Hz, the positive median is 2.67 s for training and 2.47 s for validation. Maximum gaps exceed 30 s.

Training histogram:

| Gap frames | Count |
|---:|---:|
| overlap `<0` | 594 |
| 0 | 26 |
| 1–5 | 142 |
| 6–15 | 259 |
| 16–30 | 309 |
| 31–64 | 432 |
| 65–128 | 600 |
| 129–256 | 552 |
| 257–512 | 271 |
| 513–1000 | 43 |

The complete row-level values are in [`calvin_language_segment_gaps.csv`](../analysis/calvin_language_segment_gaps.csv), including both raw-annotation and deduplicated-event adjacency.

## 7. Why the small-gap threshold is 30 frames

Definition B uses **30 frames = 1.0 s**.

This threshold is empirical and physically interpretable:

- it is near the lower quartile of positive training gaps (p25 ≈27 frames);
- it is near the p30 of validation positive gaps (≈30 frames);
- it limits the unlabeled interval to one second at the actual 30 Hz control rate;
- it is less permissive than the 65-frame typical annotated clip duration.

A larger threshold quickly admits the bulk of arbitrary play: the median positive gap is 80 train frames, and 1–128-frame gaps alone contain 1,742 training edges. A smaller 5-frame threshold would be clean but too sparse to test five-stage scale meaningfully.

## 8. Compared chain definitions

### A. Strict adjacency from the requested inequality

```text
next.start <= prev.end + 1
```

Equivalently, `gap <= 0`. This includes interval overlap. It is reported exactly as requested but is not automatically a sequential demonstration: two task labels can describe the same frames.

### B. Empirical small gap

```text
0 <= gap <= 30 frames
```

Events do not overlap, and the unlabeled interval is no more than one second. This is the primary training-evidence definition.

### C. Continuous physical episode

```text
same ep_start_end_ids interval; no reset
```

Any gap is allowed. This measures what a full-play curriculum could mine, but does not solve semantic conditioning during unlabeled frames.

## 9. Training counts

Each cell is `occurrences / unique task patterns`.

| Definition | 2 stages | 3 stages | 4 stages | 5 stages |
|---|---:|---:|---:|---:|
| A strict | 620 / 281 | 145 / 141 | 32 / 32 | 7 / 7 |
| **B ≤1 s** | **736 / 413** | **165 / 163** | **31 / 31** | **9 / 9** |
| C same episode | 3,228 / 918 | 3,197 / 2,846 | 3,166 / 3,141 | 3,135 / 3,133 |

Under B:

- 8 of 9 five-event windows contain at least four distinct task IDs;
- no five-stage pattern repeats;
- no four-stage pattern repeats;
- only two three-stage patterns repeat twice;
- the most frequent two-stage pattern occurs 11 times;
- there are no patterns with 20 or more occurrences at any length.

## 10. Validation counts

| Definition | 2 stages | 3 stages | 4 stages | 5 stages |
|---|---:|---:|---:|---:|
| A strict | 109 / 84 | 17 / 17 | 4 / 4 | 1 / 1 |
| **B ≤1 s** | **156 / 131** | **42 / 41** | **4 / 4** | **0 / 0** |
| C same episode | 602 / 413 | 598 / 584 | 594 / 594 | 590 / 590 |

The absence of even one B-definition validation five-stage chain prevents an offline matched-chain validation protocol without changing the split or continuity rule.

## 11. Scale classification

Requested per-pattern rubric:

- `>=500`: very strong;
- `100–499`: strong;
- `20–99`: exploratory;
- `<20`: very weak.

Every B-definition four- and five-stage pattern is **very weak** because each occurs exactly once. C does not rescue pattern scale: among 3,133 training five-stage patterns, 3,131 occur once and only two occur twice.

## 12. Semantic dependency scoring

Dependency is computed from the official condition/effect dictionary in [`multistep_sequences.py`](../external/calvin/calvin_models/calvin_agent/evaluation/multistep_sequences.py):

- **HIGH edge**: the preceding task changes an official object/fixture state key to a value required by the next task, such as `open_drawer -> lift_*_block_drawer` or `lift_* -> place_*`; the generic `grasped` flag alone is excluded to avoid false dependencies between different objects;
- **MEDIUM edge**: tasks share a semantic entity/state key but do not have a direct changed-effect/precondition match;
- **LOW edge**: no such official relation is found.

Chain score:

- HIGH only when every edge is HIGH;
- MEDIUM when at least one edge is HIGH or at least half are MEDIUM;
- LOW otherwise.

B-definition occurrence scores:

| Split/length | HIGH | MEDIUM | LOW |
|---|---:|---:|---:|
| train 2 | 100 | 235 | 401 |
| train 3 | 4 | 103 | 58 |
| train 4 | 0 | 20 | 11 |
| train 5 | 0 | 7 | 2 |
| validation 4 | 1 | 1 | 2 |
| validation 5 | 0 | 0 | 0 |

There is no all-HIGH five-stage short-gap training chain.

## 13. Actual B-definition five-stage candidates

All nine training candidates are data-derived. They are not the hypothetical chains proposed in R0-A.

| Pattern | Gaps | Dependency |
|---|---:|---|
| rotate pink left → push red right → push red left → rotate pink right → open drawer | `0,8,23,6` | LOW |
| place in slider → lift pink from drawer → turn on LED → move slider left → lift blue from slider | `24,18,12,7` | MEDIUM |
| lift pink from drawer → turn on LED → move slider left → lift blue from slider → place in drawer | `18,12,7,10` | MEDIUM |
| push pink left → unstack → stack → turn on lightbulb → unstack | `5,15,4,7` | MEDIUM |
| unstack → stack → turn on lightbulb → unstack → turn on LED | `15,4,7,6` | MEDIUM |
| stack → turn on lightbulb → unstack → turn on LED → lift pink from slider | `4,7,6,11` | MEDIUM |
| rotate blue left → rotate blue left → rotate blue right → rotate blue right → lift blue from table | `9,7,24,21` | MEDIUM |
| close drawer → move slider left → lift red from slider → lift blue from table → move slider right | `25,24,21,22` | MEDIUM |
| push blue left → rotate blue left → lift red from slider → move slider right → turn off lightbulb | `4,11,13,1` | MEDIUM |

Several are chronological but weakly causal. Repeated rotate/unstack stages also show why “five annotations” should not be treated automatically as a clean five-goal program.

## 14. Full-play Definition C

Definition C is attractive numerically:

- 3,135 train five-event windows;
- 590 validation five-event windows;
- 31 and 4 independent physical episodes respectively.

But the window counts are highly overlapping. Sliding one event forward creates the next chain, so 3,135 is not the number of independent demonstrations. Pattern repetition is effectively absent.

The gaps carry real actions and observations in the full dataset, but no language segment identifies what the demonstrator intended. A full-play training protocol must choose among:

- HOLD during all gaps and include BC loss;
- HOLD during gaps but mask BC loss;
- keep the previous stage instruction during gaps;
- reveal the next stage early;
- drop gap frames, which breaks recurrent-time continuity.

Each option changes the learning problem. None is specified by the official dataset.

## 15. HOLD and state-retention risk

For the transition/HOLD method, an unlabeled gap could be represented by HOLD, but HOLD would then mean two different things:

1. ordinary within-stage continuation after a transition event;
2. activity between two independently mined language clips whose intent is unknown.

This confounds the proposed semantic-memory mechanism. The model can learn that HOLD spans unrelated teleoperation rather than continued execution of the last announced stage.

For the current-subinstruction method, the ambiguity is worse: there is no official current subinstruction during a gap. Assigning the next instruction early leaks future task identity; retaining the previous one can be false; masking language/loss changes the baseline asymmetrically.

## 16. Offline and online transition boundaries

Offline:

```text
auto_lang_ann interval [start,end]
```

Online official evaluation:

```text
start_info = env.get_info()
step action
task_oracle(start_info,current_info,{subtask})
switch instruction on first success
```

These are related but not equal. The offline clip end was generated around a detected task interval and can contain pre/post-success slack. Online switching is the first detected oracle success and has no deliberate unlabeled delay.

Before using offline ends as oracle transition labels, a calibration sample must replay or reconstruct states and measure:

```text
offset = offline_end - first_oracle_success_frame
```

No such calibration was possible from metadata alone, and the debug sample does not provide a repeated five-stage set on which to establish it.

## 17. Compound instruction token audit

For every A/B/C four- and five-stage occurrence, one official utterance per event was selected deterministically and concatenated with `; then `. The exact Hugging Face CLIP ViT-B/32 tokenizer used by MaIL was loaded locally with special tokens enabled.

| Group | Count | Min | Mean | Median | Max | Overflow >77 |
|---|---:|---:|---:|---:|---:|---:|
| Train B, 4 stages | 31 | 34 | 40.35 | 40 | 48 | 0 |
| Train B, 5 stages | 9 | 44 | 49.00 | 49 | 55 | 0 |
| Train C, 4 stages | 3,166 | 24 | 39.92 | 40 | 53 | 0 |
| Train C, 5 stages | 3,135 | 33 | 49.64 | 50 | 64 | 0 |
| Validation C, 5 stages | 590 | 38 | 50.23 | 50 | 68 | 0 |
| All audited compounds | 7,573 | 24 | 44.81 | 45 | 68 | 0 |

Tokenizer overflow is not a blocker. The raw rows are in [`calvin_compound_instruction_tokens.csv`](../analysis/calvin_compound_instruction_tokens.csv).

## 18. Human validation plan and burden

Minimal validation if one still studies the nine B candidates:

1. render each event plus 30 context frames before/after;
2. show both cameras, task ID, raw utterance, `[start,end]`, and gap count;
3. verify that every event represents a distinct completed semantic action;
4. reject overlapping task-oracle effects or annotation aliases;
5. record the first visually evident success frame and compare with `end`;
6. report all nine, not a favorable subset.

Estimated burden is 45–75 minutes for all 40 B-definition train 4/5-stage windows, assuming cached videos and structured forms. Exhaustive review of all 3,259 deduplicated events is roughly 45–80 hours. A 5% stratified event sample is 2.5–4 hours but cannot establish per-pattern five-chain scale.

## 19. Training recommendation

Do not build the primary matched-supervision experiment on these chains.

CALVIN remains viable for either:

- official single-instruction training followed by compositional long-horizon evaluation; or
- a clearly labeled, exploratory full-play curriculum study whose gap/HOLD rules are themselves an experimental contribution.

Neither is the originally requested matched supervised comparison. Calling either one a small adapter change would understate the methodological difference.

## 20. Gate conclusion

Training Gate E is **FAIL** because:

- the primary non-overlap threshold yields only 9 five-stage train windows and no validation windows;
- every train five-stage pattern is a singleton;
- none of those nine has HIGH dependency across every edge;
- the relaxed C count is dominated by overlapping windows, singleton patterns, and unlabeled spans;
- converting full play into aligned FULL/current/HOLD supervision requires a new protocol assumption.

This critical failure sets the overall migration decision to **REJECT**. See [`CALVIN_SMALL_VALIDATION.md`](CALVIN_SMALL_VALIDATION.md) for the complete A–G gate and next-benchmark recommendation.
