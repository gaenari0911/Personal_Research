# R1.6 WashFruitColander stage-definition audit

Commit: `a07e365c958c4216cd6bbd5f30b47f09a65c6f00`  
Verdict: four conceptual subtasks are source-backed, but four reproducible
per-episode boundaries are **not** validated.

## Official task logic

`WashFruitColander` directly subclasses `Kitchen`. It registers the sink and a
nearby counter, initializes the faucet off, creates one to three fruits plus one
colander on the counter, and emits the instruction:

```text
Put the colander in the sink, put the fruit in the colander, and turn on the
sink faucet and pour water over the colander.
```

The actual success condition is only:

```text
all fruits satisfy OU.check_obj_in_receptacle(fruit, colander)
AND
sink.check_obj_under_water(colander)
```

The official task-attribute metadata records `num_subtasks=4`. The documentation
skill tags associate the task with PickPlace, knob twist, and lever turn. The
task class itself has no ordered state machine or intermediate completion log.

## Candidate stages

### S1 — colander in sink

- Meaning: place the colander in a sink basin.
- Predicate: `Sink.get_obj_basin_loc(..., partial_check=False) != "none"`.
- Source: task language/object configuration plus the official sink basin helper.
- Persistence: reversible; moving the colander can make it false.
- Inputs: colander bbox/pose and sink basin geometry.

### S2 — all fruits in colander

- Meaning: place every episode fruit into the colander.
- Predicate: conjunction of official `OU.check_obj_in_receptacle` calls.
- Source: this exact conjunction appears in `_check_success`.
- Persistence: reversible.
- Inputs: MuJoCo fruit-colander contact, positions, and colander radius.

### S3 — colander aligned with water site

- Meaning: establish the geometric under-spout condition, potentially by turning
  the spout.
- Predicate candidate: the xy/z portion of `Sink.check_obj_under_water`, before
  applying `water_on`.
- Source: official sink helper and atomic `TurnSinkSpout` task; the documentation
  tags WashFruitColander with knob twist.
- Persistence: reversible.
- Unresolved: WashFruitColander records no desired left/right spout orientation.
  In the fixed five samples, only episode 189 changed the spout by more than
  0.1 rad. The other four do not provide a universal spout-completion event.

### S4 — water on / full task completion

- Meaning: activate water while the fruit-filled colander is under the stream.
- Predicate: S2 and S3 and official `water_on`.
- Source: exact decomposition of `_check_success` and
  `Sink.check_obj_under_water`.
- Persistence: reversible.

## Stage-count verdict

| question | result |
|---|---|
| official metadata count | 4 |
| official raw annotation phase count | unknown; bytes unavailable |
| task-language conceptual goals | 4 only when spout alignment is separated |
| deterministic research stages | not validated |

The fourth conceptual action is plausible, but an independent S3 transition is
not universal in the sample and its generation rule is absent. R1.6 therefore
does not force four boundaries.
