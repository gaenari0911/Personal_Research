# R1-RC RoboCerebra Long-Horizon Memory Dataset

## 결과

R1-RC gate는 **PASS**, strict-clean strength는 **STRONG**이다. 공식 training
metadata 1,000개와 공개된 원본 HDF5 996개를 전수 대조한 결과, 수동 보정 없이
914개(91.4%)가 strict clean continuous trajectory가 되었다. 이 결과는 이전 R0-E
REJECT를 수정한 것이 아니라 `ROBOCEREBRA_RESEARCH_PROTOCOL_UPDATE.md`의 새 연구
정의로 별도 평가한 것이다.

## Source와 materialization

- 공식 데이터: `qiukingballball/RoboCerebra`
- 고정 revision: `5d2e1e361bf65aabbe4d18179515f5a10936cc96`
- 로컬 root: `/ssd1/itaein/datasets/RoboCerebra`
- training metadata: 1,000 rows
- 공개된 training HDF5: 996 files, 2,456,899,944 bytes
- materialized HDF5: 996/996
- 대표 visual review MP4: 5 files, 11,762,470 bytes

metadata에는 있으나 해당 revision에 HDF5 object가 없는 ID는
`coffee_table/case485`, `case487`, `case500`, `case622` 네 개다. 전체 RGB release는
다운로드하지 않았고, strict validation에 필요한 HDF5 전부와 대표 review video만
resume 가능한 방식으로 받았다. 기존 `analysis/robocerebra_public_samples`는 보존했다.

## Strict boundary validation

interval convention은 half-open `[start, end)`이다. official converter가 이전 end부터
새 end까지 Python slice로 Step을 쓰는 방식, 연속 interval의 `previous.end ==
next.start`, 실제 action/state length `T`와 terminal end의 일치 여부를 함께
검증했다. 어떤 off-by-one 수정도 하지 않았다.

| 항목 | 결과 |
|---|---:|
| metadata candidates | 1,000 |
| previous metadata-only preliminary clean | 974 |
| final strict clean | 914 |
| invalid | 86 |
| valid rate | 91.4% |

invalid reason은 서로 겹칠 수 있다. `terminal_mismatch` 61, nonpositive interval 21,
gap/overlap 10, source HDF5 missing 4, first start nonzero 2다. 모든 invalid row에는
`frame_assignment_not_exact`도 파생 reason으로 기록된다. 조합별로는 pure terminal
mismatch 56, pure nonpositive 13, nonpositive+gap/overlap 6, pure gap/overlap 4,
missing source 4, 기타 중첩 3이다. 원본 파일이나 annotation은 삭제·수정하지 않았다.

## Canonical representation

`analysis/robocerebra_memory_episode_index.json`은 914개 episode에 대해 FULL,
ordered official Steps, exact boundary, source SHA-256, state/action source, visual source
contract를 저장한다. per-frame 문자열/label을 대량 복제하지 않으며 loader가 lazy하게
계산한다.

대표적인 `coffee_table/case515`는 8,120 frames, 19 Steps다.

```text
FULL: Set up a cooking and preparation station to heat milk in a frypan on a
      stove and create a dessert-drink blend by mixing chocolate pudding,
      orange juice, and wine in a black bowl before transferring it to a basket.

S1  [0, 367)       Pick up the frypan from the coffee table.
S2  [367, 698)     Place the frypan on the flat stove at cook region.
S3  [698, 1207)    Turn on the flat stove at cook region.
...
S7  [2642, 3048)   Pick up the akita black bowl from the coffee table.
...
S13 [4377, 5946)   Pour the orange juice into the akita black bowl.
...
S19 [7802, 8120)   Place the akita black bowl into the basket at contain region.
```

## Long-horizon statistics

| 분포 | min | mean | median | p75 | p90 | max |
|---|---:|---:|---:|---:|---:|---:|
| trajectory frames | 258 | 2,945.33 | 2,811.5 | 3,503 | 4,282.1 | 8,120 |
| trajectory seconds | 12.9 | 147.27 | 140.575 | 175.15 | 214.105 | 406.0 |
| Steps / trajectory | 2 | 8.96 | 9 | 10 | 12 | 23 |
| transitions / trajectory | 1 | 7.96 | 8 | 9 | 11 | 22 |
| Step duration frames | 27 | 328.66 | 288 | 410 | 558 | 2,551 |
| Step duration seconds | 1.35 | 16.43 | 14.4 | 20.5 | 27.9 | 127.55 |
| Step duration / window-5 | 5.4× | 65.73× | 57.6× | 82× | 111.6× | 510.2× |

전체 transition은 7,277개다. 한 episode 안의 maximum
`steps_since_transition`은 median 570.5, p90 920.8, max 2,550 frames다. 따라서
5-frame local context보다 훨씬 긴 memory-depth 분석 구간을 제공한다.

## State input audit

996개 HDF5 모두 `states`가 `float64 [T, width]`이고 width는 scene/object 구성에
따라 47..162다. embedded MuJoCo XML에서 계산한 `1 + nq + nv`와 width가 996/996
일치했고, 첫 qpos 9개는 996/996에서 Panda joint 7개 + gripper finger 2개였다.
timestamp는 996/996에서 strictly monotonic이며 median delta는 0.05초다.

raw state 전체는 time + robot/object/fixture qpos + qvel을 포함하는 privileged
simulator state다. 모델에는 이를 넣지 않고 `states[t, 1:10]`의 robot qpos 9D만
권장한다. EEF pose는 직접 저장되지 않아 공식 simulator forward kinematics가
필요하다.

action은 원본 7D `[Δx, Δθ, Δgrip]` 20 Hz stream을 삭제하지 않고 optional loader
field로 보존했다. memory experiment의 기본 model input이나 mandatory prediction
target으로는 사용하지 않는다.

## Visual alignment와 review

원본 HDF5 안에는 RGB가 없고 metadata의 external MP4를 사용한다. 기존 3개 audit
sample과 새 대표 5개 모두 video image count가 `T+1`이었다. model timestep에는
image `0..T-1`을 직접 대응하고 extra final image는 제외한다. MP4 container는 60
fps지만 state/action의 논리 주기는 20 Hz이므로 매 3번째 image를 고르지 않는다.

공식 converter는 retained source index마다 state를 set하고 external/wrist를 render한
뒤 같은-index action과 묶는다. 그러나 no-op을 제거하고 Step별 episode로 slice하므로
converted local index를 원본 continuous timestep으로 쓰면 안 된다. wrist가 필요하면
원본 state `t`를 filtering 없이 replay-render하는 adapter를 사용한다. 현재 R2는
state + original external RGB로 시작 가능하다.

short, median length, long, median transition, high transition 대표 5개에서 모든
boundary 직전/시작 image를 decode했다. 모두 `T+1`, decode GOOD, 명백한 gross
boundary 문제 없음으로 기록했다. annotation 수정은 하지 않았다.

## Conditioning과 labels

FULL, CURRENT, HOLD는 deterministic하게 생성된다. HOLD는 각 Step 시작에서 새 command,
그 외 `[HOLD]`이며 hidden state freeze/reset을 의미하지 않는다. 실제 3 trajectory
전체 timeline CSV는 `analysis/robocerebra_conditioning_samples/`에 있다.

analysis-only label은 current Step, transition event, steps since transition,
cumulative transition count, previous-1..5다. `frame`, boundary, progress label은
`model_input` dictionary에 포함되지 않는다.

## Split

seed 42로 scene, Step-count quartile, duration quartile을 stratify하고 source SHA-256
group을 함께 배치했다.

- train: 734
- val: 85
- test: 95

trajectory ID, source path, source hash의 split 간 overlap은 모두 0이다. duplicate hash
group도 0이다. exact FULL text 한 개가 split을 넘지만 서로 다른 trajectory이므로
입력 text 중복 기록일 뿐 source leakage가 아니다.

## 구현과 검증

- loader: `src/robocerebra_memory/interface.py`
- builder: `tools/build_robocerebra_memory_dataset.py`
- visual review: `tools/render_robocerebra_memory_review.py`
- tests: `tests/test_robocerebra_memory.py`,
  `tests/test_robocerebra_memory_artifacts.py`

단위/interface 17개와 full-metadata smoke 7개, 총 24/24가 통과했다. smoke test는
914개 episode 가독성, 모든 text/boundary, action/state length, terminal assignment,
transition count, split disjointness와 hash leakage를 검사한다.

## Scope stop

이 Task에서는 모델 학습, Behavior Cloning, Mamba 구현, probe 학습, simulator
rollout, semantic regrouping, 수동 boundary correction, GPU job, git push를 수행하지
않았다.
