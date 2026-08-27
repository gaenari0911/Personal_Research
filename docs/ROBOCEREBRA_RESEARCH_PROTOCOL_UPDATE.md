# RoboCerebra 연구 프로토콜 변경 기록 (R1-RC)

## 변경 범위

이 문서는 R0-E 결과를 덮어쓰지 않는다. `ROBOCEREBRA_FAST_AUDIT.md`와 기존
`analysis/robocerebra_*` audit 산출물은 당시 판정의 재현 기록으로 그대로
보존한다. R1-RC는 연구 질문이 달라졌기 때문에 별도의 적합성 기준을 적용한다.

## OLD R0-E criterion

R0-E는 `Pick`, `Place`, `Open`, `Pour` 같은 원자적 command보다 상위인 semantic
macro-subgoal을 요구했다. 예를 들어 여러 command를 임의의 `Store X` parent
stage로 묶지 못한다는 이유로 ordered semantic subtask 관련 gate가 실패했다.

## CURRENT R1-RC criterion

RoboCerebra가 공개한 ordered natural-language `Step:` 자체를 유효한 subtask로
정의한다. 각 Step에 붙은 공식 temporal range를 subtask transition ground truth로
사용한다. `Pick + Place -> Store` 같은 새 grouping, 수동 boundary annotation,
off-by-one correction은 수행하지 않는다.

이 정의는 decomposition quality나 Behavior Cloning 성능이 아니라 다음 질문에
맞춰져 있다.

> 연속 trajectory를 순차 입력할 때 persistent temporal representation이 현재 및
> 과거의 language-grounded Step progress를 얼마나 오래 보존하는가?

따라서 R0-E의 `G2 ordered semantic subtasks = FAIL` 및 `G4 >=4 semantic stages =
FAIL`은 현재 연구의 gate로 재사용하지 않는다. exact FULL instruction 반복 수와
semantic macro-parent 유무도 R1-RC gate가 아니다.

## 데이터 해석 계약

- Primary episode는 공식 원본 HDF5의 continuous trajectory이다.
- 공식 Step 순서와 text는 그대로 보존한다.
- interval은 공식 변환 코드의 slicing과 실제 길이 검증에 따라 half-open
  `[start, end)`로 해석한다.
- interval이 실제 trajectory length와 맞지 않으면 고치지 않고 strict-clean
  dataset에서 제외한다.
- step transition에서 episode나 모델 state를 reset하지 않는다. reset signal은
  trajectory 시작에서만 제공한다.
- raw MuJoCo full state는 privileged state이므로 모델 입력에서 제외한다.
- model input과 progress/transition analysis label은 서로 다른 dictionary에 둔다.

## 출처 고정

- 공식 데이터셋: <https://huggingface.co/datasets/qiukingballball/RoboCerebra>
- 데이터 revision: `5d2e1e361bf65aabbe4d18179515f5a10936cc96`
- 공식 코드: <https://github.com/qiuboxiang/RoboCerebra>
- 로컬 코드 revision: `2573426c13dfcd5e7d7831c15587b058aaa1c0c0`

