# Stage B Memory Evaluation

## 목적과 Stage A와의 분리

Stage A는 future visual representation prediction으로 B0/B1/B2/B3 encoder와 Mamba를
학습한다. Stage B는 그 학습이 끝난 뒤 backbone을 완전히 freeze하고, 이미 형성된
representation에서 현재 및 과거 official Step 정보가 선형으로 복원되는지만 측정한다.
두 단계를 분리하면 Step label이 representation learning objective로 역전파되는 것을
막고, memory 정보를 새로 학습하는 probe가 아니라 기존 representation을 읽는 probe로
해석할 수 있다.

Stage B는 Behavior Cloning이나 action prediction이 아니다. 외부 MaIL 코드를 수정하지
않고 R3 `MemoryExperimentModel`과 Stage A `best_val.pt` checkpoint를 그대로 읽는다.

## Representation과 probe

- `r_t`는 현재 RGB CLIP feature와 robot qpos 9D를 융합한 128D 값이다. Language와
  temporal Mamba 이전 값이므로 instantaneous observation/state control이다.
- `z_t`는 현재 timestep token에 대응하는 Mamba 마지막 LayerNorm 출력 128D다. Raw
  convolution/SSM cache가 아니다.

Stage B는 bias가 없는 `Linear(128,512)` 다섯 개를 학습한다. `current`, `prev1`,
`prev2`, `prev3` 네 probe는 모두 같은 현재 timestep의 동일한 `z_t`를 입력받는다.
과거 timestep의 `z`를 입력하는 구조가 아니다. 각 probe는 서로 다른 weight storage와
optimizer selection을 가지므로 ranking도 독립적이다. 다섯 번째 probe는 `r_t`에서
current Step을 복원하는 독립 control이며 `z_t` current probe와 weight를 공유하지 않는다.
Linear probe는 capacity가 작은 고정 측정기라서 backbone 자체에 linearly recoverable한
정보가 있는지를 보는 데 사용한다. Probe backward 후에도 frozen backbone gradient는
`None`이어야 한다.

## Candidate, GT label, ranking

Candidate set은 각 trajectory에 실제 등장한 ordered official Step text만으로 구성한다.
각 text의 candidate representation은 Stage A cache에 있는 동일 frozen CLIP text encoder의
정규화된 512D embedding이다. 다른 trajectory의 Step은 negative로 합치지 않는다.

GT는 embedding 그 자체가 아니라 Step identity, 즉 casing/whitespace를 정규화한 official
Step text label이다. Current GT는 현재 Step이고 previous-k GT는
`current_step_index-k`다. 이전 Step이 아직 존재하지 않으면 해당 loss와 metric에서
제외한다.

**GT와 embedding 자체를 metric에서 직접 비교하는 것이 아니다. 각 probe의 q와
trajectory-local Step candidate embedding 전체의 similarity를 계산하여 ranking을 만들고,
그 ranking에서 해당 probe의 GT Step이 몇 위인지 찾은 뒤 Recall@1과 Reciprocal Rank를
계산한다.**

구체적인 계산은 다음과 같다.

1. 각 probe가 `q=normalize(Wz_t)`를 만든다. Control은 `normalize(W_r r_t)`다.
2. `q`와 모든 normalized candidate embedding 사이 cosine similarity를 계산한다.
3. probe마다 similarity 내림차순 ranking을 독립적으로 만든다.
4. 해당 probe의 normalized GT와 같은 candidate가 처음 나타나는 one-based rank를 찾는다.
5. rank가 1이면 Recall@1은 1, 아니면 0이다. Reciprocal Rank는 `1/rank`이며 MRR은
   이를 aggregate한 값이다.

한 trajectory에 normalize 후 같은 text가 여러 번 있으면 R2 규칙대로 모든 동일-text
index를 한 positive equivalence class로 처리한다. Probe loss는 multi-positive softmax
cross-entropy이며 rank는 첫 positive occurrence를 사용한다. Sequence consistency는
predicted/GT index가 아니라 normalized identity로 비교하고, 중복이 없는 subset도 별도로
기록한다.

## Sampling과 집계

`src/robocerebra_memory/sampling.py`의 frozen R2 구현을 재사용한다. 각
`trajectory × Step × steps_since_transition bin` cell에서 최대 4 frame을 deterministic
even spacing으로 선택한다. Seed는 42다. Distance bins는 `0-4`, `5-19`, `20-49`,
`50-99`, `100-199`, `200-399`, `400+`이며 transition bins는 `0`부터 `7`, `8+`다.

Primary point estimate는 frame micro-average가 아니다. Cell 내부 frame 평균, Step 내부
distance-bin 평균, trajectory 내부 Step 평균, 마지막 trajectory 평균 순서로 계산한다.
95% CI는 seed 42로 trajectory를 2,000회 bootstrap한다. Frame bootstrap은 하지 않는다.
각 table에는 trajectory/sample count와 aggregation/bootstrap metadata가 들어간다.

## Metrics와 해석

- Current/Previous-1/2/3 Recall@1과 MRR: 같은 현재 `z_t`에서 각 relative Step identity가
  몇 위로 복원되는지 측정한다.
- Current-Step Retention: current probe의 Recall@1/MRR을
  `steps_since_transition` distance bin별로 측정한다. Command transition 후 시간이
  지나도 현재 context가 남는지를 본다.
- Transition Robustness: current metric을 `cumulative_transition_count`의 `0..7,8+`
  bin으로 나눠 여러 semantic update 뒤에도 context가 유지되는지를 본다.
- Memory Depth: relative offset `k=0,1,2,3`별 current/prev1/prev2/prev3 metric을 비교한다.
- `sequence_exact_match_at_4`: 네 target이 모두 valid한 sample에서 네 temporal position을
  전부 정확히 맞춘 경우만 1이다. Step들을 알고 있어도 current/previous 위치를 바꾸면
  0인 secondary metric이다.
- Instantaneous control: 같은 sample의 `r_t` current metric과 `z_t` current metric을
  비교한다. 두 값이 비슷하면 현재 RGB/qpos만으로 Step을 알 수 있는 observation confound가
  크므로 current score를 memory 증거로 해석하기 어렵다. Control distance curve도 함께 쓴다.

B2는 current official Step text를 every timestep 입력받는다. 따라서 B2 current retrieval은
oracle-like upper bound이며 memory evidence가 아니다. B2 결과 파일에는 이 경고가
자동 기록된다.

## Train/val/test leakage 방지

기존 split manifest의 train 734, val 85, test 95 trajectory를 그대로 사용한다. Phase 1은
train/val representation만 추출하고, probe는 train cache로만 update하며 val의
trajectory-macro multi-positive loss로 checkpoint를 선택한다. Test 경로는 CLI가
`--final-test`를 검사하기 전에는 읽지 않는다. Final test는 선택된 val probe를 freeze한
후 별도 명시 command로만 실행되며, 완료 sentinel이 있으면 두 번째 metric 실행을
거부한다. 이번 구현/검증에서는 test representation extraction과 test metric을 실행하지
않았다.

## Cache와 실행 역할

GPU 역할은 frozen Stage A backbone을 forward해 balanced sample의 `r_t`와 `z_t`를
추출하는 것뿐이다. Shard에는 trajectory/frame, Step index 및 normalized text,
distance/transition bin, previous validity, 네 GT label index, local candidate text/CLIP
embedding, `r_t`, `z_t`, Stage A checkpoint identity가 저장된다. 각 shard와 manifest는
temporary file을 fsync한 뒤 atomic rename하며, 재시작 시 완성된 shard를 validate하고
건너뛴다. Partial file은 final cache로 인정하지 않는다.

Resume 시에는 variant/split/trajectory뿐 아니라 Stage A state-dict hash, update, selected
epoch까지 일치해야 한다. 각 shard는 frame/GT/bin으로 sampling identity SHA-256을 만들고,
manifest는 이를 다시 split-level hash로 묶는다. B0/B1/B2/B3의 validation sampling hash가
전부 같을 때만 comparison table을 생성한다. Probe checkpoint에는 train/val sampling hash와
Stage A checkpoint identity가 고정되며, evaluation cache의 backbone identity가 다르면
실행을 거부한다. Split 수와 중복 trajectory, candidate normalization, GT/previous validity,
distance/transition value-bin 일관성도 cache load 전에 검사한다.

Probe training, candidate ranking, metrics, trajectory bootstrap, JSON/CSV는 CPU에서
실행한다. Probe hyperparameter 값은 R2가 고정하지 않았으므로 다음은 명시적인
**Stage B implementation choice**다: AdamW, learning rate `1e-3`, weight decay `0.01`,
temperature `1.0`, 최대 20 epochs, val-loss patience 5. 이 값은
`configs/stage_b_memory_eval.yaml`에 있고 test로 조정하지 않는다.

## Stage A 완료 후 명령

아래 명령에서 Python은 Stage A와 같은 환경을 사용한다.

```bash
PY=/home/itaein/.conda/envs/egodex-box/bin/python

for variant in B0 B1 B2 B3; do
  "$PY" tools/extract_stage_b_representations.py --variant "$variant" --split train
  "$PY" tools/extract_stage_b_representations.py --variant "$variant" --split val
  "$PY" tools/train_stage_b_probes.py --variant "$variant"
done
```

위 extraction만 CUDA compute node에서 실행한다. Probe train/eval은 CPU node에서 실행할
수 있다. Validation은 probe checkpoint 선택에만 사용한다. 네 probe가 모두 고정된 후
다음 final-test phase를 정확히 한 번 실행한다.

```bash
"$PY" tools/cache_stage_a_features.py \
  --model-cache /ssd1/itaein/datasets/RoboCerebra/model_cache/r4_clip \
  --split-scope test --final-test \
  --audit analysis/stage_b/test_clip_cache_audit.json

for variant in B0 B1 B2 B3; do
  "$PY" tools/extract_stage_b_representations.py --variant "$variant" --split test --final-test
  "$PY" tools/eval_stage_b.py --variant "$variant" --split test --final-test
done
```

수동으로 validation Metric을 요청한 경우 output은 `analysis/stage_b/B0` ... `B3`, final test는
`analysis/stage_b/final_test/B0` ... 에 기록된다. Variant별로 `summary.json/csv`, distance,
transition, memory-depth CSV, sequence consistency JSON, instantaneous control JSON을 만들고,
최종 비교 CSV와 figure는 `analysis/stage_b/final_test/comparison`에 기록한다.

## Stage A와 같은 GPU allocation에서 한 번에 실행

두 GPU PBS Job은 Stage A `training-gate`가 B0/B1/B2/B3의 최종 checkpoint를 확인한 직후
`tools/run_stage_b_after_stage_a.py`를 자동 호출한다. 이 runner는 다음 순서를 중단 없이
수행한다.

1. B2/B3를 포함한 네 Stage A checkpoint의 완료·schema·variant·공통 초기화 검증
2. B0/B1, B2/B3를 각각 두 GPU pair로 train/val representation 추출
3. train으로 네 variant의 독립 probe를 학습하고 val loss로 probe를 선택·고정
4. 선택이 끝난 뒤에만 명시적 final-test gate로 test95 frozen CLIP cache 생성
5. 고정된 probe에 대해 B0/B1, B2/B3 test95 representation 추출
6. test95 Metric을 variant별 정확히 한 번 계산하고 최종 comparison CSV, SVG/HTML 생성

Validation split은 성능 보고가 아니라 probe 선택에만 사용하며, 최종 성능 수치와 그림은
test95에서만 만든다. Variant별 `FINAL_TEST_COMPLETED.json` sentinel이 재평가를 차단한다.
실행 상태는 `analysis/stage_b/pipeline_status.json`에 원자적으로 기록된다. 실제 GPU
allocation 없이 실행 순서만 확인하려면 다음을 사용한다.
실행 전 남은 PBS walltime이 12시간 이상인지 검사하고, 각 extraction/probe/Metric 단계로
넘어갈 때 `current_stage`를 갱신한다. Stage B가 실패해도 완료된 Stage A gate는 `PASS`로
보존되며 Stage B만 `FAIL`로 기록된다.

```bash
python tools/run_stage_b_after_stage_a.py --plan-only
```

평가가 끝나면 추가 dependency 없이 SVG와 HTML dashboard도 자동 생성한다.

- 최종 Variant별: `analysis/stage_b/final_test/B*/figures/`
  - `current_retention.svg`: `z_t` current와 `r_t` control, 95% CI, bin별 sample 수
  - `transition_robustness.svg`: 누적 transition별 current Recall@1
  - `memory_depth.svg`: k=0..3 Recall@1/MRR
  - `instantaneous_control.svg`: overall `z_t` 대 `r_t`
  - `sequence_consistency.svg`: 전체 valid/unambiguous subset Exact Match@4
  - `dashboard.html`: 위 다섯 figure를 한 화면에서 검토
- 최종 전체 비교: `analysis/stage_b/final_test/comparison/figures/`
  - B0–B3 distance, transition, memory-depth, instantaneous-control, sequence-consistency SVG
  - `dashboard.html`

B0–B3에는 color-blind-friendly 고정 색을 사용하고 B2는 dashed oracle style과 경고를
표시한다. Figure는 trajectory-macro point estimate, trajectory-bootstrap 95% CI,
balanced sample count를 함께 보여준다. SVG는 VS Code에서 직접 열 수 있고 vector이므로
논문용으로 확대해도 깨지지 않는다. 기존 metric 파일에서 figure만 다시 만들려면 다음을
실행한다.

```bash
"$PY" tools/plot_stage_b.py --variant B0
"$PY" tools/plot_stage_b.py --comparison
```

## CPU-only 검증

실제 checkpoint나 split을 읽지 않는 synthetic 전체 dry run은 다음과 같다.

```bash
tmp_dir=$(mktemp -d /tmp/stage-b-dry-run.XXXXXX)
/home/itaein/.conda/envs/egodex-box/bin/python tools/dry_run_stage_b.py --output-dir "$tmp_dir"
```

이 경로는 synthetic `r_t/z_t`와 Step embedding에서 다섯 probe 학습, 독립 ranking, GT rank,
R@1/MRR, 두 curve, memory depth, sequence exact match, control, JSON/CSV 출력까지 CPU에서
통과시키며 RoboCerebra test95를 읽거나 평가하지 않는다.
