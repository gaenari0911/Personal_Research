# R2 Mamba Hidden-State Memory Evaluation Protocol

## 1. Scope와 연구 질문

이 protocol은 RoboCerebra의 long-horizon continuous trajectory를 순차 입력했을 때
Mamba representation이 현재 및 과거 language-grounded Step 정보를 얼마나 오래
보존하는지 평가한다.

Primary questions는 다음과 같다.

1. 현재 official Step text가 representation에서 retrieval 가능한가?
2. `steps_since_transition`이 증가할 때 그 정보가 얼마나 감소하는가?
3. `cumulative_transition_count`가 증가할 때 progress 정보가 overwrite되는가?
4. previous-1/2/3 Step text가 현재 representation에서 retrieval 가능한가?

Policy success, action prediction, Behavior Cloning 성능은 primary outcome이 아니다.
이번 R2에서는 model/probe를 학습하지 않는다.

## 2. Input contract

각 timestep의 입력은 external RGB, Panda qpos 9D, language condition이다. 원본
MuJoCo full state, object/fixture qpos·qvel, frame/progress label, previous action은
입력하지 않는다.

- visual: frozen OpenAI CLIP ViT-B/32 image embedding 512D
- proprioception: `states[t,1:10]`, z-score 후 작은 projection
- language: 같은 frozen CLIP ViT-B/32 text encoder의 512D embedding
- fusion: modality별 projection을 합한 뒤 LayerNorm, output 128D

동일 CLIP의 image/text encoder를 고정하는 이유는 future visual target과 trajectory-local
Step candidate를 재현 가능한 공통 feature contract로 만들고, temporal backbone이
encoder capacity 차이로 이득을 얻지 않게 하기 위해서다. Checkpoint checksum은 R3에서
모든 모델에 하나로 고정한다.

## 3. “Mamba hidden state”의 정확한 정의

MaIL의 `external/MaIL/agents/models/oc_ddpm/mamba.py::MixerModel`은 각 Mamba layer의
recurrent cache와 별도로, 마지막 residual을 `norm_f`에 통과시킨 sequence output을
반환한다. source의 `forward`는 layer loop 후 non-fused 경로에서
`hidden_states = self.norm_f(residual)`을 수행하고 이를 반환한다.

두 representation을 구분한다.

- internal recurrent state: layer별 causal-convolution state와 selective-SSM state.
  streaming `inference_params`/cache에 보관되는 구현 상태이며 primary probe 대상이
  아니다.
- primary representation `z_t`: 현재 observation token에 대응하는 final `norm_f`
  output, future-prediction head 적용 전. Shape은 `[B,T,128]` 중 `[B,t,128]`이다.

보고서에서는 편의상 `h_t`라고 쓸 수 있지만 반드시 “final normalized temporal
representation `z_t`, not the raw SSM cache”라고 정의한다. Internal cache 및
early/middle/late layer probe는 secondary analysis다.

## 4. Non-BC representation learning

Random initialization을 probe하지 않는다. B0–B3 모두 다음 하나의 objective로 먼저
학습한 뒤 backbone을 freeze한다.

**Language-conditioned causal future visual contrast**

- anchor: causal representation `z_t`
- horizon: `t+20` frames, 즉 20 Hz에서 1.0초
- positive target: frozen CLIP external-RGB embedding `v_{t+20}`
- prediction head: lightweight projection from 128D to 512D
- loss: in-batch InfoNCE, temperature 0.07
- negatives: 같은 training batch의 다른 valid future targets
- trajectory terminal을 넘는 anchor 제외
- action, Step ID, boundary, future text는 target/input에 사용하지 않음
- auxiliary horizon 없음

20 frames는 1-step identity prediction보다 덜 즉각적이면서 median Step 288 frames보다
충분히 짧아 대부분의 Step에서 target을 제공한다. 실제 dry-run상 20-frame anchor는
train 2,145,676, val 248,488, test 279,587개다. 한 horizon만 사용해 objective sweep이
memory 결과에 맞춰지는 것을 막는다.

학습 anchor ID와 optimizer/update 수는 B0–B3가 동일하다. Persistent model은 episode
순서로 256-frame chunk를 처리하고 cache를 다음 chunk로 전달하되 training graph는
chunk 경계에서 detach한다. Evaluation에서는 detach 여부와 무관하게 episode 끝까지
forward cache를 유지한다. Cache reset은 episode 시작에서만 한다.

## 5. Comparison models

| ID | temporal contract | language contract | probe token |
|---|---|---|---|
| B0 Fixed-Window FULL | 매 t에서 최근 5 input만 재실행, window 밖 state 없음 | FULL every timestep | window 마지막 `norm_f` output |
| B1 Persistent FULL | episode 전체 cache 유지 | FULL every timestep | 현재 `norm_f` output |
| B2 Persistent CURRENT | episode 전체 cache 유지 | 현재 Step every timestep | 현재 `norm_f` output |
| B3 Persistent HOLD | episode 전체 cache 유지 | Step 시작에서 command, 나머지 `[HOLD]` | 현재 `norm_f` output |

B3가 primary research condition이다. B2 current retrieval은 current Step text를 직접
받는 oracle-conditioning upper bound이며 memory evidence로 해석하지 않는다.

B4 Persistent NO-LANGUAGE는 language contribution을 분리하는 가치가 있지만 primary
4-model table에는 넣지 않는다. Compute budget이 허용될 때 사전 등록된 secondary
ablation으로만 실행한다.

모든 모델은 visual/state/text encoder, fusion, Mamba 128D×16 layers, SSM 설정,
future objective, optimizer, trajectory/anchor sampling, updates, probe를 공유한다.
오직 persistence와 conditioning strategy만 다르다.

## 6. Instantaneous observation-state control

동일 common encoder의 language addition 직전 observation+state fusion `r_t`를 사용한다.
Temporal Mamba history와 language는 모두 제외하고 같은 linear retrieval probe를
적용한다. 별도 강력한 MLP를 학습하지 않는다.

이 control이 Mamba와 비슷하면 현재 RGB/qpos만으로 label을 식별할 수 있다는 뜻이며,
current-subtask 결과를 memory evidence로 사용할 수 없다.

## 7. Probe contract

Representation-learning 완료 후 encoder/backbone/future head를 freeze한다. Probe는
bias 없는 linear map `W: R^128 -> R^512` 하나다. `Wz_t`와 같은 trajectory의 ordered
Step text embedding들 사이 cosine similarity로 retrieval한다. Text candidate는
conditioning과 같은 frozen CLIP text encoder를 사용한다.

각 model×target depth마다 별도의 `W`를 학습하되 architecture, initialization seed,
optimizer, train frame IDs, val selection rule은 동일하게 유지한다. Training loss는
trajectory-local candidate에 대한 multi-positive softmax cross-entropy다. Probe끼리
parameter를 공유하거나 한 모델의 test score에 맞춰 capacity를 바꾸지 않는다.

Current target은 현재 Step, previous-k target은 `current_step_index-k`다. Primary
depth는 0/1/2/3이고 metadata previous-4/5는 보존한다. 한 trajectory 안에 casing 및
whitespace 정규화 후 동일한 Step text가 반복되면 그 occurrence들은 하나의 positive
equivalence class로 취급한다. 이 규칙은 불가능한 identical-embedding tie를 벌하지
않으며 occurrence-level progress는 별도의 transition-count curve로 측정한다.

Probe 학습에는 train representation만, 선택에는 val만, final metric에는 test만 쓴다.
Test Step text/boundary는 test evaluation target/candidate 생성에만 사용하며 backbone이나
probe update에는 사용하지 않는다.

## 8. Pre-registered bins와 balanced sampling

Distance bins는 `[0,5)`, `[5,20)`, `[20,50)`, `[50,100)`, `[100,200)`,
`[200,400)`, `[400,∞)`다. Transition bins는 `0,1,2,3,4,5,6,7,8+`다. 결과를 본 뒤
bin을 바꾸지 않는다.

Probe sample은 `trajectory × Step × distance-bin` cell마다 최대 4 frame을 half-open
cell 전체에서 deterministic하게 균등 간격으로 뽑는다. 긴 Step의 frame 수가 probe를
지배하지 않게 하며, split/model에 관계없이 동일 frame IDs를 사용한다.

Point estimate는 frame micro-average가 아니라 다음 순서의 macro-average를 사용한다.

1. cell 내부 frame 평균
2. Step 내부 bin 평균
3. trajectory 내부 Step 평균
4. trajectory 평균

구현 metric helper는 trajectory-macro를 강제한다. 최종 구현에서는 cell/Step hierarchy도
sample metadata로 보존한다. Confidence interval은 seed 42, 2,000회 trajectory bootstrap,
95% percentile CI다. Frame bootstrap은 사용하지 않는다.

## 9. Primary metrics와 figures

Metrics:

- Current Step Recall@1 / MRR
- Recall@1 vs `steps_since_transition`
- Recall@1 vs `cumulative_transition_count`
- previous-1/2/3 Recall@1 / MRR
- Recall@1 vs previous-k depth

Primary outputs:

- Figure A: current Recall@1 vs distance, B0/B1/B2/B3
- Figure B: current Recall@1 vs accumulated transitions, B0/B1/B2/B3
- Figure C: previous Recall@1 vs k=1/2/3, B0/B1/B2/B3
- Table: overall Recall@1/MRR + instantaneous control, trajectory-bootstrap CI

## 10. Hypotheses와 rejection conditions

사전 가설은 B0가 distance에 따라 더 빠르게 감소하고, B1이 이를 완화하며, B3가 sparse
transition event를 사용해 먼 거리와 previous-k에서 더 안정적이라는 것이다. 이는 예상
결과이지 protocol 판정에 반영된 사실이 아니다.

- B0 ≈ B1: persistence가 task-progress retention에 추가 이점을 주지 않음
- B1 ≈ B3: sparse transition conditioning 이점 없음
- instantaneous ≈ Mamba: observation confound가 지배하며 current probe는 memory 측정에 부적합
- current high, previous-k low: current progress representation은 있으나 historical memory는 약함
- B2 current high only: oracle text injection 효과이며 memory claim 불가
- 모든 모델 chance 수준: objective/encoder/probe point가 Step 정보를 형성하지 못함

Primary claim은 overall score 1등이 아니라 distance/transition/depth에 따른 retention
slope와 CI로 판단한다.

## 11. R3 implementation plan

R3에서 다음 순서의 CLI를 구현한다. 아래는 command contract이며 현재 실행하지 않는다.

```text
prepare-rgb-features --config configs/robocerebra_memory_protocol.yaml
train-representation --model B0|B1|B2|B3 --config ...
extract-representations --split train|val|test --model ...
fit-linear-probe --targets current|previous-1|previous-2|previous-3 --model ...
evaluate-memory --model ... --bootstrap-trajectories 2000
```

R3 시작 전 모든 필요한 MP4를 materialize하고 `T+1` alignment를 검사하며, CLIP
checkpoint checksum을 고정해야 한다. Full training이나 GPU job은 사용자 지시 없이
자동 시작하지 않는다.
