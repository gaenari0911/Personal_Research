# R2 Memory Metrics and Sampling Definitions

## Trajectory-local retrieval

Trajectory `i`의 ordered Step text embedding을 `E_i={e_i1,...,e_iK}`라고 한다.
Primary probe representation은 final normalized Mamba output `z_t∈R^128`이다. Frozen
backbone 위의 유일한 probe parameter는 bias-free linear map `W∈R^(512×128)`이다.

```text
q_t = normalize(W z_t)
s_tj = cosine(q_t, normalize(e_ij))
rank_t = first rank containing a positive candidate
Recall@1_t = 1[rank_t = 1]
MRR_t = 1 / rank_t
```

Current probe의 positive는 현재 Step text다. Previous-k probe는 현재 Step index가 `m`일
때 `m-k` text를 positive로 사용하며 `m<k`인 frame은 해당 depth에서 제외한다.

동일 trajectory에 casefold/whitespace normalization 후 같은 text가 여러 번 나오면
동일-text index 전체가 positive set이다. Rank는 첫 positive occurrence의 순위다. R2
dry-run에서 이런 trajectory는 train 20, val 3, test 3개다.

## Distance retention

`d_t = t - start(current Step)`이다.

| bin | all raw frames | all balanced samples | test trajectories | test balanced samples |
|---|---:|---:|---:|---:|
| 0–4 | 40,955 | 32,764 | 95 | 3,516 |
| 5–19 | 122,865 | 32,764 | 95 | 3,516 |
| 20–49 | 245,576 | 32,764 | 95 | 3,516 |
| 50–99 | 405,339 | 32,693 | 95 | 3,505 |
| 100–199 | 724,789 | 31,870 | 95 | 3,415 |
| 200–399 | 791,633 | 24,601 | 95 | 2,576 |
| 400+ | 360,874 | 8,579 | 76 | 877 |

가장 먼 bin에도 test 76 trajectories가 있으므로 initial bins를 유지한다. 각 bin에서
Recall@1과 MRR, balanced sample count, contributing trajectory count, 95% trajectory
bootstrap CI를 함께 보고한다.

## Transition accumulation

`c_t = current_step_index`이며 bins는 0..7과 8+다.

| bin | all raw frames | all balanced samples | test trajectories | test balanced samples |
|---|---:|---:|---:|---:|
| 0 | 341,394 | 22,664 | 95 | 2,384 |
| 1 | 286,495 | 21,705 | 95 | 2,270 |
| 2 | 267,421 | 21,127 | 95 | 2,158 |
| 3 | 319,796 | 21,979 | 95 | 2,268 |
| 4 | 289,020 | 21,559 | 95 | 2,275 |
| 5 | 279,720 | 20,711 | 94 | 2,194 |
| 6 | 275,939 | 19,433 | 84 | 2,012 |
| 7 | 228,126 | 16,297 | 75 | 1,738 |
| 8+ | 404,120 | 30,560 | 53 | 3,622 |

Distance curve와 transition curve를 별도로 보고해 긴 시간 경과와 여러 command에 의한
overwrite를 구분한다. 두 변수를 사후 결합해 유리한 slice를 고르는 분석은 secondary로
표기한다.

## Previous-k targets

Primary depth는 k=1,2,3이다.

| split | previous-1 | previous-2 | previous-3 |
|---|---:|---:|---:|
| train balanced | 138,649 | 121,214 | 104,211 |
| val balanced | 16,185 | 14,185 | 12,219 |
| test balanced | 18,537 | 16,267 | 14,109 |

모든 test trajectory가 k=1/2/3 target을 적어도 하나 제공한다. Primary Figure C는 각
depth의 trajectory-macro Recall@1과 CI를 사용한다. Previous-4/5는 metadata에 남기지만
Phase-1 primary table에는 넣지 않는다.

## Balanced point estimate

Sampling은 각 `trajectory × Step × distance-bin` cell에서 최대 4개의 deterministic
evenly-spaced frame만 사용한다. 총 raw 2,692,031 frames는 train 156,821, val 18,293,
test 20,921, 합계 196,035 balanced samples가 된다.

Probe optimization에서 각 cell contribution을 같게 하고, evaluation은 trajectory를
최종 독립 단위로 평균한다. 동일 trajectory의 많은 인접 frame을 독립 표본으로 간주하지
않는다.

## Confidence intervals

- sampling unit: trajectory
- resamples: 2,000
- seed: 42
- interval: two-sided 95% percentile bootstrap
- bin별로 해당 bin에 기여하는 trajectory만 resample
- frame-level bootstrap 금지

Model comparison에서는 동일 resampled trajectory IDs를 사용하는 paired bootstrap을
R3에 구현한다. Primary difference는 B1−B0, B3−B1, B3−B0다.

## Interpretation guardrails

- B2 current metric은 oracle-conditioning upper bound다.
- Overall micro frame accuracy는 primary metric이 아니다.
- Sample/trajectory count 없는 curve는 보고하지 않는다.
- Test 결과를 본 뒤 bin, cap, horizon, probe capacity를 변경하지 않는다.
- Hyperparameter 변경이 필요하면 val에서 새 protocol version을 만들고 test를 다시 보지 않는다.

