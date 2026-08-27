# LIBERO-10 Dataset Inspection — Task 2.6

## 1. Purpose

이 문서는 Task 2에서 BDDL만으로 분석했던 LIBERO-10 후보를 실제 demonstration으로 검증한다. 범위는 안전한 데이터 확보, HDF5 구조·action 의미·시간 통계 확인, 후보 Task 3/4/9의 대표 trajectory 시각화, Task 3 annotation을 위한 preliminary evidence 정리까지다. **Oracle boundary는 만들지 않았으며 아래 stage 경계는 모두 review proxy**이다.

이전 결정과의 연결은 다음과 같다. Task 2.5의 기본 실험은 persistent temporal encoder + stateless action decoder, current observation 입력, `predict horizon=10`, `execute=1`, MaIL 비교 기준 `obs_seq=5`이다. 본 분석은 이 설계가 실제 task 시간 척도와 맞는지 확인한다.

## 2. Storage Inspection

검사 시점 호스트는 `pleiades1`, 작업 경로는 `/home/itaein/Personal_Research`였다.

| Mount | Filesystem | Total | Used | Available | Use | 판단 |
|---|---:|---:|---:|---:|---:|---|
| `/home` | xfs | 7.0T | 6.9T | 139G | 99% | 제외 |
| `/ssd1` | ext4 | 880G | 356G | 약 479G | 43% | 선택 |
| `/ssd2` | ext4 | 880G | 607G | 228G | 73% | 대안 |
| `/hdd1` | ext4 | 1.8T | 1.6T | 106G | 94% | 제외 |
| `/hdd2` | ext4 | 1.8T | 1.6T | 113G | 94% | 제외 |
| `/data1` | nfs4 | 15T | 15T | 371G | 98% | 제외 |
| `/data2` | nfs4 | 15T | 15T | 122G | 100% | 제외 |
| `/data3` | nfs4 | 7.3T | 6.8T | 140G | 99% | 제외 |
| `/data4` | nfs4 | 15T | 15T | 19G | 100% | 제외 |
| `/data5` | nfs4 | 15T | 15T | 95G | 100% | 제외 |
| `/data6` | nfs4 | 7.3T | 6.9T | 398G | 95% | 제외 |
| `/data7` | nfs4 | 7.3T | 7.2T | 93G | 99% | 제외 |

`/ssd1/itaein`은 사용자 소유(`itaein:iteain`)이고 dataset 디렉터리는 쓰기 가능하다. 대용량 write test는 하지 않았다. quota 명령은 NFS RPC 거부로 값을 얻지 못했지만, 로컬 ext4의 실제 여유 공간과 소유권을 확인했다. 다운로드 전 `/home/itaein`, `/ssd1`, `/ssd2`, `/hdd1`, `/hdd2`를 bounded search했고 LIBERO HDF5 중복본은 없었다. `/data1`–`/data7`은 각 mount를 제한된 깊이와 timeout으로 검색했으며 match가 없었지만, timeout 때문에 전체 트리를 완전 탐색했다고 주장하지 않는다.

## 3. Dataset Source

공식 LIBERO 저장소 README가 안내하는 Hugging Face dataset 저장소를 사용했다.

- 공식 코드: [Lifelong-Robot-Learning/LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO)
- 공식 안내: [LIBERO README](https://github.com/Lifelong-Robot-Learning/LIBERO/blob/master/README.md)
- 파일 목록: [Hugging Face LIBERO-10 tree](https://huggingface.co/datasets/yifengzhu-hf/LIBERO-datasets/tree/main/libero_10)
- 확인 당시 공개 tree: 10 HDF5, 표시 크기 13.7 GB, 개별 파일 0.941–2.06 GB.

공식 downloader CLI는 `libero_100` 전체 묶음을 선택하는 형태이므로, 불필요한 90-task 데이터까지 받지 않도록 동일한 공식 HF 저장소의 `libero_10/` 아래 10개 파일만 `resolve/main` URL로 받았다. `wget -c`를 사용해 TLS 단절 시 byte-range resume했다.

## 4. Dataset Location

정상 dataset 위치:

```text
/ssd1/itaein/datasets/LIBERO/libero_10
```

정상 HDF5 10개 합계는 `13,730,608,904 bytes`(13.731 GB, 약 12.79 GiB)다. 분석 결과와 작은 영상은 workspace의 `analysis/`에 두고, 대용량 HDF5를 project나 `/home`으로 복사하지 않았다.

## 5. Dataset Integrity

최종 결과는 10/10 파일 open 성공, task당 50 demos, 총 500 demos, 총 138,090 timesteps다. 각 demo에 대해 action이 `(T,7)`인지, `states/rewards/dones/robot_states`와 모든 `obs/*`의 첫 차원이 T인지, `data.attrs.total`과 길이 합, `num_demos`와 실제 demo 수가 일치하는지 전수 검사했다. 오류는 0개다. 세부 schema, 파일 크기, SHA-256은 `analysis/libero10_hdf5_inspection.json`에 있다.

초기 병렬 다운로드 때 Task 8 파일 하나가 중복 writer 때문에 `message not aligned`로 열리지 않았고, 크기도 2,095,674,688 bytes로 공식 응답 2,061,429,892 bytes와 달랐다. 손상본을 격리하고 단일 writer로 다시 받은 뒤 전체 검사를 통과했다. 정상 Task 8 SHA-256은 `e329bb21a8ded3457854faf6a23513c90cb4b34f0e40f3f4e9e70451fc9ba504`이며, 검증 후 격리 손상본은 삭제해 2.1 GB를 회수했다. 공개 페이지에서 독립적인 checksum 목록은 찾지 못했으므로, 저장한 SHA-256은 향후 로컬 재검증 기준이지 upstream checksum 대조 결과는 아니다.

## 6. HDF5 Schema

모든 파일의 root key는 `data` 하나다. `data` attributes는 `bddl_file_name`, `env_args`, `env_name`, `macros_image_convention`, `num_demos`, `problem_info`, `tag`, `total`이며 `macros_image_convention=opengl`, `tag=libero-v1`이다. `data/demo_0`부터 `demo_49`까지 있고 각 demo attributes는 `init_state`, `model_file`, `num_samples`다.

각 demo의 공통 dataset은 다음과 같다.

| Key | Shape | dtype | 설명 |
|---|---|---|---|
| `actions` | `(T,7)` | float64 | OSC pose + gripper command |
| `states` | `(T,S)` | float64 | flattened simulator state; S는 task별 45–123 |
| `robot_states` | `(T,9)` | float64 | LIBERO robot state vector |
| `rewards` | `(T,)` | uint8 | 처리 과정에서 마지막 sample만 1 |
| `dones` | `(T,)` | uint8 | 처리 과정에서 마지막 sample만 1 |
| `obs/*` | 아래 절 참조 | mixed | camera와 proprioception |

중요한 정렬 위험이 있다. 공식 [`create_dataset.py`](https://github.com/Lifelong-Robot-Learning/LIBERO/blob/master/scripts/create_dataset.py)는 먼저 `env.step(action)`을 호출하고 반환된 `obs`를 같은 index action과 함께 저장한다. 따라서 HDF5의 `obs[t]`는 `action[t]` 실행 **후** 관측이다. MaIL loader는 같은 slice의 image와 action을 그대로 반환하므로, causal imitation target을 정의할 때 한 step shift가 필요한지 Task 4 구현 전에 명시적으로 audit해야 한다.

## 7. Observation Keys

10개 task 모두 observation key와 dtype/shape가 동일하다.

| Key | Shape | dtype |
|---|---|---|
| `obs/agentview_rgb` | `(T,128,128,3)` | uint8 |
| `obs/eye_in_hand_rgb` | `(T,128,128,3)` | uint8 |
| `obs/ee_pos` | `(T,3)` | float64 |
| `obs/ee_ori` | `(T,3)` | float64, end-effector axis-angle state |
| `obs/ee_states` | `(T,6)` | float64, position + axis-angle |
| `obs/gripper_states` | `(T,2)` | float64, finger joint qpos |
| `obs/joint_states` | `(T,7)` | float64, Panda joint position |

MaIL이 사용하는 `agentview_rgb + eye_in_hand_rgb`를 시각화에도 그대로 사용했다. named object pose, contact, drawer joint, microwave door angle, depth, segmentation, timestamp는 없다. `states` 안에는 simulator state가 포함되지만 ordering이 task/model별이고 MaIL observable도 아니므로, model XML을 이용한 명시적 decoding 없이는 completion signal로 취급하면 안 된다.

## 8. Action Representation

HDF5 action은 `(T,7)`, float64다. 모든 138,090 sample에서 처음 6차원은 controller input 범위 안이고 gripper는 정확히 `{-1,+1}`이다. controller는 `OSC_POSE`, `impedance_mode=fixed`, `control_delta=true`다.

| Dimension | Meaning | Representation | Evidence |
|---:|---|---|---|
| 0 | end-effector x translation | normalized relative position command | HDF5 `env_args`; robosuite OSC |
| 1 | end-effector y translation | normalized relative position command | same |
| 2 | end-effector z translation | normalized relative position command | same |
| 3 | rotation x component | normalized relative axis-angle vector | robosuite OSC input path |
| 4 | rotation y component | normalized relative axis-angle vector | same |
| 5 | rotation z component | normalized relative axis-angle vector | same |
| 6 | gripper | discrete `-1=open`, `+1=close` | robosuite `input2action`; data/qpos cross-check |

## 9. Exact Action Semantics

robosuite v1.4.0의 [`input2action`](https://github.com/ARISE-Initiative/robosuite/blob/v1.4.0/robosuite/utils/input_utils.py)은 OSC_POSE action을 `[dpos(3), drotation(3), grasp]`로 결합하고 grasp를 `-1=open`, `+1=closed`로 매핑한다. [`OSC`](https://github.com/ARISE-Initiative/robosuite/blob/v1.4.0/robosuite/controllers/osc.py)는 `control_delta=true`일 때 input을 scale하고 position delta와 axis-angle orientation delta로 목표 pose를 갱신한다. v1.4.0 [`osc_pose.json`](https://github.com/ARISE-Initiative/robosuite/blob/v1.4.0/robosuite/controllers/config/osc_pose.json)과 실제 HDF5 metadata 모두 input `[-1,1]`, output translation `[-0.05,0.05] m`, rotation `[-0.5,0.5] rad`를 기록한다.

따라서 HDF5의 처음 6차원은 물리 단위의 absolute pose도, 이미 z-score된 학습 target도 아니다. controller가 받는 normalized delta command이며, 최대 command는 축별 0.05 m와 0.5 rad의 상대 목표 변화로 scale된다. 실제 실행 displacement는 동역학과 controller tracking 때문에 이 최대값과 같다고 가정할 수 없다. gripper convention은 Task 3 demo에서 `+1` 직후 finger qpos가 약 0.040에서 0.005로 닫히고 `-1` 직후 다시 커지는 것도 확인했다.

## 10. Action Statistics

전체 per-task/per-dimension min, p01/p05/p25/median/p75/p95/p99, max, mean, std는 `analysis/libero10_action_stats.csv`에 있다. 핵심 후보의 `mean ± std [min,max]`은 다음과 같다.

| Task | a0 | a1 | a2 | a3 | a4 | a5 | a6 |
|---:|---|---|---|---|---|---|---|
| 3 | .054±.231 [-.777,.846] | .118±.353 [-.938,.927] | -.097±.344 [-.817,.938] | .023±.053 [-.151,.329] | .004±.066 [-.299,.370] | -.045±.094 [-.360,.264] | -.311±.950 [-1,1] |
| 4 | .019±.231 [-.694,.938] | .101±.378 [-.884,.892] | -.052±.323 [-.769,.881] | -.002±.027 [-.134,.159] | -.001±.039 [-.210,.195] | -.007±.044 [-.184,.252] | .014±1.000 [-1,1] |
| 9 | .012±.359 [-.938,.938] | .122±.318 [-.921,.919] | -.055±.273 [-.938,.924] | .010±.068 [-.230,.304] | .008±.078 [-.303,.313] | .007±.146 [-.368,.375] | -.484±.875 [-1,1] |

Gripper command transition count 분포는 Task 3 `{2:49, 4:1}`, Task 4 `{3:3, 4:40, 6:7}`, Task 9 `{2:45, 4:4, 6:1}`, secondary Task 6 `{3:27,4:9,5:7,6:6,8:1}`, Task 0 `{4:41,6:8,8:1}`이다. 이는 retry 후보를 찾는 유용한 proxy지만, finger가 물체를 실제로 잡았다는 증거는 아니다.

## 11. LIBERO-10 Demonstration Counts

공식 task order index 0 기준 Task 0–9 각각 정확히 50 demos, `demo_0`–`demo_49`다. 전체 500 demos다. 누락·중복 demo ID는 없다.

## 12. Trajectory Length Statistics

20 Hz 기준 초 환산도 함께 표시한다. std는 population std다.

| Task | Demos | Total steps | Min | Mean | Median | Max | Std | Mean sec |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 50 | 14,700 | 230 | 294.00 | 285.5 | 388 | 35.29 | 14.70 |
| 1 | 50 | 13,021 | 234 | 260.42 | 255.0 | 322 | 20.43 | 13.02 |
| 2 | 50 | 13,298 | 219 | 265.96 | 262.5 | 340 | 27.65 | 13.30 |
| 3 | 50 | 12,434 | 199 | 248.68 | 245.5 | 317 | 23.85 | 12.43 |
| 4 | 50 | 12,909 | 216 | 258.18 | 258.0 | 331 | 22.21 | 12.91 |
| 5 | 50 | 9,470 | 150 | 189.40 | 186.5 | 259 | 21.74 | 9.47 |
| 6 | 50 | 12,756 | 203 | 255.12 | 249.0 | 342 | 32.34 | 12.76 |
| 7 | 50 | 13,476 | 219 | 269.52 | 268.0 | 334 | 25.65 | 13.48 |
| 8 | 50 | 20,794 | 341 | 415.88 | 408.5 | 517 | 41.25 | 20.79 |
| 9 | 50 | 15,232 | 224 | 304.64 | 300.5 | 449 | 46.60 | 15.23 |

CSV에는 p25/p75도 포함되어 있다: `analysis/libero10_trajectory_stats.csv`.

## 13. Sampling / Control Frequency

- environment control frequency: 20 Hz.
- action/dataset sampling frequency: 20 Hz.
- RGB observation frequency: 20 Hz, 각 저장 control step마다 두 카메라 한 frame.
- policy/control timestep: 0.05 s.

근거는 모든 HDF5의 `env_args.env_kwargs.control_freq=20`과 공식 dataset 생성 코드의 `control_freq=20`이다. 생성 코드는 raw action을 매 step replay하고 처음 5 step만 force-sensor 안정화를 위해 제거하며 이후 downsampling하지 않는다. timestamp 자체는 없으므로 clock jitter는 측정할 수 없고, nominal 20 Hz로 환산했다.

## 14. Candidate Task Detailed Statistics

| Task | Demos | Length min/mean/median/max/std | Mean duration | Gripper transitions | Reward timing |
|---:|---:|---|---:|---|---|
| 4 | 50 | 216/258.18/258/331/22.21 | 12.91 s | mean 4.22, range 3–6 | 모든 demo 마지막 index만 1 |
| 3 | 50 | 199/248.68/245.5/317/23.85 | 12.43 s | mean 2.04, range 2–4 | 마지막 index만 1 |
| 9 | 50 | 224/304.64/300.5/449/46.60 | 15.23 s | mean 2.24, range 2–6 | 마지막 index만 1 |
| 6 | 50 | 203/255.12/249/342/32.34 | 12.76 s | mean 3.92, range 3–8 | 마지막 index만 1 |
| 0 | 50 | 230/294.00/285.5/388/35.29 | 14.70 s | mean 4.40, range 4–8 | 마지막 index만 1 |

`rewards/dones`는 실제 success onset을 보존하지 않는다. 공식 생성 코드가 모두 0으로 만든 뒤 마지막 sample만 1로 쓰므로 completion boundary 근거로 사용할 수 없다. major action-change 영역은 대표 trajectory에서 gripper transition과 큰 end-effector 이동 구간에 겹쳤지만, Task 3에서 optical/object-state 기반 정밀 검출이 필요하다.

## 15. Representative Demonstrations

길이순 short / upper-median / long을 선택했다.

| Task | Short | Median | Long |
|---:|---|---|---|
| 3 | `demo_2`, T=199 | `demo_20`, T=247 | `demo_32`, T=317 |
| 4 | `demo_19`, T=216 | `demo_17`, T=259 | `demo_33`, T=331 |
| 9 | `demo_29`, T=224 | `demo_9`, T=301 | `demo_36`, T=449 |

각 demo는 side-by-side agentview/eye-in-hand MP4와 16-frame contact sheet로 확인했다. 추가 retry audit는 Task 3 `demo_34`, Task 9 `demo_3`, Task 4 전체 first/last grasp frame sheet를 사용했다.

## 16. Actual Task Execution Order

- **Task 4:** white mug를 left plate에 먼저 놓고 yellow-and-white mug를 right plate에 놓는다. 대표 3개뿐 아니라 50개 demo의 first/last close frame audit에서도 첫 grasp는 white, 마지막 grasp는 yellow-and-white로 일관됐다.
- **Task 3:** 대표 3개 모두 black bowl grasp/placement/release 후 bottom drawer를 닫았다. 순서 역전은 보지 못했다.
- **Task 9:** 대표 3개 모두 mug insertion/release 후 microwave door를 닫았다. 순서 역전은 보지 못했다.

Task 3/9의 50개 전체 semantic order를 frame-by-frame 수동 판독한 것은 아니다. 다만 대부분의 2-transition gripper sequence와 task success 구조가 대표 영상의 순서를 지지한다.

## 17. Retry / Recovery Behavior

- **Task 3 — MOSTLY monotonic:** 49/50은 2 transitions. `demo_34`만 4 transitions이며 contact sheet에서 bowl 조작 중 재접근/재grasp 후 성공하는 recovery가 보인다.
- **Task 4 — MOSTLY monotonic:** 40 demos는 4, 7 demos는 6, 3 demos는 3 transitions. long `demo_33`은 second mug placement 중 regrasp/reposition이 보인다. 3-transition trajectory는 terminal release command가 없거나 접촉 상태가 다른 경우일 수 있어 실패로 해석하지 않는다.
- **Task 9 — MOSTLY monotonic:** 45 demos는 2, 4 demos는 4, `demo_3`은 6 transitions. `demo_3` contact sheet는 mug 단계에서 regrasp/correction 후 door closure를 보여준다.

따라서 단조 `S1→S2` label만 강제하면 recovery 중 semantic regression을 잘못 표현할 수 있다. Task 3에서는 stage label과 함께 completion predicate가 유지되는지 확인하고, 재grasp 때 이전 stage로 돌아갈지 protocol에 명시해야 한다.

## 18. Transition-Relevant Signals

실제 HDF5에서 바로 쓸 수 있는 신호는 RGB 두 시점, gripper command, finger qpos, joint states, end-effector position/orientation이다. named object pose/contact/articulation은 없다.

- Task 3 bowl placement: RGB로 bowl-inside-drawer, `+1→-1` release, finger qpos opening을 결합하는 것이 후보. Drawer close 시작/완료는 RGB motion/geometry 또는 model XML로 `states`의 drawer joint를 명시적으로 decode해야 한다.
- Task 9 mug placement: RGB로 mug-inside-microwave와 release를 결합. Door manipulation/closure는 RGB 또는 decoded microwave joint가 후보.
- Task 4: RGB object identity와 plate target, 각 close/open transition을 결합. 단순 transition index만으로 white/yellow를 구분할 수 없다.

`rewards/dones`는 terminal marker라 transition signal이 아니다. flattened `states`는 유용할 수 있으나 그대로는 observable/semantic key가 아니며 Task 3에서 schema-to-joint mapping을 검증해야 한다.

## 19. Preliminary Semantic Stages

아래는 **Oracle이 아닌 preliminary review decomposition**이다.

- Task 3: S1-like = bowl 접근·grasp·drawer 내부 placement·release; S2-like = drawer 접근·close.
- Task 4: S1-like = white mug를 left plate에 placement; S2-like = yellow-and-white mug를 right plate에 placement.
- Task 9: S1-like = mug를 microwave 안에 placement·release; S2-like = microwave door close.

통계 계산에는 일관된 자동 proxy가 필요해 각 후보의 **두 번째 gripper command transition**을 S1-like 종료로 사용했다. retry가 있으면 이 proxy가 semantic completion과 어긋날 수 있으므로 annotation으로 쓰지 않는다.

## 20. Preliminary Stage Duration

50-demo proxy median과 대표 3개의 값은 다음과 같다. 괄호는 20 Hz 환산이다.

| Task | All-demo S1 median | All-demo S2 median | Short S1/S2 | Median S1/S2 | Long S1/S2 |
|---:|---:|---:|---:|---:|---:|
| 3 | 161.5 / 8.08s | 80.5 / 4.03s | 133/66 | 168/79 | 192/125 |
| 4 | 105.5 / 5.28s | 145.5 / 7.28s | 96/120 | 111/148 | 135/196 |
| 9 | 157.5 / 7.88s | 123.0 / 6.15s | 125/99 | 144/157 | 276/173 |

대표 demo의 transition index는 Task 3 `[64,133]`, `[83,168]`, `[80,192]`; Task 4 `[42,96,163,209]`, `[38,111,184,247]`, `[42,135,224,258,271,317]`; Task 9 `[77,125]`, `[86,144]`, `[168,276]`이다. 절대 timestep은 영상 overlay와 JSON에서 재현 가능하다.

## 21. Temporal Gap Analysis

대표 trajectory에서 S1 proxy부터 terminal까지의 gap은 Task 3 66/79/125 steps(3.30/3.95/6.25s), Task 4 120/148/196(6.00/7.40/9.80s), Task 9 99/157/173(4.95/7.85/8.65s)다. Task 4에서 first release→second close gap은 67/73/89 steps로, 첫 완료 이후 두 번째 object를 집기까지도 3.35–4.45s가 필요하다.

즉 첫 semantic completion의 정보가 다음 decision point와 terminal까지 수십~수백 step 유지되어야 한다. 현재 RGB만으로 과거에 어느 object/placement를 완료했는지 항상 식별 가능하지 않으므로 persistent state 가설을 시험할 empirical 간격이 충분하다. 단, 정확한 transition-to-transition distribution은 Task 3 Oracle annotation 뒤 다시 계산해야 한다.

## 22. MaIL obs_seq=5 Comparison

5-frame context는 20 Hz에서 0.25s다. proxy median stage를 5로 나누면:

- Task 3: S1 32.3×, S2 16.1×.
- Task 4: S1 21.1×, S2 29.1×.
- Task 9: S1 31.5×, S2 24.6×.

대표 stage도 최소 66 steps(13.2× context)부터 최대 276 steps(55.2×)까지다. 따라서 `obs_seq=5`는 local motion/grasp cue에는 유용하지만 semantic stage 전체나 이전 completion을 직접 담기에는 현저히 짧다. 이것이 stateful execution을 정당화하지만, 성능 향상을 보장하는 결과는 아니다.

## 23. Action-Horizon / Boundary Analysis

`predict=10`, `execute=1`에서 한 proxy boundary를 포함하는 full-horizon start는 interior boundary마다 9개다. 모든 demo start를 분모로 한 preliminary crossing fraction 평균은 Task 3 3.79%, Task 4 3.64%, Task 9 3.12%다. 이는 두 번째 gripper transition 하나만 센 값이다. Task 4처럼 실제 의미 경계가 여러 개면 전체 crossing 확률은 더 커질 수 있다.

`execute=1`은 한 번에 10-step chunk를 open-loop 실행하지 않으므로 실제 제어가 경계를 넘는 위험을 크게 제한한다. 다만 decoder training target에는 경계 양쪽 action이 함께 들어갈 수 있으므로 Task 3 annotation 후 `boundary_crossing_horizon_fraction`을 Oracle 기준으로 재계산해야 한다.

## 24. Dataset Split / Filter Structure

10개 파일 모두 root에 `mask` group/filter key가 없고 official train/valid/test split도 포함하지 않는다. demo ID는 task마다 `demo_0`–`demo_49`다. MaIL `benchmark_libero10.yaml`의 `hdf5_filter_key`와 validation key는 기본값이 비어 있어 loader가 모든 demo를 읽으며, 별도 분리 없이 train/validation을 구성하면 leakage 위험이 있다.

Task 2.6에서는 split을 만들거나 HDF5를 수정하지 않았다. 이후에는 task별로 고정 seed의 trajectory-level 40/5/5 train/val/test를 외부 JSON manifest로 먼저 만들고, 필요할 때만 원본 복사본에 filter를 기록하는 방식을 권장한다. timestep/window 단위 random split은 같은 trajectory의 인접 frame leakage 때문에 금지해야 한다.

## 25. Visualization Outputs

`analysis/trajectory_videos/visualization_manifest.json`이 대표 9개 MP4/contact sheet의 source of truth다. MP4는 20 fps, 256×128, MPEG-4 Part 2(`mp4v`)이고 agentview와 eye-in-hand를 좌우로 배치했다. HDF5가 `opengl` image convention이므로 수직 반전 후 저장했다. overlay는 task, demo, frame/T, gripper action, reward를 포함한다.

주요 디렉터리:

```text
analysis/trajectory_videos/task_3/
analysis/trajectory_videos/task_4/
analysis/trajectory_videos/task_9/
```

전체 500 demos를 변환하지 않았고, intermediate frame PNG도 생성하지 않았다. 추가 audit contact sheet만 보존했다.

## 26. Candidate Task Re-ranking

1. **Task 4:** 50/50에서 object order가 일관되고, 동일한 pick/place action을 서로 다른 identity/target에 적용하므로 persistent semantic memory 가설이 가장 직접적이다. 다만 7/50 extra-transition과 identity-aware annotation 부담이 있다.
2. **Task 3:** stage가 가장 명확하고 49/50이 nominal 2-transition이라 annotation pilot로 가장 쉽다. bowl completion 뒤 drawer closure까지 긴 gap이 있다.
3. **Task 9:** Task 3과 같은 placement→articulation 구조이며 gap이 길다. 다만 length variance와 recovery proxy가 더 크다.
4. **Task 6:** 두 distinct object/goal로 memory가 필요하지만 articulation-based 명확한 후반 stage가 없고 transition count 변이가 크다.
5. **Task 0:** 두 유사한 basket placement가 길게 이어져 memory 후보지만, order/identity boundary ambiguity가 커 이번에 영상 전수 확인하지 않은 secondary다.

초기 ranking 4→3→9→6→0은 유지되지만, **annotation 난이도만 보면 Task 3이 가장 좋은 pilot**이라는 결론이 추가됐다.

## 27. Recommended Tasks for Task 3

Primary 3개는 **Task 4, Task 3, Task 9**다. annotation 작업 순서는 Task 3으로 protocol을 먼저 검증하고 Task 9, Task 4로 확장하는 것을 권장한다.

- granularity: 접근/motion primitive가 아니라 semantic completion 중심의 coarse stages. recovery를 표현할 수 있도록 completion predicate와 regression 규칙을 함께 둔다.
- Task 3 predicates: bowl-inside-bottom-drawer + release; drawer closed.
- Task 9 predicates: mug-inside-microwave + release; microwave door closed.
- Task 4 predicates: white-on-left-plate; yellow-white-on-right-plate. object identity를 반드시 포함한다.

RGB 판정과 decoded simulator state를 서로 검증하되, action transition만 Oracle로 사용하면 안 된다.

## 28. Information Passed to Task 3

Task 3에서 바로 사용할 입력은 다음과 같다.

- 20 Hz, 0.05 s/step; 각 task 50 demos.
- 대표 demo와 MP4/contact sheet manifest.
- all-demo length/action/gripper-transition JSON/CSV.
- 두 번째 gripper transition 기반 preliminary proxy CSV; **Oracle이 아님**.
- reward/done은 terminal synthetic marker라 completion onset에 부적합.
- named object/contact/articulation signal 부재; RGB 또는 simulator-state decoding 필요.
- `obs[t]`가 `action[t]` 후 관측이라는 alignment risk.
- recovery 후보: Task 3 `demo_34`, Task 4 `demo_33` 및 6-transition demos, Task 9 `demo_3`.
- Task 4 order audit: white first, yellow-white last across 50 demos.

## 29. Remaining Unknowns

1. 정확한 Oracle boundary와 completion persistence/regression 규칙은 아직 정의하지 않았다.
2. flattened `states`에서 bowl/object pose, drawer joint, microwave joint를 찾는 task별 index mapping이 없다.
3. RGB 기반 completion 판정의 inter-annotator agreement와 tolerance가 미정이다.
4. Task 3/9 전체 50 demos의 semantic order를 frame-by-frame 수동 검수하지 않았다.
5. 공식 upstream checksum 목록이 없어 SHA-256은 로컬 baseline이다.
6. nominal 20 Hz는 확인했지만 timestamp가 없어 실제 collection jitter는 알 수 없다.
7. post-action observation/action alignment가 기존 MaIL 학습 의미에 미치는 영향은 별도 controlled audit가 필요하다.
8. train/val/test split은 의도적으로 생성하지 않았다.

Task 2.6은 여기서 종료한다. Task 3 Oracle annotation, 모델 구현, 학습, GPU job은 수행하지 않았다.
