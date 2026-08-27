# R1-RC Canonical Sequential Data Interface

구현은 `src/robocerebra_memory/interface.py`에 있다. index는
`analysis/robocerebra_memory_episode_index.json`, split은
`splits/robocerebra_memory_{train,val,test}.json`을 사용한다.

## 로딩

```python
from robocerebra_memory import RoboCerebraMemoryDataset

dataset = RoboCerebraMemoryDataset(
    "analysis/robocerebra_memory_episode_index.json"
)
episode = dataset.load_episode(dataset.trajectory_ids()[0])

with episode:
    for sample in episode.iter_frames(mode="HOLD"):
        model_input = sample["model_input"]
        analysis = sample["analysis"]
```

HDF5는 `load_episode` 시점이 아니라 state/action을 처음 요청할 때 연다. iterator는
항상 `0..T-1` 순서이며 timestep shuffle API를 제공하지 않는다. `episode_start`와
`episode_end`를 제공하고, subtask transition은 reset 신호가 아니다.

## Model input

`get_frame(t)`의 `model_input`에는 다음만 있다.

- `observation`: 주입한 visual adapter의 결과. adapter가 없으면 `None`.
- `robot_state`: `states[t, 1:10]`, Panda arm qpos 7D + gripper qpos 2D.
- `condition`: 선택한 FULL/CURRENT/HOLD text.
- `action`: `include_action=True`일 때만 반환하는 optional 7D stream.

raw state 전체는 `get_raw_sim_state(t)`로 audit할 수 있지만 privileged object/fixture
state가 포함되므로 기본 model input에 들어가지 않는다. frame index, boundary,
transition count도 model input에 넣지 않는다.

## Analysis-only labels

`analysis` dictionary는 `trajectory_id`, `frame`, `step_index`, `step_text`,
`transition_event`, `steps_since_transition`, `cumulative_transition_count`,
`previous_1..previous_5`를 반환한다. 이 label은 evaluation용이며 encoder/Mamba의
입력 feature가 아니다.

Boundary convention은 `start <= t < end`이다. S1 시작은
`episode_start_event=True`, S2 이후 각 Step 시작은 `transition_event=True`다.

## Conditioning

- `FULL`: 모든 t에서 trajectory의 full instruction.
- `CURRENT`: 모든 t에서 현재 official Step text.
- `HOLD`: 각 Step 시작 frame에서 새 Step text, 그 외에는 `[HOLD]`.

HOLD에서도 observation, robot state, 모델 temporal state update는 계속된다.
HOLD가 hidden state freeze나 episode reset을 의미하지 않는다.

## Observation adapter contract

`RoboCerebraMemoryDataset(..., observation_adapter=fn)`의 `fn(trajectory_id, t)`가
observation을 반환한다. 우선 권장은 원본 external MP4의 image index `t`를 original
trajectory timestep `t`에 직접 대응하는 것이다. 표본은 모두 `T+1` images이며 마지막
extra image는 model timestep에서 제외한다. 60 fps container 값은 재생 속도 metadata일
뿐이므로 임의로 매 3번째 frame을 고르면 안 된다.

두 view가 필요하면 원본 MuJoCo state `t`를 simulator에 설정하고 `agentview` 및
`robot0_eye_in_hand`을 render한다. 공식 converter는 이 replay 원리를 확인해 주지만
no-op action을 제거하고 각 Step을 잘라 저장한다. 따라서 converted local image index를
unfiltered continuous timestep처럼 사용하면 안 되며, converter가 기록한 retained source
index mapping을 사용하거나 filtering 없이 원본 state를 replay해야 한다.

## 오류 정책

index에 들어간 episode는 construction 시 strict boundary를 다시 검사한다. source
state/action 길이가 이후 달라지면 HDF5 open 시 예외를 발생시킨다. 범위 밖 frame,
미할당 frame, ambiguous frame도 조용히 보정하지 않고 예외로 처리한다.
