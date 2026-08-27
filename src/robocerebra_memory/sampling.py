"""Pre-registered bins and deterministic balanced sampling for R2."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable, Mapping, Optional, Sequence

from .probes import CandidateSet, make_probe_target


@dataclass(frozen=True)
class DistanceBin:
    name: str
    start: int
    end: Optional[int]

    def contains(self, value: int) -> bool:
        return value >= self.start and (self.end is None or value < self.end)


DISTANCE_BINS = (
    DistanceBin("0-4", 0, 5),
    DistanceBin("5-19", 5, 20),
    DistanceBin("20-49", 20, 50),
    DistanceBin("50-99", 50, 100),
    DistanceBin("100-199", 100, 200),
    DistanceBin("200-399", 200, 400),
    DistanceBin("400+", 400, None),
)

TRANSITION_BINS = tuple(str(value) for value in range(8)) + ("8+",)


def distance_bin(value: int, bins: Sequence[DistanceBin] = DISTANCE_BINS) -> str:
    if value < 0:
        raise ValueError("steps_since_transition must be non-negative")
    for item in bins:
        if item.contains(value):
            return item.name
    raise ValueError(f"no distance bin for {value}")


def transition_bin(value: int) -> str:
    if value < 0:
        raise ValueError("cumulative_transition_count must be non-negative")
    return str(value) if value < 8 else "8+"


def evenly_spaced_frames(start: int, end: int, cap: int) -> tuple[int, ...]:
    """Select up to cap inclusive-grid points from the half-open interval."""
    if cap <= 0:
        raise ValueError("cap must be positive")
    if end <= start:
        return ()
    count = end - start
    if count <= cap:
        return tuple(range(start, end))
    if cap == 1:
        return (start + (count - 1) // 2,)
    values = tuple(start + (index * (count - 1)) // (cap - 1) for index in range(cap))
    return tuple(dict.fromkeys(values))


@dataclass(frozen=True)
class ProbeSample:
    split: str
    trajectory_id: str
    frame: int
    step_index: int
    distance_bin: str
    transition_bin: str
    candidate_count: int
    unique_candidate_text_count: int
    current_target: int
    previous_1_target: int
    previous_2_target: int
    previous_3_target: int

    def to_dict(self) -> dict:
        return asdict(self)


def iter_step_distance_cells(
    episode: Mapping[str, object],
    bins: Sequence[DistanceBin] = DISTANCE_BINS,
):
    """Yield (step, bin, absolute_start, absolute_end) nonempty cells."""
    for step in episode["steps"]:
        step_start = int(step["start"])
        duration = int(step["end"]) - step_start
        for item in bins:
            local_end = duration if item.end is None else min(duration, item.end)
            local_start = min(duration, item.start)
            if local_end > local_start:
                yield step, item, step_start + local_start, step_start + local_end


def build_balanced_samples(
    episodes: Iterable[Mapping[str, object]],
    split: str,
    cap_per_trajectory_step_distance_bin: int = 4,
) -> list[ProbeSample]:
    result = []
    for episode in episodes:
        candidates = CandidateSet.from_episode(episode)
        for step, bin_spec, start, end in iter_step_distance_cells(episode):
            current = int(step["step_index"])
            for frame in evenly_spaced_frames(
                start, end, cap_per_trajectory_step_distance_bin
            ):
                targets = [
                    make_probe_target(candidates, frame, current, depth).target_step_index
                    for depth in range(4)
                ]
                result.append(
                    ProbeSample(
                        split=split,
                        trajectory_id=candidates.trajectory_id,
                        frame=frame,
                        step_index=current,
                        distance_bin=bin_spec.name,
                        transition_bin=transition_bin(current),
                        candidate_count=len(candidates.texts),
                        unique_candidate_text_count=candidates.unique_text_count,
                        current_target=targets[0],
                        previous_1_target=targets[1],
                        previous_2_target=targets[2],
                        previous_3_target=targets[3],
                    )
                )
    return result

