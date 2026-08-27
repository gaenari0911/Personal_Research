"""Source-backed semantic-oracle primitives for WashFruitColander.

This module deliberately does not infer simulator contacts from robot state,
images, gripper events, or frame percentages. The primitive inputs must come
from the official RoboCasa / MuJoCo predicate evaluation route.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable, List, Optional, Sequence

import numpy as np


PREDICATE_VERSION = "washfruitcolander-r1.6-source-audit-v1"
TASK_SOURCE_COMMIT = "a07e365c958c4216cd6bbd5f30b47f09a65c6f00"


def predicate_s1_colander_in_sink(basin_location: str) -> bool:
    """Reduce Sink.get_obj_basin_loc output; geometry must be evaluated upstream."""
    return str(basin_location) not in {"", "none", "None"}


def predicate_s2_all_fruits_in_colander(
    official_contact_and_xy_results: Sequence[bool],
) -> bool:
    """Reduce per-fruit OU.check_obj_in_receptacle results without approximation."""
    return bool(official_contact_and_xy_results) and all(
        bool(value) for value in official_contact_and_xy_results
    )


def predicate_s3_colander_aligned_with_water_site(
    colander_position: Sequence[float],
    water_site_position: Sequence[float],
    colander_horizontal_radius: float,
    water_site_axial_size: float,
) -> bool:
    """The geometric portion of Sink.check_obj_under_water, excluding water_on."""
    obj = np.asarray(colander_position, dtype=np.float64)
    water = np.asarray(water_site_position, dtype=np.float64)
    if obj.shape != (3,) or water.shape != (3,):
        raise ValueError("colander and water-site positions must both be 3D")
    xy = np.linalg.norm(obj[:2] - water[:2]) < float(colander_horizontal_radius)
    z = obj[2] < water[2] + float(water_site_axial_size)
    return bool(xy and z)


def predicate_s4_full_success(
    all_fruits_in_colander: bool,
    colander_aligned_with_water_site: bool,
    water_on: bool,
) -> bool:
    """Exact Boolean decomposition of WashFruitColander._check_success."""
    return bool(
        all_fruits_in_colander and colander_aligned_with_water_site and water_on
    )


@dataclass(frozen=True)
class PredicateSnapshot:
    p1: bool
    p2: bool
    p3: bool
    p4: bool


@dataclass(frozen=True)
class StageBoundaries:
    frame_count: int
    transition_c1: Optional[int]
    transition_c2: Optional[int]
    transition_c3: Optional[int]
    terminal_completion: Optional[int]
    valid: bool
    failure_reason: str

    def to_dict(self) -> dict:
        return asdict(self)


def derive_ordered_boundaries(
    snapshots: Iterable[PredicateSnapshot],
) -> StageBoundaries:
    """First-completion monotonic state machine, allowing at most one jump/frame."""
    values = list(snapshots)
    fields = ("p1", "p2", "p3", "p4")
    found: List[Optional[int]] = [None, None, None, None]
    progress = 0
    for frame, snapshot in enumerate(values):
        if progress < len(fields) and bool(getattr(snapshot, fields[progress])):
            found[progress] = frame
            progress += 1
    missing = next((fields[i] for i, value in enumerate(found) if value is None), None)
    valid = missing is None and all(found[i] < found[i + 1] for i in range(3))
    reason = "" if valid else (f"missing_{missing}" if missing else "ordering_violation")
    return StageBoundaries(
        frame_count=len(values),
        transition_c1=found[0],
        transition_c2=found[1],
        transition_c3=found[2],
        terminal_completion=found[3],
        valid=valid,
        failure_reason=reason,
    )


def per_frame_stage_ids(boundaries: StageBoundaries) -> np.ndarray:
    """Serialize four stage IDs; a completion frame remains in its completed stage."""
    if not boundaries.valid:
        raise ValueError(f"cannot label invalid boundaries: {boundaries.failure_reason}")
    labels = np.zeros(boundaries.frame_count, dtype=np.uint8)
    labels[boundaries.transition_c1 + 1 :] = 1
    labels[boundaries.transition_c2 + 1 :] = 2
    labels[boundaries.transition_c3 + 1 :] = 3
    return labels


def label_provenance() -> dict:
    return {
        "label_source": "derived_from_official_robocasa_simulator_predicates",
        "official_dataset_annotation": False,
        "manual_boundary": False,
        "heuristic_only": False,
        "predicate_version": PREDICATE_VERSION,
        "task_source_commit": TASK_SOURCE_COMMIT,
        "generator_version": "r1.6",
    }
