"""Deterministic simulator-GT rules for LIBERO semantic annotation.

Task 3.1 separates the causal transition event from future-looking QA. A
placement candidate begins at the first frame in a run where the official
predicate is true and a causal release event has already been observed. Future
frames may invalidate that candidate because of regression / regrasp, but they
never move the selected run's boundary from its first frame to a later stability
confirmation frame.
"""

from typing import Dict, List, Tuple


STABLE_WINDOW = 5
FPS = 20
DISTANCE_BINS = ["0-5", "6-10", "11-20", "21-40", "41-80", "81+"]

TASK_SPECS = {
    3: {
        "task_name": "KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it",
        "s1_instruction": "place the black bowl inside the bottom drawer",
        "s1_predicate_key": "object_in_target",
        "s1_predicate": "In(akita_black_bowl_1, white_cabinet_1_bottom_region) AND released",
        "s1_object": "akita_black_bowl_1",
        "s2_instruction": "close the bottom drawer",
        "s2_predicate_key": "articulation_closed",
        "s2_predicate": "Close(white_cabinet_1_bottom_region)",
        "s2_kind": "predicate",
        "expected_gripper_cycles": 1,
    },
    4: {
        "task_name": "LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate",
        "s1_instruction": "place the white mug on the left plate",
        "s1_predicate_key": "white_mug_on_left_plate",
        "s1_predicate": "On(porcelain_mug_1, plate_1) AND released",
        "s1_object": "porcelain_mug_1",
        "s2_instruction": "place the yellow-and-white mug on the right plate",
        "s2_predicate_key": "yellow_white_mug_on_right_plate",
        "s2_predicate": "On(white_yellow_mug_1, plate_2) AND released",
        "s2_kind": "placement",
        "s2_object": "white_yellow_mug_1",
        "expected_gripper_cycles": 2,
    },
    9: {
        "task_name": "KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it",
        "s1_instruction": "place the yellow-and-white mug inside the microwave",
        "s1_predicate_key": "object_in_target",
        "s1_predicate": "In(white_yellow_mug_1, microwave_1_heating_region) AND released",
        "s1_object": "white_yellow_mug_1",
        "s2_instruction": "close the microwave door",
        "s2_predicate_key": "articulation_closed",
        "s2_predicate": "Close(microwave_1)",
        "s2_kind": "predicate",
        "expected_gripper_cycles": 1,
    },
}


def true_runs(values: List[bool]) -> List[Tuple[int, int]]:
    runs = []
    start = None
    for i, value in enumerate(values + [False]):
        if value and start is None:
            start = i
        elif not value and start is not None:
            runs.append((start, i - 1))
            start = None
    return runs


def command_runs(rows, closed=True, min_length=5):
    values = [(row["action_gripper"] > 0) == closed for row in rows]
    return [(a, b) for a, b in true_runs(values) if b - a + 1 >= min_length]


def open_onsets(rows):
    return [
        i
        for i in range(1, len(rows))
        if rows[i - 1]["action_gripper"] > 0 and rows[i]["action_gripper"] < 0
    ]


def causal_release_trace(rows, object_name):
    """Build a causal release latch using only observations up through t."""
    grasp = [bool(row["grasps"][object_name]) for row in rows]
    latched = False
    source = None
    released, source_at_t, events = [], [], []
    for t, row in enumerate(rows):
        if grasp[t]:
            if latched:
                events.append({"index": t, "type": "regrasp", "object": object_name})
            latched, source = False, None
        else:
            contact_loss = t > 0 and grasp[t - 1]
            open_command = (
                t > 0
                and rows[t - 1]["action_gripper"] > 0
                and row["action_gripper"] < 0
            )
            if contact_loss or open_command:
                source = (
                    "robosuite_bilateral_finger_contact_loss"
                    if contact_loss
                    else "close_to_open_gripper_command_fallback"
                )
                latched = True
                events.append(
                    {"index": t, "type": "release", "source": source, "object": object_name}
                )
        released.append(bool(latched and not grasp[t]))
        source_at_t.append(source if released[-1] else None)
    return {
        "grasp": grasp,
        "released": released,
        "source_at_t": source_at_t,
        "events": events,
        "bilateral_grasp_observed": any(grasp),
    }


def _invalidation_reason(rows, predicate, release, index):
    reasons = []
    if index < len(rows):
        if not predicate[index]:
            reasons.append("predicate_regression")
        if release["grasp"][index]:
            reasons.append("regrasp")
        if not release["released"][index] and not release["grasp"][index]:
            reasons.append("release_evidence_lost")
    return "+".join(reasons) if reasons else "trajectory_end"


def causal_placement_stage(rows, predicate_key, object_name, terminal=False):
    predicate = [bool(row["predicates"][predicate_key]) for row in rows]
    release = causal_release_trace(rows, object_name)
    condition = [p and r for p, r in zip(predicate, release["released"])]
    runs = true_runs(condition)
    final_run = runs[-1] if runs and runs[-1][1] == len(rows) - 1 else None
    invalidated = []
    earlier_runs = runs[:-1] if final_run else runs
    for start, end in earlier_runs:
        invalidated.append(
            {
                "type": "causal_candidate_invalidated",
                "candidate_boundary": start,
                "candidate_run": [start, end],
                "invalidation_index": end + 1 if end + 1 < len(rows) else None,
                "reason": _invalidation_reason(rows, predicate, release, end + 1),
            }
        )
    if final_run is None:
        return {
            "completion": None,
            "completion_status": "unresolved",
            "stability_status": "unresolved",
            "predicate_key": predicate_key,
            "object_name": object_name,
            "candidate_runs": [list(x) for x in runs],
            "invalidated_candidates": invalidated,
            "release_events": release["events"],
            "bilateral_grasp_observed": release["bilateral_grasp_observed"],
            "completion_evidence": None,
            "future_qa": {"used_to_delay_boundary": False},
        }
    completion = final_run[0]
    run_length = final_run[1] - final_run[0] + 1
    stability_status = "stable" if run_length >= STABLE_WINDOW else "right_censored"
    result = {
        "completion": completion,
        "completion_status": (
            "observed"
            if not terminal or stability_status == "stable"
            else "right_censored"
        ),
        "stability_status": stability_status,
        "predicate_key": predicate_key,
        "object_name": object_name,
        "candidate_runs": [list(x) for x in runs],
        "invalidated_candidates": invalidated,
        "release_events": release["events"],
        "bilateral_grasp_observed": release["bilateral_grasp_observed"],
        "completion_evidence": {
            "official_predicate": True,
            "released": True,
            "grasped": False,
            "release_source": release["source_at_t"][completion],
        },
        "future_qa": {
            "used_to_delay_boundary": False,
            "window_frames": STABLE_WINDOW,
            "available_condition_frames": run_length,
            "checked_through_index": min(len(rows) - 1, completion + STABLE_WINDOW - 1),
            "result": stability_status,
        },
    }
    return result


def causal_predicate_stage(rows, predicate_key, terminal=True):
    predicate = [bool(row["predicates"][predicate_key]) for row in rows]
    runs = true_runs(predicate)
    final_run = runs[-1] if runs and runs[-1][1] == len(rows) - 1 else None
    invalidated = [
        {
            "type": "causal_candidate_invalidated",
            "candidate_boundary": start,
            "candidate_run": [start, end],
            "invalidation_index": end + 1 if end + 1 < len(rows) else None,
            "reason": "predicate_regression",
        }
        for start, end in (runs[:-1] if final_run else runs)
    ]
    if final_run is None:
        return {
            "completion": None,
            "completion_status": "unresolved",
            "stability_status": "unresolved",
            "predicate_key": predicate_key,
            "candidate_runs": [list(x) for x in runs],
            "invalidated_candidates": invalidated,
            "completion_evidence": None,
            "future_qa": {"used_to_delay_boundary": False},
        }
    completion = final_run[0]
    run_length = final_run[1] - final_run[0] + 1
    stability_status = "stable" if run_length >= STABLE_WINDOW else "right_censored"
    return {
        "completion": completion,
        "completion_status": "observed" if terminal and stability_status == "stable" else stability_status,
        "stability_status": stability_status,
        "predicate_key": predicate_key,
        "candidate_runs": [list(x) for x in runs],
        "invalidated_candidates": invalidated,
        "completion_evidence": {"official_predicate": True},
        "future_qa": {
            "used_to_delay_boundary": False,
            "window_frames": STABLE_WINDOW,
            "available_condition_frames": run_length,
            "checked_through_index": min(len(rows) - 1, completion + STABLE_WINDOW - 1),
            "result": stability_status,
        },
    }


def infer_annotation_evidence(trace: Dict):
    rows, task = trace["rows"], int(trace["task_id"])
    spec = TASK_SPECS[task]
    s1 = causal_placement_stage(rows, spec["s1_predicate_key"], spec["s1_object"])
    if spec["s2_kind"] == "placement":
        s2 = causal_placement_stage(rows, spec["s2_predicate_key"], spec["s2_object"], terminal=True)
    else:
        s2 = causal_predicate_stage(rows, spec["s2_predicate_key"], terminal=True)

    closed_runs = command_runs(rows, closed=True)
    extra_cycles = max(0, len(closed_runs) - spec["expected_gripper_cycles"])
    recovery_events = [
        {"stage": stage, **event}
        for stage, detail in (("S1", s1), ("S2", s2))
        for event in detail["invalidated_candidates"]
    ]
    if extra_cycles:
        recovery_events.append(
            {
                "stage": "trajectory",
                "type": "extra_gripper_cycle_candidate",
                "approx_start": closed_runs[spec["expected_gripper_cycles"] - 1][0],
                "extra_cycle_count": extra_cycles,
            }
        )

    conditioning_eligible = s1["completion"] is not None
    confidence = "needs_review"
    if conditioning_eligible:
        source = s1["completion_evidence"]["release_source"]
        nonterminal_extra_cycles = max(
            0, len([x for x in closed_runs if x[0] < s1["completion"]]) - 1
        )
        near_recovery = any(
            e["candidate_run"][1] >= s1["completion"] - 20
            for e in s1["invalidated_candidates"]
        )
        confidence = (
            "high"
            if s1["stability_status"] == "stable"
            and source == "robosuite_bilateral_finger_contact_loss"
            and not near_recovery
            and nonterminal_extra_cycles == 0
            else "medium"
        )

    initial_candidate = min((x[0] for x in s1["candidate_runs"]), default=None)
    final_boundary = s1["completion"]
    return {
        "s1": s1,
        "s2": s2,
        "closed_command_runs": [list(x) for x in closed_runs],
        "conditioning_eligible": conditioning_eligible,
        "needs_review": not conditioning_eligible,
        "confidence": confidence,
        "recovery": {
            "has_recovery": bool(recovery_events),
            "recovery_count": len(recovery_events),
            "events": recovery_events,
            "initial_candidate": initial_candidate,
            "final_selected_boundary": final_boundary,
            "difference_steps": (
                final_boundary - initial_candidate
                if initial_candidate is not None and final_boundary is not None
                else None
            ),
        },
    }


def infer_boundaries(trace: Dict):
    """Compatibility wrapper used by the review renderer."""
    result = infer_annotation_evidence(trace)
    return {
        "s1": result["s1"],
        "s2": result["s2"],
        "closed_command_runs": result["closed_command_runs"],
        "recovery": result["recovery"],
        "confidence": result["confidence"],
    }
