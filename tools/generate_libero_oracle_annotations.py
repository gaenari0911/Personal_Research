#!/usr/bin/env python3
"""Generate Task 3.1 causal simulator-GT Oracle annotations and summaries."""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np

from semantic_annotation_core import DISTANCE_BINS, FPS, TASK_SPECS, infer_annotation_evidence


ROOT = Path(__file__).resolve().parents[1]
TRACE_ROOT = ROOT / "analysis/semantic_state_traces"
DEFAULT_OUTPUT = ROOT / "annotations/libero10_semantic"
DEFAULT_SUMMARY = ROOT / "analysis/oracle_annotation_summary.csv"
PROXY_CSV = ROOT / "analysis/libero10_preliminary_stage_proxy_stats.csv"


def load_traces():
    traces = []
    for task in (3, 4, 9):
        path = TRACE_ROOT / f"task_{task}_all.jsonl"
        with path.open(encoding="utf-8") as handle:
            traces.extend(json.loads(line) for line in handle if line.strip())
    return sorted(traces, key=lambda x: (x["task_id"], int(x["demo_id"].split("_")[-1])))


def load_proxy():
    result = {}
    with PROXY_CSV.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            task = int(row["task_id"])
            if task in TASK_SPECS:
                result[(task, row["demo_id"])] = int(
                    row["proxy_boundary_second_gripper_transition"]
                )
    return result


def subtask_record(subtask_id, instruction, action_start, terminal, detail, predicate, confidence):
    completion = detail["completion"]
    return {
        "subtask_id": subtask_id,
        "instruction": instruction,
        "action_start": action_start,
        "is_terminal": terminal,
        "completion_obs_index": completion,
        "completion_action_index": completion,
        "next_subtask_action_start": None if terminal else completion + 1,
        "completion_predicate": predicate,
        "completion_evidence": detail["completion_evidence"],
        "completion_status": detail["completion_status"],
        "stability_status": detail["stability_status"],
        "future_qa": detail["future_qa"],
        "confidence": confidence if not terminal else None,
    }


def make_annotation(trace, proxy_boundary):
    task, T = int(trace["task_id"]), int(trace["T"])
    spec = TASK_SPECS[task]
    evidence = infer_annotation_evidence(trace)
    s1 = evidence["s1"]
    s1_completion = s1["completion"]
    s2_start = s1_completion + 1 if s1_completion is not None else None
    retention = T - s2_start if s2_start is not None else None
    nonterminal_extra_cycles = max(
        0,
        len([x for x in evidence["closed_command_runs"] if x[0] < s1_completion]) - 1,
    ) if s1_completion is not None else 0
    nonterminal_recovery = bool(
        evidence["s1"]["invalidated_candidates"] or nonterminal_extra_cycles
    )
    annotation = {
        "annotation_version": "libero_semantic_subtask_v1",
        "protocol_revision": "task_3_1_causal_boundary",
        "dataset": "LIBERO-10",
        "task_id": task,
        "task_name": spec["task_name"],
        "demo_id": trace["demo_id"],
        "trajectory_length": T,
        "alignment": {
            "observation_semantics": "post_action",
            "rule": "obs[t] is stored after executing action[t]",
            "next_action_rule": "completion_obs_index + 1",
        },
        "conditioning_eligible": evidence["conditioning_eligible"],
        "transition_confidence": evidence["confidence"],
        "subtasks": [
            subtask_record(
                0, spec["s1_instruction"], 0, False, s1,
                spec["s1_predicate"], evidence["confidence"],
            ),
            subtask_record(
                1, spec["s2_instruction"], s2_start, True, evidence["s2"],
                spec["s2_predicate"], evidence["confidence"],
            ),
        ],
        "transition_metadata": {
            "s1_completion_obs_index": s1_completion,
            "s2_action_start": s2_start,
            "retention_length_steps": retention,
            "retention_length_seconds_at_20hz": round(retention / FPS, 4) if retention is not None else None,
            "steps_since_last_transition_definition": "for action t >= s2_action_start: t - s2_action_start",
            "suggested_distance_bins": DISTANCE_BINS,
        },
        "has_recovery": evidence["recovery"]["has_recovery"],
        "recovery_events": evidence["recovery"]["events"],
        "recovery_summary": {
            key: evidence["recovery"][key]
            for key in ("recovery_count", "initial_candidate", "final_selected_boundary", "difference_steps")
        } | {
            "nonterminal_recovery": nonterminal_recovery,
            "nonterminal_extra_gripper_cycles": nonterminal_extra_cycles,
        },
        "proxy_comparison": {
            "proxy_name": "second_gripper_command_transition",
            "proxy_boundary": proxy_boundary,
            "signed_oracle_minus_proxy": s1_completion - proxy_boundary if s1_completion is not None else None,
            "absolute_difference_frames": abs(s1_completion - proxy_boundary) if s1_completion is not None else None,
        },
        "needs_review": evidence["needs_review"],
        "review_reasons": [] if not evidence["needs_review"] else ["non_terminal_transition_unresolved"],
        "notes": "Terminal completion is not required for S1→S2 conditioning eligibility.",
    }
    return annotation, evidence


def validate(annotation):
    T = annotation["trajectory_length"]
    s1, s2 = annotation["subtasks"]
    errors = []
    if annotation["conditioning_eligible"]:
        c = s1["completion_obs_index"]
        if not isinstance(c, int) or not 0 <= c < T:
            errors.append("s1_completion_out_of_bounds")
        if s1["completion_action_index"] != c:
            errors.append("s1_action_obs_mismatch")
        if s1["next_subtask_action_start"] != c + 1:
            errors.append("next_start_mismatch")
        if not s1["next_subtask_action_start"] < T:
            errors.append("next_start_out_of_bounds")
        if s2["action_start"] != c + 1:
            errors.append("s2_action_start_mismatch")
        if not s1["completion_evidence"]["official_predicate"]:
            errors.append("official_predicate_missing")
        if not s1["completion_evidence"]["released"]:
            errors.append("release_missing")
        if s1["future_qa"].get("used_to_delay_boundary") is not False:
            errors.append("future_leakage_flag")
    if s2["next_subtask_action_start"] is not None:
        errors.append("terminal_next_not_null")
    if s2["completion_obs_index"] is not None:
        if not 0 <= s2["completion_obs_index"] < T:
            errors.append("s2_completion_out_of_bounds")
        if s2["completion_action_index"] != s2["completion_obs_index"]:
            errors.append("s2_action_obs_mismatch")
        if annotation["conditioning_eligible"] and not s1["completion_obs_index"] < s2["completion_obs_index"]:
            errors.append("boundary_ordering")
    return errors


def descriptive(values):
    a = np.asarray(values, dtype=float)
    return {
        "count": int(a.size), "min": float(np.min(a)), "mean": float(np.mean(a)),
        "median": float(np.median(a)), "max": float(np.max(a)), "std": float(np.std(a)),
        "p25": float(np.percentile(a, 25)), "p75": float(np.percentile(a, 75)),
    }


def proxy_stats(rows):
    values = [r["proxy_abs_difference"] for r in rows]
    a = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(a)), "median": float(np.median(a)),
        "p90": float(np.percentile(a, 90)), "max": float(np.max(a)),
    }


def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def select_review(rows):
    selected = {}
    def add(row, reason):
        key = (row["task_id"], row["demo_id"])
        selected.setdefault(key, {"task_id": key[0], "demo_id": key[1], "reasons": []})
        if reason not in selected[key]["reasons"]:
            selected[key]["reasons"].append(reason)
    for task in (3, 4, 9):
        task_rows = [r for r in rows if r["task_id"] == task]
        high = sorted((r for r in task_rows if r["transition_confidence"] == "high"), key=lambda r: r["trajectory_length"])
        if high:
            for i in sorted(set((0, len(high) // 2, len(high) - 1))): add(high[i], "high_confidence_sample")
        medium = sorted((r for r in task_rows if r["transition_confidence"] == "medium"), key=lambda r: r["proxy_abs_difference"], reverse=True)
        for r in medium[:2]: add(r, "medium_confidence_sample")
        add(max(task_rows, key=lambda r: r["remaining_episode_length"]), "largest_retention")
        add(max(task_rows, key=lambda r: r["proxy_abs_difference"]), "largest_proxy_disagreement")
        recovery = [r for r in task_rows if r["nonterminal_recovery"]]
        if recovery: add(max(recovery, key=lambda r: r["candidate_difference_steps"]), "largest_nonterminal_recovery")
    return sorted(selected.values(), key=lambda x: (x["task_id"], int(x["demo_id"].split("_")[-1])))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    args = parser.parse_args()
    traces, proxy = load_traces(), load_proxy()
    annotations, summary_rows, qa_errors = [], [], []
    args.output_root.mkdir(parents=True, exist_ok=True)
    for trace in traces:
        key = (int(trace["task_id"]), trace["demo_id"])
        annotation, evidence = make_annotation(trace, proxy[key])
        errors = validate(annotation)
        if errors:
            qa_errors.append({"task_id": key[0], "demo_id": key[1], "errors": errors})
        annotations.append(annotation)
        task_dir = args.output_root / f"task_{key[0]}"
        task_dir.mkdir(parents=True, exist_ok=True)
        (task_dir / f"{key[1]}.json").write_text(json.dumps(annotation, indent=2) + "\n", encoding="utf-8")
        s1, s2 = annotation["subtasks"]
        nonterminal_extra = max(0, len([x for x in evidence["closed_command_runs"] if x[0] < s1["completion_obs_index"]]) - 1)
        nonterminal_recovery = bool(evidence["s1"]["invalidated_candidates"] or nonterminal_extra)
        summary_rows.append({
            "task_id": key[0], "demo_id": key[1], "trajectory_length": trace["T"],
            "s1_completion_obs": s1["completion_obs_index"], "s2_action_start": s2["action_start"],
            "s1_duration": s1["completion_obs_index"] + 1,
            "remaining_episode_length": annotation["transition_metadata"]["retention_length_steps"],
            "retention_seconds_at_20hz": annotation["transition_metadata"]["retention_length_seconds_at_20hz"],
            "conditioning_eligible": annotation["conditioning_eligible"],
            "transition_confidence": annotation["transition_confidence"],
            "has_recovery": annotation["has_recovery"], "recovery_count": evidence["recovery"]["recovery_count"],
            "nonterminal_recovery": nonterminal_recovery,
            "candidate_initial": evidence["recovery"]["initial_candidate"],
            "candidate_final": evidence["recovery"]["final_selected_boundary"],
            "candidate_difference_steps": evidence["recovery"]["difference_steps"],
            "terminal_completion_status": s2["completion_status"],
            "terminal_completion_obs": s2["completion_obs_index"],
            "needs_review": annotation["needs_review"],
            "proxy_boundary": proxy[key],
            "proxy_signed_oracle_minus": annotation["proxy_comparison"]["signed_oracle_minus_proxy"],
            "proxy_abs_difference": annotation["proxy_comparison"]["absolute_difference_frames"],
        })
    if qa_errors:
        raise RuntimeError(f"annotation QA failed: {qa_errors[:5]}")

    aggregate = args.output_root / "annotations.jsonl"
    aggregate.write_text("".join(json.dumps(x, separators=(",", ":")) + "\n" for x in annotations), encoding="utf-8")
    fields = list(summary_rows[0])
    write_csv(args.summary, summary_rows, fields)

    task_stats = {}
    for task in (3, 4, 9):
        rows = [r for r in summary_rows if r["task_id"] == task]
        task_stats[str(task)] = {
            "coverage": {
                "total": len(rows), "conditioning_eligible": sum(r["conditioning_eligible"] for r in rows),
                "confidence": dict(Counter(r["transition_confidence"] for r in rows)),
                "needs_review": sum(r["needs_review"] for r in rows),
                "terminal_status": dict(Counter(r["terminal_completion_status"] for r in rows)),
            },
            "trajectory_steps": descriptive([r["trajectory_length"] for r in rows]),
            "s1_duration_steps": descriptive([r["s1_duration"] for r in rows]),
            "retention_steps": descriptive([r["remaining_episode_length"] for r in rows]),
            "retention_seconds": descriptive([r["retention_seconds_at_20hz"] for r in rows]),
            "retention_over_mail_obs_seq_5": descriptive([r["remaining_episode_length"] / 5 for r in rows]),
            "recovery": {
                "any_stage_demos": sum(r["has_recovery"] for r in rows),
                "nonterminal_demos": sum(r["nonterminal_recovery"] for r in rows),
                "max_initial_to_final_difference": max(r["candidate_difference_steps"] for r in rows),
            },
            "proxy_absolute_difference_frames": proxy_stats(rows),
        }
    all_proxy = proxy_stats(summary_rows)
    recovery_proxy = proxy_stats([r for r in summary_rows if r["nonterminal_recovery"]])
    statistics = {
        "annotation_version": "libero_semantic_subtask_v1",
        "protocol_revision": "task_3_1_causal_boundary",
        "fps": FPS,
        "total_demos": len(summary_rows),
        "automatically_accepted": sum(r["conditioning_eligible"] and not r["needs_review"] for r in summary_rows),
        "needs_review": sum(r["needs_review"] for r in summary_rows),
        "terminal_right_censored": sum(r["terminal_completion_status"] == "right_censored" for r in summary_rows),
        "terminal_unresolved": sum(r["terminal_completion_status"] == "unresolved" for r in summary_rows),
        "task_stats": task_stats,
        "proxy_absolute_difference_frames_all": all_proxy,
        "proxy_absolute_difference_frames_nonterminal_recovery": recovery_proxy,
    }
    stats_path = ROOT / "analysis/oracle_annotation_statistics.json"
    stats_path.write_text(json.dumps(statistics, indent=2) + "\n", encoding="utf-8")

    motivation_rows = []
    for task in (3, 4, 9):
        s = task_stats[str(task)]
        motivation_rows.append({
            "task_id": task,
            "median_trajectory_steps": s["trajectory_steps"]["median"],
            "median_s1_duration_steps": s["s1_duration_steps"]["median"],
            "median_post_transition_retention_steps": s["retention_steps"]["median"],
            "median_post_transition_retention_seconds": s["retention_seconds"]["median"],
            "mail_context_steps": 5,
            "retention_over_context": s["retention_steps"]["median"] / 5,
        })
    write_csv(ROOT / "analysis/oracle_experiment_motivation.csv", motivation_rows, list(motivation_rows[0]))

    queue_fields = ["task", "demo", "candidate_boundary", "reason", "confidence", "recovery_evidence", "video_or_contact_sheet"]
    queue_rows = [{
        "task": r["task_id"], "demo": r["demo_id"], "candidate_boundary": r["candidate_final"],
        "reason": "non_terminal_transition_unresolved", "confidence": r["transition_confidence"],
        "recovery_evidence": r["nonterminal_recovery"], "video_or_contact_sheet": "",
    } for r in summary_rows if r["needs_review"]]
    write_csv(ROOT / "analysis/annotation_review/human_review_queue.csv", queue_rows, queue_fields)

    selection = {"policy": "3 high/task, 2 medium/task, recovery/outlier extrema", "items": select_review(summary_rows)}
    (ROOT / "analysis/annotation_review/automatic_review_selection.json").write_text(json.dumps(selection, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "annotation_version": "libero_semantic_subtask_v1",
        "protocol_revision": "task_3_1_causal_boundary",
        "dataset": "LIBERO-10", "tasks": [3, 4, 9], "total_annotations": len(annotations),
        "conditioning_eligible": statistics["automatically_accepted"],
        "needs_review": statistics["needs_review"],
        "terminal_right_censored": statistics["terminal_right_censored"],
        "terminal_unresolved": statistics["terminal_unresolved"],
        "aggregate_jsonl": str(aggregate.relative_to(ROOT)),
        "summary_csv": str(args.summary.relative_to(ROOT)),
        "statistics_json": str(stats_path.relative_to(ROOT)),
        "qa": {"status": "PASS", "validated_annotations": len(annotations), "errors": 0},
    }
    (args.output_root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (ROOT / "analysis/oracle_annotation_qa.json").write_text(json.dumps({
        "status": "PASS", "checked": len(annotations), "errors": [],
        "checks": ["index_bounds", "action_observation_equality", "next_start", "ordering_when_observed", "terminal_next_null", "official_predicate", "release", "future_not_used_to_delay_boundary", "schema"],
    }, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
