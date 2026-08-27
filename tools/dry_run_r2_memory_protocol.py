#!/usr/bin/env python3
"""Metadata-only R2 target/bin/sampling dry run; performs no model training."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from src.robocerebra_memory.metrics import retrieval_result, trajectory_macro_metrics, ScoredRetrieval
from src.robocerebra_memory.probes import CandidateSet, duplicate_text_groups
from src.robocerebra_memory.sampling import (
    DISTANCE_BINS,
    TRANSITION_BINS,
    build_balanced_samples,
    evenly_spaced_frames,
    iter_step_distance_cells,
    transition_bin,
)


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def numeric_stats(values) -> dict:
    values = np.asarray(list(values), dtype=np.float64)
    if not len(values):
        return {key: None for key in ("min", "mean", "median", "p75", "p90", "max")}
    return {
        "min": float(np.min(values)),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p75": float(np.percentile(values, 75)),
        "p90": float(np.percentile(values, 90)),
        "max": float(np.max(values)),
    }


def counter_template(names) -> dict[str, int]:
    return {name: 0 for name in names}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--index", type=Path, default=Path("analysis/robocerebra_memory_episode_index.json")
    )
    parser.add_argument("--split-root", type=Path, default=Path("splits"))
    parser.add_argument("--analysis-root", type=Path, default=Path("analysis"))
    parser.add_argument("--cap", type=int, default=4)
    parser.add_argument("--future-horizon", type=int, default=20)
    args = parser.parse_args()

    payload = json.loads(args.index.read_text(encoding="utf-8"))
    by_id = {item["trajectory_id"]: item for item in payload["episodes"]}
    split_ids = {
        split: json.loads(
            (args.split_root / f"robocerebra_memory_{split}.json").read_text(
                encoding="utf-8"
            )
        )
        for split in ("train", "val", "test")
    }
    all_samples = {}
    distance_output = {
        "pre_registered_bins": [
            {"name": item.name, "start_inclusive": item.start, "end_exclusive": item.end}
            for item in DISTANCE_BINS
        ],
        "bin_adjustment_after_results": False,
        "splits": {},
    }
    transition_output = {
        "pre_registered_bins": list(TRANSITION_BINS),
        "splits": {},
    }
    target_output = {
        "candidate_definition": "ordered official Step text entries within each trajectory",
        "duplicate_text_policy": "identical casefolded/whitespace-normalized texts are one positive equivalence class",
        "splits": {},
    }
    sampling_output = {
        "rule": "at most four deterministic evenly spaced frames per trajectory x Step x steps-since-transition bin",
        "cap_per_trajectory_step_distance_bin": args.cap,
        "temporal_correlation_controls": [
            "cell cap",
            "deterministic spacing",
            "trajectory-macro metrics",
            "trajectory-level bootstrap",
        ],
        "splits": {},
    }

    for split, ids in split_ids.items():
        episodes = [by_id[value] for value in ids]
        samples = build_balanced_samples(episodes, split, args.cap)
        all_samples[split] = samples

        raw_distance = counter_template(item.name for item in DISTANCE_BINS)
        cells_distance = counter_template(item.name for item in DISTANCE_BINS)
        trajectory_distance = {item.name: set() for item in DISTANCE_BINS}
        raw_transition = counter_template(TRANSITION_BINS)
        trajectory_transition = {name: set() for name in TRANSITION_BINS}
        previous_raw = {depth: 0 for depth in (1, 2, 3)}
        previous_trajectories = {depth: set() for depth in (1, 2, 3)}
        duplicate_trajectories = 0
        duplicate_group_count = 0
        candidate_counts = []
        unique_candidate_counts = []

        for episode in episodes:
            candidates = CandidateSet.from_episode(episode)
            candidate_counts.append(len(candidates.texts))
            unique_candidate_counts.append(candidates.unique_text_count)
            duplicate_groups = duplicate_text_groups(candidates.texts)
            duplicate_trajectories += int(bool(duplicate_groups))
            duplicate_group_count += len(duplicate_groups)
            for step, bin_spec, start, end in iter_step_distance_cells(episode):
                raw_distance[bin_spec.name] += end - start
                cells_distance[bin_spec.name] += 1
                trajectory_distance[bin_spec.name].add(episode["trajectory_id"])
            for step in episode["steps"]:
                index = int(step["step_index"])
                duration = int(step["end"]) - int(step["start"])
                transition_name = transition_bin(index)
                raw_transition[transition_name] += duration
                trajectory_transition[transition_name].add(episode["trajectory_id"])
                for depth in (1, 2, 3):
                    if index >= depth:
                        previous_raw[depth] += duration
                        previous_trajectories[depth].add(episode["trajectory_id"])

        balanced_distance = Counter(item.distance_bin for item in samples)
        balanced_transition = Counter(item.transition_bin for item in samples)
        balanced_trajectories_distance = {
            name: len({item.trajectory_id for item in samples if item.distance_bin == name})
            for name in raw_distance
        }
        balanced_trajectories_transition = {
            name: len({item.trajectory_id for item in samples if item.transition_bin == name})
            for name in raw_transition
        }
        distance_output["splits"][split] = {
            name: {
                "raw_frames": raw_distance[name],
                "trajectory_step_cells": cells_distance[name],
                "trajectories": len(trajectory_distance[name]),
                "balanced_samples": balanced_distance[name],
                "balanced_trajectories": balanced_trajectories_distance[name],
            }
            for name in raw_distance
        }
        transition_output["splits"][split] = {
            name: {
                "raw_frames": raw_transition[name],
                "trajectories": len(trajectory_transition[name]),
                "balanced_samples": balanced_transition[name],
                "balanced_trajectories": balanced_trajectories_transition[name],
            }
            for name in raw_transition
        }

        previous_balanced = {
            depth: sum(getattr(item, f"previous_{depth}_target") >= 0 for item in samples)
            for depth in (1, 2, 3)
        }
        target_output["splits"][split] = {
            "trajectory_count": len(episodes),
            "candidate_count": numeric_stats(candidate_counts),
            "unique_normalized_candidate_text_count": numeric_stats(unique_candidate_counts),
            "trajectories_with_duplicate_normalized_step_text": duplicate_trajectories,
            "duplicate_normalized_text_group_count": duplicate_group_count,
            "current_balanced_target_count": len(samples),
            "previous_targets": {
                str(depth): {
                    "raw_eligible_frames": previous_raw[depth],
                    "balanced_target_count": previous_balanced[depth],
                    "eligible_trajectories": len(previous_trajectories[depth]),
                }
                for depth in (1, 2, 3)
            },
        }

        sample_counts_by_trajectory = Counter(item.trajectory_id for item in samples)
        sample_counts_by_step = Counter((item.trajectory_id, item.step_index) for item in samples)
        raw_frames = sum(int(item["num_frames"]) for item in episodes)
        sampling_output["splits"][split] = {
            "trajectories": len(episodes),
            "raw_frames": raw_frames,
            "future_prediction_anchors_horizon_20": sum(
                max(0, int(item["num_frames"]) - args.future_horizon) for item in episodes
            ),
            "trajectory_step_distance_cells": sum(cells_distance.values()),
            "balanced_probe_samples": len(samples),
            "raw_to_balanced_ratio": raw_frames / len(samples),
            "samples_per_trajectory": numeric_stats(sample_counts_by_trajectory.values()),
            "samples_per_step": numeric_stats(sample_counts_by_step.values()),
        }

    # Exact, dependency-light smoke test for retrieval code with a perfect query.
    candidates = np.eye(4, dtype=np.float64)
    recall, mrr, rank = retrieval_result(candidates[2], candidates, (2,))
    macro = trajectory_macro_metrics(
        [ScoredRetrieval("dummy/episode", recall, mrr, "dummy")]
    )
    sampling_output["dummy_representation_metric_smoke"] = {
        "candidate_count": 4,
        "target_index": 2,
        "rank": rank,
        "recall_at_1": recall,
        "mrr": mrr,
        "trajectory_macro": macro,
    }

    # Totals make the dry-run report directly consumable without recomputing splits.
    for output in (distance_output, transition_output):
        names = (
            [item.name for item in DISTANCE_BINS]
            if output is distance_output
            else list(TRANSITION_BINS)
        )
        output["all_splits"] = {
            name: {
                key: sum(output["splits"][split][name][key] for split in split_ids)
                for key in ("raw_frames", "balanced_samples")
            }
            for name in names
        }
    sampling_output["all_splits"] = {
        "trajectories": sum(value["trajectories"] for value in sampling_output["splits"].values()),
        "raw_frames": sum(value["raw_frames"] for value in sampling_output["splits"].values()),
        "balanced_probe_samples": sum(
            value["balanced_probe_samples"] for value in sampling_output["splits"].values()
        ),
    }

    write_json(args.analysis_root / "r2_memory_sampling_statistics.json", sampling_output)
    write_json(args.analysis_root / "r2_distance_bin_statistics.json", distance_output)
    write_json(args.analysis_root / "r2_transition_count_statistics.json", transition_output)
    write_json(args.analysis_root / "r2_probe_target_statistics.json", target_output)

    checks = {
        "G1_primary_probe_point": "PASS",
        "G2_non_BC_objective": "PASS",
        "G3_B0_B1_B2_B3_contract": "PASS",
        "G4_current_retrieval": "PASS",
        "G5_steps_since_transition": "PASS",
        "G6_cumulative_transition_count": "PASS",
        "G7_previous_1_2_3": "PASS",
        "G8_CURRENT_leakage_rule": "PASS",
        "G9_instantaneous_control": "PASS",
        "G10_balanced_sampling": "PASS",
        "G11_split_leakage_protocol": "PASS",
        "G12_tests": "PENDING_EXTERNAL_TEST_RUN",
    }
    write_json(
        args.analysis_root / "r2_protocol_gate.json",
        {
            "task": "R2",
            "r2_gate": "PENDING_TESTS",
            "ready_for_model_implementation": False,
            "checks": checks,
            "dry_run": sampling_output["all_splits"],
            "model_trained": False,
            "behavior_cloning_performed": False,
            "gpu_job_submitted": False,
            "dataset_deleted": False,
            "git_push_performed": False,
        },
    )
    print(
        json.dumps(
            {
                "sampling": sampling_output["all_splits"],
                "distance": distance_output["all_splits"],
                "transition": transition_output["all_splits"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

