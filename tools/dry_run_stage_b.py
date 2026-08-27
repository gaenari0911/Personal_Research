#!/usr/bin/env python3
"""Synthetic CPU-only end-to-end Stage B validation; never reads RoboCerebra splits."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from robocerebra_memory.eval.metrics import evaluate_shards, write_evaluation_outputs  # noqa: E402
from robocerebra_memory.eval.representation_extractor import atomic_json  # noqa: E402
from robocerebra_memory.eval.training import fit_probe_bank  # noqa: E402
from robocerebra_memory.eval.visualization import write_variant_visualizations  # noqa: E402
from robocerebra_memory.probes import normalize_step_text  # noqa: E402
from robocerebra_memory.sampling import distance_bin, transition_bin  # noqa: E402


def synthetic_shard(trajectory_id: str, split: str, offset: int = 0) -> dict:
    texts = ("S1", "S2", "S3", "S4")
    candidates = torch.zeros(4, 512)
    candidates[:, :4] = torch.eye(4)
    samples = []
    z_values = []
    r_values = []
    for step_index in range(4):
        for local_distance in (0, 6, 24, 55):
            frame = offset + step_index * 100 + local_distance
            samples.append(
                {
                    "trajectory_id": trajectory_id, "frame": frame, "step_index": step_index,
                    "distance_bin": distance_bin(local_distance), "transition_bin": transition_bin(step_index),
                    "gt_current": step_index, "gt_prev1": step_index - 1,
                    "gt_prev2": step_index - 2, "gt_prev3": step_index - 3,
                    "valid_prev1": step_index >= 1, "valid_prev2": step_index >= 2,
                    "valid_prev3": step_index >= 3, "normalized_step_text": normalize_step_text(texts[step_index]),
                    "steps_since_transition": local_distance, "cumulative_transition_count": step_index,
                }
            )
            z = torch.zeros(128)
            z[step_index] = 1.0
            z_values.append(z)
            r_values.append(z.clone())
    return {
        "schema_version": "stage-b-representation-v1", "variant": "B3", "split": split,
        "trajectory_id": trajectory_id, "samples": samples, "candidate_texts": texts,
        "normalized_candidate_texts": tuple(normalize_step_text(text) for text in texts),
        "candidate_embeddings": candidates, "r_t": torch.stack(r_values), "z_t": torch.stack(z_values),
        "normalized": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    train = [synthetic_shard("synthetic/train1", "train"), synthetic_shard("synthetic/train2", "train", 1)]
    val = [synthetic_shard("synthetic/val1", "val")]
    probes, selection = fit_probe_bank(
        train, val, epochs=12, learning_rate=0.05, weight_decay=0.0,
        temperature=1.0, patience=4, seed=42,
    )
    evaluated = evaluate_shards(val, probes, resamples=200, seed=42)
    write_evaluation_outputs(args.output_dir, "B3", "synthetic_val", evaluated)
    write_variant_visualizations(args.output_dir, "B3", evaluated)
    report = {
        "status": "PASS",
        "device": "cpu",
        "gpu_used": False,
        "real_split_files_read": False,
        "test_split_evaluated": False,
        "stages": [
            "synthetic_r_t_z_t", "independent_probe_training", "trajectory_local_similarity",
            "independent_ranking", "gt_rank", "recall_at_1_mrr", "distance_curve",
            "transition_curve", "memory_depth", "sequence_exact_match_at_4", "instantaneous_control",
            "json_csv_outputs",
            "svg_html_visualizations",
        ],
        "selection": selection,
    }
    atomic_json(args.output_dir / "dry_run_report.json", report)
    print(json.dumps({"status": "PASS", "output": str(args.output_dir), "gpu_used": False}, indent=2))


if __name__ == "__main__":
    main()
