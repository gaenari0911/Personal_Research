#!/usr/bin/env python3
"""Extract balanced r_t/z_t Stage B caches from a frozen Stage A checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from robocerebra_memory.eval.representation_extractor import (  # noqa: E402
    SCHEMA,
    atomic_json,
    atomic_torch_save,
    balanced_samples_for_episode,
    build_representation_shard,
    checkpoint_identity_matches,
    combined_sampling_sha256,
    extract_selected_representations,
    load_frozen_stage_a_model,
    load_split_ids,
    shard_filename,
    validate_split_count,
    validate_representation_shard,
)
from robocerebra_memory.stage_a import VARIANTS, load_cache  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), required=True)
    parser.add_argument("--final-test", action="store_true")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/stage_b_memory_eval.yaml")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    # This check must precede every test split file read.
    split_path = ROOT / f"splits/robocerebra_memory_{args.split}.json"
    ids = load_split_ids(split_path, args.split, args.final_test)
    if args.device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("real Stage B backbone extraction requires CUDA; use dry_run_stage_b.py for CPU validation")
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    validate_split_count(ids, args.split, int(config["dataset"]["expected_trajectories"][args.split]))
    checkpoint = args.checkpoint or ROOT / config["stage_a_checkpoints"][args.variant]
    output = args.output_dir or ROOT / config["outputs"]["representation_root"] / args.variant / args.split
    episodes = json.loads((ROOT / config["dataset"]["episode_index"]).read_text(encoding="utf-8"))["episodes"]
    by_id = {episode["trajectory_id"]: episode for episode in episodes}
    if set(ids) - set(by_id):
        raise RuntimeError("split contains a trajectory absent from the episode index")
    model, checkpoint_metadata = load_frozen_stage_a_model(checkpoint, args.variant, torch.device("cuda"))
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("Stage B extraction backbone is not frozen")
    shards = []
    trajectory_hashes = []
    for position, trajectory_id in enumerate(ids, start=1):
        filename = shard_filename(trajectory_id)
        target = output / filename
        if target.is_file():
            existing = torch.load(target, map_location="cpu", weights_only=False)
            validate_representation_shard(existing)
            if existing["trajectory_id"] != trajectory_id or existing["variant"] != args.variant or existing["split"] != args.split:
                raise RuntimeError(f"resume shard identity mismatch: {target}")
            if not checkpoint_identity_matches(existing, checkpoint_metadata):
                raise RuntimeError(f"resume shard was extracted from a different Stage A checkpoint: {target}")
            shards.append(filename)
            trajectory_hashes.append((trajectory_id, existing["sampling_sha256"]))
            continue
        episode = by_id[trajectory_id]
        samples = balanced_samples_for_episode(
            episode, args.split, int(config["sampling"]["cap_per_trajectory_step_distance_bin"])
        )
        stage_a_cache = load_cache(trajectory_id, by_id)
        frames = [sample.frame for sample in samples]
        r_t, z_t = extract_selected_representations(
            model, stage_a_cache, args.variant, frames, torch.device("cuda")
        )
        shard = build_representation_shard(
            episode, args.split, args.variant, samples, stage_a_cache, r_t, z_t,
            checkpoint_metadata, torch.float16,
        )
        validate_representation_shard(shard)
        atomic_torch_save(target, shard)
        validate_representation_shard(torch.load(target, map_location="cpu", weights_only=False))
        shards.append(filename)
        trajectory_hashes.append((trajectory_id, shard["sampling_sha256"]))
        atomic_json(
            output / "manifest.json",
            {
                "schema_version": SCHEMA, "status": "IN_PROGRESS", "variant": args.variant,
                "split": args.split, "trajectory_count": len(shards), "expected_trajectory_count": len(ids),
                "shards": shards, "checkpoint": checkpoint_metadata,
                "sampling_sha256": combined_sampling_sha256(trajectory_hashes),
                "test_split": args.split == "test", "final_test_gate": args.final_test,
            },
        )
        print(f"{args.variant} {args.split}: {position}/{len(ids)} {trajectory_id}", flush=True)
    atomic_json(
        output / "manifest.json",
        {
            "schema_version": SCHEMA, "status": "COMPLETE", "variant": args.variant,
            "split": args.split, "trajectory_count": len(shards), "expected_trajectory_count": len(ids),
            "shards": shards, "checkpoint": checkpoint_metadata,
            "sampling_sha256": combined_sampling_sha256(trajectory_hashes),
            "test_split": args.split == "test", "final_test_gate": args.final_test,
        },
    )


if __name__ == "__main__":
    main()
