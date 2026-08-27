"""Frozen Stage A checkpoint loading and resume-safe Stage B extraction."""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Sequence

import torch
from torch import Tensor

from robocerebra_memory.models import MemoryExperimentModel
from robocerebra_memory.probes import CandidateSet, normalize_step_text
from robocerebra_memory.sampling import (
    DISTANCE_BINS,
    TRANSITION_BINS,
    ProbeSample,
    build_balanced_samples,
    distance_bin,
    transition_bin,
)
from robocerebra_memory.stage_a import (
    VARIANTS,
    assert_b3_hold_contract,
    language_inputs,
    load_cache,
    state_dict_sha256,
)


SCHEMA = "stage-b-representation-v1"
SAMPLE_ID_FIELDS = (
    "frame", "step_index", "distance_bin", "transition_bin",
    "gt_current", "gt_prev1", "gt_prev2", "gt_prev3",
)


def guard_split_access(split: str, final_test: bool) -> None:
    if split not in {"train", "val", "test"}:
        raise ValueError(f"unknown split: {split}")
    if split == "test" and not final_test:
        raise PermissionError("test split requires the explicit --final-test gate")


def load_split_ids(path: str | Path, split: str, final_test: bool = False) -> list[str]:
    # Deliberately gate before touching the path; this is asserted in leakage tests.
    guard_split_access(split, final_test)
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    ids = list(payload["trajectory_ids"] if isinstance(payload, dict) else payload)
    if not ids or any(not isinstance(value, str) or not value for value in ids):
        raise RuntimeError(f"{split} split contains an invalid trajectory ID")
    if len(set(ids)) != len(ids):
        raise RuntimeError(f"{split} split contains duplicate trajectory IDs")
    return ids


def validate_split_count(ids: Sequence[str], split: str, expected_count: int) -> None:
    if len(ids) != expected_count:
        raise RuntimeError(
            f"{split} split count mismatch: expected {expected_count}, found {len(ids)}"
        )


def atomic_torch_save(path: str | Path, value: object) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".stage-b-part")
    try:
        with temporary.open("wb") as stream:
            torch.save(value, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def atomic_json(path: str | Path, value: object) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".stage-b-part")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def load_frozen_stage_a_model(
    checkpoint_path: str | Path, variant: str, device: torch.device
) -> tuple[MemoryExperimentModel, dict]:
    if variant not in VARIANTS:
        raise ValueError(variant)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if payload.get("schema_version") != "stage-a-v1" or payload.get("variant") != variant:
        raise RuntimeError("Stage A best checkpoint contract mismatch")
    if (
        int(payload.get("global_update", 0)) <= 0
        or int(payload.get("completed_epoch", 0)) <= 0
        or not math.isfinite(float(payload.get("best_val", float("nan"))))
    ):
        raise RuntimeError("Stage A checkpoint is not a completed val-selected training checkpoint")
    model = MemoryExperimentModel(variant)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
    metadata = {
        "path": str(Path(checkpoint_path)),
        "variant": variant,
        "global_update": int(payload.get("global_update", -1)),
        "completed_epoch": int(payload.get("completed_epoch", -1)),
        "best_val": float(payload["best_val"]),
        "state_dict_sha256": state_dict_sha256(model.state_dict()),
    }
    return model, metadata


@torch.no_grad()
def extract_selected_representations(
    model: MemoryExperimentModel,
    cache_payload: dict,
    variant: str,
    frames: Sequence[int],
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    """Extract r_t and final normalized z_t without the Stage A future-target restriction."""
    if not frames:
        raise ValueError("at least one Stage B sample is required")
    length = int(cache_payload["num_frames"])
    if min(frames) < 0 or max(frames) >= length:
        raise IndexError("Stage B sample outside trajectory")
    visual = cache_payload["visual_features"].float().to(device).unsqueeze(0)
    qpos = cache_payload["robot_qpos"].float().to(device).unsqueeze(0)
    language, mask = language_inputs(cache_payload, variant, device)
    if variant == "B3":
        assert_b3_hold_contract(cache_payload, mask)
    selected = torch.tensor(frames, dtype=torch.long, device=device)
    if variant == "B0":
        r_parts: dict[int, Tensor] = {}
        z_parts: dict[int, Tensor] = {}
        by_length: dict[int, list[tuple[int, int]]] = defaultdict(list)
        for output_index, frame in enumerate(frames):
            start = max(0, frame - model.window_size + 1)
            by_length[frame - start + 1].append((output_index, frame))
        for window_length, entries in by_length.items():
            indices = torch.tensor(
                [
                    [frame - window_length + 1 + offset for offset in range(window_length)]
                    for _output_index, frame in entries
                ],
                device=device,
            )
            instantaneous, tokens = model.encoder(
                visual[0, indices], qpos[0, indices], language[0, indices], mask[0, indices]
            )
            temporal, _state = model.backbone.forward_sequence(tokens)
            for local, (output_index, _frame) in enumerate(entries):
                r_parts[output_index] = instantaneous[local, -1]
                z_parts[output_index] = temporal[local, -1]
        r_t = torch.stack([r_parts[index] for index in range(len(frames))])
        z_t = torch.stack([z_parts[index] for index in range(len(frames))])
    else:
        instantaneous, tokens = model.encoder(visual, qpos, language, mask)
        state = model.reset_episode_state(1, device=device, dtype=tokens.dtype)
        temporal, state = model.backbone.forward_sequence(tokens, state)
        if state.steps != length:
            raise RuntimeError("persistent Stage B extraction did not traverse full episode")
        r_t, z_t = instantaneous[0, selected], temporal[0, selected]
        model.discard_episode_state()
    if r_t.shape != (len(frames), 128) or z_t.shape != (len(frames), 128):
        raise RuntimeError("Stage B r_t/z_t shape contract mismatch")
    return r_t.cpu(), z_t.cpu()


def _sample_rows(samples: Iterable[ProbeSample]) -> list[dict]:
    return [sample.to_dict() for sample in samples]


def build_representation_shard(
    episode: dict,
    split: str,
    variant: str,
    samples: Sequence[ProbeSample],
    cache_payload: dict,
    r_t: Tensor,
    z_t: Tensor,
    checkpoint: dict,
    storage_dtype: torch.dtype = torch.float16,
) -> dict:
    if len(samples) != len(r_t) or r_t.shape != z_t.shape:
        raise RuntimeError("Stage B sample/representation count mismatch")
    candidates = CandidateSet.from_episode(episode)
    rows = _sample_rows(samples)
    for row in rows:
        index = int(row["step_index"])
        row.update(
            {
                "normalized_step_text": candidates.normalized_texts[index],
                "steps_since_transition": int(row["frame"]) - int(episode["steps"][index]["start"]),
                "cumulative_transition_count": index,
                "valid_prev1": row["previous_1_target"] >= 0,
                "valid_prev2": row["previous_2_target"] >= 0,
                "valid_prev3": row["previous_3_target"] >= 0,
                "gt_current": row["current_target"],
                "gt_prev1": row["previous_1_target"],
                "gt_prev2": row["previous_2_target"],
                "gt_prev3": row["previous_3_target"],
            }
        )
    return {
        "schema_version": SCHEMA,
        "variant": variant,
        "split": split,
        "trajectory_id": episode["trajectory_id"],
        "checkpoint": checkpoint,
        "samples": rows,
        "candidate_texts": candidates.texts,
        "normalized_candidate_texts": candidates.normalized_texts,
        "candidate_embeddings": cache_payload["step_text_features"].to(storage_dtype).cpu(),
        "r_t": r_t.to(storage_dtype).cpu(),
        "z_t": z_t.to(storage_dtype).cpu(),
        "normalized": True,
        "sampling_sha256": sample_identity_sha256(episode["trajectory_id"], rows),
    }


def sample_identity_sha256(trajectory_id: str, samples: Sequence[dict]) -> str:
    identity = {
        "trajectory_id": trajectory_id,
        "samples": [
            {field: sample[field] for field in SAMPLE_ID_FIELDS}
            for sample in samples
        ],
    }
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def combined_sampling_sha256(trajectory_hashes: Sequence[tuple[str, str]]) -> str:
    encoded = json.dumps(list(trajectory_hashes), separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_representation_shard(payload: dict) -> None:
    required = {
        "schema_version", "variant", "split", "trajectory_id", "samples",
        "checkpoint", "candidate_texts", "normalized_candidate_texts", "candidate_embeddings",
        "r_t", "z_t", "normalized", "sampling_sha256",
    }
    missing = required - set(payload)
    if missing or payload.get("schema_version") != SCHEMA:
        raise RuntimeError(f"invalid Stage B shard fields: {sorted(missing)}")
    if payload.get("variant") not in VARIANTS or payload.get("split") not in {"train", "val", "test"}:
        raise RuntimeError("invalid Stage B shard variant/split")
    if not isinstance(payload.get("trajectory_id"), str) or not payload["trajectory_id"]:
        raise RuntimeError("invalid Stage B trajectory ID")
    checkpoint = payload.get("checkpoint")
    if not isinstance(checkpoint, dict) or checkpoint.get("variant") != payload["variant"] or not checkpoint.get("state_dict_sha256"):
        raise RuntimeError("invalid Stage B checkpoint identity")
    if payload.get("normalized") is not True:
        raise RuntimeError("Stage B shard normalization flag is not true")
    if not isinstance(payload["samples"], list):
        raise RuntimeError("Stage B samples must be a list")
    count = len(payload["samples"])
    candidates = len(payload["candidate_texts"])
    if count == 0 or candidates == 0:
        raise RuntimeError("Stage B shard cannot be empty")
    normalized = tuple(payload["normalized_candidate_texts"])
    expected_normalized = tuple(normalize_step_text(str(text)) for text in payload["candidate_texts"])
    if normalized != expected_normalized or len(normalized) != candidates:
        raise RuntimeError("Stage B normalized candidate text mismatch")
    if not all(isinstance(payload[name], Tensor) for name in ("candidate_embeddings", "r_t", "z_t")):
        raise RuntimeError("Stage B representations/candidates must be tensors")
    if tuple(payload["r_t"].shape) != (count, 128) or tuple(payload["z_t"].shape) != (count, 128):
        raise RuntimeError("invalid Stage B representation shape")
    if tuple(payload["candidate_embeddings"].shape) != (candidates, 512):
        raise RuntimeError("invalid Stage B candidate shape")
    tensors = (payload["candidate_embeddings"], payload["r_t"], payload["z_t"])
    if any(not torch.isfinite(tensor).all() for tensor in tensors):
        raise RuntimeError("non-finite Stage B tensor")
    candidate_norms = payload["candidate_embeddings"].float().norm(dim=-1)
    if not torch.allclose(candidate_norms, torch.ones_like(candidate_norms), atol=0.02, rtol=0.02):
        raise RuntimeError("Stage B candidate embeddings are not normalized")
    valid_distance_bins = {item.name for item in DISTANCE_BINS}
    valid_transition_bins = set(TRANSITION_BINS)
    required_sample_fields = {
        "trajectory_id", "split", "frame", "step_index", "distance_bin", "transition_bin",
        "normalized_step_text", "steps_since_transition", "cumulative_transition_count",
        "valid_prev1", "valid_prev2", "valid_prev3", "gt_current", "gt_prev1", "gt_prev2", "gt_prev3",
    }
    frames = set()
    for sample in payload["samples"]:
        missing_sample = required_sample_fields - set(sample)
        if missing_sample:
            raise RuntimeError(f"Stage B sample fields missing: {sorted(missing_sample)}")
        frame = int(sample["frame"])
        if frame < 0 or frame in frames:
            raise RuntimeError("Stage B sample frames must be unique and non-negative")
        frames.add(frame)
        step = int(sample["step_index"])
        if not 0 <= step < candidates or int(sample["gt_current"]) != step:
            raise RuntimeError("Stage B current GT/step mismatch")
        if sample["trajectory_id"] != payload["trajectory_id"] or sample["split"] != payload["split"]:
            raise RuntimeError("Stage B sample identity mismatch")
        if sample["normalized_step_text"] != normalized[step]:
            raise RuntimeError("Stage B sample normalized text mismatch")
        if sample["distance_bin"] not in valid_distance_bins or sample["transition_bin"] not in valid_transition_bins:
            raise RuntimeError("Stage B sample bin mismatch")
        distance = int(sample["steps_since_transition"])
        cumulative = int(sample["cumulative_transition_count"])
        if distance < 0 or cumulative != step:
            raise RuntimeError("Stage B sample transition metadata mismatch")
        if distance_bin(distance) != sample["distance_bin"] or transition_bin(cumulative) != sample["transition_bin"]:
            raise RuntimeError("Stage B sample value/bin inconsistency")
        for depth in range(1, 4):
            target = int(sample[f"gt_prev{depth}"])
            valid = bool(sample[f"valid_prev{depth}"])
            expected_target = step - depth if step >= depth else -1
            if target != expected_target or valid != (expected_target >= 0):
                raise RuntimeError("Stage B previous-k validity/GT mismatch")
    expected_sampling_hash = sample_identity_sha256(payload["trajectory_id"], payload["samples"])
    if payload["sampling_sha256"] != expected_sampling_hash:
        raise RuntimeError("Stage B sampling identity hash mismatch")


def checkpoint_identity_matches(shard: dict, checkpoint: dict) -> bool:
    existing = shard.get("checkpoint", {})
    return (
        existing.get("variant") == checkpoint.get("variant")
        and existing.get("state_dict_sha256") == checkpoint.get("state_dict_sha256")
        and int(existing.get("global_update", -1)) == int(checkpoint.get("global_update", -1))
        and int(existing.get("completed_epoch", -1)) == int(checkpoint.get("completed_epoch", -1))
    )


def shard_filename(trajectory_id: str) -> str:
    digest = hashlib.sha256(trajectory_id.encode("utf-8")).hexdigest()[:12]
    return f"{trajectory_id.replace('/', '__')}__{digest}.pt"


def balanced_samples_for_episode(episode: dict, split: str, cap: int = 4) -> list[ProbeSample]:
    return build_balanced_samples([episode], split, cap)
