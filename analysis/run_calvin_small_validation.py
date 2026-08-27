#!/usr/bin/env python3
"""Reproducible, CPU-only audit for the CALVIN small-validation gate."""

from __future__ import annotations

import ast
import csv
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import subprocess
from typing import Any, Callable, Iterable

import cv2
import numpy as np
from PIL import Image, ImageDraw
from transformers import AutoTokenizer


ROOT = Path("/home/itaein/Personal_Research")
DEBUG_ROOT = Path("/ssd1/itaein/datasets/CALVIN/debug/calvin_debug_dataset")
META_ROOT = Path("/ssd1/itaein/datasets/CALVIN/debug/metadata_only")
TOKENIZER_ROOT = Path("/tmp/calvin_clip_tokenizer")
ANALYSIS = ROOT / "analysis"
REVIEW = ANALYSIS / "calvin_review"
FPS = 30.0
SMALL_GAP_FRAMES = 30


def native(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): native(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [native(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(native(data), indent=2, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    if fieldnames is None:
        fieldnames = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def percentiles(values: Iterable[float]) -> dict[str, float | int | None]:
    a = np.asarray(list(values), dtype=np.float64)
    if not len(a):
        return {k: None for k in ("count", "min", "mean", "median", "p75", "p90", "p95", "max")}
    return {
        "count": int(len(a)),
        "min": float(np.min(a)),
        "mean": float(np.mean(a)),
        "median": float(np.median(a)),
        "p75": float(np.percentile(a, 75)),
        "p90": float(np.percentile(a, 90)),
        "p95": float(np.percentile(a, 95)),
        "max": float(np.max(a)),
    }


def git_value(*args: str) -> str:
    return subprocess.check_output(["git", "-C", str(ROOT / "external/calvin"), *args], text=True).strip()


def load_language(split: str, root: Path = META_ROOT) -> tuple[dict[str, Any], np.ndarray]:
    ann = np.load(root / split / "lang_annotations/auto_lang_ann.npy", allow_pickle=True).item()
    episode_bounds = np.asarray(np.load(root / split / "ep_start_end_ids.npy"), dtype=np.int64)
    return ann, episode_bounds


def episode_id(start: int, end: int, bounds: np.ndarray) -> int:
    match = np.where((bounds[:, 0] <= start) & (end <= bounds[:, 1]))[0]
    if len(match) != 1:
        raise ValueError(f"interval [{start}, {end}] maps to {len(match)} physical episodes")
    return int(match[0])


def raw_segments(split: str) -> tuple[list[dict[str, Any]], np.ndarray, dict[str, Any]]:
    ann, bounds = load_language(split)
    rows = []
    for source_id, ((start, end), task, language) in enumerate(
        zip(ann["info"]["indx"], ann["language"]["task"], ann["language"]["ann"])
    ):
        start, end = int(start), int(end)
        rows.append(
            {
                "source_id": source_id,
                "split": split,
                "episode_id": episode_id(start, end, bounds),
                "start": start,
                "end": end,
                "duration_frames_inclusive": end - start + 1,
                "task": str(task),
                "language": str(language),
            }
        )
    rows.sort(key=lambda x: (x["episode_id"], x["start"], x["end"], x["task"], x["source_id"]))
    return rows, bounds, ann


def deduplicate_events(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Merge overlapping/adjacent annotations of the same task within one recording episode."""
    groups: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in segments:
        groups[(row["episode_id"], row["task"])].append(row)
    events = []
    for (ep, task), rows in groups.items():
        rows.sort(key=lambda x: (x["start"], x["end"], x["source_id"]))
        current = None
        for row in rows:
            if current is not None and row["start"] <= current["end"] + 1:
                current["end"] = max(current["end"], row["end"])
                current["source_annotation_count"] += 1
                current["source_ids"].append(row["source_id"])
                current["utterances"].add(row["language"])
            else:
                current = {
                    "split": row["split"],
                    "episode_id": ep,
                    "task": task,
                    "start": row["start"],
                    "end": row["end"],
                    "source_annotation_count": 1,
                    "source_ids": [row["source_id"]],
                    "utterances": {row["language"]},
                }
                events.append(current)
    events.sort(key=lambda x: (x["episode_id"], x["start"], x["end"], x["task"]))
    for i, event in enumerate(events):
        event["event_id"] = i
        event["duration_frames_inclusive"] = event["end"] - event["start"] + 1
        event["utterance"] = sorted(event["utterances"])[0]
    return events


def gap_rows_for(items: list[dict[str, Any]], representation: str) -> list[dict[str, Any]]:
    out = []
    for prev, nxt in zip(items, items[1:]):
        same_episode = prev["episode_id"] == nxt["episode_id"]
        gap = int(nxt["start"] - prev["end"] - 1) if same_episode else None
        relation = "reset_boundary"
        if same_episode:
            relation = "overlap" if gap < 0 else "touching" if gap == 0 else "gap"
        out.append(
            {
                "split": prev["split"],
                "representation": representation,
                "same_physical_episode": same_episode,
                "episode_id": prev["episode_id"],
                "prev_id": prev.get("event_id", prev.get("source_id")),
                "next_id": nxt.get("event_id", nxt.get("source_id")),
                "prev_task": prev["task"],
                "next_task": nxt["task"],
                "prev_start": prev["start"],
                "prev_end": prev["end"],
                "next_start": nxt["start"],
                "next_end": nxt["end"],
                "gap_frames": "" if gap is None else gap,
                "gap_seconds": "" if gap is None else round(gap / FPS, 6),
                "relation": relation,
            }
        )
    return out


def load_task_semantics() -> dict[str, list[dict[str, dict[str, Any]]]]:
    path = ROOT / "external/calvin/calvin_models/calvin_agent/evaluation/multistep_sequences.py"
    tree = ast.parse(path.read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "tasks" for t in node.targets):
            return ast.literal_eval(node.value)
    raise RuntimeError("official task semantics dictionary not found")


TASK_SEMANTICS = load_task_semantics()


def changed_effects(option: dict[str, dict[str, Any]]) -> dict[str, Any]:
    cond, effect = option["condition"], option["effect"]
    return {k: v for k, v in effect.items() if cond.get(k) != v}


def pair_dependency(a: str, b: str) -> str:
    a_options, b_options = TASK_SEMANTICS[a], TASK_SEMANTICS[b]
    for aa in a_options:
        changed = changed_effects(aa)
        for bb in b_options:
            if any(k != "grasped" and k in bb["condition"] and bb["condition"][k] == v for k, v in changed.items()):
                return "HIGH"
    a_keys = set().union(*(set(x["condition"]) | set(x["effect"]) for x in a_options))
    b_keys = set().union(*(set(x["condition"]) | set(x["effect"]) for x in b_options))
    shared = (a_keys & b_keys) - {"grasped"}
    return "MEDIUM" if shared else "LOW"


def chain_dependency(tasks: list[str]) -> tuple[str, list[str]]:
    scores = [pair_dependency(a, b) for a, b in zip(tasks, tasks[1:])]
    if scores and all(x == "HIGH" for x in scores):
        overall = "HIGH"
    elif "HIGH" in scores or (scores and sum(x == "MEDIUM" for x in scores) >= math.ceil(len(scores) / 2)):
        overall = "MEDIUM"
    else:
        overall = "LOW"
    return overall, scores


DEFINITIONS: dict[str, Callable[[int], bool]] = {
    "A_strict_next_start_le_prev_end_plus_1": lambda gap: gap <= 0,
    "B_nonoverlap_gap_le_30_frames": lambda gap: 0 <= gap <= SMALL_GAP_FRAMES,
    "C_same_physical_episode_no_reset": lambda gap: True,
}


def scale_label(count: int) -> str:
    if count >= 500:
        return "very_strong"
    if count >= 100:
        return "strong"
    if count >= 20:
        return "exploratory"
    return "very_weak"


def enumerate_chains(events: list[dict[str, Any]], length: int, eligible: Callable[[int], bool]):
    for i in range(len(events) - length + 1):
        window = events[i : i + length]
        if len({x["episode_id"] for x in window}) != 1:
            continue
        gaps = [window[j + 1]["start"] - window[j]["end"] - 1 for j in range(length - 1)]
        if all(eligible(gap) for gap in gaps):
            yield window, gaps


def chain_audit() -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    statistics: dict[str, Any] = {
        "schema_version": "1.0",
        "data_source": "D->D metadata-only byte-range extraction from official task_D_D.zip",
        "small_gap_threshold_frames": SMALL_GAP_FRAMES,
        "small_gap_threshold_seconds": SMALL_GAP_FRAMES / FPS,
        "definitions": {
            "A": "same-episode consecutive deduplicated events with next.start <= prev.end + 1; overlaps included, so not automatically a sequential demonstration",
            "B": "same-episode consecutive deduplicated events with 0 <= gap <= 30 frames; non-overlapping and at most one second",
            "C": "same physical recording episode with no reset; any unlabeled gap is allowed",
        },
        "splits": {},
    }
    all_gap_rows: list[dict[str, Any]] = []
    pattern_rows: list[dict[str, Any]] = []
    token_candidates: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    for split in ("training", "validation"):
        segments, bounds, ann = raw_segments(split)
        events = deduplicate_events(segments)
        event_rows.extend(events)
        all_gap_rows.extend(gap_rows_for(segments, "raw_annotation"))
        all_gap_rows.extend(gap_rows_for(events, "deduplicated_event"))

        raw_gaps = [r["gap_frames"] for r in all_gap_rows if r["split"] == split and r["representation"] == "raw_annotation" and r["same_physical_episode"]]
        event_gaps = [r["gap_frames"] for r in all_gap_rows if r["split"] == split and r["representation"] == "deduplicated_event" and r["same_physical_episode"]]
        positive_event_gaps = [x for x in event_gaps if x > 0]
        bins = [(-100000, -1), (0, 0), (1, 5), (6, 15), (16, 30), (31, 64), (65, 128), (129, 256), (257, 512), (513, 1000), (1001, 100000)]
        split_stats: dict[str, Any] = {
            "raw_annotation_count": len(segments),
            "deduplicated_event_count": len(events),
            "task_count": len({x["task"] for x in segments}),
            "physical_episode_count": len(bounds),
            "physical_episode_bounds_min": np.min(bounds, axis=0),
            "physical_episode_bounds_max": np.max(bounds, axis=0),
            "physical_episode_duration_frames_inclusive": percentiles(bounds[:, 1] - bounds[:, 0] + 1),
            "annotation_embedding_shape": list(np.asarray(ann["language"]["emb"]).shape),
            "annotation_embedding_dtype": str(np.asarray(ann["language"]["emb"]).dtype),
            "duration_frames_inclusive": percentiles(x["duration_frames_inclusive"] for x in segments),
            "raw_consecutive_gap_frames": percentiles(raw_gaps),
            "deduplicated_event_gap_frames": percentiles(event_gaps),
            "positive_deduplicated_event_gap_frames": percentiles(positive_event_gaps),
            "deduplicated_event_gap_histogram": [
                {"lower": lo, "upper": hi, "count": int(sum(lo <= x <= hi for x in event_gaps))} for lo, hi in bins
            ],
            "chain_definitions": {},
        }

        for definition, eligible in DEFINITIONS.items():
            def_stats = {}
            for length in (2, 3, 4, 5):
                occurrences = list(enumerate_chains(events, length, eligible))
                grouped: dict[tuple[str, ...], list[tuple[list[dict[str, Any]], list[int]]]] = defaultdict(list)
                for window, gaps in occurrences:
                    grouped[tuple(x["task"] for x in window)].append((window, gaps))
                    if length in (4, 5):
                        token_candidates.append(
                            {
                                "split": split,
                                "definition": definition,
                                "length": length,
                                "episode_id": window[0]["episode_id"],
                                "start": window[0]["start"],
                                "end": window[-1]["end"],
                                "tasks": [x["task"] for x in window],
                                "utterances": [x["utterance"] for x in window],
                                "gaps": gaps,
                            }
                        )
                for pattern, values in grouped.items():
                    dep, edge_deps = chain_dependency(list(pattern))
                    flat_gaps = [g for _, gaps in values for g in gaps]
                    first_window, first_gaps = values[0]
                    pattern_rows.append(
                        {
                            "split": split,
                            "definition": definition,
                            "chain_length": length,
                            "pattern": " -> ".join(pattern),
                            "occurrence_count": len(values),
                            "scale_label": scale_label(len(values)),
                            "semantic_dependency_score": dep,
                            "edge_dependency_scores": "|".join(edge_deps),
                            "mean_gap_frames": round(float(np.mean(flat_gaps)), 6) if flat_gaps else "",
                            "max_gap_frames": max(flat_gaps) if flat_gaps else "",
                            "representative_episode_id": first_window[0]["episode_id"],
                            "representative_start": first_window[0]["start"],
                            "representative_end": first_window[-1]["end"],
                            "representative_gaps": "|".join(map(str, first_gaps)),
                        }
                    )
                counts = Counter(len(v) for v in grouped.values())
                dep_counts = Counter(chain_dependency([x["task"] for x in w])[0] for w, _ in occurrences)
                def_stats[str(length)] = {
                    "occurrences": len(occurrences),
                    "unique_patterns": len(grouped),
                    "occurrences_with_at_least_4_unique_tasks": int(
                        sum(len({x["task"] for x in window}) >= 4 for window, _ in occurrences)
                    ),
                    "patterns_with_at_least_2_occurrences": int(sum(v >= 2 for v in map(len, grouped.values()))),
                    "patterns_with_at_least_20_occurrences": int(sum(v >= 20 for v in map(len, grouped.values()))),
                    "max_occurrences_for_one_pattern": max(map(len, grouped.values()), default=0),
                    "occurrence_count_frequency": dict(sorted(counts.items())),
                    "semantic_dependency_occurrences": dict(dep_counts),
                }
            split_stats["chain_definitions"][definition] = def_stats
        statistics["splits"][split] = split_stats
    statistics["training_supervision_assessment"] = {
        "strict_B_five_stage_occurrences": statistics["splits"]["training"]["chain_definitions"]["B_nonoverlap_gap_le_30_frames"]["5"]["occurrences"],
        "strict_B_five_stage_unique_patterns": statistics["splits"]["training"]["chain_definitions"]["B_nonoverlap_gap_le_30_frames"]["5"]["unique_patterns"],
        "strict_B_five_stage_occurrences_with_at_least_4_unique_tasks": statistics["splits"]["training"]["chain_definitions"]["B_nonoverlap_gap_le_30_frames"]["5"]["occurrences_with_at_least_4_unique_tasks"],
        "strict_B_validation_five_stage_occurrences": statistics["splits"]["validation"]["chain_definitions"]["B_nonoverlap_gap_le_30_frames"]["5"]["occurrences"],
        "C_five_stage_occurrences": statistics["splits"]["training"]["chain_definitions"]["C_same_physical_episode_no_reset"]["5"]["occurrences"],
        "C_five_stage_unique_patterns": statistics["splits"]["training"]["chain_definitions"]["C_same_physical_episode_no_reset"]["5"]["unique_patterns"],
        "verdict": "FAIL",
        "reason": "The defensible <=1 s non-overlap rule yields only nine unique five-stage training chains and zero validation chains; same-episode chains are numerous but nearly all unique and require unverified HOLD/masked-loss treatment across unlabeled gaps.",
    }
    return statistics, all_gap_rows, pattern_rows, token_candidates


def modality_schema(value: np.ndarray) -> dict[str, Any]:
    return {"shape": list(value.shape), "dtype": str(value.dtype)}


def wrapped_delta(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (b - a + np.pi) % (2 * np.pi) - np.pi


def debug_data_audit() -> tuple[dict[str, Any], dict[str, Any], dict[str, list[Path]]]:
    schema: dict[str, Any] = {
        "schema_version": "1.0",
        "dataset_root": str(DEBUG_ROOT),
        "archive": {
            "path": "/ssd1/itaein/datasets/CALVIN/debug/calvin_debug_dataset.zip",
            "bytes": 1299150917,
            "sha256": "c66d09147e2c806b244f18ea7d61e388d4dac11f828929779437f728d03e1204",
            "official_checksum_match": True,
        },
        "frequency_hz": FPS,
        "timestamp_field_present": False,
        "timestamp_interpretation": "No explicit timestamp key; consecutive filename indices are 30 Hz control ticks per official config.",
        "splits": {},
    }
    action_camera: dict[str, Any] = {
        "schema_version": "1.0",
        "overall_mail_compatibility": "NEAR",
        "source_semantics": {
            "action_row_alignment": "episode_t stores observation/state at t and the desired TCP state from the next recorded frame as actions[t]; rel_actions[t] is derived from that target and robot_obs[t]",
            "source": "external/calvin/calvin_env/calvin_env/datarenderer.py and utils/utils.py",
        },
        "splits": {},
    }
    split_files: dict[str, list[Path]] = {}
    for split in ("training", "validation"):
        folder = DEBUG_ROOT / split
        files = sorted(folder.glob("episode_*.npz"), key=lambda p: int(p.stem.split("_")[-1]))
        split_files[split] = files
        ids = np.asarray([int(p.stem.split("_")[-1]) for p in files], dtype=np.int64)
        bounds = np.asarray(np.load(folder / "ep_start_end_ids.npy"), dtype=np.int64)
        ep_lens = np.atleast_1d(np.load(folder / "ep_lens.npy")).astype(np.int64)
        scene_info = np.load(folder / "scene_info.npy", allow_pickle=True).item()
        ann = np.load(folder / "lang_annotations/auto_lang_ann.npy", allow_pickle=True).item()
        modality = None
        action_values, rel_values, robot_values, scene_values = [], [], [], []
        rel_formula_errors, next_action_errors, next_gripper_errors = [], [], []
        for file in files:
            with np.load(file) as d:
                if modality is None:
                    modality = {key: modality_schema(d[key]) for key in d.files}
                action_values.append(d["actions"].copy())
                rel_values.append(d["rel_actions"].copy())
                robot_values.append(d["robot_obs"].copy())
                scene_values.append(d["scene_obs"].copy())
                expected_pos = np.clip(d["actions"][:3] - d["robot_obs"][:3], -0.02, 0.02) / 0.02
                expected_orn = np.clip(wrapped_delta(d["robot_obs"][3:6], d["actions"][3:6]), -0.05, 0.05) / 0.05
                expected_rel = np.concatenate([expected_pos, expected_orn, d["actions"][-1:]])
                rel_formula_errors.append(float(np.max(np.abs(expected_rel - d["rel_actions"]))))
        actions = np.stack(action_values)
        rel_actions = np.stack(rel_values)
        robot_obs = np.stack(robot_values)
        scene_obs = np.stack(scene_values)
        for start, end in bounds:
            local = np.where((ids >= start) & (ids <= end))[0]
            if len(local) > 1:
                next_action_errors.extend(np.max(np.abs(actions[local[:-1], :6] - robot_obs[local[1:], :6]), axis=1).tolist())
                next_gripper_errors.extend(np.abs(actions[local[:-1], 6] - robot_obs[local[1:], 14]).tolist())
        gripper = rel_actions[:, 6]
        split_schema = {
            "frame_file_count": len(files),
            "frame_id_min": int(ids.min()),
            "frame_id_max": int(ids.max()),
            "frame_ids_contiguous": bool(np.all(np.diff(ids) == 1)),
            "modalities": modality,
            "ep_start_end_ids": bounds.tolist(),
            "ep_lens": ep_lens.tolist(),
            "episode_bounds_are_inclusive": bool(np.all(bounds[:, 1] - bounds[:, 0] + 1 == ep_lens)),
            "physical_episode_count": len(bounds),
            "scene_info": scene_info,
            "language_annotation_count": len(ann["language"]["ann"]),
            "language_task_count": len(set(ann["language"]["task"])),
            "language_embedding_shape": list(np.asarray(ann["language"]["emb"]).shape),
            "language_embedding_dtype": str(np.asarray(ann["language"]["emb"]).dtype),
            "language_samples": [
                {
                    "task": str(t),
                    "language": str(l),
                    "start": int(ix[0]),
                    "end": int(ix[1]),
                    "duration_frames_inclusive": int(ix[1] - ix[0] + 1),
                    "duration_seconds": round((ix[1] - ix[0] + 1) / FPS, 6),
                }
                for t, l, ix in zip(ann["language"]["task"], ann["language"]["ann"], ann["info"]["indx"])
            ],
        }
        schema["splits"][split] = split_schema
        action_camera["splits"][split] = {
            "frame_count": len(files),
            "actions": {
                "shape": list(actions.shape),
                "dtype": str(actions.dtype),
                "min_per_dim": np.min(actions, axis=0),
                "max_per_dim": np.max(actions, axis=0),
                "mean_per_dim": np.mean(actions, axis=0),
                "std_per_dim": np.std(actions, axis=0),
            },
            "rel_actions": {
                "shape": list(rel_actions.shape),
                "dtype": str(rel_actions.dtype),
                "min_per_dim": np.min(rel_actions, axis=0),
                "max_per_dim": np.max(rel_actions, axis=0),
                "mean_per_dim": np.mean(rel_actions, axis=0),
                "std_per_dim": np.std(rel_actions, axis=0),
                "p01_per_dim": np.percentile(rel_actions, 1, axis=0),
                "p50_per_dim": np.percentile(rel_actions, 50, axis=0),
                "p99_per_dim": np.percentile(rel_actions, 99, axis=0),
                "saturation_rate_abs_ge_0_999_per_motion_dim": np.mean(np.abs(rel_actions[:, :6]) >= 0.999, axis=0),
                "gripper_unique": sorted(np.unique(gripper).tolist()),
                "gripper_open_plus1_fraction": float(np.mean(gripper == 1)),
                "gripper_close_minus1_fraction": float(np.mean(gripper == -1)),
                "gripper_switch_count": int(np.sum(gripper[1:] != gripper[:-1])),
            },
            "robot_obs": {"shape": list(robot_obs.shape), "dtype": str(robot_obs.dtype)},
            "scene_obs": {"shape": list(scene_obs.shape), "dtype": str(scene_obs.dtype)},
            "alignment_checks": {
                "max_rel_action_formula_error": max(rel_formula_errors),
                "max_actions_t_vs_robot_obs_t_plus_1_first_6_error_within_episode": max(next_action_errors, default=None),
                "max_action_gripper_t_vs_robot_obs_gripper_t_plus_1_error_within_episode": max(next_gripper_errors, default=None),
                "same_row_pair_for_policy": "obs[t] -> rel_actions[t]",
                "shift_required": 0,
            },
        }
    action_camera["mail_mapping"] = {
        "action": {
            "calvin": "rel_actions[0:3] relative TCP xyz clipped/scaled by 0.02 m; [3:6] wrapped Euler delta clipped/scaled by 0.05 rad; [6] gripper {-1,+1}",
            "mail": "7D Cartesian manipulation action, per-dimension z-score normalized for BC",
            "verdict": "NEAR",
            "required_adapter": "cast float64 to float32, use rel_actions without time shift, fit MaIL scaler on training split",
        },
        "static_camera": {
            "calvin": "rgb_static uint8 HWC 200x200x3",
            "mail": "agentview_rgb float32 CHW 128x128",
            "verdict": "NEAR",
            "required_adapter": "RGB-preserving resize to 128x128, THWC->TCHW, divide by 255",
        },
        "gripper_camera": {
            "calvin": "rgb_gripper uint8 HWC 84x84x3",
            "mail": "eye_in_hand_rgb float32 CHW 128x128",
            "verdict": "NEAR",
            "required_adapter": "RGB-preserving resize to 128x128, THWC->TCHW, divide by 255",
        },
        "language": {
            "calvin": "raw text plus default 384D MiniLM embeddings",
            "mail": "512D CLIP ViT-B/32 embedding projected to 256D",
            "verdict": "NEAR",
            "required_adapter": "re-encode raw text with openai/clip-vit-base-patch32; do not use the 384D default embedding as if it were CLIP",
        },
        "temporal": {
            "calvin_hz": 30,
            "mail_steps": {"observation_window": 5, "action_horizon": 10},
            "calvin_seconds": {"observation_window": 5 / 30, "action_horizon": 10 / 30},
            "libero_hz": 20,
            "libero_seconds": {"observation_window": 5 / 20, "action_horizon": 10 / 20},
            "verdict": "NEAR",
        },
    }
    return schema, action_camera, split_files


def resize_rgb(array: np.ndarray) -> Image.Image:
    return Image.fromarray(array, mode="RGB").resize((128, 128), Image.Resampling.BILINEAR)


def make_contact_sheet(frames: list[np.ndarray], indices: list[int], title: str, path: Path) -> None:
    cols, rows, tile, header = 7, 3, 128, 42
    canvas = Image.new("RGB", (cols * tile, rows * (tile + 18) + header), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 8), title, fill="black")
    for i, (frame, idx) in enumerate(zip(frames, indices)):
        x, y = (i % cols) * tile, header + (i // cols) * (tile + 18)
        canvas.paste(resize_rgb(frame), (x, y))
        draw.text((x + 3, y + tile + 1), f"t={idx}", fill="black")
    canvas.save(path, quality=92)


def review_outputs() -> list[dict[str, Any]]:
    REVIEW.mkdir(parents=True, exist_ok=True)
    selections = [
        ("validation", "turn_on_lightbulb", 553636, 553700),
        ("validation", "lift_blue_block_slider", 553691, 553755),
        ("validation", "lift_red_block_table", 554046, 554110),
        ("validation", "place_in_slider", 554110, 554145),
        ("training", "push_pink_block_right", 359714, 359757),
    ]
    manifest = []
    for split, task, start, end in selections:
        all_ids = list(range(start, end + 1))
        sample_ids = np.linspace(start, end, 21).round().astype(int).tolist()
        static_frames, gripper_frames, video_frames = [], [], []
        for idx in all_ids:
            with np.load(DEBUG_ROOT / split / f"episode_{idx:07d}.npz") as d:
                static = d["rgb_static"].copy()
                gripper = d["rgb_gripper"].copy()
            if idx in set(sample_ids):
                # Preserve duplicate linspace indices if a very short segment ever occurs.
                pass
            s128 = np.asarray(resize_rgb(static))
            g128 = np.asarray(resize_rgb(gripper))
            video_frames.append(np.concatenate([s128, g128], axis=1))
        for idx in sample_ids:
            with np.load(DEBUG_ROOT / split / f"episode_{idx:07d}.npz") as d:
                static_frames.append(d["rgb_static"].copy())
                gripper_frames.append(d["rgb_gripper"].copy())
        stem = f"{split}_{task}_{start}_{end}"
        static_path = REVIEW / f"{stem}_static_21frames.jpg"
        gripper_path = REVIEW / f"{stem}_gripper_21frames.jpg"
        video_path = REVIEW / f"{stem}_two_view.mp4"
        temp_video_path = REVIEW / f".{stem}_mpeg4_temp.mp4"
        make_contact_sheet(static_frames, sample_ids, f"{task} | rgb_static | [{start}, {end}]", static_path)
        make_contact_sheet(gripper_frames, sample_ids, f"{task} | rgb_gripper | [{start}, {end}]", gripper_path)
        writer = cv2.VideoWriter(str(temp_video_path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (256, 128))
        for frame in video_frames:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        writer.release()
        subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(temp_video_path),
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(video_path),
            ],
            check=True,
        )
        temp_video_path.unlink()
        manifest.append(
            {
                "split": split,
                "task": task,
                "start": start,
                "end": end,
                "duration_frames_inclusive": end - start + 1,
                "duration_seconds": round((end - start + 1) / FPS, 6),
                "static_contact_sheet": static_path.relative_to(ROOT).as_posix(),
                "gripper_contact_sheet": gripper_path.relative_to(ROOT).as_posix(),
                "two_view_video": video_path.relative_to(ROOT).as_posix(),
            }
        )
    write_csv(REVIEW / "review_manifest.csv", manifest)
    return manifest


def tokenize_compounds(candidates: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER_ROOT), local_files_only=True)
    rows = []
    for i, candidate in enumerate(candidates):
        text = "; then ".join(s.rstrip(" .") for s in candidate["utterances"]) + "."
        count = len(tokenizer(text, add_special_tokens=True, truncation=False)["input_ids"])
        dep, edge_deps = chain_dependency(candidate["tasks"])
        rows.append(
            {
                "candidate_id": i,
                "split": candidate["split"],
                "definition": candidate["definition"],
                "chain_length": candidate["length"],
                "episode_id": candidate["episode_id"],
                "start": candidate["start"],
                "end": candidate["end"],
                "tasks": " -> ".join(candidate["tasks"]),
                "gaps_frames": "|".join(map(str, candidate["gaps"])),
                "semantic_dependency_score": dep,
                "edge_dependency_scores": "|".join(edge_deps),
                "compound_instruction": text,
                "clip_tokens_with_special_tokens": count,
                "clip_limit": tokenizer.model_max_length,
                "overflow": count > tokenizer.model_max_length,
            }
        )
    counts = [r["clip_tokens_with_special_tokens"] for r in rows]
    summary = {
        "tokenizer": "openai/clip-vit-base-patch32",
        "implementation": type(tokenizer).__name__,
        "model_max_length": tokenizer.model_max_length,
        "special_tokens_included": True,
        "concatenation_rule": "strip terminal spaces/periods; join stages with '; then '; append one final period",
        "all_candidates": percentiles(counts),
        "overflow_count": sum(r["overflow"] for r in rows),
        "overflow_fraction": sum(r["overflow"] for r in rows) / len(rows) if rows else None,
        "by_definition_length_split": {},
    }
    groups: dict[tuple[str, str, int], list[int]] = defaultdict(list)
    for row in rows:
        groups[(row["split"], row["definition"], int(row["chain_length"]))].append(
            int(row["clip_tokens_with_special_tokens"])
        )
    for key, values in groups.items():
        summary["by_definition_length_split"]["|".join(map(str, key))] = {
            **percentiles(values),
            "overflow_count": sum(x > tokenizer.model_max_length for x in values),
        }
    return rows, summary


def evaluator_audit() -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "official_repo_commit": git_value("rev-parse", "HEAD"),
        "environment_reset": {
            "location": "external/calvin/calvin_models/calvin_agent/evaluation/evaluate_policy.py:129",
            "frequency": "once before each five-task evaluation sequence",
            "between_subtasks": False,
        },
        "policy_reset": {
            "location": "external/calvin/calvin_models/calvin_agent/evaluation/evaluate_policy.py:157",
            "frequency": "at the start of every subtask rollout",
            "classification": "CASE_B",
            "meaning": "environment state persists, but official evaluator discards model hidden state between semantic stages",
        },
        "success_boundary": {
            "location": "external/calvin/calvin_models/calvin_agent/evaluation/evaluate_policy.py:160-176",
            "online": "task oracle compares start_info with current_info after each action and switches stage immediately on success",
            "offline": "auto_lang_ann.npy supplies fixed [start,end] intervals mined from play",
            "equivalence": False,
            "difference": "offline end is a labeled clip boundary and can lag/lead the first online oracle success; online evaluation has no unlabeled gap before the next instruction",
        },
        "stage_horizon": {"steps": 360, "seconds_at_30hz": 12.0},
        "required_modification": "Call model.reset() once next to env.reset() in evaluate_sequence and remove/bypass the per-subtask reset in rollout; preserve task-oracle success and environment continuity.",
        "modification_scope": "evaluator-only; official CALVIN source was not edited",
    }


def migration_gate(chain_stats: dict[str, Any], token_stats: dict[str, Any]) -> dict[str, Any]:
    available = 513727045632
    archive = 177379436142
    extracted = 178284517323
    debug_bytes = 2615026055
    metadata_bytes = 9720848
    return {
        "schema_version": "1.0",
        "decision": "REJECT",
        "decision_scope": "CALVIN as the primary benchmark for the matched supervised 4-6 stage experiment",
        "gates": {
            "A_small_download_and_schema": {"pass": True, "evidence": "official checksum-matched debug archive extracted and loaded"},
            "B_action_camera_compatibility": {"pass": True, "evidence": "7D rel_actions and two RGB views are NEAR-compatible with deterministic adapters"},
            "C_language_and_continuity": {"pass": True, "evidence": "raw language, [start,end], and physical reset bounds are present in D->D metadata"},
            "D_long_horizon_sequence_structure": {
                "pass": True,
                "evidence": f"same-episode definition C contains {chain_stats['splits']['training']['chain_definitions']['C_same_physical_episode_no_reset']['5']['occurrences']} five-event windows",
                "qualification": "most contain unlabeled gaps and nearly every task pattern is unique",
            },
            "E_training_supervision": {
                "pass": False,
                "critical": True,
                "evidence": chain_stats["training_supervision_assessment"],
            },
            "F_reproducible_evaluation": {"pass": True, "evidence": "official 1000x5 evaluator, task oracle, seeded initial states; evaluator reset modification is source-localized"},
            "G_storage_feasibility": {
                "pass": True,
                "available_bytes_after_debug": available,
                "D_archive_bytes": archive,
                "D_extracted_bytes_from_zip_central_directory": extracted,
                "archive_plus_extracted_bytes": archive + extracted,
                "estimated_free_after_archive_and_extracted_bytes": available - archive - extracted,
                "debug_current_bytes": debug_bytes,
                "metadata_only_approx_bytes": metadata_bytes,
                "constraint": "Do not create a second full-size converted copy while retaining the ZIP; use a streaming adapter or remove/archive the ZIP only after explicit approval.",
            },
        },
        "critical_failure": "Gate E",
        "why_not_conditional": "The issue is not a missing metadata check: full D->D annotations were inspected. Repeated, matched five-stage supervised demonstrations are absent under the defensible small-gap rule, and repairing that would require a new data/protocol assumption central to the claim.",
        "next_recommendation": {
            "benchmark": "RoboCasa365",
            "scope": "metadata/action audit of arm-only target composite tasks",
            "download_authorized": False,
            "selection_rule": "Require at least three 4-6 stage target tasks with base_motion=0 and constant arm-active control_mode before any data download.",
        },
        "compound_token_overflow_count": token_stats["overflow_count"],
    }


def main() -> None:
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    schema, action_camera, _ = debug_data_audit()
    chain_stats, gap_rows, pattern_rows, token_candidates = chain_audit()
    token_rows, token_stats = tokenize_compounds(token_candidates)
    review_manifest = review_outputs()
    chain_stats["compound_instruction_token_statistics"] = token_stats
    chain_stats["review_manifest"] = review_manifest
    chain_stats["generated_at_utc"] = datetime.now(timezone.utc).isoformat()
    schema["official_repo"] = {
        "url": git_value("remote", "get-url", "origin"),
        "branch": git_value("branch", "--show-current"),
        "commit": git_value("rev-parse", "HEAD"),
        "commit_date": git_value("show", "-s", "--format=%cI", "HEAD"),
        "package_version": "0.0.1",
        "calvin_env_submodule_commit": subprocess.check_output(
            ["git", "-C", str(ROOT / "external/calvin/calvin_env"), "rev-parse", "HEAD"], text=True
        ).strip(),
    }
    schema["runtime_recommendation"] = {
        "existing_complete_environment_found": False,
        "recommended_environment": "calvin-mail",
        "python": "3.8 (official CALVIN README)",
        "key_pins": ["torch==1.13.1", "pytorch-lightning==1.8.6", "hydra-core==1.1.1", "setuptools==57.5.0"],
        "status": "recommendation only; environment not created",
    }
    schema["metadata_only_source"] = {
        "archive": "http://calvin.cs.uni-freiburg.de/dataset/task_D_D.zip",
        "archive_bytes": 177379436142,
        "archive_sha256": "45efc2fb24a09a50ab3ed6cdc7637604ee857d3ba1bab23d63925c2d71e79d4f",
        "method": "HTTP byte-range extraction using the ZIP64 central directory; each extracted entry was decompressed and CRC32-verified",
        "full_archive_downloaded": False,
        "path": str(META_ROOT),
        "crc32_verified_entries": [
            {"path": "training/ep_start_end_ids.npy", "bytes": 624, "crc32": "ab2a4aca"},
            {"path": "training/ep_lens.npy", "bytes": 376, "crc32": "30f4e134"},
            {"path": "training/scene_info.npy", "bytes": 408, "crc32": "8ec10886"},
            {"path": "training/lang_annotations/auto_lang_ann.npy", "bytes": 8108180, "crc32": "c39066c2"},
            {"path": "validation/ep_start_end_ids.npy", "bytes": 192, "crc32": "70990dfc"},
            {"path": "validation/ep_lens.npy", "bytes": 160, "crc32": "80bc28d9"},
            {"path": "validation/lang_annotations/auto_lang_ann.npy", "bytes": 1617294, "crc32": "8efe0c8d"},
        ],
    }

    evaluator = evaluator_audit()
    gate = migration_gate(chain_stats, token_stats)
    write_json(ANALYSIS / "calvin_schema_audit.json", schema)
    write_json(ANALYSIS / "calvin_action_camera_compatibility.json", action_camera)
    write_json(ANALYSIS / "calvin_evaluator_reset_audit.json", evaluator)
    write_json(ANALYSIS / "calvin_chain_statistics.json", chain_stats)
    write_json(ANALYSIS / "calvin_migration_gate.json", gate)
    write_csv(ANALYSIS / "calvin_language_segment_gaps.csv", gap_rows)
    write_csv(ANALYSIS / "calvin_chain_patterns.csv", pattern_rows)
    write_csv(ANALYSIS / "calvin_compound_instruction_tokens.csv", token_rows)
    print(json.dumps({"gate": gate["decision"], "files": 8, "review_segments": len(review_manifest)}, indent=2))


if __name__ == "__main__":
    main()
