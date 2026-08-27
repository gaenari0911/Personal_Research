"""Stage B scoring, sequence consistency, and result serialization."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Iterable

import torch
from torch.nn import functional as F

from robocerebra_memory.sampling import DISTANCE_BINS, TRANSITION_BINS

from .aggregation import MetricRow, aggregate_curve, aggregate_rows
from .probes import CONTROL_NAME, TARGET_NAMES, ProbeBank
from .retrieval import rank_scores


TARGET_COLUMNS = {
    "current": "gt_current",
    "prev1": "gt_prev1",
    "prev2": "gt_prev2",
    "prev3": "gt_prev3",
}


@torch.no_grad()
def score_shard(payload: dict, probes: ProbeBank) -> tuple[dict[str, list[MetricRow]], list[dict], list[MetricRow]]:
    candidates = F.normalize(payload["candidate_embeddings"].float(), dim=-1)
    z_t = payload["z_t"].float()
    r_t = payload["r_t"].float()
    temporal_queries = probes.temporal_queries(z_t)
    control_query = probes.instantaneous_query(r_t)
    normalized_texts = tuple(payload["normalized_candidate_texts"])
    result = {name: [] for name in TARGET_NAMES}
    control_rows: list[MetricRow] = []
    sequence_rows = []
    temporal_scores = {
        name: (query @ candidates.transpose(0, 1)).detach().cpu() for name, query in temporal_queries.items()
    }
    control_scores = (control_query @ candidates.transpose(0, 1)).detach().cpu()
    for sample_index, sample in enumerate(payload["samples"]):
        predictions: dict[str, str] = {}
        all_valid = True
        for name in TARGET_NAMES:
            target_index = int(sample[TARGET_COLUMNS[name]])
            if target_index < 0:
                all_valid = False
                continue
            target_text = normalized_texts[target_index]
            positive_indices = tuple(index for index, text in enumerate(normalized_texts) if text == target_text)
            scored = rank_scores(temporal_scores[name][sample_index].numpy(), positive_indices)
            predictions[name] = normalized_texts[scored.predicted_index]
            result[name].append(
                MetricRow(
                    payload["trajectory_id"], int(sample["step_index"]), sample["distance_bin"],
                    sample["transition_bin"], scored.recall_at_1, scored.reciprocal_rank,
                )
            )
        current_index = int(sample["gt_current"])
        current_text = normalized_texts[current_index]
        current_positives = tuple(index for index, text in enumerate(normalized_texts) if text == current_text)
        control = rank_scores(control_scores[sample_index].numpy(), current_positives)
        control_rows.append(
            MetricRow(
                payload["trajectory_id"], int(sample["step_index"]), sample["distance_bin"],
                sample["transition_bin"], control.recall_at_1, control.reciprocal_rank,
            )
        )
        if all_valid:
            expected = {
                name: normalized_texts[int(sample[TARGET_COLUMNS[name]])] for name in TARGET_NAMES
            }
            unambiguous = all(normalized_texts.count(expected[name]) == 1 for name in TARGET_NAMES)
            sequence_rows.append(
                {
                    "trajectory_id": payload["trajectory_id"],
                    "step_index": int(sample["step_index"]),
                    "distance_bin": sample["distance_bin"],
                    "transition_bin": sample["transition_bin"],
                    "exact": float(predictions == expected),
                    "unambiguous": unambiguous,
                }
            )
    return result, sequence_rows, control_rows


def _sequence_aggregate(rows: Iterable[dict], resamples: int, seed: int) -> dict:
    rows = list(rows)
    metric_rows = [
        MetricRow(
            row["trajectory_id"], row["step_index"], row["distance_bin"], row["transition_bin"],
            row["exact"], row["exact"],
        )
        for row in rows
    ]
    aggregate = aggregate_rows(metric_rows, resamples=resamples, seed=seed)
    return {
        "sequence_exact_match_at_4": aggregate["recall_at_1"],
        "ci_low": aggregate["recall_at_1_ci_low"],
        "ci_high": aggregate["recall_at_1_ci_high"],
        "trajectory_count": aggregate["trajectory_count"],
        "sample_count": aggregate["sample_count"],
        "aggregation": aggregate["aggregation"],
        "bootstrap_unit": aggregate["bootstrap_unit"],
        "bootstrap_resamples": resamples,
        "bootstrap_seed": seed,
    }


def evaluate_shards(payloads: Iterable[dict], probes: ProbeBank, *, resamples: int = 2000, seed: int = 42) -> dict:
    all_rows = {name: [] for name in TARGET_NAMES}
    control_rows: list[MetricRow] = []
    sequence_rows: list[dict] = []
    for payload in payloads:
        scored, sequence, control = score_shard(payload, probes)
        for name in TARGET_NAMES:
            all_rows[name].extend(scored[name])
        sequence_rows.extend(sequence)
        control_rows.extend(control)
    summary = {name: aggregate_rows(all_rows[name], resamples=resamples, seed=seed) for name in TARGET_NAMES}
    distance = aggregate_curve(
        all_rows["current"], "distance_bin", [item.name for item in DISTANCE_BINS],
        resamples=resamples, seed=seed,
    )
    transition = aggregate_curve(
        all_rows["current"], "transition_bin", TRANSITION_BINS,
        resamples=resamples, seed=seed,
    )
    control_summary = aggregate_rows(control_rows, resamples=resamples, seed=seed)
    control_distance = aggregate_curve(
        control_rows, "distance_bin", [item.name for item in DISTANCE_BINS],
        resamples=resamples, seed=seed,
    )
    sequence = _sequence_aggregate(sequence_rows, resamples, seed)
    unambiguous = _sequence_aggregate(
        [row for row in sequence_rows if row["unambiguous"]], resamples, seed
    )
    sequence["unambiguous_subset"] = unambiguous
    return {
        "summary": summary,
        "distance": distance,
        "transition": transition,
        "memory_depth": [
            {"k": index, "target": name, **summary[name]} for index, name in enumerate(TARGET_NAMES)
        ],
        "sequence": sequence,
        "control": {
            "z_t_current": summary["current"],
            "r_t_current": control_summary,
            "distance": {"z_t": distance, "r_t": control_distance},
        },
        "raw_rows": all_rows,
        "raw_control_rows": control_rows,
    }


def _atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".stage-b-part")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def write_json(path: Path, value: object) -> None:
    _atomic_text(path, json.dumps(value, indent=2, allow_nan=False) + "\n")


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"refusing to write headerless CSV: {path}")
    from io import StringIO

    buffer = StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    _atomic_text(path, buffer.getvalue())


def write_evaluation_outputs(
    output_dir: str | Path,
    variant: str,
    split: str,
    evaluated: dict,
    provenance: dict | None = None,
) -> None:
    output = Path(output_dir)
    warning = (
        "B2 current retrieval is an oracle-like upper bound because CURRENT text is injected every timestep; "
        "it is not evidence of memory."
        if variant == "B2" else None
    )
    summary_payload = {
        "schema_version": "stage-b-evaluation-v1",
        "variant": variant,
        "split": split,
        "metrics": evaluated["summary"],
        "interpretation_warning": warning,
        "test_split_evaluated": split == "test",
        "provenance": provenance or {},
    }
    write_json(output / "summary.json", summary_payload)
    write_csv(
        output / "summary.csv",
        [{"variant": variant, "split": split, "target": name, **values} for name, values in evaluated["summary"].items()],
    )
    write_csv(output / "current_retention_by_distance.csv", evaluated["distance"])
    write_csv(output / "current_retention_by_transition.csv", evaluated["transition"])
    write_csv(output / "memory_depth.csv", evaluated["memory_depth"])
    write_json(output / "sequence_consistency.json", evaluated["sequence"])
    write_json(output / "instantaneous_control.json", evaluated["control"])
