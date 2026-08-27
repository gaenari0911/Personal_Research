"""Dependency-free SVG/HTML visualizations for Stage B memory metrics."""

from __future__ import annotations

import csv
import html
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


VARIANT_COLORS = {
    "B0": "#4C78A8",
    "B1": "#F58518",
    "B2": "#E45756",
    "B3": "#2A9D8F",
}
CONTROL_COLOR = "#7A7A7A"
MRR_COLOR = "#7B61A8"
BACKGROUND = "#FBFCFE"
GRID = "#DCE2EA"
TEXT = "#17202A"
MUTED = "#5D6D7E"


@dataclass(frozen=True)
class PlotSeries:
    label: str
    values: tuple[float | None, ...]
    color: str
    lows: tuple[float | None, ...] | None = None
    highs: tuple[float | None, ...] | None = None
    dashed: bool = False


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".stage-b-plot-part")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _number(value) -> float | None:
    if value is None or value == "":
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _escape(value: object) -> str:
    return html.escape(str(value), quote=True)


def _points(values: Sequence[float | None], xs: Sequence[float], y_of) -> str:
    return " ".join(
        f"{xs[index]:.2f},{y_of(value):.2f}"
        for index, value in enumerate(values)
        if value is not None
    )


def line_chart_svg(
    *,
    title: str,
    subtitle: str,
    x_labels: Sequence[str],
    series: Sequence[PlotSeries],
    x_title: str,
    y_title: str = "Score",
    sample_counts: Sequence[int | None] | None = None,
    width: int = 960,
    height: int = 560,
) -> str:
    if not x_labels or not series:
        raise ValueError("a Stage B plot requires labels and at least one series")
    if any(len(item.values) != len(x_labels) for item in series):
        raise ValueError("plot series length does not match x-axis")
    left, right, top, bottom = 82, 32, 116, 100
    plot_width, plot_height = width - left - right, height - top - bottom
    xs = [left + (plot_width / max(1, len(x_labels) - 1)) * index for index in range(len(x_labels))]
    y_of = lambda value: top + (1.0 - min(1.0, max(0.0, float(value)))) * plot_height
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        f'<title id="title">{_escape(title)}</title>',
        f'<desc id="desc">{_escape(subtitle)}</desc>',
        f'<rect width="{width}" height="{height}" rx="18" fill="{BACKGROUND}"/>',
        f'<text x="{left}" y="38" font-family="system-ui,sans-serif" font-size="24" font-weight="700" fill="{TEXT}">{_escape(title)}</text>',
        f'<text x="{left}" y="66" font-family="system-ui,sans-serif" font-size="13" fill="{MUTED}">{_escape(subtitle)}</text>',
    ]
    legend_x = left
    for item in series:
        dash = ' stroke-dasharray="7 5"' if item.dashed else ""
        parts.extend(
            [
                f'<line x1="{legend_x}" y1="91" x2="{legend_x + 28}" y2="91" stroke="{item.color}" stroke-width="3"{dash}/>',
                f'<circle cx="{legend_x + 14}" cy="91" r="4" fill="{item.color}"/>',
                f'<text x="{legend_x + 36}" y="96" font-family="system-ui,sans-serif" font-size="13" fill="{TEXT}">{_escape(item.label)}</text>',
            ]
        )
        legend_x += 48 + max(72, len(item.label) * 8)
    for tick in range(6):
        value = tick / 5
        y = y_of(value)
        parts.append(f'<line x1="{left}" y1="{y:.2f}" x2="{width-right}" y2="{y:.2f}" stroke="{GRID}" stroke-width="1"/>')
        parts.append(f'<text x="{left-14}" y="{y+4:.2f}" text-anchor="end" font-family="system-ui,sans-serif" font-size="12" fill="{MUTED}">{value:.1f}</text>')
    parts.extend(
        [
            f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_height}" stroke="{TEXT}" stroke-width="1.5"/>',
            f'<line x1="{left}" y1="{top+plot_height}" x2="{width-right}" y2="{top+plot_height}" stroke="{TEXT}" stroke-width="1.5"/>',
        ]
    )
    for index, label in enumerate(x_labels):
        x = xs[index]
        parts.append(f'<text x="{x:.2f}" y="{top+plot_height+27}" text-anchor="middle" font-family="system-ui,sans-serif" font-size="12" fill="{TEXT}">{_escape(label)}</text>')
        if sample_counts is not None and sample_counts[index] is not None:
            parts.append(f'<text x="{x:.2f}" y="{top+plot_height+45}" text-anchor="middle" font-family="system-ui,sans-serif" font-size="10" fill="{MUTED}">n={int(sample_counts[index]):,}</text>')
    parts.append(f'<text x="{left+plot_width/2:.2f}" y="{height-18}" text-anchor="middle" font-family="system-ui,sans-serif" font-size="13" font-weight="600" fill="{TEXT}">{_escape(x_title)}</text>')
    parts.append(f'<text x="20" y="{top+plot_height/2:.2f}" text-anchor="middle" transform="rotate(-90 20 {top+plot_height/2:.2f})" font-family="system-ui,sans-serif" font-size="13" font-weight="600" fill="{TEXT}">{_escape(y_title)}</text>')
    for item in series:
        dash = ' stroke-dasharray="7 5"' if item.dashed else ""
        point_string = _points(item.values, xs, y_of)
        if point_string:
            parts.append(f'<polyline points="{point_string}" fill="none" stroke="{item.color}" stroke-width="3" stroke-linejoin="round" stroke-linecap="round"{dash}/>' )
        for index, value in enumerate(item.values):
            if value is None:
                continue
            x, y = xs[index], y_of(value)
            if item.lows is not None and item.highs is not None:
                low, high = item.lows[index], item.highs[index]
                if low is not None and high is not None:
                    y_low, y_high = y_of(low), y_of(high)
                    parts.extend(
                        [
                            f'<line x1="{x:.2f}" y1="{y_high:.2f}" x2="{x:.2f}" y2="{y_low:.2f}" stroke="{item.color}" stroke-width="1.5" opacity="0.75"/>',
                            f'<line x1="{x-5:.2f}" y1="{y_high:.2f}" x2="{x+5:.2f}" y2="{y_high:.2f}" stroke="{item.color}" stroke-width="1.5"/>',
                            f'<line x1="{x-5:.2f}" y1="{y_low:.2f}" x2="{x+5:.2f}" y2="{y_low:.2f}" stroke="{item.color}" stroke-width="1.5"/>',
                        ]
                    )
            parts.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="5" fill="{BACKGROUND}" stroke="{item.color}" stroke-width="3"><title>{_escape(item.label)}: {value:.4f}</title></circle>')
    parts.append("</svg>\n")
    return "".join(parts)


def _series_from_rows(label: str, rows: Sequence[dict], color: str, metric: str = "recall_at_1", dashed: bool = False) -> PlotSeries:
    prefix = "recall_at_1" if metric == "recall_at_1" else "mrr"
    return PlotSeries(
        label=label,
        values=tuple(_number(row.get(metric)) for row in rows),
        lows=tuple(_number(row.get(f"{prefix}_ci_low")) for row in rows),
        highs=tuple(_number(row.get(f"{prefix}_ci_high")) for row in rows),
        color=color,
        dashed=dashed,
    )


def _dashboard(path: Path, title: str, subtitle: str, figures: Sequence[tuple[str, str]], warning: str | None = None) -> None:
    warning_html = f'<div class="warning">{_escape(warning)}</div>' if warning else ""
    cards = "".join(
        f'<section><h2>{_escape(label)}</h2><img src="{_escape(filename)}" alt="{_escape(label)}"></section>'
        for label, filename in figures
    )
    content = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{_escape(title)}</title><style>
body{{margin:0;background:#EEF2F7;color:{TEXT};font-family:system-ui,sans-serif}}main{{max-width:1180px;margin:auto;padding:36px}}
h1{{margin:0 0 8px;font-size:32px}}p{{color:{MUTED};margin:0 0 24px}}.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(500px,1fr));gap:20px}}
section{{background:white;border:1px solid #DCE2EA;border-radius:16px;padding:16px;box-shadow:0 6px 24px #20304012}}h2{{font-size:16px;margin:0 0 10px}}img{{width:100%;height:auto}}
.warning{{background:#FFF1F0;border-left:5px solid {VARIANT_COLORS['B2']};padding:14px 18px;margin:18px 0 24px;border-radius:8px;font-weight:600}}
</style></head><body><main><h1>{_escape(title)}</h1><p>{_escape(subtitle)}</p>{warning_html}<div class="grid">{cards}</div></main></body></html>\n"""
    _atomic_text(path, content)


def write_variant_visualizations(output_dir: str | Path, variant: str, evaluated: dict) -> list[Path]:
    output = Path(output_dir) / "figures"
    color = VARIANT_COLORS[variant]
    distance = evaluated["distance"]
    transition = evaluated["transition"]
    depth = evaluated["memory_depth"]
    control_distance = evaluated["control"]["distance"]["r_t"]
    sequence = evaluated["sequence"]
    paths = []

    def save(name: str, svg: str) -> None:
        path = output / name
        _atomic_text(path, svg)
        paths.append(path)

    save(
        "current_retention.svg",
        line_chart_svg(
            title=f"{variant} Current-Step Retention",
            subtitle="Trajectory-macro Recall@1 with 95% trajectory-bootstrap CI; r_t is the instantaneous control.",
            x_labels=[row["bin"] for row in distance],
            series=(
                _series_from_rows(f"{variant} z_t current", distance, color),
                _series_from_rows("r_t current control", control_distance, CONTROL_COLOR, dashed=True),
            ),
            sample_counts=[row["sample_count"] for row in distance],
            x_title="Steps since transition",
            y_title="Recall@1",
        ),
    )
    save(
        "transition_robustness.svg",
        line_chart_svg(
            title=f"{variant} Transition Robustness",
            subtitle="Current-Step retrieval after accumulated semantic transitions.",
            x_labels=[row["bin"] for row in transition],
            series=(_series_from_rows(f"{variant} z_t current", transition, color),),
            sample_counts=[row["sample_count"] for row in transition],
            x_title="Cumulative transition count",
            y_title="Recall@1",
        ),
    )
    save(
        "memory_depth.svg",
        line_chart_svg(
            title=f"{variant} Memory Depth",
            subtitle="Information recovered from the same current z_t at relative Step offsets k=0..3.",
            x_labels=[f"k={row['k']} {row['target']}" for row in depth],
            series=(
                _series_from_rows("Recall@1", depth, color),
                _series_from_rows("MRR", depth, MRR_COLOR, metric="mrr", dashed=True),
            ),
            sample_counts=[row["sample_count"] for row in depth],
            x_title="Relative subtask offset",
            y_title="Retrieval score",
        ),
    )
    z_current = evaluated["control"]["z_t_current"]
    r_current = evaluated["control"]["r_t_current"]
    save(
        "instantaneous_control.svg",
        line_chart_svg(
            title=f"{variant} Temporal vs Instantaneous Control",
            subtitle="Overall trajectory-macro retrieval. A small gap indicates a strong observation confound.",
            x_labels=("Recall@1", "MRR"),
            series=(
                PlotSeries("z_t temporal", (_number(z_current["recall_at_1"]), _number(z_current["mrr"])), color),
                PlotSeries("r_t instantaneous", (_number(r_current["recall_at_1"]), _number(r_current["mrr"])), CONTROL_COLOR, dashed=True),
            ),
            x_title="Metric",
            y_title="Score",
        ),
    )
    unambiguous = sequence["unambiguous_subset"]
    save(
        "sequence_consistency.svg",
        line_chart_svg(
            title=f"{variant} Sequence Consistency",
            subtitle="Secondary exact-match metric: current/previous-1/2/3 temporal positions must all be correct.",
            x_labels=("All valid", "Unambiguous text"),
            series=(
                PlotSeries(
                    "Exact Match@4",
                    (_number(sequence["sequence_exact_match_at_4"]), _number(unambiguous["sequence_exact_match_at_4"])),
                    color,
                    lows=(_number(sequence["ci_low"]), _number(unambiguous["ci_low"])),
                    highs=(_number(sequence["ci_high"]), _number(unambiguous["ci_high"])),
                ),
            ),
            sample_counts=(sequence["sample_count"], unambiguous["sample_count"]),
            x_title="Eligible subset",
            y_title="Sequence Exact Match@4",
        ),
    )
    warning = (
        "B2 injects the current official Step every timestep. Its current score is an oracle-like upper bound, not memory evidence."
        if variant == "B2" else None
    )
    dashboard = output / "dashboard.html"
    _dashboard(
        dashboard,
        f"Stage B Memory Evaluation — {variant}",
        "Frozen-backbone linear retrieval probes · trajectory-macro estimates · 95% trajectory bootstrap",
        (
            ("Current retention", "current_retention.svg"),
            ("Transition robustness", "transition_robustness.svg"),
            ("Memory depth", "memory_depth.svg"),
            ("Instantaneous control", "instantaneous_control.svg"),
            ("Sequence consistency", "sequence_consistency.svg"),
        ),
        warning,
    )
    paths.append(dashboard)
    return paths


def _read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def load_variant_evaluation(output_dir: str | Path) -> tuple[str, dict]:
    directory = Path(output_dir)
    summary_payload = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    variant = summary_payload["variant"]
    if variant not in VARIANT_COLORS:
        raise RuntimeError(f"unknown Stage B visualization variant: {variant}")
    return variant, {
        "summary": summary_payload["metrics"],
        "distance": _read_csv(directory / "current_retention_by_distance.csv"),
        "transition": _read_csv(directory / "current_retention_by_transition.csv"),
        "memory_depth": _read_csv(directory / "memory_depth.csv"),
        "sequence": json.loads((directory / "sequence_consistency.json").read_text(encoding="utf-8")),
        "control": json.loads((directory / "instantaneous_control.json").read_text(encoding="utf-8")),
    }


def write_comparison_visualizations(report_root: str | Path) -> list[Path]:
    root = Path(report_root)
    variants = ("B0", "B1", "B2", "B3")
    per_variant = {}
    for variant in variants:
        directory = root / variant
        if not (directory / "summary.json").is_file():
            return []
        summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
        control = json.loads((directory / "instantaneous_control.json").read_text(encoding="utf-8"))
        per_variant[variant] = {
            "summary": summary,
            "control": control,
            "distance": _read_csv(directory / "current_retention_by_distance.csv"),
            "transition": _read_csv(directory / "current_retention_by_transition.csv"),
            "depth": _read_csv(directory / "memory_depth.csv"),
            "sequence": json.loads((directory / "sequence_consistency.json").read_text(encoding="utf-8")),
        }
    output = root / "comparison" / "figures"
    paths = []

    def save(name: str, svg: str) -> None:
        path = output / name
        _atomic_text(path, svg)
        paths.append(path)

    first = per_variant["B0"]
    save(
        "retention_distance_comparison.svg",
        line_chart_svg(
            title="B0–B3 Current-Step Retention",
            subtitle="Trajectory-macro Recall@1 with 95% trajectory-bootstrap CI. B2 is oracle-conditioned.",
            x_labels=[row["bin"] for row in first["distance"]],
            series=tuple(
                _series_from_rows(variant, per_variant[variant]["distance"], VARIANT_COLORS[variant], dashed=variant == "B2")
                for variant in variants
            ),
            sample_counts=[int(row["sample_count"]) for row in first["distance"]],
            x_title="Steps since transition",
            y_title="Recall@1",
        ),
    )
    save(
        "transition_robustness_comparison.svg",
        line_chart_svg(
            title="B0–B3 Transition Robustness",
            subtitle="Current-Step retrieval as semantic transitions accumulate.",
            x_labels=[row["bin"] for row in first["transition"]],
            series=tuple(
                _series_from_rows(variant, per_variant[variant]["transition"], VARIANT_COLORS[variant], dashed=variant == "B2")
                for variant in variants
            ),
            sample_counts=[int(row["sample_count"]) for row in first["transition"]],
            x_title="Cumulative transition count",
            y_title="Recall@1",
        ),
    )
    save(
        "memory_depth_comparison.svg",
        line_chart_svg(
            title="B0–B3 Memory Depth",
            subtitle="Recall@1 at current and previous-1/2/3 targets, all decoded from the same current z_t.",
            x_labels=[f"k={row['k']} {row['target']}" for row in first["depth"]],
            series=tuple(
                _series_from_rows(variant, per_variant[variant]["depth"], VARIANT_COLORS[variant], dashed=variant == "B2")
                for variant in variants
            ),
            sample_counts=[int(row["sample_count"]) for row in first["depth"]],
            x_title="Relative subtask offset",
            y_title="Recall@1",
        ),
    )
    z_values = tuple(_number(per_variant[variant]["control"]["z_t_current"]["recall_at_1"]) for variant in variants)
    r_values = tuple(_number(per_variant[variant]["control"]["r_t_current"]["recall_at_1"]) for variant in variants)
    save(
        "instantaneous_control_comparison.svg",
        line_chart_svg(
            title="Temporal Representation vs Instantaneous Control",
            subtitle="Overall current Recall@1. The z_t − r_t gap helps separate memory from visual/proprioceptive confounds.",
            x_labels=variants,
            series=(
                PlotSeries("z_t current", z_values, "#1F5A94"),
                PlotSeries("r_t control", r_values, CONTROL_COLOR, dashed=True),
            ),
            x_title="Model variant",
            y_title="Recall@1",
        ),
    )
    sequence_values = tuple(
        _number(per_variant[variant]["sequence"]["sequence_exact_match_at_4"])
        for variant in variants
    )
    sequence_lows = tuple(_number(per_variant[variant]["sequence"]["ci_low"]) for variant in variants)
    sequence_highs = tuple(_number(per_variant[variant]["sequence"]["ci_high"]) for variant in variants)
    save(
        "sequence_consistency_comparison.svg",
        line_chart_svg(
            title="B0–B3 Sequence Consistency",
            subtitle="Secondary Exact Match@4: all four temporal-offset identities must be simultaneously correct.",
            x_labels=variants,
            series=(PlotSeries("Sequence Exact Match@4", sequence_values, "#6F4E7C", sequence_lows, sequence_highs),),
            sample_counts=tuple(per_variant[variant]["sequence"]["sample_count"] for variant in variants),
            x_title="Model variant",
            y_title="Exact Match@4",
        ),
    )
    dashboard = output / "dashboard.html"
    _dashboard(
        dashboard,
        "Stage B Memory Evaluation — B0/B1/B2/B3",
        "Primary comparison views · frozen-backbone linear probes · trajectory-level uncertainty",
        (
            ("Retention distance", "retention_distance_comparison.svg"),
            ("Transition robustness", "transition_robustness_comparison.svg"),
            ("Memory depth", "memory_depth_comparison.svg"),
            ("Instantaneous control", "instantaneous_control_comparison.svg"),
            ("Sequence consistency", "sequence_consistency_comparison.svg"),
        ),
        "B2 receives CURRENT Step text every timestep. Treat its current score only as an oracle-like upper bound.",
    )
    paths.append(dashboard)
    return paths
