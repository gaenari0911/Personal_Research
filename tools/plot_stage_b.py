#!/usr/bin/env python3
"""Regenerate Stage B SVG/HTML figures from existing metric outputs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from robocerebra_memory.eval.visualization import (  # noqa: E402
    load_variant_evaluation,
    write_comparison_visualizations,
    write_variant_visualizations,
)
from robocerebra_memory.stage_a import VARIANTS  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-root", type=Path, default=ROOT / "analysis/stage_b")
    parser.add_argument("--variant", choices=VARIANTS)
    parser.add_argument("--comparison", action="store_true")
    args = parser.parse_args()
    if not args.variant and not args.comparison:
        parser.error("select --variant or --comparison")
    generated = []
    if args.variant:
        variant, evaluated = load_variant_evaluation(args.report_root / args.variant)
        generated.extend(write_variant_visualizations(args.report_root / variant, variant, evaluated))
    if args.comparison:
        generated.extend(write_comparison_visualizations(args.report_root))
    if not generated:
        raise RuntimeError("required Stage B metric outputs are incomplete")
    print(json.dumps({"status": "PASS", "generated": [str(path) for path in generated]}, indent=2))


if __name__ == "__main__":
    main()
