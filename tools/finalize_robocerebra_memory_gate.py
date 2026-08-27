#!/usr/bin/env python3
"""Record the external test run and close the R1-RC gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gate", type=Path, default=Path("analysis/robocerebra_memory_gate.json"))
    parser.add_argument("--total", type=int, required=True)
    parser.add_argument("--passed", type=int, required=True)
    parser.add_argument("--failed", type=int, required=True)
    args = parser.parse_args()
    gate = json.loads(args.gate.read_text(encoding="utf-8"))
    tests_pass = args.total > 0 and args.passed == args.total and args.failed == 0
    gate["checks"]["G12_tests"] = "PASS" if tests_pass else "FAIL"
    gate["test_summary"] = {
        "total": args.total,
        "passed": args.passed,
        "failed": args.failed,
    }
    all_pass = all(value == "PASS" for value in gate["checks"].values())
    gate["r1_rc_gate"] = "PASS" if all_pass else "FAIL"
    gate["ready_for_memory_protocol"] = all_pass
    args.gate.write_text(
        json.dumps(gate, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(gate, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
