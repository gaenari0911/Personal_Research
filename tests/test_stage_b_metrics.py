import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from robocerebra_memory.eval.aggregation import MetricRow, aggregate_curve, aggregate_rows  # noqa: E402


def row(trajectory, step, distance, transition, value):
    return MetricRow(trajectory, step, distance, transition, value, value)


class StageBMetricTests(unittest.TestCase):
    def test_hierarchical_trajectory_macro_not_frame_micro(self):
        rows = [row("A", 0, "0-4", "0", 0.0) for _ in range(100)]
        rows += [row("A", 0, "5-19", "0", 1.0)]
        rows += [row("B", 0, "0-4", "0", 1.0)]
        result = aggregate_rows(rows, resamples=100, seed=42)
        # A: mean(cell0=0, cell1=1)=0.5; B=1; trajectory macro=(0.5+1)/2=.75.
        self.assertAlmostEqual(result["recall_at_1"], 0.75)
        self.assertNotAlmostEqual(result["recall_at_1"], 2 / 102)
        self.assertEqual(result["trajectory_count"], 2)
        self.assertEqual(result["sample_count"], 102)

    def test_bootstrap_is_deterministic_and_trajectory_level(self):
        rows = [row("A", 0, "0-4", "0", 0.0), row("B", 0, "0-4", "0", 1.0)]
        left = aggregate_rows(rows, resamples=500, seed=42)
        right = aggregate_rows(rows, resamples=500, seed=42)
        self.assertEqual(left, right)
        self.assertEqual(left["bootstrap_unit"], "trajectory")
        self.assertEqual(left["bootstrap_resamples"], 500)
        self.assertEqual(left["bootstrap_seed"], 42)

    def test_distance_and_transition_curves_report_counts(self):
        rows = [
            row("A", 0, "0-4", "0", 1.0),
            row("A", 1, "5-19", "1", 0.0),
            row("B", 0, "0-4", "0", 1.0),
        ]
        distance = aggregate_curve(rows, "distance_bin", ["0-4", "5-19"], resamples=50)
        transition = aggregate_curve(rows, "transition_bin", ["0", "1"], resamples=50)
        self.assertEqual(distance[0]["trajectory_count"], 2)
        self.assertEqual(distance[0]["sample_count"], 2)
        self.assertEqual(transition[1]["trajectory_count"], 1)


if __name__ == "__main__":
    unittest.main()
