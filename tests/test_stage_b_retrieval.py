import sys
import unittest
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from robocerebra_memory.eval.metrics import score_shard  # noqa: E402
from robocerebra_memory.eval.probes import ProbeBank  # noqa: E402
from robocerebra_memory.eval.retrieval import (  # noqa: E402
    compute_gt_rank,
    rank_scores,
    recall_at_1_from_rank,
    reciprocal_rank,
)


class StageBRetrievalTests(unittest.TestCase):
    def test_hand_computable_four_probe_rankings_and_sequence_failure(self):
        bank = ProbeBank()
        rankings = {
            "current": [0.1, 0.2, 0.3, 0.4],       # S4>S3>S2>S1
            "prev1": [0.1, 0.2, 0.3, 0.4],         # GT S3 rank 2
            "prev2": [0.2, 0.4, 0.3, 0.1],         # S2>S3>S1>S4
            "prev3": [0.3, 0.4, 0.2, 0.1],         # S2>S1>S3>S4
            "instantaneous_current": [0.1, 0.2, 0.3, 0.4],
        }
        with torch.no_grad():
            for name, values in rankings.items():
                bank.probes[name].projection.weight.zero_()
                bank.probes[name].projection.weight[:4, 0] = torch.tensor(values)
        candidates = torch.zeros(4, 512)
        candidates[:, :4] = torch.eye(4)
        representation = torch.zeros(1, 128)
        representation[0, 0] = 1.0
        payload = {
            "trajectory_id": "tau", "candidate_embeddings": candidates,
            "normalized_candidate_texts": ("s1", "s2", "s3", "s4"),
            "z_t": representation, "r_t": representation.clone(),
            "samples": [{
                "step_index": 3, "distance_bin": "0-4", "transition_bin": "3",
                "gt_current": 3, "gt_prev1": 2, "gt_prev2": 1, "gt_prev3": 0,
            }],
        }
        rows, sequence, _control = score_shard(payload, bank)
        observed = {
            name: (values[0].recall_at_1, values[0].reciprocal_rank)
            for name, values in rows.items()
        }
        self.assertEqual(observed["current"], (1.0, 1.0))
        self.assertEqual(observed["prev1"], (0.0, 0.5))
        self.assertEqual(observed["prev2"], (1.0, 1.0))
        self.assertEqual(observed["prev3"], (0.0, 0.5))
        self.assertEqual(sequence[0]["exact"], 0.0)

    def test_explicit_gt_rank_recall_and_rr(self):
        rank = compute_gt_rank([0.4, 0.9, 0.8, 0.1], [2])
        self.assertEqual(rank, 2)
        self.assertEqual(recall_at_1_from_rank(rank), 0.0)
        self.assertEqual(reciprocal_rank(rank), 0.5)

    def test_duplicate_normalized_text_is_multi_positive(self):
        scored = rank_scores([0.2, 0.8, 0.9], [0, 2])
        self.assertEqual(scored.rank, 1)
        self.assertEqual(scored.recall_at_1, 1.0)

    def test_candidate_reordering_preserves_metric_when_gt_is_remapped(self):
        scores = np.asarray([0.2, 0.9, 0.4, 0.1])
        original = rank_scores(scores, [2])
        permutation = np.asarray([3, 2, 0, 1])
        reordered = rank_scores(scores[permutation], [int(np.where(permutation == 2)[0][0])])
        self.assertEqual((original.rank, original.recall_at_1, original.reciprocal_rank),
                         (reordered.rank, reordered.recall_at_1, reordered.reciprocal_rank))

    def test_ranking_uses_only_trajectory_local_candidates(self):
        local = rank_scores([0.9, 0.8], [0])
        another_trajectory_distractor = 1.0
        self.assertEqual(local.rank, 1)
        self.assertGreater(another_trajectory_distractor, 0.9)


if __name__ == "__main__":
    unittest.main()
