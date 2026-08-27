import sys
import unittest
from pathlib import Path

import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from robocerebra_memory.eval.probes import (  # noqa: E402
    CONTROL_NAME,
    TARGET_NAMES,
    ProbeBank,
    assert_frozen_backbone,
    freeze_backbone,
    positive_mask,
    probe_loss,
)
from robocerebra_memory.eval.representation_extractor import extract_selected_representations  # noqa: E402
from robocerebra_memory.models import MemoryExperimentModel  # noqa: E402


class StageBProbeTests(unittest.TestCase):
    def test_four_temporal_probes_are_independent(self):
        bank = ProbeBank(seed=42)
        bank.assert_independent()
        pointers = [bank.probes[name].projection.weight.data_ptr() for name in TARGET_NAMES]
        self.assertEqual(len(set(pointers)), 4)

    def test_same_current_z_is_input_to_all_four_probes(self):
        bank = ProbeBank(seed=42)
        z_t = torch.randn(3, 128)
        seen = []
        handles = []
        for name in TARGET_NAMES:
            handles.append(bank.probes[name].register_forward_pre_hook(lambda _module, inputs: seen.append(inputs[0])))
        output = bank.temporal_queries(z_t)
        for handle in handles:
            handle.remove()
        self.assertEqual(set(output), set(TARGET_NAMES))
        self.assertEqual(len(seen), 4)
        self.assertTrue(all(value is z_t for value in seen))

    def test_previous_k_invalid_rows_are_masked(self):
        candidates = ("s1", "s2", "s3", "s4")
        targets = torch.tensor([-1, 1, -1, 3])
        mask = positive_mask(targets, candidates)
        self.assertFalse(mask[0].any())
        self.assertTrue(mask[1, 1])
        self.assertFalse(mask[2].any())
        self.assertTrue(mask[3, 3])
        probe = ProbeBank().probes["prev2"]
        loss = probe_loss(probe, torch.randn(4, 128), torch.randn(4, 512), targets, candidates, 1.0)
        self.assertTrue(torch.isfinite(loss))

    def test_instantaneous_and_temporal_current_weights_are_independent(self):
        bank = ProbeBank()
        self.assertNotEqual(
            bank.probes[CONTROL_NAME].projection.weight.data_ptr(),
            bank.probes["current"].projection.weight.data_ptr(),
        )

    def test_probe_backward_never_populates_frozen_backbone_grad(self):
        backbone = freeze_backbone(nn.Sequential(nn.Linear(6, 128), nn.LayerNorm(128)))
        z_t = backbone(torch.randn(5, 6))
        probe = ProbeBank().probes["current"]
        loss = probe_loss(
            probe, z_t, torch.randn(4, 512), torch.tensor([0, 1, 2, 3, 0]),
            ("s1", "s2", "s3", "s4"), 1.0,
        )
        loss.backward()
        assert_frozen_backbone(backbone)
        self.assertIsNotNone(probe.projection.weight.grad)

    def test_cpu_synthetic_extraction_returns_r_t_and_final_z_t_for_all_variants(self):
        torch.manual_seed(42)
        common = MemoryExperimentModel("B0", n_layer=1, d_state=2, d_conv=2, expand=1)
        common_state = common.state_dict()
        payload = {
            "num_frames": 7,
            "visual_features": torch.randn(7, 512).half(),
            "robot_qpos": torch.randn(7, 9),
            "full_text_feature": torch.randn(512),
            "step_text_features": torch.randn(2, 512),
            "step_boundaries": [(0, 3), (3, 7)],
        }
        instantaneous = []
        for variant in ("B0", "B1", "B2", "B3"):
            model = MemoryExperimentModel(variant, n_layer=1, d_state=2, d_conv=2, expand=1)
            model.load_state_dict(common_state, strict=True)
            freeze_backbone(model)
            r_t, z_t = extract_selected_representations(
                model, payload, variant, [0, 2, 3, 6], torch.device("cpu")
            )
            self.assertEqual(r_t.shape, (4, 128))
            self.assertEqual(z_t.shape, (4, 128))
            self.assertTrue(torch.isfinite(r_t).all())
            self.assertTrue(torch.isfinite(z_t).all())
            self.assertIsNone(model.episode_state)
            instantaneous.append(r_t)
        for r_t in instantaneous[1:]:
            self.assertTrue(torch.allclose(r_t, instantaneous[0], atol=1e-6, rtol=1e-5))


if __name__ == "__main__":
    unittest.main()
