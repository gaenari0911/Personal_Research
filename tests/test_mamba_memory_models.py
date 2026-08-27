"""R3 executable model, state, objective, and probe contract tests."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from robocerebra_memory.interface import RoboCerebraMemoryDataset  # noqa: E402
from robocerebra_memory.losses import future_info_nce  # noqa: E402
from robocerebra_memory.models import (  # noqa: E402
    CommonEncoder,
    FrozenCLIPFeatureEncoder,
    InMemoryFeatureCache,
    MemoryExperimentModel,
    build_condition_schedule,
)
from robocerebra_memory.probes import (  # noqa: E402
    CandidateSet,
    LinearRetrievalProbe,
    make_probe_target,
    multi_positive_probe_loss,
)


def tiny_model(variant: str, **overrides) -> MemoryExperimentModel:
    values = dict(
        d_model=16,
        state_feature_dim=8,
        n_layer=2,
        d_state=4,
        d_conv=3,
        expand=1,
        future_dim=32,
        window_size=5,
    )
    values.update(overrides)
    return MemoryExperimentModel(variant, **values)


def features(time: int = 7, batch: int = 1):
    return (
        torch.randn(batch, time, 512),
        torch.randn(batch, time, 9),
        torch.randn(batch, time, 512),
    )


EPISODE = {
    "trajectory_id": "scene/case",
    "full_instruction": "Do both tasks",
    "num_frames": 7,
    "steps": [
        {"step_index": 0, "text": "Pick bowl", "start": 0, "end": 3},
        {"step_index": 1, "text": "Place bowl", "start": 3, "end": 7},
    ],
}


class TinyClip(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1))

    def get_image_features(self, pixel_values):
        return torch.ones(pixel_values.shape[0], 512, device=pixel_values.device) * self.weight

    def get_text_features(self, input_ids, attention_mask=None):
        return torch.ones(input_ids.shape[0], 512, device=input_ids.device) * self.weight


class TinyProcessor:
    @classmethod
    def from_pretrained(cls, *_args, **_kwargs):
        return cls()

    def __call__(self, *, images=None, text=None, **_kwargs):
        if images is not None:
            return {"pixel_values": torch.ones(len(images), 3, 2, 2)}
        return {
            "input_ids": torch.ones(len(text), 2, dtype=torch.long),
            "attention_mask": torch.ones(len(text), 2, dtype=torch.long),
        }


class R3MambaMemoryModelTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    def test_01_common_encoder_shapes_and_taps(self):
        encoder = CommonEncoder()
        r, x = encoder(*features(time=3))
        self.assertEqual(r.shape, (1, 3, 128))
        self.assertEqual(x.shape, (1, 3, 128))
        self.assertFalse(torch.equal(r, x))

    def test_02_privileged_state_width_is_rejected(self):
        encoder = CommonEncoder()
        v, _, language = features(time=2)
        with self.assertRaisesRegex(ValueError, "exactly 9"):
            encoder(v, torch.randn(1, 2, 116), language)

    def test_03_clip_constructor_freezes_parameters(self):
        cache = InMemoryFeatureCache()
        with patch("transformers.CLIPModel.from_pretrained", return_value=TinyClip()), patch(
            "transformers.CLIPProcessor.from_pretrained", return_value=TinyProcessor()
        ):
            encoder = FrozenCLIPFeatureEncoder(feature_cache=cache)
        self.assertTrue(all(not p.requires_grad for p in encoder.clip.parameters()))
        self.assertFalse(encoder.clip.training)

    def test_04_clip_feature_cache_and_shapes(self):
        cache = InMemoryFeatureCache()
        with patch("transformers.CLIPModel.from_pretrained", return_value=TinyClip()), patch(
            "transformers.CLIPProcessor.from_pretrained", return_value=TinyProcessor()
        ):
            encoder = FrozenCLIPFeatureEncoder(feature_cache=cache)
        image = encoder.encode_images([object()], cache_keys=["frame-0"])
        text = encoder.encode_texts(["Pick bowl"])
        self.assertEqual((image.shape, text.shape), ((1, 512), (1, 512)))
        self.assertTrue(torch.equal(image, encoder.encode_images([None], cache_keys=["frame-0"])))

    def test_05_b0_window_is_five_without_padding(self):
        model = tiny_model("B0")
        output = model.forward_sequence(*features())
        self.assertEqual(model.last_window_lengths, (1, 2, 3, 4, 5, 5, 5))
        self.assertEqual(output["temporal"].shape, (1, 7, 16))

    def test_06_b0_has_no_persistent_cache(self):
        model = tiny_model("B0")
        output = model.forward_sequence(*features(time=2))
        self.assertIsNone(output["state"])
        self.assertIsNone(model.episode_state)
        with self.assertRaises(RuntimeError):
            model.reset_episode_state(1)

    def test_07_b1_persists_across_calls(self):
        model = tiny_model("B1")
        v, s, language = features(time=3)
        first = model.forward_sequence(v, s, language)
        self.assertEqual(first["state"].steps, 3)
        second = model.forward_sequence(v[:, :2], s[:, :2], language[:, :2], reset_episode=False)
        self.assertEqual(second["state"].steps, 5)
        self.assertEqual(model.episode_reset_count, 1)

    def test_08_b2_current_text_selection(self):
        schedule = build_condition_schedule(EPISODE, "B2")
        self.assertEqual(schedule.texts, ("Pick bowl",) * 3 + ("Place bowl",) * 4)
        self.assertTrue(all(schedule.injection_mask))

    def test_09_b3_transition_command_injection(self):
        schedule = build_condition_schedule(EPISODE, "B3")
        self.assertEqual(schedule.texts[0], "Pick bowl")
        self.assertEqual(schedule.texts[3], "Place bowl")
        self.assertEqual(schedule.injection_mask, (True, False, False, True, False, False, False))

    def test_10_b3_hold_is_not_a_text_embedding(self):
        schedule = build_condition_schedule(EPISODE, "B3")
        self.assertNotIn("[HOLD]", schedule.texts)
        self.assertIsNone(schedule.texts[2])

    def test_11_zero_mask_removes_language_contribution(self):
        encoder = CommonEncoder(d_model=16, state_feature_dim=8)
        v, s, language = features(time=2)
        r, x_a = encoder(v, s, language, torch.zeros(1, 2, dtype=torch.bool))
        _, x_b = encoder(v, s, torch.randn_like(language), torch.zeros(1, 2, dtype=torch.bool))
        self.assertTrue(torch.equal(x_a, x_b))
        self.assertEqual(x_a.shape, r.shape)

    def test_12_b3_transition_does_not_reset(self):
        model = tiny_model("B3")
        v, s, language = features(time=3)
        model.reset_episode_state(1)
        for t, event in enumerate((True, False, True)):
            model.forward_step(v[:, t], s[:, t], language[:, t], torch.tensor([event]))
        self.assertEqual(model.episode_reset_count, 1)
        self.assertEqual(model.episode_state.steps, 3)

    def test_13_episode_reset_exactly_once_in_sequence(self):
        model = tiny_model("B1")
        model.forward_sequence(*features(time=4))
        self.assertEqual(model.episode_reset_count, 1)
        model.discard_episode_state()
        self.assertIsNone(model.episode_state)

    def test_14_primary_and_future_output_shapes(self):
        model = MemoryExperimentModel("B1", n_layer=1)
        output = model.forward_sequence(*features(time=2))
        self.assertEqual(output["instantaneous"].shape, (1, 2, 128))
        self.assertEqual(output["temporal"].shape, (1, 2, 128))
        self.assertEqual(output["future_prediction"].shape, (1, 2, 512))

    def test_15_future_target_has_no_gradient(self):
        prediction = torch.randn(4, 8, requires_grad=True)
        target = torch.randn(4, 8, requires_grad=True)
        loss, _ = future_info_nce(prediction, target)
        loss.backward()
        self.assertIsNotNone(prediction.grad)
        self.assertIsNone(target.grad)

    def test_16_info_nce_is_finite(self):
        loss, stats = future_info_nce(torch.randn(4, 8), torch.randn(4, 8))
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(stats["logits"].shape, (4, 4))

    def test_17_current_candidate_construction(self):
        candidates = CandidateSet.from_episode(EPISODE)
        target = make_probe_target(candidates, frame=4, current_step_index=1, depth=0)
        self.assertEqual(target.target_step_index, 1)

    def test_18_previous_k_construction(self):
        candidates = CandidateSet.from_episode(EPISODE)
        self.assertEqual(make_probe_target(candidates, 4, 1, 1).target_step_index, 0)
        self.assertFalse(make_probe_target(candidates, 1, 0, 1).eligible)

    def test_19_duplicate_multi_positive_loss(self):
        scores = torch.tensor([[2.0, 2.0, -1.0]])
        positives = torch.tensor([[True, True, False]])
        loss = multi_positive_probe_loss(scores, positives)
        self.assertLess(loss.item(), 0.1)

    def test_20_linear_probe_supports_both_taps(self):
        probe = LinearRetrievalProbe(128, 512)
        candidates = torch.randn(3, 512)
        for representation in (torch.randn(2, 128), torch.randn(2, 128)):
            scores = probe.scores(representation, candidates)
            self.assertEqual(scores.shape, (2, 3))
            self.assertTrue(torch.allclose(probe(representation).norm(dim=-1), torch.ones(2)))

    def test_21_common_initialization_reproducible(self):
        torch.manual_seed(123)
        left = tiny_model("B0")
        torch.manual_seed(123)
        right = tiny_model("B3")
        self.assertEqual(left.state_dict().keys(), right.state_dict().keys())
        self.assertTrue(all(torch.equal(left.state_dict()[k], right.state_dict()[k]) for k in left.state_dict()))

    def test_22_split_loader_compatibility(self):
        dataset = RoboCerebraMemoryDataset(ROOT / "analysis/robocerebra_memory_episode_index.json")
        seen = set()
        for split in ("train", "val", "test"):
            payload = json.loads((ROOT / f"splits/robocerebra_memory_{split}.json").read_text())
            ids = payload["trajectory_ids"] if isinstance(payload, dict) else payload
            episode = dataset.load_episode(ids[0])
            self.assertEqual(episode.get_robot_state(0).shape, (9,))
            self.assertFalse(seen.intersection(ids))
            seen.update(ids)
            episode.close()

    def test_23_full_sequence_matches_recurrent_steps(self):
        model = tiny_model("B3")
        model.eval()
        v, s, language = features(time=5)
        mask = torch.tensor([[True, False, False, True, False]])
        full = model.forward_sequence(v, s, language, mask)["temporal"]
        model.reset_episode_state(1)
        stepwise = torch.stack(
            [
                model.forward_step(v[:, t], s[:, t], language[:, t], mask[:, t])["temporal"]
                for t in range(5)
            ],
            dim=1,
        )
        self.assertTrue(torch.allclose(full, stepwise, atol=1e-6, rtol=1e-5))

    def test_24_hold_still_updates_mamba_state(self):
        model = tiny_model("B3")
        v, s, language = features(time=2)
        model.reset_episode_state(1)
        model.forward_step(v[:, 0], s[:, 0], language[:, 0], torch.tensor([True]))
        before = model.episode_state
        model.forward_step(v[:, 1], s[:, 1], language[:, 1], torch.tensor([False]))
        after = model.episode_state
        self.assertEqual((before.steps, after.steps), (1, 2))
        self.assertFalse(torch.equal(before.layers[0].ssm, after.layers[0].ssm))

    def test_25_b3_requires_explicit_event_mask(self):
        with self.assertRaisesRegex(ValueError, "requires"):
            tiny_model("B3").forward_sequence(*features(time=2))

    def test_26_future_target_is_not_a_model_input(self):
        names = MemoryExperimentModel.forward_sequence.__annotations__
        self.assertNotIn("future_target", names)


if __name__ == "__main__":
    unittest.main()
