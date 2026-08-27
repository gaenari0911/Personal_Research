"""One architecture class implementing B0/B1/B2/B3 execution semantics."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Mapping, Sequence

import torch
from torch import Tensor, nn

from .encoders import CommonEncoder, FrozenCLIPFeatureEncoder
from .future_head import FutureRepresentationHead
from .mamba_memory import MambaMemoryBackbone, MambaState


class ModelVariant(str, Enum):
    B0 = "B0"
    B1 = "B1"
    B2 = "B2"
    B3 = "B3"


@dataclass(frozen=True)
class ConditionSchedule:
    """None means semantic no-op and is never passed to the text encoder."""

    texts: tuple[str | None, ...]
    injection_mask: tuple[bool, ...]


def _get(value: object, name: str):
    return value[name] if isinstance(value, Mapping) else getattr(value, name)


def build_condition_schedule(
    episode: object,
    variant: ModelVariant | str,
    frames: Sequence[int] | None = None,
) -> ConditionSchedule:
    variant = ModelVariant(variant)
    num_frames = int(_get(episode, "num_frames"))
    frames = tuple(range(num_frames)) if frames is None else tuple(map(int, frames))
    if any(frame < 0 or frame >= num_frames for frame in frames):
        raise IndexError("conditioning frame outside episode")
    full_instruction = str(_get(episode, "full_instruction"))
    steps = tuple(_get(episode, "steps"))
    by_frame: dict[int, tuple[int, str]] = {}
    for step in steps:
        start, end = int(_get(step, "start")), int(_get(step, "end"))
        text = str(_get(step, "text"))
        for frame in frames:
            if start <= frame < end:
                by_frame[frame] = (start, text)
    if len(by_frame) != len(frames):
        raise ValueError("every requested frame must have exactly one official Step")
    texts: list[str | None] = []
    masks: list[bool] = []
    for frame in frames:
        start, step_text = by_frame[frame]
        if variant in (ModelVariant.B0, ModelVariant.B1):
            texts.append(full_instruction)
            masks.append(True)
        elif variant is ModelVariant.B2:
            texts.append(step_text)
            masks.append(True)
        elif frame == start:
            texts.append(step_text)
            masks.append(True)
        else:
            # Crucially: no literal "[HOLD]" text exists in this schedule.
            texts.append(None)
            masks.append(False)
    return ConditionSchedule(tuple(texts), tuple(masks))


def encode_condition_schedule(
    schedule: ConditionSchedule,
    encoder: FrozenCLIPFeatureEncoder,
    *,
    device: torch.device | None = None,
) -> tuple[Tensor, Tensor]:
    """Encode command events only and materialize HOLD rows as exact zeros."""
    device = encoder.device if device is None else device
    features = torch.zeros(len(schedule.texts), encoder.output_dim, device=device)
    event_indices = [i for i, text in enumerate(schedule.texts) if text is not None]
    if event_indices:
        values = encoder.encode_texts([schedule.texts[i] for i in event_indices])
        features[event_indices] = values.to(device)
    mask = torch.tensor(schedule.injection_mask, dtype=torch.bool, device=device)
    return features, mask


class MemoryExperimentModel(nn.Module):
    """Unified B0--B3 model; only conditioning and state execution differ."""

    def __init__(
        self,
        variant: ModelVariant | str,
        *,
        visual_dim: int = 512,
        state_dim: int = 9,
        state_feature_dim: int = 64,
        language_dim: int = 512,
        d_model: int = 128,
        n_layer: int = 16,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        future_dim: int = 512,
        window_size: int = 5,
    ) -> None:
        super().__init__()
        self.variant = ModelVariant(variant)
        self.window_size = window_size
        self.encoder = CommonEncoder(
            visual_dim, state_dim, state_feature_dim, language_dim, d_model
        )
        self.backbone = MambaMemoryBackbone(
            d_model, n_layer, d_state, d_conv, expand
        )
        self.future_head = FutureRepresentationHead(d_model, future_dim)
        self._episode_state: MambaState | None = None
        self.episode_reset_count = 0
        self.last_window_lengths: tuple[int, ...] = ()

    @property
    def persistent(self) -> bool:
        return self.variant is not ModelVariant.B0

    @property
    def episode_state(self) -> MambaState | None:
        return self._episode_state

    def reset_episode_state(
        self,
        batch_size: int,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> MambaState:
        if not self.persistent:
            raise RuntimeError("B0 has no persistent episode state")
        self._episode_state = self.backbone.initial_state(
            batch_size, device=device, dtype=dtype
        )
        self.episode_reset_count += 1
        return self._episode_state

    def discard_episode_state(self) -> None:
        """Episode-end discard; this is not a transition reset."""
        self._episode_state = None

    def _encode(
        self,
        visual_features: Tensor,
        robot_state: Tensor,
        language_features: Tensor,
        language_injection_mask: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        if self.variant is ModelVariant.B3 and language_injection_mask is None:
            raise ValueError("B3 requires its command-event injection mask")
        return self.encoder(
            visual_features,
            robot_state,
            language_features,
            language_injection_mask,
        )

    def forward_sequence(
        self,
        visual_features: Tensor,
        robot_state: Tensor,
        language_features: Tensor,
        language_injection_mask: Tensor | None = None,
        *,
        reset_episode: bool = True,
    ) -> dict[str, Tensor | MambaState | None]:
        instantaneous, tokens = self._encode(
            visual_features, robot_state, language_features, language_injection_mask
        )
        if tokens.ndim != 3:
            raise ValueError("sequence inputs must have shape [batch, time, feature]")
        if self.variant is ModelVariant.B0:
            temporal_values = []
            lengths = []
            for timestep in range(tokens.shape[1]):
                start = max(0, timestep - self.window_size + 1)
                window_output, _ = self.backbone.forward_sequence(tokens[:, start : timestep + 1])
                temporal_values.append(window_output[:, -1])
                lengths.append(timestep - start + 1)
            temporal = torch.stack(temporal_values, dim=1)
            self.last_window_lengths = tuple(lengths)
            state = None
        else:
            if reset_episode:
                self.reset_episode_state(
                    tokens.shape[0], device=tokens.device, dtype=tokens.dtype
                )
            if self._episode_state is None:
                raise RuntimeError("reset_episode_state must be called before persistent forward")
            temporal, state = self.backbone.forward_sequence(tokens, self._episode_state)
            self._episode_state = state
        return {
            "instantaneous": instantaneous,
            "temporal": temporal,
            "future_prediction": self.future_head(temporal),
            "state": state,
        }

    def forward_step(
        self,
        visual_features: Tensor,
        robot_state: Tensor,
        language_features: Tensor,
        language_injection_mask: Tensor | None = None,
    ) -> dict[str, Tensor | MambaState]:
        if not self.persistent:
            raise RuntimeError("B0 step execution requires an explicit local window")
        if self._episode_state is None:
            raise RuntimeError("reset_episode_state must be called exactly once at episode start")
        instantaneous, token = self._encode(
            visual_features, robot_state, language_features, language_injection_mask
        )
        if token.ndim != 2:
            raise ValueError("step inputs must have shape [batch, feature]")
        temporal, state = self.backbone.forward_step(token, self._episode_state)
        self._episode_state = state
        return {
            "instantaneous": instantaneous,
            "temporal": temporal,
            "future_prediction": self.future_head(temporal),
            "state": state,
        }
