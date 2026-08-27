"""Frozen CLIP feature interface and the shared observation/language encoder."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any, Optional, Protocol

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class FeatureCache(Protocol):
    """Small cache contract; R4 may replace this with an on-disk implementation."""

    def get(self, namespace: str, key: str) -> Optional[Tensor]: ...

    def put(self, namespace: str, key: str, value: Tensor) -> None: ...


class InMemoryFeatureCache:
    """Process-local cache used by smoke tests and interactive inspection."""

    def __init__(self) -> None:
        self._values: dict[tuple[str, str], Tensor] = {}

    def get(self, namespace: str, key: str) -> Optional[Tensor]:
        value = self._values.get((namespace, key))
        return None if value is None else value.clone()

    def put(self, namespace: str, key: str, value: Tensor) -> None:
        self._values[(namespace, key)] = value.detach().cpu().clone()


class FrozenCLIPFeatureEncoder(nn.Module):
    """Hugging Face interface for frozen OpenAI CLIP ViT-B/32 features.

    Heavy dependencies are imported only when this class is instantiated, so the
    existing NumPy-only R1/R2 protocol remains importable in its original env.
    """

    output_dim = 512

    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        *,
        cache_dir: str | Path | None = None,
        local_files_only: bool = True,
        feature_cache: FeatureCache | None = None,
        device: str | torch.device = "cpu",
    ) -> None:
        super().__init__()
        try:
            from transformers import CLIPModel, CLIPProcessor
        except ImportError as error:  # pragma: no cover - environment diagnostic
            raise ImportError("FrozenCLIPFeatureEncoder requires transformers") from error
        kwargs = {
            "cache_dir": None if cache_dir is None else str(cache_dir),
            "local_files_only": local_files_only,
        }
        self.model_name = model_name
        self.processor = CLIPProcessor.from_pretrained(model_name, **kwargs)
        self.clip = CLIPModel.from_pretrained(model_name, **kwargs).to(device)
        self.clip.requires_grad_(False)
        self.clip.eval()
        self.feature_cache = feature_cache

    def train(self, mode: bool = True) -> "FrozenCLIPFeatureEncoder":
        # The feature encoder is a fixed measurement instrument, even when the
        # surrounding experiment model is put into train mode.
        super().train(False)
        self.clip.eval()
        return self

    @property
    def device(self) -> torch.device:
        return next(self.clip.parameters()).device

    @torch.no_grad()
    def encode_images(
        self, images: Iterable[Any], *, cache_keys: Iterable[str] | None = None
    ) -> Tensor:
        images = list(images)
        keys = None if cache_keys is None else list(cache_keys)
        if keys is not None and len(keys) != len(images):
            raise ValueError("one cache key is required per image")
        cached: list[Tensor | None] = [None] * len(images)
        missing = list(range(len(images)))
        if keys is not None and self.feature_cache is not None:
            cached = [self.feature_cache.get("image", key) for key in keys]
            missing = [index for index, value in enumerate(cached) if value is None]
        if missing:
            batch = self.processor(
                images=[images[index] for index in missing], return_tensors="pt"
            )
            pixels = batch["pixel_values"].to(self.device)
            values = F.normalize(self.clip.get_image_features(pixel_values=pixels), dim=-1)
            for index, value in zip(missing, values):
                cached[index] = value.detach().cpu()
                if keys is not None and self.feature_cache is not None:
                    self.feature_cache.put("image", keys[index], value)
        return torch.stack([value for value in cached if value is not None]).to(self.device)

    @torch.no_grad()
    def encode_texts(
        self, texts: Iterable[str], *, cache_keys: Iterable[str] | None = None
    ) -> Tensor:
        texts = list(texts)
        keys = list(texts) if cache_keys is None else list(cache_keys)
        if len(keys) != len(texts):
            raise ValueError("one cache key is required per text")
        cached = (
            [self.feature_cache.get("text", key) for key in keys]
            if self.feature_cache is not None
            else [None] * len(texts)
        )
        missing = [index for index, value in enumerate(cached) if value is None]
        if missing:
            batch = self.processor(
                text=[texts[index] for index in missing],
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
            inputs = {
                name: value.to(self.device)
                for name, value in batch.items()
                if name in {"input_ids", "attention_mask"}
            }
            values = F.normalize(self.clip.get_text_features(**inputs), dim=-1)
            for index, value in zip(missing, values):
                cached[index] = value.detach().cpu()
                if self.feature_cache is not None:
                    self.feature_cache.put("text", keys[index], value)
        return torch.stack([value for value in cached if value is not None]).to(self.device)


class CommonEncoder(nn.Module):
    """Shared B0--B3 encoder with an explicit pre-language representation tap."""

    def __init__(
        self,
        visual_dim: int = 512,
        state_dim: int = 9,
        state_feature_dim: int = 64,
        language_dim: int = 512,
        d_model: int = 128,
    ) -> None:
        super().__init__()
        self.visual_dim = visual_dim
        self.state_dim = state_dim
        self.language_dim = language_dim
        self.d_model = d_model
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, state_feature_dim),
            nn.GELU(),
        )
        self.observation_projection = nn.Linear(
            visual_dim + state_feature_dim, d_model
        )
        self.observation_norm = nn.LayerNorm(d_model)
        self.language_projection = nn.Linear(language_dim, d_model)
        self.token_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        visual_features: Tensor,
        robot_state: Tensor,
        language_features: Tensor,
        language_injection_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        if visual_features.shape[:-1] != robot_state.shape[:-1]:
            raise ValueError("visual and robot state leading shapes must match")
        if visual_features.shape[:-1] != language_features.shape[:-1]:
            raise ValueError("visual and language leading shapes must match")
        if visual_features.shape[-1] != self.visual_dim:
            raise ValueError(f"visual feature width must be {self.visual_dim}")
        # Exact width is an executable safeguard against feeding full privileged
        # MuJoCo state instead of Panda qpos[7] + gripper qpos[2].
        if robot_state.shape[-1] != self.state_dim:
            raise ValueError(f"robot state width must be exactly {self.state_dim}")
        if language_features.shape[-1] != self.language_dim:
            raise ValueError(f"language feature width must be {self.language_dim}")
        state_features = self.state_encoder(robot_state)
        instantaneous = self.observation_norm(
            self.observation_projection(torch.cat((visual_features, state_features), dim=-1))
        )
        language = self.language_projection(language_features)
        if language_injection_mask is not None:
            if language_injection_mask.shape != visual_features.shape[:-1]:
                raise ValueError("language mask must match batch/time dimensions")
            language = language * language_injection_mask.to(language.dtype).unsqueeze(-1)
        temporal_token = self.token_norm(instantaneous + language)
        return instantaneous, temporal_token
