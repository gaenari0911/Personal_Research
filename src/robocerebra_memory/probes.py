"""Trajectory-local retrieval targets for R2 memory probes."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Mapping, Sequence

try:  # Keep the R1/R2 metadata helpers usable in the base NumPy environment.
    import torch
    from torch import Tensor, nn
    from torch.nn import functional as F
except ImportError:  # pragma: no cover - exercised only by the torch-free env
    torch = None
    Tensor = object
    nn = None
    F = None


_WHITESPACE = re.compile(r"\s+")


def normalize_step_text(text: str) -> str:
    """Normalize only casing/whitespace when defining equivalent text targets."""
    return _WHITESPACE.sub(" ", text.strip()).casefold()


@dataclass(frozen=True)
class CandidateSet:
    trajectory_id: str
    texts: tuple[str, ...]
    normalized_texts: tuple[str, ...]

    @classmethod
    def from_episode(cls, episode: Mapping[str, object]) -> "CandidateSet":
        texts = tuple(str(step["text"]) for step in episode["steps"])
        return cls(
            trajectory_id=str(episode["trajectory_id"]),
            texts=texts,
            normalized_texts=tuple(normalize_step_text(text) for text in texts),
        )

    def positive_indices(self, target_index: int) -> tuple[int, ...]:
        """Accept identical normalized Step texts as an equivalence class."""
        if target_index < 0 or target_index >= len(self.texts):
            return ()
        target = self.normalized_texts[target_index]
        return tuple(
            index for index, value in enumerate(self.normalized_texts) if value == target
        )

    @property
    def unique_text_count(self) -> int:
        return len(set(self.normalized_texts))


@dataclass(frozen=True)
class ProbeTarget:
    trajectory_id: str
    frame: int
    depth: int
    target_step_index: int
    positive_candidate_indices: tuple[int, ...]

    @property
    def eligible(self) -> bool:
        return self.target_step_index >= 0 and bool(self.positive_candidate_indices)


def make_probe_target(
    candidate_set: CandidateSet,
    frame: int,
    current_step_index: int,
    depth: int = 0,
) -> ProbeTarget:
    if depth < 0:
        raise ValueError("depth must be non-negative")
    target_index = current_step_index - depth
    positive = candidate_set.positive_indices(target_index)
    return ProbeTarget(
        trajectory_id=candidate_set.trajectory_id,
        frame=int(frame),
        depth=depth,
        target_step_index=target_index if target_index >= 0 else -1,
        positive_candidate_indices=positive,
    )


def duplicate_text_groups(texts: Sequence[str]) -> tuple[tuple[int, ...], ...]:
    groups: dict[str, list[int]] = {}
    for index, text in enumerate(texts):
        groups.setdefault(normalize_step_text(text), []).append(index)
    return tuple(tuple(values) for values in groups.values() if len(values) > 1)


if nn is not None:

    class LinearRetrievalProbe(nn.Module):
        """Bias-free 128-to-512 probe shared by temporal and instantaneous taps."""

        def __init__(self, input_dim: int = 128, output_dim: int = 512) -> None:
            super().__init__()
            self.projection = nn.Linear(input_dim, output_dim, bias=False)

        def forward(self, representation: Tensor) -> Tensor:
            return F.normalize(self.projection(representation), dim=-1)

        def scores(self, representation: Tensor, candidates: Tensor) -> Tensor:
            query = self(representation)
            candidates = F.normalize(candidates.detach(), dim=-1)
            if candidates.ndim == 2:
                return query @ candidates.transpose(0, 1)
            if candidates.ndim == 3:
                return torch.einsum("bd,bkd->bk", query, candidates)
            raise ValueError("candidates must have shape [K,D] or [B,K,D]")


    def multi_positive_probe_loss(scores: Tensor, positive_mask: Tensor) -> Tensor:
        """Trajectory-local softmax loss accepting duplicate-text positives."""
        if scores.ndim != 2 or positive_mask.shape != scores.shape:
            raise ValueError("scores and positive_mask must have shape [batch,candidates]")
        positive_mask = positive_mask.bool()
        if not torch.all(positive_mask.any(dim=-1)):
            raise ValueError("every row requires at least one positive")
        positive_scores = scores.masked_fill(~positive_mask, float("-inf"))
        return (torch.logsumexp(scores, dim=-1) - torch.logsumexp(positive_scores, dim=-1)).mean()

else:

    class LinearRetrievalProbe:  # pragma: no cover - diagnostic fallback
        def __init__(self, *_args, **_kwargs) -> None:
            raise ImportError("LinearRetrievalProbe requires PyTorch")

    def multi_positive_probe_loss(*_args, **_kwargs):  # pragma: no cover
        raise ImportError("multi_positive_probe_loss requires PyTorch")
