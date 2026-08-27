"""Independent bias-free linear probes for Stage B."""

from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import Tensor, nn

from robocerebra_memory.probes import LinearRetrievalProbe, multi_positive_probe_loss


TARGET_NAMES = ("current", "prev1", "prev2", "prev3")
CONTROL_NAME = "instantaneous_current"


def _seeded_probe(seed: int, input_dim: int, output_dim: int) -> LinearRetrievalProbe:
    # Every target receives the same deterministic initialization but owns distinct storage.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return LinearRetrievalProbe(input_dim, output_dim)


class ProbeBank(nn.Module):
    """Four temporal probes plus an independent instantaneous-current control."""

    def __init__(self, input_dim: int = 128, output_dim: int = 512, seed: int = 42) -> None:
        super().__init__()
        names = TARGET_NAMES + (CONTROL_NAME,)
        self.probes = nn.ModuleDict(
            {name: _seeded_probe(seed, input_dim, output_dim) for name in names}
        )
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.seed = seed

    def temporal_queries(self, z_t: Tensor) -> dict[str, Tensor]:
        """Apply all four independent temporal probes to the exact same current z_t."""
        return {name: self.probes[name](z_t) for name in TARGET_NAMES}

    def instantaneous_query(self, r_t: Tensor) -> Tensor:
        return self.probes[CONTROL_NAME](r_t)

    def assert_independent(self) -> None:
        parameters = [self.probes[name].projection.weight for name in self.probes]
        if len({parameter.data_ptr() for parameter in parameters}) != len(parameters):
            raise AssertionError("Stage B probes share parameter storage")


def freeze_backbone(backbone: nn.Module) -> nn.Module:
    backbone.eval()
    for parameter in backbone.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
    return backbone


def assert_frozen_backbone(backbone: nn.Module) -> None:
    if any(parameter.requires_grad for parameter in backbone.parameters()):
        raise AssertionError("backbone parameter still requires gradients")
    if any(parameter.grad is not None for parameter in backbone.parameters()):
        raise AssertionError("probe backward populated a backbone gradient")


def positive_mask(
    target_indices: Tensor,
    normalized_candidate_texts: Iterable[str],
    *,
    device: torch.device | None = None,
) -> Tensor:
    texts = tuple(normalized_candidate_texts)
    target_indices = target_indices.to(dtype=torch.long)
    mask = torch.zeros(len(target_indices), len(texts), dtype=torch.bool, device=device)
    for row, target_index in enumerate(target_indices.tolist()):
        if target_index < 0:
            continue
        target = texts[target_index]
        mask[row] = torch.tensor([text == target for text in texts], device=device)
    return mask


def probe_loss(
    probe: LinearRetrievalProbe,
    representations: Tensor,
    candidates: Tensor,
    target_indices: Tensor,
    normalized_candidate_texts: Iterable[str],
    temperature: float,
) -> Tensor:
    if temperature <= 0:
        raise ValueError("probe temperature must be positive")
    valid = target_indices >= 0
    if not bool(valid.any()):
        raise ValueError("probe batch has no valid target")
    scores = probe.scores(representations[valid], candidates) / temperature
    mask = positive_mask(
        target_indices[valid], normalized_candidate_texts, device=scores.device
    )
    return multi_positive_probe_loss(scores, mask)
