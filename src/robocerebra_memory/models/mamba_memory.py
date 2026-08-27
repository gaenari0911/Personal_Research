"""Causal Mamba-1 memory backbone with an explicit recurrent-state contract.

The implementation is a readable PyTorch selective-scan reference path.  It
uses the MaIL Mamba-1 dimensions and equations, while avoiding the optional
CUDA fused kernels so R3 consistency tests can run on CPU.  R4 can swap the
kernel implementation without changing the state/model interface.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class MambaLayerState:
    conv: Tensor
    ssm: Tensor


@dataclass(frozen=True)
class MambaState:
    """Per-layer causal convolution and selective-SSM state."""

    layers: tuple[MambaLayerState, ...]
    steps: int = 0

    def detach(self) -> "MambaState":
        """Explicit opt-in only; no model path calls this automatically in R3."""
        return MambaState(
            tuple(MambaLayerState(x.conv.detach(), x.ssm.detach()) for x in self.layers),
            self.steps,
        )


class ReferenceMambaBlock(nn.Module):
    """Mamba-1 block expressed as a timestep recurrence for CPU verification."""

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dt_rank: int | None = None,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.inner_dim = expand * d_model
        self.dt_rank = dt_rank or math.ceil(d_model / 16)
        self.norm = nn.LayerNorm(d_model)
        self.in_proj = nn.Linear(d_model, 2 * self.inner_dim, bias=False)
        self.conv_weight = nn.Parameter(torch.empty(self.inner_dim, d_conv))
        self.conv_bias = nn.Parameter(torch.zeros(self.inner_dim))
        self.x_proj = nn.Linear(
            self.inner_dim, self.dt_rank + 2 * d_state, bias=False
        )
        self.dt_proj = nn.Linear(self.dt_rank, self.inner_dim, bias=True)
        self.A_log = nn.Parameter(
            torch.log(torch.arange(1, d_state + 1, dtype=torch.float32))
            .unsqueeze(0)
            .repeat(self.inner_dim, 1)
        )
        self.D = nn.Parameter(torch.ones(self.inner_dim))
        self.out_proj = nn.Linear(self.inner_dim, d_model, bias=False)
        nn.init.kaiming_uniform_(self.conv_weight, a=math.sqrt(5))
        # MaIL config fixes dt into this range. Bias initialization makes the
        # untrained reference numerically stable while preserving selectivity.
        dt = torch.exp(
            torch.empty(self.inner_dim).uniform_(math.log(0.001), math.log(0.1))
        ).clamp(min=1e-4)
        with torch.no_grad():
            self.dt_proj.bias.copy_(dt + torch.log(-torch.expm1(-dt)))

    def initial_state(
        self, batch_size: int, *, device: torch.device, dtype: torch.dtype
    ) -> MambaLayerState:
        return MambaLayerState(
            conv=torch.zeros(
                batch_size, self.inner_dim, self.d_conv - 1, device=device, dtype=dtype
            ),
            ssm=torch.zeros(
                batch_size, self.inner_dim, self.d_state, device=device, dtype=dtype
            ),
        )

    def forward_step(
        self, hidden: Tensor, state: MambaLayerState
    ) -> tuple[Tensor, MambaLayerState]:
        residual = hidden
        projected, gate = self.in_proj(self.norm(hidden)).chunk(2, dim=-1)
        conv_sequence = torch.cat((state.conv, projected.unsqueeze(-1)), dim=-1)
        convolved = (conv_sequence * self.conv_weight.unsqueeze(0)).sum(dim=-1)
        convolved = F.silu(convolved + self.conv_bias)
        new_conv = conv_sequence[..., 1:]

        delta_raw, B, C = torch.split(
            self.x_proj(convolved), [self.dt_rank, self.d_state, self.d_state], dim=-1
        )
        delta = F.softplus(self.dt_proj(delta_raw))
        A = -torch.exp(self.A_log.float()).to(convolved.dtype)
        discrete_A = torch.exp(delta.unsqueeze(-1) * A.unsqueeze(0))
        discrete_Bx = (
            delta.unsqueeze(-1)
            * B.unsqueeze(1)
            * convolved.unsqueeze(-1)
        )
        new_ssm = discrete_A * state.ssm + discrete_Bx
        scanned = (new_ssm * C.unsqueeze(1)).sum(dim=-1) + self.D * convolved
        mixed = self.out_proj(scanned * F.silu(gate))
        return residual + mixed, MambaLayerState(new_conv, new_ssm)

    def forward_sequence(
        self, hidden: Tensor, state: MambaLayerState
    ) -> tuple[Tensor, MambaLayerState]:
        """Vectorize projections/convolution while retaining the exact causal scan."""
        residual = hidden
        projected, gate = self.in_proj(self.norm(hidden)).chunk(2, dim=-1)
        prefix = state.conv.transpose(1, 2)
        convolution_input = torch.cat((prefix, projected), dim=1).transpose(1, 2)
        convolved = F.conv1d(
            convolution_input,
            self.conv_weight.unsqueeze(1),
            self.conv_bias,
            groups=self.inner_dim,
        ).transpose(1, 2)
        convolved = F.silu(convolved)
        new_conv = torch.cat((state.conv, projected.transpose(1, 2)), dim=-1)[
            ..., -(self.d_conv - 1) :
        ]
        delta_raw, B, C = torch.split(
            self.x_proj(convolved), [self.dt_rank, self.d_state, self.d_state], dim=-1
        )
        delta = F.softplus(self.dt_proj(delta_raw))
        A = -torch.exp(self.A_log.float()).to(convolved.dtype)
        ssm = state.ssm
        scanned = []
        for timestep in range(hidden.shape[1]):
            dt = delta[:, timestep]
            input_t = convolved[:, timestep]
            discrete_A = torch.exp(dt.unsqueeze(-1) * A.unsqueeze(0))
            ssm = (
                discrete_A * ssm
                + dt.unsqueeze(-1)
                * B[:, timestep].unsqueeze(1)
                * input_t.unsqueeze(-1)
            )
            scanned.append(
                (ssm * C[:, timestep].unsqueeze(1)).sum(dim=-1)
                + self.D * input_t
            )
        scanned_tensor = torch.stack(scanned, dim=1)
        mixed = self.out_proj(scanned_tensor * F.silu(gate))
        return residual + mixed, MambaLayerState(new_conv, ssm)


class MambaMemoryBackbone(nn.Module):
    """MaIL-aligned 16-layer Mamba-1 temporal encoder.

    The primary representation is the final LayerNorm output after processing
    the current token. Full sequence and recurrent APIs share one exact step
    implementation and therefore have no hidden reset or detach boundary.
    """

    backend_name = "torch_reference_mamba1_selective_scan"

    def __init__(
        self,
        d_model: int = 128,
        n_layer: int = 16,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
    ) -> None:
        super().__init__()
        if n_layer < 1 or d_conv < 1:
            raise ValueError("n_layer and d_conv must be positive")
        self.d_model = d_model
        self.n_layer = n_layer
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.gradient_checkpointing = False
        self.layers = nn.ModuleList(
            [
                ReferenceMambaBlock(d_model, d_state, d_conv, expand)
                for _ in range(n_layer)
            ]
        )
        self.norm_f = nn.LayerNorm(d_model)

    def initial_state(
        self,
        batch_size: int,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> MambaState:
        parameter = next(self.parameters())
        device = parameter.device if device is None else device
        dtype = parameter.dtype if dtype is None else dtype
        return MambaState(
            tuple(
                layer.initial_state(batch_size, device=device, dtype=dtype)
                for layer in self.layers
            )
        )

    def forward_step(self, token: Tensor, state: MambaState) -> tuple[Tensor, MambaState]:
        if token.ndim != 2 or token.shape[-1] != self.d_model:
            raise ValueError(f"token must have shape [batch, {self.d_model}]")
        if len(state.layers) != self.n_layer:
            raise ValueError("state layer count does not match backbone")
        hidden = token
        new_layers = []
        for layer, layer_state in zip(self.layers, state.layers):
            hidden, new_state = layer.forward_step(hidden, layer_state)
            new_layers.append(new_state)
        return self.norm_f(hidden), MambaState(tuple(new_layers), state.steps + 1)

    def forward_sequence(
        self, tokens: Tensor, state: MambaState | None = None
    ) -> tuple[Tensor, MambaState]:
        if tokens.ndim != 3 or tokens.shape[-1] != self.d_model:
            raise ValueError(f"tokens must have shape [batch, time, {self.d_model}]")
        if tokens.shape[1] < 1:
            raise ValueError("sequence must contain at least one timestep")
        if state is None:
            state = self.initial_state(
                tokens.shape[0], device=tokens.device, dtype=tokens.dtype
            )
        hidden = tokens
        new_layers = []
        for layer, layer_state in zip(self.layers, state.layers):
            if self.gradient_checkpointing and self.training:
                from torch.utils.checkpoint import checkpoint

                def run_layer(value, conv, ssm, module=layer):
                    output, updated = module.forward_sequence(
                        value, MambaLayerState(conv, ssm)
                    )
                    return output, updated.conv, updated.ssm

                hidden, new_conv, new_ssm = checkpoint(
                    run_layer,
                    hidden,
                    layer_state.conv,
                    layer_state.ssm,
                    use_reentrant=False,
                )
                new_state = MambaLayerState(new_conv, new_ssm)
            else:
                hidden, new_state = layer.forward_sequence(hidden, layer_state)
            new_layers.append(new_state)
        return self.norm_f(hidden), MambaState(
            tuple(new_layers), state.steps + tokens.shape[1]
        )
