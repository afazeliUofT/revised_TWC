from __future__ import annotations

"""Interpretable multi-scale positive-semidefinite channel posterior.

Every feature block is a fixed random Fourier basis. Trainable nonnegative
feature weights and optional context-conditioned nonnegative scale gains define
an explicitly PSD prior K = B B^H. Conditioning on NR DMRS is exact under the
resulting Gaussian model.
"""

from dataclasses import dataclass
from typing import Iterable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .channels import rff_bank
from .models import PosteriorOutput, project_latent_covariance_to_grid


MULTISCALE_POSTERIOR_VERSION = "context_psd_multiscale_v1"


@dataclass(frozen=True)
class RFFScale:
    rank: int
    length_f: float
    length_t: float

    def validate(self) -> None:
        if int(self.rank) <= 0:
            raise ValueError("RFF scale rank must be positive")
        if float(self.length_f) <= 0.0 or float(self.length_t) <= 0.0:
            raise ValueError("RFF scale lengths must be positive")


class MultiScalePosteriorOperator(nn.Module):
    """Exact Gaussian posterior with a multi-scale PSD feature prior.

    Parameters are deliberately small and interpretable:

    * one nonnegative spectral weight per random feature;
    * one effective pilot-noise scale;
    * optionally, a linear context-to-scale map whose softplus outputs are
      nonnegative scale gains.

    Grid-dependent coordinates and pilot indices are buffers. Multiple NR grids
    can therefore share the same trainable parameter objects while keeping their
    own deterministic feature matrices.
    """

    def __init__(
        self,
        coords: torch.Tensor,
        pilot_idx: torch.Tensor,
        n_layers: int,
        scales: Sequence[RFFScale],
        *,
        seed: int,
        context: torch.Tensor | None = None,
        context_conditioned: bool = False,
    ) -> None:
        super().__init__()
        self.n_layers = int(n_layers)
        self.operator_seed = int(seed)
        self.scales = tuple(scales)
        if not self.scales:
            raise ValueError("At least one RFF scale is required")
        for item in self.scales:
            item.validate()
        self.rank = int(sum(int(item.rank) for item in self.scales))
        self.context_conditioned = bool(context_conditioned)

        self.register_buffer("coords", coords.detach().clone())
        self.register_buffer("pilot_idx", pilot_idx.detach().clone().long())

        blocks: list[torch.Tensor] = []
        scale_index: list[int] = []
        for index, item in enumerate(self.scales):
            block = rff_bank(
                coords.detach(),
                int(item.rank),
                float(item.length_f),
                float(item.length_t),
                seed=int(seed) + 1009 * index,
                bank_rank=int(item.rank),
            )
            blocks.append(block)
            scale_index.extend([index] * int(item.rank))
        self.register_buffer("base_features", torch.cat(blocks, dim=-1))
        self.register_buffer(
            "feature_scale_index",
            torch.tensor(scale_index, dtype=torch.long, device=coords.device),
        )

        if context is None:
            context = torch.zeros(1, dtype=torch.float32, device=coords.device)
        context = context.detach().clone().to(coords.device, torch.float32).flatten()
        self.register_buffer("context", context)

        self.raw_feature_weights = nn.Parameter(torch.zeros(self.rank))
        self.log_noise_scale = nn.Parameter(torch.tensor(0.0))

        if self.context_conditioned:
            num_scales = len(self.scales)
            self.raw_scale_bias = nn.Parameter(torch.zeros(num_scales))
            self.context_to_scale = nn.Parameter(
                torch.zeros(num_scales, int(context.numel()))
            )
            # Small deterministic initialization makes context active without
            # initially overwhelming the common feature spectrum.
            with torch.no_grad():
                for i in range(num_scales):
                    self.context_to_scale[i, i % int(context.numel())] = (
                        0.02 * (i + 1)
                    )
        else:
            self.register_parameter("raw_scale_bias", None)
            self.register_parameter("context_to_scale", None)

    @property
    def num_scales(self) -> int:
        return len(self.scales)

    def scale_gains(self) -> torch.Tensor:
        if not self.context_conditioned:
            gains = torch.ones(
                self.num_scales,
                dtype=torch.float32,
                device=self.base_features.device,
            )
        else:
            logits = self.raw_scale_bias + self.context_to_scale @ self.context
            gains = F.softplus(logits) + 1e-4
        # Each scale block has unit feature energy. L2 normalization therefore
        # keeps the prior's marginal power comparable across scale counts.
        return gains / torch.sqrt(torch.sum(gains.square()).clamp_min(1e-8))

    def weighted_features(self) -> torch.Tensor:
        feature_weights = F.softplus(self.raw_feature_weights) + 1e-4
        # Normalize within each scale so scale gains have a direct interpretation.
        normalized = torch.empty_like(feature_weights)
        for index in range(self.num_scales):
            selector = self.feature_scale_index == index
            block = feature_weights[selector]
            normalized[selector] = block / torch.sqrt(
                torch.mean(block.square()).clamp_min(1e-8)
            )
        gains = self.scale_gains()[self.feature_scale_index]
        weights = normalized * gains
        return self.base_features * weights.to(self.base_features.dtype).view(1, -1)

    def _observation_basis(self, phi: torch.Tensor) -> torch.Tensor:
        u_p = self.weighted_features()[self.pilot_idx, :]
        p = int(u_p.shape[0])
        blocks = [phi[n, :].view(p, 1) * u_p for n in range(self.n_layers)]
        return torch.cat(blocks, dim=1)

    def forward(
        self,
        y_p: torch.Tensor,
        phi: torch.Tensor,
        noise_var: torch.Tensor,
    ) -> PosteriorOutput:
        batch_size, n_rx, n_pilots = y_p.shape
        device = y_p.device
        observation_basis = self._observation_basis(phi).to(device)
        n_latent = int(observation_basis.shape[1])
        sigma2 = (
            noise_var * torch.exp(self.log_noise_scale).clamp(0.125, 8.0)
            + 1e-6
        )
        eye_p = torch.eye(n_pilots, dtype=torch.complex64, device=device)
        observation_cov = (
            observation_basis @ observation_basis.conj().transpose(0, 1)
            + sigma2.to(torch.complex64) * eye_p
        )
        chol = torch.linalg.cholesky(observation_cov + 1e-5 * eye_p)

        y_flat = y_p.reshape(batch_size * n_rx, n_pilots).T.contiguous()
        alpha = torch.cholesky_solve(y_flat, chol)
        latent_mean = observation_basis.conj().transpose(0, 1) @ alpha
        latent_mean = latent_mean.T.reshape(
            batch_size, n_rx, self.n_layers, self.rank
        )
        features = self.weighted_features().to(device)
        channel_mean = torch.einsum(
            "rq,bxnq->bnxr", features, latent_mean
        ).contiguous()

        cov_solve = torch.cholesky_solve(observation_basis, chol)
        latent_cov = (
            torch.eye(n_latent, dtype=torch.complex64, device=device)
            - observation_basis.conj().transpose(0, 1) @ cov_solve
        )
        latent_cov = 0.5 * (
            latent_cov + latent_cov.conj().transpose(0, 1)
        )
        local_cov, var_diag = project_latent_covariance_to_grid(
            features,
            latent_cov,
            self.n_layers,
            self.rank,
        )
        return PosteriorOutput(
            mean=channel_mean,
            var_diag=var_diag,
            local_cov=local_cov,
            latent_cov=latent_cov,
            effective_noise=sigma2,
        )

    def parameter_report(self) -> dict[str, object]:
        return {
            "version": MULTISCALE_POSTERIOR_VERSION,
            "rank": self.rank,
            "num_scales": self.num_scales,
            "context_conditioned": self.context_conditioned,
            "context_dim": int(self.context.numel()),
            "scale_gains": [float(x) for x in self.scale_gains().detach().cpu()],
            "trainable_parameters": int(
                sum(p.numel() for p in self.parameters() if p.requires_grad)
            ),
        }


def bind_shared_multiscale_parameters(
    operators: Sequence[MultiScalePosteriorOperator],
) -> None:
    if not operators:
        raise ValueError("At least one operator is required")
    master = operators[0]
    for operator in operators[1:]:
        if operator.rank != master.rank:
            raise ValueError("Shared multi-scale operators must have equal rank")
        if operator.num_scales != master.num_scales:
            raise ValueError("Shared multi-scale operators must have equal scale count")
        if operator.context_conditioned != master.context_conditioned:
            raise ValueError("Shared operators disagree on context conditioning")
        operator.raw_feature_weights = master.raw_feature_weights
        operator.log_noise_scale = master.log_noise_scale
        if master.context_conditioned:
            operator.raw_scale_bias = master.raw_scale_bias
            operator.context_to_scale = master.context_to_scale


def unique_parameter_count(modules: Iterable[nn.Module]) -> int:
    seen: set[int] = set()
    count = 0
    for module in modules:
        for parameter in module.parameters():
            if parameter.requires_grad and id(parameter) not in seen:
                seen.add(id(parameter))
                count += int(parameter.numel())
    return count
