from __future__ import annotations

"""LS-anchored localized Bayesian residual channel posterior.

The model is intentionally small and receiver-theoretic. It never receives the
true channel at inference. It uses

* Sionna's observable LS estimate and error variance,
* the received NR DMRS samples,
* the exact DMRS matrix,
* a fixed localized delay--Doppler basis selected by the preceding oracle gate,
* a smooth learned prior over ordered localized modes, and
* exact complex Gaussian conditioning of residual coefficients.

The final channel estimate is the LS estimate plus a damped posterior residual.
The LS path is therefore a safe nested special case of the model.
"""

from dataclasses import dataclass
import math
from typing import Any, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .models import PosteriorOutput, project_latent_covariance_to_grid


IMPLEMENTABLE_LOCALIZED_VERSION = "ls_anchored_localized_residual_v1"
SHARED_PARAMETER_NAMES = (
    "raw_variance_knots",
    "log_residual_scale",
    "raw_gain_real_knots",
    "raw_gain_imag_knots",
    "gain_context",
    "variance_context",
    "log_pilot_noise_scale",
    "residual_gate_bias",
    "residual_gate_context",
    "residual_gate_noise_slope",
    "raw_ls_variance_weight",
    "raw_disagreement_weight",
    "log_output_variance_scale",
)


def _inverse_softplus(value: float) -> float:
    value = max(float(value), 1e-8)
    return math.log(math.expm1(value))


def _positive_curve(knots: torch.Tensor, length: int) -> torch.Tensor:
    if int(length) <= 0:
        raise ValueError("curve length must be positive")
    source = F.softplus(knots).view(1, 1, -1) + 1e-5
    if int(length) == int(knots.numel()):
        values = source.reshape(-1)
    else:
        values = F.interpolate(
            source,
            size=int(length),
            mode="linear",
            align_corners=True,
        ).reshape(-1)
    return values / values.mean().clamp_min(1e-8)


def _signed_curve(knots: torch.Tensor, length: int) -> torch.Tensor:
    source = knots.view(1, 1, -1)
    if int(length) == int(knots.numel()):
        return source.reshape(-1)
    return F.interpolate(
        source,
        size=int(length),
        mode="linear",
        align_corners=True,
    ).reshape(-1)


def _diagonal_covariance(var_diag: torch.Tensor) -> torch.Tensor:
    """Convert [N,R] real variances to [N,N,R] complex covariance."""
    if var_diag.ndim != 2:
        raise ValueError(f"Expected var_diag[N,R], got {tuple(var_diag.shape)}")
    return torch.diag_embed(var_diag.transpose(0, 1).to(torch.complex64)).permute(
        1, 2, 0
    ).contiguous()


def _scalar_noise(noise_var: torch.Tensor, device: torch.device) -> torch.Tensor:
    value = torch.as_tensor(noise_var, device=device).real.to(torch.float32)
    if value.numel() == 0:
        raise ValueError("noise variance is empty")
    return value.mean().clamp_min(1e-8)


@dataclass
class LSAnchoredLocalizedResult:
    posterior: PosteriorOutput
    residual_posterior: PosteriorOutput
    residual_gate: torch.Tensor
    correction_power: torch.Tensor
    prior_variance: torch.Tensor
    localized_correction: torch.Tensor
    ls_mean: torch.Tensor
    ls_var_diag: torch.Tensor
    diagnostics: dict[str, Any]


class LSAnchoredLocalizedResidualPosterior(nn.Module):
    """Exact Gaussian localized residual posterior anchored to an LS estimate.

    The trainable quantities are smooth mode curves and a few scalar calibration
    parameters. Grid-specific bases and pilot indices are deterministic buffers.
    Parameters can be shared across 4-, 8-, and 12-PRB instances.
    """

    def __init__(
        self,
        *,
        features: torch.Tensor,
        pilot_idx: torch.Tensor,
        n_layers: int,
        nominal_rank: int,
        context: torch.Tensor,
        num_knots: int = 8,
    ) -> None:
        super().__init__()
        if features.ndim != 2 or not torch.is_complex(features):
            raise ValueError("features must be a complex matrix [R,Q]")
        if int(features.shape[1]) <= 0:
            raise ValueError("localized basis must have positive effective rank")
        if int(nominal_rank) < int(features.shape[1]):
            raise ValueError("nominal rank cannot be smaller than effective rank")
        if int(num_knots) < 3:
            raise ValueError("at least three spectral knots are required")
        self.n_layers = int(n_layers)
        self.nominal_rank = int(nominal_rank)
        self.effective_rank = int(features.shape[1])
        self.num_knots = int(num_knots)
        self.register_buffer("features", features.detach().clone().to(torch.complex64))
        self.register_buffer("pilot_idx", pilot_idx.detach().clone().long())
        self.register_buffer("context", context.detach().clone().float().flatten())

        # The residual prior starts small so the model is initially close to LS.
        self.raw_variance_knots = nn.Parameter(
            torch.full((self.num_knots,), _inverse_softplus(0.05))
        )
        self.log_residual_scale = nn.Parameter(torch.tensor(math.log(0.05)))

        # Smooth complex correction of localized coefficients predicted from LS.
        self.raw_gain_real_knots = nn.Parameter(torch.zeros(self.num_knots))
        self.raw_gain_imag_knots = nn.Parameter(torch.zeros(self.num_knots))
        self.gain_context = nn.Parameter(torch.zeros(int(self.context.numel())))
        self.variance_context = nn.Parameter(torch.zeros(int(self.context.numel())))

        # Exact DMRS residual conditioning uses a learned effective-noise scale.
        self.log_pilot_noise_scale = nn.Parameter(torch.tensor(0.0))

        # A conservative residual gate makes ordinary LS a nested special case.
        self.residual_gate_bias = nn.Parameter(torch.tensor(-1.4))
        self.residual_gate_context = nn.Parameter(
            torch.zeros(int(self.context.numel()))
        )
        self.residual_gate_noise_slope = nn.Parameter(torch.tensor(0.0))

        # Calibrated output uncertainty.
        self.raw_ls_variance_weight = nn.Parameter(
            torch.tensor(_inverse_softplus(1.0))
        )
        self.raw_disagreement_weight = nn.Parameter(
            torch.tensor(_inverse_softplus(0.05))
        )
        self.log_output_variance_scale = nn.Parameter(torch.tensor(0.0))

    def mode_parameters(self) -> dict[str, torch.Tensor]:
        q = self.effective_rank
        variance_profile = _positive_curve(self.raw_variance_knots, q)
        context_variance = torch.exp(
            torch.clamp(torch.dot(self.variance_context, self.context), -2.0, 2.0)
        )
        prior_variance = (
            torch.exp(self.log_residual_scale).clamp(1e-3, 1e3)
            * context_variance
            * variance_profile
        ).clamp_min(1e-7)

        real = torch.tanh(_signed_curve(self.raw_gain_real_knots, q))
        imag = torch.tanh(_signed_curve(self.raw_gain_imag_knots, q))
        context_gain = torch.exp(
            torch.clamp(torch.dot(self.gain_context, self.context), -2.0, 2.0)
        )
        delta_gain = context_gain * torch.complex(real, imag)
        return {
            "prior_variance": prior_variance,
            "delta_gain": delta_gain,
        }

    def _observation_basis(self, phi: torch.Tensor) -> torch.Tensor:
        phi = torch.as_tensor(phi, device=self.features.device).to(torch.complex64)
        if phi.ndim != 2 or int(phi.shape[0]) != self.n_layers:
            raise ValueError(
                f"phi must have shape [{self.n_layers},P], got {tuple(phi.shape)}"
            )
        u_p = self.features[self.pilot_idx]
        if int(phi.shape[1]) != int(u_p.shape[0]):
            raise ValueError("DMRS matrix and pilot-index lengths disagree")
        blocks = [
            phi[layer].reshape(-1, 1) * u_p
            for layer in range(self.n_layers)
        ]
        return torch.cat(blocks, dim=1)

    def residual_posterior(
        self,
        *,
        y_p: torch.Tensor,
        phi: torch.Tensor,
        noise_var: torch.Tensor,
        ls_mean: torch.Tensor,
    ) -> tuple[PosteriorOutput, torch.Tensor, torch.Tensor, dict[str, Any]]:
        if ls_mean.ndim != 4:
            raise ValueError("ls_mean must have shape [B,N,RX,R]")
        batch_size, n_layers, n_rx, n_re = ls_mean.shape
        if n_layers != self.n_layers or n_re != int(self.features.shape[0]):
            raise ValueError("LS estimate shape disagrees with localized operator")
        if y_p.shape != (batch_size, n_rx, int(self.pilot_idx.numel())):
            raise ValueError(
                "pilot observation shape mismatch: "
                f"observed={tuple(y_p.shape)} expected="
                f"{(batch_size, n_rx, int(self.pilot_idx.numel()))}"
            )

        mode = self.mode_parameters()
        prior_q = mode["prior_variance"]
        delta_gain = mode["delta_gain"].to(ls_mean.dtype)
        q = int(prior_q.numel())
        prior = prior_q.repeat(self.n_layers)
        basis = self._observation_basis(phi)

        # Predict a structured residual mean from observable LS coefficients.
        ls_coeff = torch.einsum(
            "rq,bnxr->bnxq", self.features.conj(), ls_mean
        )
        prior_mean = ls_coeff * delta_gain.view(1, 1, 1, q)
        prior_mean_flat = prior_mean.permute(0, 2, 1, 3).reshape(
            batch_size, n_rx, self.n_layers * q
        )

        ls_pilot = ls_mean[..., self.pilot_idx]
        predicted_ls_pilot = torch.einsum("np,bnxp->bxp", phi, ls_pilot)
        residual_observation = y_p - predicted_ls_pilot
        predicted_prior_residual = torch.einsum(
            "pl,bxl->bxp", basis, prior_mean_flat
        )
        innovation = residual_observation - predicted_prior_residual

        raw_noise = _scalar_noise(noise_var, ls_mean.device)
        effective_noise = (
            raw_noise * torch.exp(self.log_pilot_noise_scale).clamp(0.125, 8.0)
            + 1e-7
        )
        weighted_basis = basis * prior.view(1, -1).to(basis.dtype)
        observation_cov = (
            weighted_basis @ basis.conj().transpose(0, 1)
            + effective_noise.to(torch.complex64)
            * torch.eye(basis.shape[0], dtype=torch.complex64, device=basis.device)
        )
        jitter = 1e-6 * torch.diagonal(observation_cov).real.mean().clamp_min(1.0)
        chol = torch.linalg.cholesky(
            observation_cov
            + jitter
            * torch.eye(
                observation_cov.shape[0],
                dtype=observation_cov.dtype,
                device=observation_cov.device,
            )
        )
        rhs = innovation.reshape(batch_size * n_rx, -1).T.contiguous()
        alpha = torch.cholesky_solve(rhs, chol)
        correction = (
            prior.view(-1, 1).to(basis.dtype)
            * (basis.conj().transpose(0, 1) @ alpha)
        )
        posterior_flat = prior_mean_flat + correction.T.reshape(
            batch_size, n_rx, self.n_layers * q
        )
        posterior_coeff = posterior_flat.reshape(
            batch_size, n_rx, self.n_layers, q
        )
        localized_correction = torch.einsum(
            "rq,bxnq->bnxr", self.features, posterior_coeff
        ).contiguous()

        solve_weighted = torch.cholesky_solve(weighted_basis, chol)
        latent_cov = (
            torch.diag(prior.to(torch.complex64))
            - weighted_basis.conj().transpose(0, 1) @ solve_weighted
        )
        latent_cov = 0.5 * (
            latent_cov + latent_cov.conj().transpose(0, 1)
        )
        local_cov, var_diag = project_latent_covariance_to_grid(
            self.features,
            latent_cov,
            self.n_layers,
            q,
        )
        residual = PosteriorOutput(
            mean=localized_correction,
            var_diag=var_diag,
            local_cov=local_cov,
            latent_cov=latent_cov,
            effective_noise=effective_noise,
        )
        diagnostics = {
            "effective_rank": q,
            "nominal_rank": self.nominal_rank,
            "effective_noise": effective_noise,
            "innovation_power": innovation.abs().square().mean().real,
            "predicted_residual_power": localized_correction.abs().square().mean().real,
        }
        return residual, prior_q, delta_gain, diagnostics

    def forward(
        self,
        *,
        y_p: torch.Tensor,
        phi: torch.Tensor,
        noise_var: torch.Tensor,
        ls_mean: torch.Tensor,
        ls_var_diag: torch.Tensor,
    ) -> LSAnchoredLocalizedResult:
        residual, prior_variance, delta_gain, diagnostics = self.residual_posterior(
            y_p=y_p,
            phi=phi,
            noise_var=noise_var,
            ls_mean=ls_mean,
        )
        ls_var = torch.as_tensor(
            ls_var_diag,
            device=ls_mean.device,
            dtype=torch.float32,
        )
        if ls_var.ndim == 3:
            ls_var = ls_var.mean(dim=0)
        if ls_var.shape != residual.var_diag.shape:
            raise ValueError(
                f"LS variance shape {tuple(ls_var.shape)} does not match "
                f"localized variance {tuple(residual.var_diag.shape)}"
            )

        raw_noise = _scalar_noise(noise_var, ls_mean.device)
        gate_logit = (
            self.residual_gate_bias
            + torch.dot(self.residual_gate_context, self.context)
            + self.residual_gate_noise_slope * torch.log(raw_noise)
        )
        gate = torch.sigmoid(gate_logit)
        localized_correction = gate.to(ls_mean.dtype) * residual.mean
        mean = ls_mean + localized_correction

        correction_power = localized_correction.abs().square().mean(
            dim=(0, 2)
        ).real
        ls_weight = F.softplus(self.raw_ls_variance_weight) + 1e-5
        disagreement_weight = F.softplus(self.raw_disagreement_weight)
        output_scale = torch.exp(self.log_output_variance_scale).clamp(0.125, 8.0)
        diagonal_extra = (
            ls_weight * ls_var
            + disagreement_weight * correction_power
        ).clamp_min(1e-8)
        local_cov = (
            gate.square().to(residual.local_cov.dtype) * residual.local_cov
            + _diagonal_covariance(diagonal_extra)
        )
        local_cov = output_scale.to(local_cov.dtype) * local_cov
        local_cov = 0.5 * (
            local_cov + local_cov.transpose(0, 1).conj()
        )
        var_diag = torch.stack(
            [local_cov[index, index].real for index in range(self.n_layers)],
            dim=0,
        ).clamp_min(1e-8)
        posterior = PosteriorOutput(
            mean=mean,
            var_diag=var_diag,
            local_cov=local_cov,
            latent_cov=residual.latent_cov,
            effective_noise=residual.effective_noise,
        )
        diagnostics = {
            **diagnostics,
            "version": IMPLEMENTABLE_LOCALIZED_VERSION,
            "residual_gate": gate,
            "ls_variance_weight": ls_weight,
            "disagreement_weight": disagreement_weight,
            "output_variance_scale": output_scale,
            "delta_gain_mean_abs": delta_gain.abs().mean(),
            "inference_uses_true_channel": False,
        }
        return LSAnchoredLocalizedResult(
            posterior=posterior,
            residual_posterior=residual,
            residual_gate=gate,
            correction_power=correction_power,
            prior_variance=prior_variance,
            localized_correction=localized_correction,
            ls_mean=ls_mean,
            ls_var_diag=ls_var,
            diagnostics=diagnostics,
        )

    def parameter_report(self) -> dict[str, Any]:
        return {
            "version": IMPLEMENTABLE_LOCALIZED_VERSION,
            "nominal_rank": self.nominal_rank,
            "effective_rank": self.effective_rank,
            "num_knots": self.num_knots,
            "context_dim": int(self.context.numel()),
            "trainable_parameters": int(
                sum(parameter.numel() for parameter in self.parameters())
            ),
            "inference_inputs": [
                "received_dmrs",
                "dmrs_matrix",
                "noise_variance",
                "sionna_ls_mean",
                "sionna_ls_error_variance",
                "known_nr_context",
            ],
            "inference_uses_true_channel": False,
        }


def bind_shared_localized_parameters(
    operators: Sequence[LSAnchoredLocalizedResidualPosterior],
) -> None:
    if not operators:
        raise ValueError("at least one localized operator is required")
    master = operators[0]
    for operator in operators[1:]:
        if operator.num_knots != master.num_knots:
            raise ValueError("localized operators disagree on knot count")
        if operator.context.numel() != master.context.numel():
            raise ValueError("localized operators disagree on context dimension")
        for name in SHARED_PARAMETER_NAMES:
            setattr(operator, name, getattr(master, name))


def shared_localized_state(
    operator: LSAnchoredLocalizedResidualPosterior,
) -> dict[str, torch.Tensor]:
    return {
        name: getattr(operator, name).detach().cpu().clone()
        for name in SHARED_PARAMETER_NAMES
    }


def load_shared_localized_state(
    operator: LSAnchoredLocalizedResidualPosterior,
    state: dict[str, torch.Tensor],
) -> None:
    with torch.no_grad():
        for name in SHARED_PARAMETER_NAMES:
            if name not in state:
                raise RuntimeError(f"localized checkpoint is missing {name}")
            target = getattr(operator, name)
            value = torch.as_tensor(
                state[name], dtype=target.dtype, device=target.device
            )
            if value.shape != target.shape:
                raise RuntimeError(
                    f"localized checkpoint shape mismatch for {name}: "
                    f"{tuple(value.shape)} != {tuple(target.shape)}"
                )
            target.copy_(value)


def unique_localized_parameters(
    operators: Sequence[LSAnchoredLocalizedResidualPosterior],
) -> list[nn.Parameter]:
    seen: set[int] = set()
    result: list[nn.Parameter] = []
    for operator in operators:
        for parameter in operator.parameters():
            if parameter.requires_grad and id(parameter) not in seen:
                seen.add(id(parameter))
                result.append(parameter)
    return result


def mathematical_self_test(device: torch.device | str = "cpu") -> dict[str, Any]:
    dev = torch.device(device)
    torch.manual_seed(33017)
    n_re, rank, n_layers, n_rx, batch = 24, 8, 2, 3, 4
    raw = (
        torch.randn(n_re, rank, device=dev)
        + 1j * torch.randn(n_re, rank, device=dev)
    ).to(torch.complex64)
    features, _ = torch.linalg.qr(raw, mode="reduced")
    pilot_idx = torch.tensor([1, 5, 9, 13, 17, 21], device=dev)
    context = torch.tensor([0.5, 0.1], device=dev)
    model = LSAnchoredLocalizedResidualPosterior(
        features=features,
        pilot_idx=pilot_idx,
        n_layers=n_layers,
        nominal_rank=10,
        context=context,
    ).to(dev)
    phi = torch.zeros(n_layers, pilot_idx.numel(), dtype=torch.complex64, device=dev)
    phi[0, ::2] = 1.0
    phi[1, 1::2] = 1.0
    ls_mean = (
        torch.randn(batch, n_layers, n_rx, n_re, device=dev)
        + 1j * torch.randn(batch, n_layers, n_rx, n_re, device=dev)
    ).to(torch.complex64) / math.sqrt(2.0)
    y_p = torch.einsum("np,bnxp->bxp", phi, ls_mean[..., pilot_idx])
    y_p = y_p + 0.1 * (
        torch.randn_like(y_p) + 1j * torch.randn_like(y_p)
    )
    ls_var = torch.full((n_layers, n_re), 0.2, device=dev)
    result = model(
        y_p=y_p,
        phi=phi,
        noise_var=torch.tensor(0.1, device=dev),
        ls_mean=ls_mean,
        ls_var_diag=ls_var,
    )
    eigenvalues = torch.linalg.eigvalsh(
        result.residual_posterior.latent_cov.to(torch.complex128)
    ).real
    loss = (
        result.posterior.mean.abs().square().mean()
        + result.posterior.var_diag.mean()
        + result.localized_correction.abs().mean()
    )
    loss.backward()
    parameters = unique_localized_parameters([model])
    gradients = [
        parameter.grad is not None
        and bool(torch.isfinite(parameter.grad).all().item())
        for parameter in parameters
    ]
    before = shared_localized_state(model)
    clone = LSAnchoredLocalizedResidualPosterior(
        features=features,
        pilot_idx=pilot_idx,
        n_layers=n_layers,
        nominal_rank=10,
        context=context,
    ).to(dev)
    load_shared_localized_state(clone, before)
    clone_result = clone(
        y_p=y_p,
        phi=phi,
        noise_var=torch.tensor(0.1, device=dev),
        ls_mean=ls_mean,
        ls_var_diag=ls_var,
    )
    with torch.no_grad():
        model.residual_gate_bias.fill_(-40.0)
        nested = model(
            y_p=y_p,
            phi=phi,
            noise_var=torch.tensor(0.1, device=dev),
            ls_mean=ls_mean,
            ls_var_diag=ls_var,
        )
    ls_nested_error = float(torch.max(torch.abs(nested.posterior.mean - ls_mean)).item())
    checks = {
        "posterior_mean_finite": bool(torch.isfinite(result.posterior.mean).all().item()),
        "posterior_variance_positive": bool((result.posterior.var_diag > 0).all().item()),
        "residual_covariance_psd": float(eigenvalues.min().item()) > -1e-7,
        "residual_gate_valid": 0.0 < float(result.residual_gate.item()) < 1.0,
        "all_parameter_gradients_present_and_finite": all(gradients),
        "checkpoint_roundtrip": bool(
            torch.allclose(
                result.posterior.mean,
                clone_result.posterior.mean,
                rtol=0.0,
                atol=0.0,
            )
            and torch.allclose(
                result.posterior.var_diag,
                clone_result.posterior.var_diag,
                rtol=0.0,
                atol=0.0,
            )
        ),
        "inference_contract_excludes_truth": (
            model.parameter_report()["inference_uses_true_channel"] is False
        ),
        "ls_is_nested_special_case": ls_nested_error < 1e-6,
    }
    return {
        "version": IMPLEMENTABLE_LOCALIZED_VERSION,
        "passed": all(checks.values()),
        "checks": checks,
        "minimum_residual_covariance_eigenvalue": float(eigenvalues.min().item()),
        "residual_gate": float(result.residual_gate.item()),
        "ls_nested_max_abs_error": ls_nested_error,
        "parameter_report": model.parameter_report(),
    }
