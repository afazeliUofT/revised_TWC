from __future__ import annotations

"""Evidence-weighted mixture LMMSE channel estimator.

This module implements a small, interpretable Bayesian channel estimator for
5G NR PUSCH.  A finite bank of structured Gaussian channel priors is defined in
one precision-safe localized delay--Doppler basis.  For every prior component,
the conditional channel posterior is the exact complex-Gaussian LMMSE
posterior.  The component weights are not attention scores: they are exact
Bayesian model posterior probabilities obtained from the pilot marginal
likelihood.

The estimator therefore reduces to ordinary LMMSE channel estimation when the
number of components is one.  With multiple components, its posterior mean is
nonlinear in the pilot observations and is the MMSE estimator under the stated
mixture prior.  All trainable parameters have a direct probabilistic meaning.
"""

from dataclasses import dataclass
import math
from typing import Any, Iterable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .models import PosteriorOutput, project_latent_covariance_to_grid


EVIDENCE_MIXTURE_LMMSE_VERSION = "evidence_mixture_lmmse_ce_v1"
SHARED_PARAMETER_NAMES = (
    "raw_profile_knots",
    "log_scale_base",
    "raw_scale_gaps",
    "prior_logits",
    "context_to_prior",
    "noise_to_prior",
    "log_noise_scales",
    "log_residual_floor",
)


def _inverse_softplus(value: float) -> float:
    value = max(float(value), 1e-8)
    return math.log(math.expm1(value))


def _interpolate_rows(values: torch.Tensor, length: int) -> torch.Tensor:
    if values.ndim != 2:
        raise ValueError("values must have shape [K,L]")
    if int(length) <= 0:
        raise ValueError("length must be positive")
    if int(values.shape[1]) == int(length):
        return values
    return F.interpolate(
        values[:, None, :],
        size=int(length),
        mode="linear",
        align_corners=True,
    )[:, 0, :]


def _diagonal_covariance(var_diag: torch.Tensor) -> torch.Tensor:
    if var_diag.ndim != 2:
        raise ValueError(f"Expected var_diag[N,R], got {tuple(var_diag.shape)}")
    return torch.diag_embed(var_diag.transpose(0, 1).to(torch.complex64)).permute(
        1, 2, 0
    ).contiguous()


def _project_batch_latent_covariance_to_grid(
    features: torch.Tensor,
    latent_covariance: torch.Tensor,
    n_layers: int,
    rank: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Project [B,NQ,NQ] covariance to [B,N,N,R] local marginals."""
    if latent_covariance.ndim != 3:
        raise ValueError("latent_covariance must have shape [B,NQ,NQ]")
    rows: list[torch.Tensor] = []
    for n in range(int(n_layers)):
        row: list[torch.Tensor] = []
        for m in range(int(n_layers)):
            block = latent_covariance[
                :,
                n * rank : (n + 1) * rank,
                m * rank : (m + 1) * rank,
            ]
            cov_nm = torch.einsum(
                "rq,bqs,rs->br",
                features,
                block,
                features.conj(),
            )
            row.append(cov_nm)
        rows.append(torch.stack(row, dim=1))
    local_cov = torch.stack(rows, dim=1)
    local_cov = 0.5 * (local_cov + local_cov.transpose(1, 2).conj())
    var_diag = torch.stack(
        [local_cov[:, n, n].real.clamp_min(1e-8) for n in range(int(n_layers))],
        dim=1,
    )
    return local_cov, var_diag


def _scalar_noise(noise_var: torch.Tensor, device: torch.device) -> torch.Tensor:
    value = torch.as_tensor(noise_var, device=device).real.to(torch.float32)
    if value.numel() == 0:
        raise ValueError("noise variance is empty")
    return value.mean().clamp_min(1e-9)


def _safe_complex_logdet_from_cholesky(chol: torch.Tensor) -> torch.Tensor:
    diagonal = torch.diagonal(chol).real.clamp_min(1e-15)
    return 2.0 * torch.log(diagonal).sum()


@dataclass
class EvidenceMixtureResult:
    posterior: PosteriorOutput
    weights: torch.Tensor
    context_prior: torch.Tensor
    component_log_evidence: torch.Tensor
    component_variances: torch.Tensor
    component_noise: torch.Tensor
    evidence_entropy: torch.Tensor
    effective_component_count: torch.Tensor
    normalized_negative_log_evidence: torch.Tensor
    mode: str
    diagnostics: dict[str, Any]


class EvidenceMixtureLMMSEPosterior(nn.Module):
    """Exact evidence-weighted mixture of structured LMMSE estimators.

    Parameters are shared across grid widths.  Grid-dependent localized bases,
    pilot indices, and known NR context are deterministic buffers.  The mode
    spectra are represented by a small number of smooth knots and component
    total powers are strictly ordered, which removes the ordinary label-swap
    ambiguity except when two components become identical.
    """

    def __init__(
        self,
        *,
        features: torch.Tensor,
        pilot_idx: torch.Tensor,
        n_layers: int,
        nominal_rank: int,
        context: torch.Tensor,
        num_components: int = 4,
        num_knots: int = 8,
    ) -> None:
        super().__init__()
        if features.ndim != 2 or not torch.is_complex(features):
            raise ValueError("features must be a complex matrix [R,Q]")
        if int(features.shape[1]) <= 0:
            raise ValueError("localized basis must have positive effective rank")
        if int(nominal_rank) < int(features.shape[1]):
            raise ValueError("nominal rank cannot be smaller than effective rank")
        if int(n_layers) <= 0:
            raise ValueError("n_layers must be positive")
        if int(num_components) <= 0:
            raise ValueError("num_components must be positive")
        if int(num_knots) < 3:
            raise ValueError("num_knots must be at least three")

        self.n_layers = int(n_layers)
        self.nominal_rank = int(nominal_rank)
        self.effective_rank = int(features.shape[1])
        self.num_components = int(num_components)
        self.num_knots = int(num_knots)
        self.register_buffer("features", features.detach().clone().to(torch.complex64))
        self.register_buffer("pilot_idx", pilot_idx.detach().clone().long())
        self.register_buffer("context", context.detach().clone().float().flatten())

        # Diverse smooth initial spectra.  Rows remain normalized to unit mean;
        # component total power is controlled separately by ordered scales.
        grid = torch.linspace(-1.0, 1.0, self.num_knots)
        slopes = torch.linspace(2.0, -2.0, self.num_components)
        initial = slopes[:, None] * grid[None, :]
        initial = initial + 0.15 * torch.cos(
            math.pi * torch.arange(self.num_components)[:, None] * grid[None, :]
        )
        self.raw_profile_knots = nn.Parameter(initial)

        # Ordered total component powers.  Basis columns have unit norm, hence
        # R/Q is the natural grid-dependent coefficient-variance scale.
        self.log_scale_base = nn.Parameter(torch.tensor(math.log(0.22)))
        if self.num_components > 1:
            self.raw_scale_gaps = nn.Parameter(
                torch.full(
                    (self.num_components - 1,),
                    _inverse_softplus(0.65),
                )
            )
        else:
            self.raw_scale_gaps = nn.Parameter(torch.empty(0))

        self.prior_logits = nn.Parameter(torch.zeros(self.num_components))
        self.context_to_prior = nn.Parameter(
            torch.zeros(self.num_components, int(self.context.numel()))
        )
        self.noise_to_prior = nn.Parameter(torch.zeros(self.num_components))
        self.log_noise_scales = nn.Parameter(torch.zeros(self.num_components))
        self.log_residual_floor = nn.Parameter(torch.tensor(math.log(0.01)))

    def ordered_log_scales(self) -> torch.Tensor:
        if self.num_components == 1:
            return self.log_scale_base.reshape(1)
        gaps = F.softplus(self.raw_scale_gaps) + 0.05
        increments = torch.cat(
            [torch.zeros(1, device=gaps.device, dtype=gaps.dtype), gaps]
        )
        return self.log_scale_base + torch.cumsum(increments, dim=0)

    def component_variance_profiles(self) -> torch.Tensor:
        raw = _interpolate_rows(self.raw_profile_knots, self.effective_rank)
        profile = F.softplus(raw) + 1e-5
        profile = profile / profile.mean(dim=1, keepdim=True).clamp_min(1e-8)
        grid_scale = float(self.features.shape[0]) / float(self.effective_rank)
        total_scales = torch.exp(self.ordered_log_scales()).clamp(1e-3, 1e3)
        return (
            grid_scale
            * total_scales[:, None]
            * profile
        ).clamp_min(1e-7)

    def context_prior_logits(self, noise_var: torch.Tensor) -> torch.Tensor:
        raw_noise = _scalar_noise(noise_var, self.features.device)
        log_noise = torch.log(raw_noise)
        return (
            self.prior_logits
            + self.context_to_prior @ self.context
            + self.noise_to_prior * log_noise
        )

    def context_prior_probabilities(self, noise_var: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.context_prior_logits(noise_var), dim=0)

    def _observation_basis(self, phi: torch.Tensor) -> torch.Tensor:
        phi = torch.as_tensor(phi, device=self.features.device).to(torch.complex64)
        if phi.ndim != 2 or int(phi.shape[0]) != self.n_layers:
            raise ValueError(
                f"phi must have shape [{self.n_layers},P], got {tuple(phi.shape)}"
            )
        u_p = self.features[self.pilot_idx]
        if int(phi.shape[1]) != int(u_p.shape[0]):
            raise ValueError("DMRS matrix and pilot-index lengths disagree")
        p = int(u_p.shape[0])
        blocks = [phi[layer].reshape(p, 1) * u_p for layer in range(self.n_layers)]
        return torch.cat(blocks, dim=1).contiguous()

    def _single_component(
        self,
        *,
        y_p: torch.Tensor,
        observation_basis: torch.Tensor,
        mode_variance: torch.Tensor,
        effective_noise: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, n_rx, n_pilots = y_p.shape
        prior = mode_variance.repeat(self.n_layers).to(torch.float32)
        weighted_basis = observation_basis * prior[None, :].to(observation_basis.dtype)
        eye = torch.eye(
            n_pilots,
            dtype=torch.complex64,
            device=observation_basis.device,
        )
        observation_cov = (
            weighted_basis @ observation_basis.conj().transpose(0, 1)
            + effective_noise.to(torch.complex64) * eye
        )
        jitter = 1e-6 * torch.diagonal(observation_cov).real.mean().clamp_min(1.0)
        chol = torch.linalg.cholesky(observation_cov + jitter.to(eye.dtype) * eye)

        y_flat = y_p.reshape(batch_size * n_rx, n_pilots).T.contiguous()
        alpha = torch.cholesky_solve(y_flat, chol)
        latent_mean_flat = (
            prior[:, None].to(observation_basis.dtype)
            * (observation_basis.conj().transpose(0, 1) @ alpha)
        )
        latent_mean = latent_mean_flat.T.reshape(
            batch_size, n_rx, self.n_layers * self.effective_rank
        )

        solve_weighted = torch.cholesky_solve(weighted_basis, chol)
        latent_cov = (
            torch.diag(prior.to(torch.complex64))
            - weighted_basis.conj().transpose(0, 1) @ solve_weighted
        )
        latent_cov = 0.5 * (latent_cov + latent_cov.conj().transpose(0, 1))

        quadratic = torch.sum(y_flat.conj() * alpha, dim=0).real.reshape(
            batch_size, n_rx
        )
        logdet = _safe_complex_logdet_from_cholesky(chol)
        log_evidence = -(quadratic.sum(dim=1) + float(n_rx) * logdet)
        return latent_mean, latent_cov, log_evidence

    def _posterior_from_moments(
        self,
        *,
        latent_mean: torch.Tensor,
        latent_cov: torch.Tensor,
        effective_noise: torch.Tensor,
    ) -> PosteriorOutput:
        batch_size, n_rx, _ = latent_mean.shape
        coeff = latent_mean.reshape(
            batch_size, n_rx, self.n_layers, self.effective_rank
        )
        channel_mean = torch.einsum(
            "rq,bxnq->bnxr", self.features, coeff
        ).contiguous()
        if latent_cov.ndim == 2:
            local_cov, var_diag = project_latent_covariance_to_grid(
                self.features,
                latent_cov,
                self.n_layers,
                self.effective_rank,
            )
            residual_floor = torch.exp(self.log_residual_floor).clamp(1e-6, 1.0)
            var_diag = var_diag + residual_floor
            local_cov = (
                local_cov
                + _diagonal_covariance(torch.ones_like(var_diag) * residual_floor)
            )
            local_cov = 0.5 * (local_cov + local_cov.transpose(0, 1).conj())
        elif latent_cov.ndim == 3:
            local_cov, var_diag = _project_batch_latent_covariance_to_grid(
                self.features,
                latent_cov,
                self.n_layers,
                self.effective_rank,
            )
            residual_floor = torch.exp(self.log_residual_floor).clamp(1e-6, 1.0)
            var_diag = var_diag + residual_floor
            diagonal = torch.diag_embed(
                (torch.ones_like(var_diag) * residual_floor)
                .permute(0, 2, 1)
                .to(torch.complex64)
            ).permute(0, 2, 3, 1).contiguous()
            local_cov = local_cov + diagonal
            local_cov = 0.5 * (local_cov + local_cov.transpose(1, 2).conj())
        else:
            raise ValueError("latent_cov must have shape [L,L] or [B,L,L]")
        return PosteriorOutput(
            mean=channel_mean,
            var_diag=var_diag.real.clamp_min(1e-8),
            local_cov=local_cov,
            latent_cov=latent_cov,
            effective_noise=effective_noise,
        )

    def forward(
        self,
        *,
        y_p: torch.Tensor,
        phi: torch.Tensor,
        noise_var: torch.Tensor,
        mode: str = "mixture",
    ) -> EvidenceMixtureResult:
        if y_p.ndim != 3 or not torch.is_complex(y_p):
            raise ValueError("y_p must be a complex tensor [B,RX,P]")
        if int(y_p.shape[-1]) != int(self.pilot_idx.numel()):
            raise ValueError("pilot observation length mismatch")
        if mode not in {"mixture", "hard", "uniform", "moment"}:
            raise ValueError(f"Unsupported inference mode: {mode}")

        observation_basis = self._observation_basis(phi)
        variances = self.component_variance_profiles()
        raw_noise = _scalar_noise(noise_var, y_p.device)
        residual_floor = torch.exp(self.log_residual_floor).clamp(1e-6, 1.0)
        component_noise = (
            raw_noise
            * torch.exp(self.log_noise_scales).clamp(0.125, 8.0)
            + residual_floor
            + 1e-7
        )
        context_logits = self.context_prior_logits(noise_var)
        context_prior = torch.softmax(context_logits, dim=0)

        if mode == "moment":
            moment_variance = torch.sum(
                context_prior[:, None] * variances, dim=0
            )
            moment_noise = torch.sum(context_prior * component_noise)
            mean, covariance, log_evidence = self._single_component(
                y_p=y_p,
                observation_basis=observation_basis,
                mode_variance=moment_variance,
                effective_noise=moment_noise,
            )
            covariance_batch = covariance[None, ...].expand(
                int(y_p.shape[0]), -1, -1
            ).contiguous()
            posterior = self._posterior_from_moments(
                latent_mean=mean,
                latent_cov=covariance_batch,
                effective_noise=moment_noise.expand(int(y_p.shape[0])),
            )
            weights = context_prior[None, :].expand(y_p.shape[0], -1)
            entropy = -torch.sum(
                weights * torch.log(weights.clamp_min(1e-12)), dim=1
            )
            normalized_nle = -log_evidence / float(
                max(int(y_p.shape[1] * y_p.shape[2]), 1)
            )
            return EvidenceMixtureResult(
                posterior=posterior,
                weights=weights,
                context_prior=context_prior,
                component_log_evidence=log_evidence[:, None],
                component_variances=variances,
                component_noise=component_noise,
                evidence_entropy=entropy,
                effective_component_count=torch.exp(entropy),
                normalized_negative_log_evidence=normalized_nle,
                mode=mode,
                diagnostics={
                    "version": EVIDENCE_MIXTURE_LMMSE_VERSION,
                    "inference_uses_true_channel": False,
                    "moment_matched_lmmse": True,
                },
            )

        means: list[torch.Tensor] = []
        covariances: list[torch.Tensor] = []
        log_evidence_rows: list[torch.Tensor] = []
        for index in range(self.num_components):
            mean, covariance, log_evidence = self._single_component(
                y_p=y_p,
                observation_basis=observation_basis,
                mode_variance=variances[index],
                effective_noise=component_noise[index],
            )
            means.append(mean)
            covariances.append(covariance)
            log_evidence_rows.append(log_evidence)

        component_means = torch.stack(means, dim=2)  # [B,RX,K,L]
        component_covariances = torch.stack(covariances, dim=0)  # [K,L,L]
        log_evidence = torch.stack(log_evidence_rows, dim=1)  # [B,K]
        log_context_prior = torch.log_softmax(context_logits, dim=0)
        posterior_logits = log_context_prior[None, :] + log_evidence
        soft_weights = torch.softmax(posterior_logits, dim=1)
        if mode == "uniform":
            weights = torch.full_like(soft_weights, 1.0 / float(self.num_components))
        elif mode == "hard":
            winner = torch.argmax(posterior_logits, dim=1)
            weights = F.one_hot(winner, self.num_components).to(soft_weights.dtype)
        else:
            weights = soft_weights

        latent_mean = torch.einsum(
            "bk,bxkl->bxl", weights.to(component_means.dtype), component_means
        )
        within = torch.einsum(
            "bk,kij->bij",
            weights.to(component_covariances.dtype),
            component_covariances,
        )
        difference = component_means - latent_mean[:, :, None, :]
        between_by_rx = torch.einsum(
            "bk,bxki,bxkj->bxij",
            weights.to(difference.dtype),
            difference,
            difference.conj(),
        )
        between = between_by_rx.mean(dim=1)
        latent_cov = within + between
        latent_cov = 0.5 * (latent_cov + latent_cov.conj().transpose(-1, -2))
        effective_noise = torch.sum(weights * component_noise[None, :], dim=1)
        average_weights = weights.mean(dim=0)
        posterior = self._posterior_from_moments(
            latent_mean=latent_mean,
            latent_cov=latent_cov,
            effective_noise=effective_noise,
        )
        entropy = -torch.sum(
            weights * torch.log(weights.clamp_min(1e-12)), dim=1
        )
        log_marginal = torch.logsumexp(posterior_logits, dim=1)
        normalized_nle = -log_marginal / float(
            max(int(y_p.shape[1] * y_p.shape[2]), 1)
        )
        return EvidenceMixtureResult(
            posterior=posterior,
            weights=weights,
            context_prior=context_prior,
            component_log_evidence=log_evidence,
            component_variances=variances,
            component_noise=component_noise,
            evidence_entropy=entropy,
            effective_component_count=torch.exp(entropy),
            normalized_negative_log_evidence=normalized_nle,
            mode=mode,
            diagnostics={
                "version": EVIDENCE_MIXTURE_LMMSE_VERSION,
                "inference_uses_true_channel": False,
                "ordered_total_scales": torch.exp(self.ordered_log_scales()),
                "mean_weights": average_weights,
                "moment_matched_lmmse": False,
            },
        )

    def prior_coefficient_nll(
        self,
        h_true: torch.Tensor,
        noise_var: torch.Tensor,
    ) -> torch.Tensor:
        """Supervised proper-score term used only during simulation training."""
        if h_true.ndim != 4 or int(h_true.shape[-1]) != int(self.features.shape[0]):
            raise ValueError("h_true must have shape [B,N,RX,R]")
        coefficients = torch.einsum(
            "rq,bnxr->bnxq", self.features.conj(), h_true
        )
        variances = self.component_variance_profiles()
        context_logits = self.context_prior_logits(noise_var)
        log_prior = torch.log_softmax(context_logits, dim=0)
        terms: list[torch.Tensor] = []
        dimension = float(
            max(
                int(coefficients.shape[1])
                * int(coefficients.shape[2])
                * int(coefficients.shape[3]),
                1,
            )
        )
        for index in range(self.num_components):
            variance = variances[index].view(1, 1, 1, -1)
            score = (
                coefficients.abs().square() / variance
                + torch.log(variance)
            ).sum(dim=(1, 2, 3))
            terms.append(log_prior[index] - score)
        log_probability = torch.logsumexp(torch.stack(terms, dim=1), dim=1)
        return -log_probability.mean() / dimension

    def diversity_penalty(self) -> torch.Tensor:
        if self.num_components <= 1:
            return torch.zeros((), device=self.features.device)
        profiles = self.component_variance_profiles()
        normalized = F.normalize(profiles, dim=1)
        similarity = normalized @ normalized.T
        eye = torch.eye(
            self.num_components,
            dtype=torch.bool,
            device=similarity.device,
        )
        off = similarity.masked_select(~eye)
        return torch.square(F.relu(off - 0.94)).mean()

    def parameter_report(self) -> dict[str, Any]:
        return {
            "version": EVIDENCE_MIXTURE_LMMSE_VERSION,
            "num_components": self.num_components,
            "num_knots": self.num_knots,
            "effective_rank": self.effective_rank,
            "nominal_rank": self.nominal_rank,
            "context_dim": int(self.context.numel()),
            "trainable_parameters": int(
                sum(parameter.numel() for parameter in self.parameters())
            ),
            "inference_uses_true_channel": False,
            "routing_rule": "exact_bayesian_pilot_marginal_likelihood",
            "component_experts": "exact_structured_lmmse_posteriors",
            "single_component_reduction": "ordinary_lmmse_channel_estimation",
            "label_ordering": "strictly_ordered_total_component_power",
            "unmodeled_channel_term": "independent_white_residual_in_pilot_noise_and_output_covariance",
        }


def bind_shared_evidence_parameters(
    operators: Sequence[EvidenceMixtureLMMSEPosterior],
) -> None:
    if not operators:
        raise ValueError("At least one operator is required")
    master = operators[0]
    for operator in operators[1:]:
        if operator.num_components != master.num_components:
            raise ValueError("component-count mismatch")
        if operator.num_knots != master.num_knots:
            raise ValueError("knot-count mismatch")
        if int(operator.context.numel()) != int(master.context.numel()):
            raise ValueError("context-dimension mismatch")
        for name in SHARED_PARAMETER_NAMES:
            setattr(operator, name, getattr(master, name))


def unique_evidence_parameters(
    operators: Iterable[EvidenceMixtureLMMSEPosterior],
) -> list[nn.Parameter]:
    result: list[nn.Parameter] = []
    seen: set[int] = set()
    for operator in operators:
        for parameter in operator.parameters():
            if id(parameter) not in seen:
                seen.add(id(parameter))
                result.append(parameter)
    return result


def shared_evidence_state(
    operator: EvidenceMixtureLMMSEPosterior,
) -> dict[str, torch.Tensor]:
    return {
        name: getattr(operator, name).detach().cpu().clone()
        for name in SHARED_PARAMETER_NAMES
    }


def load_shared_evidence_state(
    operator: EvidenceMixtureLMMSEPosterior,
    state: dict[str, Any],
) -> None:
    with torch.no_grad():
        for name in SHARED_PARAMETER_NAMES:
            if name not in state:
                raise RuntimeError(f"Missing state tensor: {name}")
            target = getattr(operator, name)
            value = torch.as_tensor(
                state[name], dtype=target.dtype, device=target.device
            )
            if value.shape != target.shape:
                raise RuntimeError(
                    f"State shape mismatch for {name}: {tuple(value.shape)} != {tuple(target.shape)}"
                )
            target.copy_(value)


def evidence_state_to_jsonable(state: dict[str, torch.Tensor]) -> dict[str, Any]:
    return {
        name: {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "values": value.detach().cpu().reshape(-1).tolist(),
        }
        for name, value in state.items()
    }


def mathematical_self_test(device: torch.device | str = "cpu") -> dict[str, Any]:
    dev = torch.device(device)
    torch.manual_seed(51023)
    batch, n_layers, n_rx, n_re, rank = 3, 3, 2, 24, 8
    raw = (
        torch.randn(n_re, rank, device=dev)
        + 1j * torch.randn(n_re, rank, device=dev)
    ).to(torch.complex64)
    features, _ = torch.linalg.qr(raw, mode="reduced")
    pilot_idx = torch.tensor([1, 4, 7, 10, 14, 18, 21], device=dev)
    phi = torch.zeros(
        n_layers, pilot_idx.numel(), dtype=torch.complex64, device=dev
    )
    for layer in range(n_layers):
        phi[layer, layer::n_layers] = 1.0
    context = torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0, 1.5, 3.0, 0.35], device=dev)
    model = EvidenceMixtureLMMSEPosterior(
        features=features,
        pilot_idx=pilot_idx,
        n_layers=n_layers,
        nominal_rank=rank,
        context=context,
        num_components=4,
        num_knots=6,
    ).to(dev)
    y_p = (
        torch.randn(batch, n_rx, pilot_idx.numel(), device=dev)
        + 1j * torch.randn(batch, n_rx, pilot_idx.numel(), device=dev)
    ).to(torch.complex64) / math.sqrt(2.0)
    result = model(
        y_p=y_p,
        phi=phi,
        noise_var=torch.tensor(0.2, device=dev),
        mode="mixture",
    )
    eig = torch.linalg.eigvalsh(result.posterior.latent_cov.to(torch.complex128)).real

    # K=1 must reduce exactly to the moment-matched LMMSE path.
    single = EvidenceMixtureLMMSEPosterior(
        features=features,
        pilot_idx=pilot_idx,
        n_layers=n_layers,
        nominal_rank=rank,
        context=context,
        num_components=1,
        num_knots=6,
    ).to(dev)
    single_mixture = single(
        y_p=y_p,
        phi=phi,
        noise_var=torch.tensor(0.2, device=dev),
        mode="mixture",
    )
    single_moment = single(
        y_p=y_p,
        phi=phi,
        noise_var=torch.tensor(0.2, device=dev),
        mode="moment",
    )
    k1_mean_error = float(
        torch.max(
            torch.abs(single_mixture.posterior.mean - single_moment.posterior.mean)
        ).item()
    )
    k1_cov_error = float(
        torch.max(
            torch.abs(
                single_mixture.posterior.local_cov
                - single_moment.posterior.local_cov
            )
        ).item()
    )

    # Layer permutation equivariance.
    permutation = torch.tensor([2, 0, 1], device=dev)
    inverse = torch.argsort(permutation)
    permuted = model(
        y_p=y_p,
        phi=phi[permutation],
        noise_var=torch.tensor(0.2, device=dev),
        mode="mixture",
    )
    equivariance_error = float(
        torch.max(
            torch.abs(
                permuted.posterior.mean[:, inverse]
                - result.posterior.mean
            )
        ).item()
    )

    loss = (
        result.posterior.mean.abs().square().mean()
        + result.posterior.var_diag.mean()
        + result.normalized_negative_log_evidence.mean()
        + 0.1 * model.diversity_penalty()
    )
    loss.backward()
    parameters = unique_evidence_parameters([model])
    gradient_finite = all(
        parameter.grad is not None
        and bool(torch.isfinite(parameter.grad).all().item())
        for parameter in parameters
    )
    before = shared_evidence_state(model)
    optimizer = torch.optim.Adam(parameters, lr=1e-3)
    optimizer.step()
    after = shared_evidence_state(model)
    changed = any(
        not torch.equal(before[name], after[name]) for name in SHARED_PARAMETER_NAMES
    )

    clone = EvidenceMixtureLMMSEPosterior(
        features=features,
        pilot_idx=pilot_idx,
        n_layers=n_layers,
        nominal_rank=rank,
        context=context,
        num_components=4,
        num_knots=6,
    ).to(dev)
    load_shared_evidence_state(clone, after)
    model_after = model(
        y_p=y_p,
        phi=phi,
        noise_var=torch.tensor(0.2, device=dev),
        mode="mixture",
    )
    clone_after = clone(
        y_p=y_p,
        phi=phi,
        noise_var=torch.tensor(0.2, device=dev),
        mode="mixture",
    )
    roundtrip = bool(
        torch.allclose(
            model_after.posterior.mean,
            clone_after.posterior.mean,
            rtol=0.0,
            atol=0.0,
        )
        and torch.allclose(
            model_after.posterior.var_diag,
            clone_after.posterior.var_diag,
            rtol=0.0,
            atol=0.0,
        )
    )

    ordered = model.ordered_log_scales().detach()
    checks = {
        "posterior_mean_finite": bool(
            torch.isfinite(result.posterior.mean).all().item()
        ),
        "posterior_variance_positive": bool(
            (result.posterior.var_diag > 0).all().item()
        ),
        "latent_covariance_psd": float(eig.min().item()) > -1e-6,
        "sample_specific_posterior_covariance": (
            result.posterior.latent_cov.ndim == 3
            and int(result.posterior.latent_cov.shape[0]) == batch
            and result.posterior.local_cov.ndim == 4
            and int(result.posterior.local_cov.shape[0]) == batch
        ),
        "weights_sum_to_one": bool(
            torch.allclose(
                result.weights.sum(dim=1),
                torch.ones(batch, device=dev),
                atol=1e-6,
                rtol=0.0,
            )
        ),
        "k1_reduces_to_lmmse_mean": k1_mean_error < 1e-5,
        "k1_reduces_to_lmmse_covariance": k1_cov_error < 1e-5,
        "layer_permutation_equivariance": equivariance_error < 2e-4,
        "ordered_component_power": bool(
            torch.all(ordered[1:] > ordered[:-1]).item()
        ) if ordered.numel() > 1 else True,
        "gradients_present_and_finite": gradient_finite,
        "optimizer_updates_parameters": changed,
        "checkpoint_roundtrip": roundtrip,
        "small_parameter_count": model.parameter_report()["trainable_parameters"] <= 128,
        "inference_contract_excludes_truth": (
            model.parameter_report()["inference_uses_true_channel"] is False
        ),
    }
    return {
        "version": EVIDENCE_MIXTURE_LMMSE_VERSION,
        "passed": all(checks.values()),
        "checks": checks,
        "minimum_latent_eigenvalue": float(eig.min().item()),
        "k1_mean_max_abs_error": k1_mean_error,
        "k1_covariance_max_abs_error": k1_cov_error,
        "permutation_max_abs_error": equivariance_error,
        "parameter_report": model.parameter_report(),
        "mean_component_weights": result.weights.mean(dim=0).detach().cpu().tolist(),
    }
