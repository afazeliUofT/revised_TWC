from __future__ import annotations

"""One-step soft-data-aided Gaussian posterior refinement.

The pilot posterior is represented in the same low-rank latent basis used by
``MultiScalePosteriorOperator``.  Reliable detector symbol moments are then
used as fractional pseudo-pilot observations.  Symbol uncertainty is converted
into additive moment-matched observation noise, and a fractional-likelihood
coefficient prevents the same received data from being counted as an
independent observation with full weight.

No trainable parameter is introduced by this module.
"""

from dataclasses import dataclass
import math
from typing import Any

import torch

from .models import PosteriorOutput
from .multiscale_posterior import MultiScalePosteriorOperator


TURBO_POSTERIOR_VERSION = "fractional_soft_data_gaussian_update_v1"


@dataclass
class LatentPosteriorState:
    posterior: PosteriorOutput
    latent_mean: torch.Tensor  # [B,RX,N,Q]
    latent_cov: torch.Tensor   # [NQ,NQ]
    features: torch.Tensor     # [R,Q]


@dataclass
class TurboUpdateDiagnostics:
    version: str
    selected_observations: int
    data_observations: int
    information_damping: float
    selected_symbol_variance_mean: float
    pseudo_noise_mean: float
    latent_trace_before: float
    latent_trace_after: float
    latent_trace_reduction_fraction: float
    max_hermitian_error: float
    finite: bool


def _effective_noise(
    operator: MultiScalePosteriorOperator,
    noise_var: torch.Tensor,
) -> torch.Tensor:
    return (
        torch.as_tensor(noise_var, device=operator.base_features.device)
        * torch.exp(operator.log_noise_scale).clamp(0.125, 8.0)
        + 1e-6
    )


def latent_posterior_from_pilots(
    operator: MultiScalePosteriorOperator,
    y_p: torch.Tensor,
    phi: torch.Tensor,
    noise_var: torch.Tensor,
) -> LatentPosteriorState:
    """Reproduce the operator's exact pilot posterior and expose latent mean.

    This is deliberately source-locked to ``MultiScalePosteriorOperator`` and
    follows its forward equations exactly.  A smoke test compares this result
    against the operator's public forward output before any turbo update is
    allowed.
    """
    if not isinstance(operator, MultiScalePosteriorOperator):
        raise TypeError("Turbo posterior v1 requires MultiScalePosteriorOperator")
    if y_p.ndim != 3:
        raise ValueError("y_p must have shape [B,RX,P]")
    batch_size, n_rx, n_pilots = y_p.shape
    device = y_p.device
    observation_basis = operator._observation_basis(phi).to(device)  # noqa: SLF001
    if int(observation_basis.shape[0]) != int(n_pilots):
        raise ValueError("Pilot observation basis does not match y_p")
    n_latent = int(observation_basis.shape[1])
    sigma2 = _effective_noise(operator, noise_var).to(device)
    eye_p = torch.eye(n_pilots, dtype=torch.complex64, device=device)
    observation_cov = (
        observation_basis @ observation_basis.conj().transpose(0, 1)
        + sigma2.to(torch.complex64) * eye_p
    )
    chol = torch.linalg.cholesky(observation_cov + 1e-5 * eye_p)

    y_flat = y_p.reshape(batch_size * n_rx, n_pilots).T.contiguous()
    alpha = torch.cholesky_solve(y_flat, chol)
    latent_mean_flat = observation_basis.conj().transpose(0, 1) @ alpha
    latent_mean = latent_mean_flat.T.reshape(
        batch_size, n_rx, operator.n_layers, operator.rank
    )
    features = operator.weighted_features().to(device)
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
    local_cov, var_diag = project_batched_latent_covariance(
        features,
        latent_cov.unsqueeze(0),
        operator.n_layers,
        operator.rank,
    )
    posterior = PosteriorOutput(
        mean=channel_mean,
        var_diag=var_diag[0],
        local_cov=local_cov[0],
        latent_cov=latent_cov,
        effective_noise=sigma2,
    )
    return LatentPosteriorState(
        posterior=posterior,
        latent_mean=latent_mean,
        latent_cov=latent_cov,
        features=features,
    )


def project_batched_latent_covariance(
    features: torch.Tensor,
    latent_cov: torch.Tensor,
    n_layers: int,
    rank: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Project [B,NQ,NQ] latent covariance to [B,N,N,R]."""
    if latent_cov.ndim != 3:
        raise ValueError("latent_cov must have shape [B,NQ,NQ]")
    batch = int(latent_cov.shape[0])
    expected = int(n_layers) * int(rank)
    if latent_cov.shape[-2:] != (expected, expected):
        raise ValueError("latent covariance dimension mismatch")
    rows: list[torch.Tensor] = []
    for n in range(int(n_layers)):
        row: list[torch.Tensor] = []
        for m in range(int(n_layers)):
            block = latent_cov[
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
    if local_cov.shape[:3] != (batch, int(n_layers), int(n_layers)):
        raise RuntimeError("Batched covariance projection shape failure")
    local_cov = 0.5 * (
        local_cov + local_cov.transpose(1, 2).conj()
    )
    var_diag = torch.stack(
        [
            local_cov[:, n, n].real.clamp_min(1e-8)
            for n in range(int(n_layers))
        ],
        dim=1,
    )
    return local_cov, var_diag


def select_reliable_data_indices(
    symbol_var: torch.Tensor,
    *,
    fraction: float,
    min_observations: int,
    max_observations: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select the lowest aggregate symbol-variance REs for each batch item."""
    if symbol_var.ndim != 3:
        raise ValueError("symbol_var must have shape [B,N,D]")
    _, _, n_data = symbol_var.shape
    if not 0.0 < float(fraction) <= 1.0:
        raise ValueError("fraction must lie in (0,1]")
    desired = int(math.ceil(float(fraction) * int(n_data)))
    count = min(int(n_data), int(max_observations))
    count = min(count, max(int(min_observations), desired))
    if count <= 0:
        raise ValueError("selected observation count must be positive")
    aggregate_variance = symbol_var.real.mean(dim=1)
    values, indices = torch.topk(
        aggregate_variance,
        k=count,
        dim=-1,
        largest=False,
        sorted=True,
    )
    return indices, values


def fractional_gaussian_condition(
    latent_mean: torch.Tensor,
    latent_cov: torch.Tensor,
    observation_matrix: torch.Tensor,
    observations: torch.Tensor,
    observation_noise: torch.Tensor,
    *,
    information_damping: float,
    jitter: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fractionally condition a Gaussian posterior on batched linear data.

    ``latent_mean`` is [B,RX,L], ``latent_cov`` is [L,L] or [B,L,L],
    ``observation_matrix`` is [B,K,L], observations are [B,RX,K], and
    observation noise is [B,K].  The power-EP/fractional-likelihood coefficient
    is implemented by replacing R with R/rho.
    """
    rho = float(information_damping)
    if not 0.0 <= rho <= 1.0:
        raise ValueError("information_damping must lie in [0,1]")
    if latent_mean.ndim != 3 or observation_matrix.ndim != 3:
        raise ValueError("latent_mean and observation_matrix must be batched")
    batch, n_rx, latent_dim = latent_mean.shape
    if observation_matrix.shape[0] != batch or observation_matrix.shape[2] != latent_dim:
        raise ValueError("observation matrix shape mismatch")
    n_obs = int(observation_matrix.shape[1])
    if observations.shape != (batch, n_rx, n_obs):
        raise ValueError("observations shape mismatch")
    if observation_noise.shape != (batch, n_obs):
        raise ValueError("observation_noise shape mismatch")
    if latent_cov.ndim == 2:
        covariance = latent_cov.unsqueeze(0).expand(batch, -1, -1)
    elif latent_cov.ndim == 3 and latent_cov.shape[0] == batch:
        covariance = latent_cov
    else:
        raise ValueError("latent_cov must be [L,L] or [B,L,L]")
    covariance = covariance.to(latent_mean.device)
    if rho == 0.0:
        return latent_mean, covariance.clone()

    matrix = observation_matrix.to(latent_mean.dtype)
    effective_noise = observation_noise.real.clamp_min(1e-8) / rho
    a_cov = matrix @ covariance
    innovation_cov = a_cov @ matrix.conj().transpose(-1, -2)
    diagonal = torch.diag_embed(effective_noise).to(innovation_cov.dtype)
    innovation_cov = innovation_cov + diagonal
    scale = torch.diagonal(innovation_cov, dim1=-2, dim2=-1).real.mean(-1)
    eye = torch.eye(n_obs, dtype=innovation_cov.dtype, device=innovation_cov.device)
    innovation_cov = 0.5 * (
        innovation_cov + innovation_cov.conj().transpose(-1, -2)
    ) + (float(jitter) * scale.clamp_min(1.0))[..., None, None] * eye
    chol = torch.linalg.cholesky(innovation_cov)

    cross = covariance @ matrix.conj().transpose(-1, -2)  # [B,L,K]
    gain = torch.cholesky_solve(
        cross.conj().transpose(-1, -2), chol
    ).conj().transpose(-1, -2)
    prediction = torch.einsum("bkl,brl->brk", matrix, latent_mean)
    innovation = observations - prediction
    updated_mean = latent_mean + torch.einsum(
        "blk,brk->brl", gain, innovation
    )
    updated_cov = covariance - gain @ a_cov
    updated_cov = 0.5 * (
        updated_cov + updated_cov.conj().transpose(-1, -2)
    )
    return updated_mean, updated_cov


def _batch_noise_grid(
    noise_var: torch.Tensor,
    *,
    batch: int,
    n_data: int,
    device: torch.device,
) -> torch.Tensor:
    value = torch.as_tensor(noise_var, device=device).real.float()
    if value.ndim == 0:
        return value.expand(batch, n_data)
    if value.ndim == 1 and value.numel() == batch:
        return value[:, None].expand(batch, n_data)
    if value.ndim == 2 and value.shape == (batch, n_data):
        return value
    return value.mean().expand(batch, n_data)


def soft_data_posterior_update(
    state: LatentPosteriorState,
    *,
    y: torch.Tensor,
    data_idx: torch.Tensor,
    noise_var: torch.Tensor,
    symbol_mean: torch.Tensor,
    symbol_var: torch.Tensor,
    information_damping: float,
    data_fraction: float,
    min_observations: int,
    max_observations: int,
    jitter: float = 1e-6,
) -> tuple[PosteriorOutput, TurboUpdateDiagnostics, torch.Tensor]:
    """Apply one fractional, moment-matched soft-data posterior update."""
    initial = state.posterior
    if symbol_mean.shape != symbol_var.shape or symbol_mean.ndim != 3:
        raise ValueError("symbol_mean/symbol_var must have shape [B,N,D]")
    batch, n_layers, n_data = symbol_mean.shape
    if int(n_layers) != int(state.latent_mean.shape[2]):
        raise ValueError("symbol stream count disagrees with latent posterior")
    if int(data_idx.numel()) != int(n_data):
        raise ValueError("symbol data dimension disagrees with data_idx")
    n_rx = int(y.shape[1])
    rank = int(state.latent_mean.shape[-1])
    selected, selected_variance = select_reliable_data_indices(
        symbol_var,
        fraction=float(data_fraction),
        min_observations=int(min_observations),
        max_observations=int(max_observations),
    )
    count = int(selected.shape[1])
    data_idx = torch.as_tensor(data_idx, dtype=torch.long, device=y.device)
    features_data = state.features[data_idx]
    selected_features = features_data[selected]  # [B,K,Q]
    gather_symbols = selected[:, None, :].expand(batch, n_layers, count)
    x_mean = torch.gather(symbol_mean, 2, gather_symbols)
    x_var = torch.gather(symbol_var.real, 2, gather_symbols)
    observation_matrix = torch.einsum(
        "bnk,bkq->bknq", x_mean, selected_features
    ).reshape(batch, count, n_layers * rank)

    y_data = y[..., data_idx]
    y_selected = torch.gather(
        y_data,
        2,
        selected[:, None, :].expand(batch, n_rx, count),
    )
    posterior_mean_data = initial.mean[..., data_idx]
    mean_power = posterior_mean_data.abs().square().mean(dim=2)
    if initial.var_diag.ndim == 2:
        marginal_variance = initial.var_diag[None, :, data_idx].expand(batch, -1, -1)
    elif initial.var_diag.ndim == 3:
        marginal_variance = initial.var_diag[..., data_idx]
    else:
        raise ValueError("posterior var_diag shape is unsupported")
    channel_second_moment = mean_power + marginal_variance.real.clamp_min(0.0)
    selected_second_moment = torch.gather(
        channel_second_moment,
        2,
        gather_symbols,
    )
    data_noise = _batch_noise_grid(
        noise_var,
        batch=batch,
        n_data=n_data,
        device=y.device,
    )
    selected_noise = torch.gather(data_noise, 1, selected)
    pseudo_noise = (
        selected_noise
        + torch.sum(x_var * selected_second_moment, dim=1)
    ).real.clamp_min(1e-7)

    latent_mean_flat = state.latent_mean.reshape(batch, n_rx, n_layers * rank)
    updated_mean_flat, updated_cov = fractional_gaussian_condition(
        latent_mean_flat,
        state.latent_cov,
        observation_matrix,
        y_selected,
        pseudo_noise,
        information_damping=float(information_damping),
        jitter=float(jitter),
    )
    updated_latent_mean = updated_mean_flat.reshape(
        batch, n_rx, n_layers, rank
    )
    channel_mean = torch.einsum(
        "rq,bxnq->bnxr", state.features, updated_latent_mean
    ).contiguous()
    local_cov, var_diag = project_batched_latent_covariance(
        state.features,
        updated_cov,
        n_layers,
        rank,
    )
    refined = PosteriorOutput(
        mean=channel_mean,
        var_diag=var_diag,
        local_cov=local_cov,
        latent_cov=updated_cov,
        effective_noise=initial.effective_noise,
    )

    trace_before = torch.diagonal(state.latent_cov).real.sum()
    trace_after = torch.diagonal(updated_cov, dim1=-2, dim2=-1).real.sum(-1).mean()
    hermitian = torch.max(
        torch.abs(updated_cov - updated_cov.conj().transpose(-1, -2))
    )
    finite = bool(
        torch.isfinite(channel_mean).all().item()
        and torch.isfinite(local_cov).all().item()
        and torch.isfinite(updated_cov).all().item()
        and torch.isfinite(updated_latent_mean).all().item()
    )
    diagnostics = TurboUpdateDiagnostics(
        version=TURBO_POSTERIOR_VERSION,
        selected_observations=count,
        data_observations=n_data,
        information_damping=float(information_damping),
        selected_symbol_variance_mean=float(selected_variance.mean().item()),
        pseudo_noise_mean=float(pseudo_noise.mean().item()),
        latent_trace_before=float(trace_before.item()),
        latent_trace_after=float(trace_after.item()),
        latent_trace_reduction_fraction=float(
            ((trace_before - trace_after) / trace_before.clamp_min(1e-8)).item()
        ),
        max_hermitian_error=float(hermitian.item()),
        finite=finite,
    )
    return refined, diagnostics, selected


def posterior_batch_metrics(
    posterior: PosteriorOutput,
    truth: torch.Tensor,
    data_idx: torch.Tensor,
) -> dict[str, float]:
    """Channel metrics supporting both shared and batch-dependent variance."""
    mean = posterior.mean[..., data_idx]
    h = truth[..., data_idx]
    if posterior.var_diag.ndim == 2:
        var = posterior.var_diag[None, :, None, data_idx].to(mean.device)
    elif posterior.var_diag.ndim == 3:
        var = posterior.var_diag[:, :, None, data_idx].to(mean.device)
    else:
        raise ValueError("posterior var_diag shape is unsupported")
    error = torch.abs(mean - h).square()
    nmse = error.mean() / torch.abs(h).square().mean().clamp_min(1e-8)
    normalized = error / var.clamp_min(1e-8)
    threshold = -math.log(0.05) * var
    return {
        "channel_nmse": float(nmse.real.item()),
        "normalized_error_mean": float(normalized.real.mean().item()),
        "coverage95": float((error <= threshold).float().mean().item()),
    }


def mathematical_self_test(device: torch.device | str = "cpu") -> dict[str, Any]:
    dev = torch.device(device)
    torch.manual_seed(9317)
    batch, n_rx, latent_dim, n_obs = 2, 3, 8, 5
    raw = (
        torch.randn(latent_dim, latent_dim, device=dev)
        + 1j * torch.randn(latent_dim, latent_dim, device=dev)
    ) / math.sqrt(2.0 * latent_dim)
    cov = raw @ raw.conj().T + 0.4 * torch.eye(
        latent_dim, dtype=torch.complex64, device=dev
    )
    mean = (
        torch.randn(batch, n_rx, latent_dim, device=dev)
        + 1j * torch.randn(batch, n_rx, latent_dim, device=dev)
    ) / math.sqrt(2.0)
    matrix = (
        torch.randn(batch, n_obs, latent_dim, device=dev)
        + 1j * torch.randn(batch, n_obs, latent_dim, device=dev)
    ) / math.sqrt(2.0 * latent_dim)
    observations = (
        torch.randn(batch, n_rx, n_obs, device=dev)
        + 1j * torch.randn(batch, n_rx, n_obs, device=dev)
    ) / math.sqrt(2.0)
    noise = torch.full((batch, n_obs), 0.7, device=dev)
    rho = 0.35
    updated_mean, updated_cov = fractional_gaussian_condition(
        mean, cov, matrix, observations, noise,
        information_damping=rho,
    )

    direct_mean: list[torch.Tensor] = []
    direct_cov: list[torch.Tensor] = []
    precision0 = torch.linalg.inv(cov)
    for b in range(batch):
        a = matrix[b]
        r_inv = torch.diag(rho / noise[b]).to(torch.complex64)
        precision = precision0 + a.conj().T @ r_inv @ a
        covariance_b = torch.linalg.inv(precision)
        eta0 = torch.einsum("lm,rm->rl", precision0, mean[b])
        eta = eta0 + torch.einsum(
            "lk,kr->rl", a.conj().T @ r_inv, observations[b].T
        )
        mean_b = torch.einsum("lm,rm->rl", covariance_b, eta)
        direct_mean.append(mean_b)
        direct_cov.append(covariance_b)
    direct_mean_t = torch.stack(direct_mean)
    direct_cov_t = torch.stack(direct_cov)
    mean_error = float(torch.max(torch.abs(updated_mean - direct_mean_t)).item())
    cov_error = float(torch.max(torch.abs(updated_cov - direct_cov_t)).item())

    unchanged_mean, unchanged_cov = fractional_gaussian_condition(
        mean, cov, matrix, observations, noise,
        information_damping=0.0,
    )
    zero_mean_error = float(torch.max(torch.abs(unchanged_mean - mean)).item())
    zero_cov_error = float(torch.max(torch.abs(unchanged_cov - cov)).item())
    eig = torch.linalg.eigvalsh(updated_cov.to(torch.complex128)).real
    trace_before = torch.diagonal(cov).real.sum()
    trace_after = torch.diagonal(updated_cov, dim1=-2, dim2=-1).real.sum(-1)

    symbol_var = torch.tensor(
        [[[0.8, 0.1, 0.4, 0.2], [0.9, 0.2, 0.5, 0.3]]],
        device=dev,
    )
    selected, values = select_reliable_data_indices(
        symbol_var,
        fraction=0.5,
        min_observations=1,
        max_observations=4,
    )
    selection_exact = selected.tolist() == [[1, 3]]

    features = (
        torch.randn(6, 4, device=dev)
        + 1j * torch.randn(6, 4, device=dev)
    ) / math.sqrt(8.0)
    small_cov = torch.eye(8, dtype=torch.complex64, device=dev).unsqueeze(0)
    local, variance = project_batched_latent_covariance(
        features, small_cov, 2, 4
    )
    projection_ok = local.shape == (1, 2, 2, 6) and variance.shape == (1, 2, 6)

    checks = {
        "fractional_mean_matches_information_form": mean_error < 2e-4,
        "fractional_covariance_matches_information_form": cov_error < 2e-4,
        "zero_information_returns_same_mean": zero_mean_error < 1e-7,
        "zero_information_returns_same_covariance": zero_cov_error < 1e-7,
        "posterior_covariance_psd": float(eig.min().item()) > -2e-7,
        "posterior_trace_not_increased": bool(
            torch.all(trace_after <= trace_before + 2e-5).item()
        ),
        "reliability_selection_exact": selection_exact and bool(values[0, 0] <= values[0, 1]),
        "batched_projection_shapes": projection_ok,
        "finite": bool(
            torch.isfinite(updated_mean).all().item()
            and torch.isfinite(updated_cov).all().item()
        ),
    }
    return {
        "version": TURBO_POSTERIOR_VERSION,
        "passed": all(checks.values()),
        "checks": checks,
        "mean_max_abs_error": mean_error,
        "covariance_max_abs_error": cov_error,
        "minimum_eigenvalue": float(eig.min().item()),
        "maximum_trace_change": float((trace_after - trace_before).max().item()),
        "selected_indices": selected.detach().cpu().tolist(),
    }
