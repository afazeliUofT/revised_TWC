from __future__ import annotations

from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F

from .channels import rff_bank
from .qam import (
    make_qam_constellation,
    symbol_logits_to_bit_logits,
    symbol_probs_to_mean_var,
)


@dataclass
class PosteriorOutput:
    mean: torch.Tensor       # [B,N,RX,R]
    var_diag: torch.Tensor   # [N,R], real marginal variance per RX antenna
    local_cov: torch.Tensor  # [N,N,R], Cov(h_n[r],h_m[r]) per RX antenna
    latent_cov: torch.Tensor # [NQ,NQ]
    effective_noise: torch.Tensor


def project_latent_covariance_to_grid(
    features: torch.Tensor,
    latent_cov: torch.Tensor,
    n_layers: int,
    rank: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Project latent covariance to local layer covariance on every RE.

    The channel representation is h_n[r] = u[r]^T z_n. Therefore
    Cov(h_n[r],h_m[r]) = u[r]^T C_nm u[r]^*, not u[r]^H C_nm u[r].
    The distinction is essential for complex Fourier features.
    """
    rows: list[torch.Tensor] = []
    for n in range(int(n_layers)):
        row: list[torch.Tensor] = []
        for m in range(int(n_layers)):
            block = latent_cov[
                n * rank:(n + 1) * rank,
                m * rank:(m + 1) * rank,
            ]
            cov_nm = torch.einsum(
                "rq,qs,rs->r",
                features,
                block,
                features.conj(),
            )
            row.append(cov_nm)
        rows.append(torch.stack(row, dim=0))
    local_cov = torch.stack(rows, dim=0)
    local_cov = 0.5 * (local_cov + local_cov.transpose(0, 1).conj())
    var_diag = torch.stack(
        [local_cov[n, n].real.clamp_min(1e-8) for n in range(int(n_layers))],
        dim=0,
    )
    return local_cov, var_diag


class LowRankPosteriorOperator(nn.Module):
    """PSD low-rank channel prior followed by exact Gaussian conditioning."""

    def __init__(
        self,
        coords: torch.Tensor,
        pilot_idx: torch.Tensor,
        n_layers: int,
        rank: int,
        length_f: float = 6.0,
        length_t: float = 2.0,
        seed: int = 1011,
        bank_rank: int | None = None,
    ):
        super().__init__()
        self.n_layers = int(n_layers)
        self.rank = int(rank)
        self.bank_rank = self.rank if bank_rank is None else int(bank_rank)
        if self.bank_rank < self.rank:
            raise ValueError("operator bank_rank must be at least the active rank")
        self.operator_seed = int(seed)
        self.register_buffer("coords", coords.detach().clone())
        self.register_buffer("pilot_idx", pilot_idx.detach().clone())
        base = rff_bank(
            coords.detach(),
            self.rank,
            length_f,
            length_t,
            seed=self.operator_seed,
            bank_rank=self.bank_rank,
        )
        self.register_buffer("base_features", base)
        # Nonnegative spectral weights guarantee K=B B^H is PSD.
        self.raw_weights = nn.Parameter(torch.zeros(self.rank))
        self.log_noise_scale = nn.Parameter(torch.tensor(0.0))

    def weighted_features(self) -> torch.Tensor:
        weights = F.softplus(self.raw_weights) + 1e-4
        weights = weights / torch.sqrt(torch.mean(weights ** 2).clamp_min(1e-8))
        return self.base_features * weights.to(self.base_features.dtype).view(1, -1)

    def _observation_basis(self, phi: torch.Tensor) -> torch.Tensor:
        """Build the pilot observation operator A B with shape [P,NQ]."""
        u_p = self.weighted_features()[self.pilot_idx, :]
        p = u_p.shape[0]
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
        n_latent = observation_basis.shape[1]
        sigma2 = (
            noise_var * torch.exp(self.log_noise_scale).clamp(0.25, 4.0)
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

    def channel_nmse(
        self,
        posterior: PosteriorOutput,
        h_true: torch.Tensor,
        idx: torch.Tensor | None = None,
    ) -> torch.Tensor:
        mu = posterior.mean if idx is None else posterior.mean[..., idx]
        h = h_true if idx is None else h_true[..., idx]
        return (
            torch.mean(torch.abs(mu - h) ** 2)
            / torch.mean(torch.abs(h) ** 2).clamp_min(1e-8)
        ).real


def coupling_matrix(
    h_mean: torch.Tensor,
    local_cov: torch.Tensor,
    data_idx: torch.Tensor,
    noise_var: torch.Tensor,
) -> torch.Tensor:
    """Posterior expected squared off-diagonal whitened Gram coupling.

    ``local_cov[n,m,r]`` is Cov(h_n[r],h_m[r]) for one receive antenna.
    Receive-antenna posterior errors are independent and identically distributed
    in Gate 0. The cross-layer covariance term is retained exactly for this model.
    """
    mu = h_mean[..., data_idx]
    cov = local_cov[..., data_idx].to(mu.device)
    var = torch.stack(
        [cov[n, n].real.clamp_min(0.0) for n in range(cov.shape[0])], dim=0
    )
    batch_size, n_layers, n_rx, n_data = mu.shape
    inv_noise = 1.0 / noise_var.clamp_min(1e-6)
    inv_noise_sq = inv_noise ** 2
    coupling = torch.zeros(
        (batch_size, n_data, n_layers, n_layers),
        dtype=torch.float32,
        device=mu.device,
    )
    for n in range(n_layers):
        for m in range(n + 1, n_layers):
            # E[e_n^H W e_m] = tr(W Sigma_mn), hence the reversed [m,n] index.
            cross_trace = float(n_rx) * cov[m, n].view(1, n_data) * inv_noise
            coherent = (
                torch.sum(mu[:, n].conj() * mu[:, m], dim=1) * inv_noise
                + cross_trace
            )
            term0 = torch.abs(coherent) ** 2
            term1 = torch.sum(
                torch.abs(mu[:, n]) ** 2
                * var[m].view(1, 1, n_data)
                * inv_noise_sq,
                dim=1,
            )
            term2 = torch.sum(
                torch.abs(mu[:, m]) ** 2
                * var[n].view(1, 1, n_data)
                * inv_noise_sq,
                dim=1,
            )
            term3 = (
                float(n_rx)
                * var[n].view(1, n_data)
                * var[m].view(1, n_data)
                * inv_noise_sq
            )
            value = (term0 + term1 + term2 + term3).real.clamp_min(0.0)
            coupling[:, :, n, m] = value
            coupling[:, :, m, n] = value
    return coupling


def coupling_selection_mask(kappa: torch.Tensor, edge_mass: float) -> torch.Tensor:
    """Keep the strongest interferers until the requested coupling mass is covered."""
    if kappa.ndim != 4 or kappa.shape[-1] != kappa.shape[-2]:
        raise ValueError("kappa must have shape [B,D,N,N].")
    mass = float(max(0.0, min(1.0, edge_mass)))
    scores = kappa.detach().real.clamp_min(0.0)
    n_layers = scores.shape[-1]
    eye = torch.eye(
        n_layers, dtype=torch.bool, device=scores.device
    ).view(1, 1, n_layers, n_layers)
    scores = scores.masked_fill(eye, 0.0)
    values, indices = torch.sort(scores, dim=-1, descending=True)
    total = values.sum(dim=-1, keepdim=True)
    previous = torch.cumsum(values, dim=-1) - values
    keep_sorted = (values > 0.0) & (previous < mass * total)
    if mass <= 0.0:
        keep_sorted = torch.zeros_like(keep_sorted)
    mask = torch.zeros_like(keep_sorted, dtype=torch.bool)
    mask.scatter_(-1, indices, keep_sorted)
    return mask & (~eye)


def edge_density(mask: torch.Tensor) -> torch.Tensor:
    """Return graph density without forcing a device synchronization."""
    n_layers = int(mask.shape[-1])
    denominator = (
        mask.shape[0] * mask.shape[1] * n_layers * max(n_layers - 1, 1)
    )
    return mask.float().sum() / max(float(denominator), 1.0)


def diagonal_interference_moments(
    mu: torch.Tensor,
    local_cov: torch.Tensor,
    x_mean: torch.Tensor,
    x_var: torch.Tensor,
    strong: torch.Tensor,
    target_layer: int,
    noise_var: torch.Tensor,
    use_uncertainty: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return moments needed by one target-layer symbol update.

    ``local_cov[i,j,d]`` is :math:`E[e_i e_j^*]` for one receive
    antenna. For a soft-cancelled interferer ``i``, the mean-channel
    residual variance is ``v_i |mu_i|^2``. For an omitted weak interferer,
    its complete second moment multiplies ``|mu_i|^2``. Channel-error
    uncertainty uses the full cross-layer posterior covariance:

    ``sum_ij E[x_i x_j^*] local_cov[i,j]``.

    The target channel error is added later for each candidate symbol. The
    returned cross coefficient ``q`` gives the candidate-dependent term
    ``2 Re{a q}``.
    """
    n_layers = mu.shape[1]
    target = int(target_layer)
    if target < 0 or target >= n_layers:
        raise IndexError("target_layer is outside the layer range")
    if local_cov.shape[:2] != (n_layers, n_layers):
        raise ValueError("local_cov must have shape [N,N,D]")

    strong = strong.bool().clone()
    strong[:, target, :] = False
    other = torch.ones_like(strong, dtype=torch.bool)
    other[:, target, :] = False
    weak = other & (~strong)

    strong_f = strong[:, :, None, :].to(mu.real.dtype)
    weak_f = weak[:, :, None, :].to(mu.real.dtype)
    other_f = other.to(mu.real.dtype)
    symbol_second = x_var + torch.abs(x_mean) ** 2

    interference_mean = torch.sum(
        mu * x_mean[:, :, None, :] * strong_f,
        dim=1,
    )
    mean_channel_variance = torch.sum(
        (
            x_var[:, :, None, :] * strong_f
            + symbol_second[:, :, None, :] * weak_f
        )
        * torch.abs(mu) ** 2,
        dim=1,
    )

    batch_size, _, _, n_data = mu.shape
    if use_uncertainty:
        covariance = local_cov.to(mu.device)
        means_other = x_mean * other_f
        # m^T C m^* plus the independent symbol-variance contribution.
        mean_quadratic = torch.einsum(
            "bid,ijd,bjd->bd",
            means_other,
            covariance,
            means_other.conj(),
        ).real
        diagonal_cov = torch.stack(
            [
                covariance[i, i].real.clamp_min(0.0)
                for i in range(n_layers)
            ],
            dim=0,
        )
        symbol_variance_term = torch.sum(
            x_var * other_f * diagonal_cov[None, :, :],
            dim=1,
        )
        channel_error_variance = (
            mean_quadratic + symbol_variance_term
        ).view(batch_size, 1, n_data)

        # q_n = sum_{m != n} C_nm E[x_m]^*. The candidate contribution is
        # |a|^2 C_nn + 2 Re{a q_n}.
        target_cross = torch.einsum(
            "md,bmd->bd",
            covariance[target],
            means_other.conj(),
        )
        target_variance = covariance[target, target].real.clamp_min(0.0)
    else:
        channel_error_variance = torch.zeros_like(mean_channel_variance)
        target_cross = torch.zeros(
            (batch_size, n_data), dtype=mu.dtype, device=mu.device
        )
        target_variance = torch.zeros(
            n_data, dtype=mu.real.dtype, device=mu.device
        )

    variance_without_target = (
        noise_var + mean_channel_variance + channel_error_variance
    ).real.clamp_min(1e-8)
    return (
        interference_mean,
        variance_without_target,
        target_cross,
        target_variance,
    )


class BayesRouteDetector(nn.Module):
    """Uncertainty-aware soft detector with coupling-adaptive cancellation.

    Strong graph neighbors are soft-cancelled. Omitted weak neighbors are kept
    as zero-mean Gaussian interference using their current second moments. Gate 0
    uses a diagonal receive-antenna covariance approximation.
    """

    def __init__(
        self,
        bits_per_symbol: int,
        n_iter: int = 3,
        use_uncertainty: bool = True,
    ):
        super().__init__()
        self.bits_per_symbol = int(bits_per_symbol)
        self.n_iter = int(n_iter)
        self.default_use_uncertainty = bool(use_uncertainty)

    def forward(
        self,
        y: torch.Tensor,
        h_mean: torch.Tensor,
        h_cov: torch.Tensor,
        data_idx: torch.Tensor,
        noise_var: torch.Tensor,
        kappa: torch.Tensor | None = None,
        edge_mass: float = 1.0,
        use_uncertainty: bool | None = None,
    ):
        device = y.device
        use_unc = (
            self.default_use_uncertainty
            if use_uncertainty is None
            else bool(use_uncertainty)
        )
        constellation, bit_table = make_qam_constellation(
            self.bits_per_symbol, device=device
        )
        y_data = y[..., data_idx]
        mu = h_mean[..., data_idx]
        local_cov = h_cov[..., data_idx].to(device)
        batch_size, n_layers, _, n_data = mu.shape
        x_mean = torch.zeros(
            (batch_size, n_layers, n_data),
            dtype=torch.complex64,
            device=device,
        )
        x_var = torch.ones(
            (batch_size, n_layers, n_data),
            dtype=torch.float32,
            device=device,
        )

        if kappa is None:
            graph_mask = torch.ones(
                (batch_size, n_data, n_layers, n_layers),
                dtype=torch.bool,
                device=device,
            )
            diagonal = torch.eye(
                n_layers, dtype=torch.bool, device=device
            ).view(1, 1, n_layers, n_layers)
            graph_mask = graph_mask & (~diagonal)
        else:
            graph_mask = coupling_selection_mask(kappa, edge_mass)

        symbol_logits = None
        for _ in range(self.n_iter):
            logits_by_layer = []
            for n in range(n_layers):
                strong = graph_mask[:, :, n, :].permute(0, 2, 1)
                (
                    mean_without_n,
                    var_without_n,
                    target_cross,
                    target_variance,
                ) = diagonal_interference_moments(
                    mu,
                    local_cov,
                    x_mean,
                    x_var,
                    strong,
                    n,
                    noise_var,
                    use_unc,
                )
                mu_n = mu[:, n]
                candidate_logits = []
                for symbol in constellation:
                    residual = y_data - mean_without_n - mu_n * symbol
                    candidate_var = var_without_n
                    if use_unc:
                        candidate_var = (
                            candidate_var
                            + torch.abs(symbol) ** 2
                            * target_variance.view(1, 1, n_data)
                            + 2.0
                            * torch.real(symbol * target_cross).view(
                                batch_size, 1, n_data
                            )
                        )
                    candidate_var = candidate_var.real.clamp_min(1e-6)
                    log_likelihood = -torch.sum(
                        torch.abs(residual) ** 2 / candidate_var
                        + torch.log(candidate_var),
                        dim=1,
                    )
                    candidate_logits.append(log_likelihood)
                logits_by_layer.append(torch.stack(candidate_logits, dim=-1))
            symbol_logits = torch.stack(logits_by_layer, dim=1)
            probabilities = torch.softmax(symbol_logits, dim=-1)
            x_mean, x_var = symbol_probs_to_mean_var(
                probabilities, constellation
            )

        bit_logits = symbol_logits_to_bit_logits(symbol_logits, bit_table)
        return bit_logits, symbol_logits, x_mean, x_var, graph_mask


class BayesRouteReceiver(nn.Module):
    def __init__(self, cfg, coords: torch.Tensor, pilot_idx: torch.Tensor):
        super().__init__()
        self.cfg = cfg
        system = cfg.system
        model_cfg = cfg.model
        operator_seed = int(
            model_cfg.get("operator_seed", int(cfg.seed) + 1009)
        )
        rank = int(model_cfg.rank)
        bank_rank = int(model_cfg.get("operator_bank_rank", rank))
        self.posterior = LowRankPosteriorOperator(
            coords=coords,
            pilot_idx=pilot_idx,
            n_layers=int(system.n_layers),
            rank=rank,
            length_f=float(system.channel_length_f),
            length_t=float(system.channel_length_t),
            seed=operator_seed,
            bank_rank=bank_rank,
        )
        self.detector = BayesRouteDetector(
            bits_per_symbol=int(system.bits_per_symbol),
            n_iter=int(model_cfg.detector_iterations),
            use_uncertainty=bool(model_cfg.get("use_uncertainty", True)),
        )

    def forward(
        self,
        batch,
        use_uncertainty: bool | None = None,
        edge_mass: float | None = None,
    ):
        pilot_observation = batch.y[..., batch.pilot_idx]
        posterior = self.posterior(
            pilot_observation, batch.phi, batch.noise_var
        )
        chosen_uncertainty = (
            self.detector.default_use_uncertainty
            if use_uncertainty is None
            else bool(use_uncertainty)
        )
        # A covariance-off ablation removes posterior covariance from both the
        # detector metric and the graph construction. The graph is hard-selected,
        # so its inputs are detached to avoid retaining an unused autograd graph.
        graph_covariance = (
            posterior.local_cov
            if chosen_uncertainty
            else torch.zeros_like(posterior.local_cov)
        )
        kappa = coupling_matrix(
            posterior.mean.detach(),
            graph_covariance.detach(),
            batch.data_idx,
            batch.noise_var.detach(),
        )
        chosen_mass = float(
            self.cfg.model.get("edge_mass", 1.0)
            if edge_mass is None
            else edge_mass
        )
        bit_logits, symbol_logits, x_mean, x_var, graph_mask = self.detector(
            batch.y,
            posterior.mean,
            posterior.local_cov,
            batch.data_idx,
            batch.noise_var,
            kappa=kappa,
            edge_mass=chosen_mass,
            use_uncertainty=chosen_uncertainty,
        )
        return {
            "bit_logits": bit_logits,
            "symbol_logits": symbol_logits,
            "x_mean": x_mean,
            "x_var": x_var,
            "posterior": posterior,
            "kappa": kappa,
            "graph_mask": graph_mask,
            "edge_density": edge_density(graph_mask),
            "edge_mass": chosen_mass,
        }


class LSReceiver(nn.Module):
    """Constant-channel orthogonal-pilot baseline for Gate 0 only."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.detector = BayesRouteDetector(
            int(cfg.system.bits_per_symbol),
            n_iter=int(cfg.model.detector_iterations),
            use_uncertainty=False,
        )

    def forward(self, batch):
        y_pilot = batch.y[..., batch.pilot_idx]
        phi = batch.phi
        n_pilots = phi.shape[-1]
        channel_average = torch.einsum(
            "np,bxp->bnx", phi.conj(), y_pilot
        ) / float(n_pilots)
        channel_mean = channel_average[..., None].expand(
            -1, -1, -1, batch.y.shape[-1]
        ).contiguous()
        channel_cov = torch.zeros(
            (phi.shape[0], phi.shape[0], batch.y.shape[-1]),
            dtype=torch.complex64,
            device=batch.y.device,
        )
        bit_logits, symbol_logits, x_mean, x_var, graph_mask = self.detector(
            batch.y,
            channel_mean,
            channel_cov,
            batch.data_idx,
            batch.noise_var,
            kappa=None,
            edge_mass=1.0,
            use_uncertainty=False,
        )
        return {
            "bit_logits": bit_logits,
            "symbol_logits": symbol_logits,
            "x_mean": x_mean,
            "x_var": x_var,
            "posterior": None,
            "kappa": None,
            "graph_mask": graph_mask,
            "edge_density": edge_density(graph_mask),
        }


class OracleReceiver(nn.Module):
    """Perfect-CSI soft-cancellation reference, not an exact ML detector."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.detector = BayesRouteDetector(
            int(cfg.system.bits_per_symbol),
            n_iter=int(cfg.model.detector_iterations),
            use_uncertainty=False,
        )

    def forward(self, batch):
        channel_cov = torch.zeros(
            (batch.h.shape[1], batch.h.shape[1], batch.h.shape[-1]),
            dtype=torch.complex64,
            device=batch.y.device,
        )
        bit_logits, symbol_logits, x_mean, x_var, graph_mask = self.detector(
            batch.y,
            batch.h,
            channel_cov,
            batch.data_idx,
            batch.noise_var,
            kappa=None,
            edge_mass=1.0,
            use_uncertainty=False,
        )
        return {
            "bit_logits": bit_logits,
            "symbol_logits": symbol_logits,
            "x_mean": x_mean,
            "x_var": x_var,
            "posterior": None,
            "kappa": None,
            "graph_mask": graph_mask,
            "edge_density": edge_density(graph_mask),
        }
