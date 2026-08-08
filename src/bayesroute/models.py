from __future__ import annotations
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
from .channels import rff_bank
from .qam import make_qam_constellation, symbol_logits_to_bit_logits, symbol_probs_to_mean_var


@dataclass
class PosteriorOutput:
    mean: torch.Tensor       # [B, N, RX, R]
    var_diag: torch.Tensor   # [N, R]
    latent_cov: torch.Tensor # [NQ, NQ]
    effective_noise: torch.Tensor


class LowRankPosteriorOperator(nn.Module):
    """PSD low-rank channel prior followed by exact Gaussian conditioning."""

    def __init__(self, coords: torch.Tensor, pilot_idx: torch.Tensor, n_layers: int,
                 rank: int, length_f: float = 6.0, length_t: float = 2.0,
                 seed: int = 1011):
        super().__init__()
        self.n_layers = int(n_layers)
        self.rank = int(rank)
        self.operator_seed = int(seed)
        self.register_buffer("coords", coords.detach().clone())
        self.register_buffer("pilot_idx", pilot_idx.detach().clone())
        base = rff_bank(coords.detach(), rank, length_f, length_t, seed=self.operator_seed)
        self.register_buffer("base_features", base)
        # Nonnegative spectral weights guarantee K=B B^H is PSD.
        self.raw_weights = nn.Parameter(torch.zeros(rank))
        self.log_noise_scale = nn.Parameter(torch.tensor(0.0))

    def weighted_features(self) -> torch.Tensor:
        w = F.softplus(self.raw_weights) + 1e-4
        w = w / torch.sqrt(torch.mean(w ** 2).clamp_min(1e-8))
        return self.base_features * w.to(self.base_features.dtype).view(1, -1)

    def _observation_basis(self, phi: torch.Tensor) -> torch.Tensor:
        """Build the pilot observation operator A B with shape [P, NQ]."""
        u_p = self.weighted_features()[self.pilot_idx, :]  # [P,Q]
        p = u_p.shape[0]
        blocks = [phi[n, :].view(p, 1) * u_p for n in range(self.n_layers)]
        return torch.cat(blocks, dim=1)

    def forward(self, y_p: torch.Tensor, phi: torch.Tensor,
                noise_var: torch.Tensor) -> PosteriorOutput:
        bsz, n_rx, p = y_p.shape
        device = y_p.device
        ab = self._observation_basis(phi).to(device)
        nq = ab.shape[1]
        sigma2 = noise_var * torch.exp(self.log_noise_scale).clamp(0.25, 4.0) + 1e-6
        eye_p = torch.eye(p, dtype=torch.complex64, device=device)
        c = ab @ ab.conj().transpose(0, 1) + sigma2.to(torch.complex64) * eye_p
        chol = torch.linalg.cholesky(c + 1e-5 * eye_p)

        y_flat = y_p.reshape(bsz * n_rx, p).T.contiguous()
        alpha = torch.cholesky_solve(y_flat, chol)
        latent_mu = ab.conj().transpose(0, 1) @ alpha
        latent_mu = latent_mu.T.reshape(bsz, n_rx, self.n_layers, self.rank)
        u = self.weighted_features().to(device)
        h_mu = torch.einsum("rq,bxnq->bnxr", u, latent_mu).contiguous()

        cinv_ab = torch.cholesky_solve(ab, chol)
        latent_cov = (
            torch.eye(nq, dtype=torch.complex64, device=device)
            - ab.conj().transpose(0, 1) @ cinv_ab
        )
        latent_cov = 0.5 * (latent_cov + latent_cov.conj().transpose(0, 1))

        vars_layers = []
        for n in range(self.n_layers):
            block = latent_cov[
                n * self.rank:(n + 1) * self.rank,
                n * self.rank:(n + 1) * self.rank,
            ]
            v = torch.einsum("rq,qs,rs->r", u.conj(), block, u).real.clamp_min(1e-8)
            vars_layers.append(v)
        var_diag = torch.stack(vars_layers, dim=0)
        return PosteriorOutput(
            mean=h_mu,
            var_diag=var_diag,
            latent_cov=latent_cov,
            effective_noise=sigma2,
        )

    def channel_nmse(self, posterior: PosteriorOutput, h_true: torch.Tensor,
                     idx: torch.Tensor | None = None) -> torch.Tensor:
        mu = posterior.mean if idx is None else posterior.mean[..., idx]
        h = h_true if idx is None else h_true[..., idx]
        return (
            torch.mean(torch.abs(mu - h) ** 2)
            / torch.mean(torch.abs(h) ** 2).clamp_min(1e-8)
        ).real


def coupling_matrix(h_mean: torch.Tensor, h_var: torch.Tensor,
                    data_idx: torch.Tensor, noise_var: torch.Tensor) -> torch.Tensor:
    """Posterior expected squared off-diagonal whitened Gram coupling.

    Returns [B,D,N,N]. The diagonal is zero. The Gate-0 implementation uses
    white effective noise and diagonal receive-antenna channel uncertainty.
    Pairwise values are computed once and mirrored because the coupling is
    theoretically symmetric.
    """
    mu = h_mean[..., data_idx]
    var = h_var[:, data_idx].to(mu.device)
    bsz, n_layers, n_rx, d = mu.shape
    inv_noise = 1.0 / noise_var.clamp_min(1e-6)
    k = torch.zeros((bsz, d, n_layers, n_layers), dtype=torch.float32, device=mu.device)
    for n in range(n_layers):
        for m in range(n + 1, n_layers):
            coherent = torch.sum(mu[:, n].conj() * mu[:, m] * inv_noise, dim=1)
            term0 = torch.abs(coherent) ** 2
            term1 = torch.sum(
                torch.abs(mu[:, n]) ** 2
                * var[m].view(1, 1, d)
                * (inv_noise ** 2),
                dim=1,
            )
            term2 = torch.sum(
                torch.abs(mu[:, m]) ** 2
                * var[n].view(1, 1, d)
                * (inv_noise ** 2),
                dim=1,
            )
            term3 = (
                float(n_rx)
                * var[n].view(1, d)
                * var[m].view(1, d)
                * (inv_noise ** 2)
            )
            value = (term0 + term1 + term2 + term3).real
            k[:, :, n, m] = value
            k[:, :, m, n] = value
    return k


def coupling_selection_mask(kappa: torch.Tensor, edge_mass: float) -> torch.Tensor:
    """Keep the strongest interferers until the requested coupling mass is covered."""
    if kappa.ndim != 4 or kappa.shape[-1] != kappa.shape[-2]:
        raise ValueError("kappa must have shape [B,D,N,N].")
    mass = float(max(0.0, min(1.0, edge_mass)))
    scores = kappa.detach().real.clamp_min(0.0)
    n_layers = scores.shape[-1]
    eye = torch.eye(n_layers, dtype=torch.bool, device=scores.device).view(1, 1, n_layers, n_layers)
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


def edge_density(mask: torch.Tensor) -> float:
    n = int(mask.shape[-1])
    denom = mask.shape[0] * mask.shape[1] * n * max(n - 1, 1)
    return float(mask.float().sum().item() / max(float(denom), 1.0))


class BayesRouteDetector(nn.Module):
    """Uncertainty-aware soft detector with coupling-adaptive cancellation.

    Strong graph neighbors are soft-cancelled. Omitted weak neighbors are kept
    as zero-mean Gaussian interference using their current second moments.
    """

    def __init__(self, bits_per_symbol: int, n_iter: int = 3,
                 use_uncertainty: bool = True):
        super().__init__()
        self.bits_per_symbol = int(bits_per_symbol)
        self.n_iter = int(n_iter)
        self.default_use_uncertainty = bool(use_uncertainty)

    def forward(self, y: torch.Tensor, h_mean: torch.Tensor, h_var: torch.Tensor,
                data_idx: torch.Tensor, noise_var: torch.Tensor,
                kappa: torch.Tensor | None = None, edge_mass: float = 1.0,
                use_uncertainty: bool | None = None):
        device = y.device
        use_unc = self.default_use_uncertainty if use_uncertainty is None else bool(use_uncertainty)
        const, bit_table = make_qam_constellation(self.bits_per_symbol, device=device)
        y_d = y[..., data_idx]
        mu = h_mean[..., data_idx]
        var = h_var[:, data_idx].to(device)
        bsz, n_layers, _, d = mu.shape
        x_mean = torch.zeros((bsz, n_layers, d), dtype=torch.complex64, device=device)
        x_var = torch.ones((bsz, n_layers, d), dtype=torch.float32, device=device)

        if kappa is None:
            graph_mask = torch.ones(
                (bsz, d, n_layers, n_layers), dtype=torch.bool, device=device
            )
            diag = torch.eye(n_layers, dtype=torch.bool, device=device).view(1, 1, n_layers, n_layers)
            graph_mask = graph_mask & (~diag)
        else:
            graph_mask = coupling_selection_mask(kappa, edge_mass)

        symbol_logits = None
        for _ in range(self.n_iter):
            channel_power = torch.abs(mu) ** 2
            if use_unc:
                channel_power = channel_power + var[None, :, None, :]
            symbol_second = x_var + torch.abs(x_mean) ** 2
            logits_n = []
            for n in range(n_layers):
                # [B,N,D]: which interferers are treated by explicit soft cancellation.
                strong = graph_mask[:, :, n, :].permute(0, 2, 1)
                strong = strong.clone()
                strong[:, n, :] = False
                weak = ~strong
                weak[:, n, :] = False
                strong_f = strong[:, :, None, :].to(channel_power.dtype)
                weak_f = weak[:, :, None, :].to(channel_power.dtype)

                mean_without_n = torch.sum(
                    mu * x_mean[:, :, None, :] * strong_f, dim=1
                )
                variance_coeff = (
                    x_var[:, :, None, :] * strong_f
                    + symbol_second[:, :, None, :] * weak_f
                )
                var_without_n = noise_var + torch.sum(
                    variance_coeff * channel_power, dim=1
                )

                mu_n = mu[:, n]
                var_n = var[n].view(1, 1, d) if use_unc else 0.0
                cand_logits = []
                for a in const:
                    residual = y_d - mean_without_n - mu_n * a
                    cvar = (
                        var_without_n + (torch.abs(a) ** 2) * var_n
                    ).real.clamp_min(1e-6)
                    ll = -torch.sum(
                        (torch.abs(residual) ** 2) / cvar + torch.log(cvar), dim=1
                    )
                    cand_logits.append(ll)
                logits_n.append(torch.stack(cand_logits, dim=-1))
            symbol_logits = torch.stack(logits_n, dim=1)
            probs = torch.softmax(symbol_logits, dim=-1)
            x_mean, x_var = symbol_probs_to_mean_var(probs, const)

        bit_logits = symbol_logits_to_bit_logits(symbol_logits, bit_table)
        return bit_logits, symbol_logits, x_mean, x_var, graph_mask


class BayesRouteReceiver(nn.Module):
    def __init__(self, cfg, coords: torch.Tensor, pilot_idx: torch.Tensor):
        super().__init__()
        self.cfg = cfg
        sys_cfg = cfg.system
        mdl = cfg.model
        operator_seed = int(mdl.get("operator_seed", int(cfg.seed) + 1009))
        self.posterior = LowRankPosteriorOperator(
            coords=coords,
            pilot_idx=pilot_idx,
            n_layers=int(sys_cfg.n_layers),
            rank=int(mdl.rank),
            length_f=float(sys_cfg.channel_length_f),
            length_t=float(sys_cfg.channel_length_t),
            seed=operator_seed,
        )
        self.detector = BayesRouteDetector(
            bits_per_symbol=int(sys_cfg.bits_per_symbol),
            n_iter=int(mdl.detector_iterations),
            use_uncertainty=bool(mdl.get("use_uncertainty", True)),
        )

    def forward(self, batch, use_uncertainty: bool | None = None,
                edge_mass: float | None = None):
        y_p = batch.y[..., batch.pilot_idx]
        post = self.posterior(y_p, batch.phi, batch.noise_var)
        kappa = coupling_matrix(post.mean, post.var_diag, batch.data_idx, batch.noise_var)
        chosen_mass = float(
            self.cfg.model.get("edge_mass", 1.0) if edge_mass is None else edge_mass
        )
        bit_logits, symbol_logits, x_mean, x_var, graph_mask = self.detector(
            batch.y,
            post.mean,
            post.var_diag,
            batch.data_idx,
            batch.noise_var,
            kappa=kappa,
            edge_mass=chosen_mass,
            use_uncertainty=use_uncertainty,
        )
        return {
            "bit_logits": bit_logits,
            "symbol_logits": symbol_logits,
            "x_mean": x_mean,
            "x_var": x_var,
            "posterior": post,
            "kappa": kappa,
            "graph_mask": graph_mask,
            "edge_density": edge_density(graph_mask),
            "edge_mass": chosen_mass,
        }


class LSReceiver(nn.Module):
    """Constant-channel orthogonal-pilot baseline for Gate-0 only."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.detector = BayesRouteDetector(
            int(cfg.system.bits_per_symbol),
            n_iter=int(cfg.model.detector_iterations),
            use_uncertainty=False,
        )

    def forward(self, batch):
        y_p = batch.y[..., batch.pilot_idx]
        phi = batch.phi
        p = phi.shape[-1]
        h_avg = torch.einsum("np,bxp->bnx", phi.conj(), y_p) / float(p)
        h_mean = h_avg[..., None].expand(-1, -1, -1, batch.y.shape[-1]).contiguous()
        h_var = torch.zeros(
            (phi.shape[0], batch.y.shape[-1]), dtype=torch.float32, device=batch.y.device
        )
        bit_logits, symbol_logits, x_mean, x_var, graph_mask = self.detector(
            batch.y, h_mean, h_var, batch.data_idx, batch.noise_var,
            kappa=None, edge_mass=1.0, use_uncertainty=False,
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
        h_var = torch.zeros(
            (batch.h.shape[1], batch.h.shape[-1]),
            dtype=torch.float32,
            device=batch.y.device,
        )
        bit_logits, symbol_logits, x_mean, x_var, graph_mask = self.detector(
            batch.y, batch.h, h_var, batch.data_idx, batch.noise_var,
            kappa=None, edge_mass=1.0, use_uncertainty=False,
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
