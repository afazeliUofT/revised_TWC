from __future__ import annotations
import math
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
    """Learned PSD low-rank channel operator with exact Gaussian conditioning.

    The channel prior is h = B z, z~CN(0,I). B is block diagonal over layers
    and built from fixed complex Fourier features with learned nonnegative weights.
    """

    def __init__(self, coords: torch.Tensor, pilot_idx: torch.Tensor, n_layers: int,
                 rank: int, length_f: float = 6.0, length_t: float = 2.0, seed: int = 11):
        super().__init__()
        self.n_layers = int(n_layers)
        self.rank = int(rank)
        self.register_buffer("coords", coords.detach().clone())
        self.register_buffer("pilot_idx", pilot_idx.detach().clone())
        base = rff_bank(coords.detach(), rank, length_f, length_t, seed=seed)
        self.register_buffer("base_features", base)
        self.raw_weights = nn.Parameter(torch.zeros(rank))
        self.log_noise_scale = nn.Parameter(torch.tensor(0.0))

    def weighted_features(self) -> torch.Tensor:
        # positive weights keep K = B B^H positive semidefinite
        w = F.softplus(self.raw_weights) + 1e-4
        w = w / torch.sqrt(torch.mean(w ** 2).clamp_min(1e-8))
        return self.base_features * w.to(self.base_features.dtype).view(1, -1)

    def _observation_basis(self, phi: torch.Tensor) -> torch.Tensor:
        """Build AB with shape [P, N*Q]."""
        u_p = self.weighted_features()[self.pilot_idx, :]  # [P,Q]
        p = u_p.shape[0]
        blocks = []
        for n in range(self.n_layers):
            blocks.append(phi[n, :].view(p, 1) * u_p)
        return torch.cat(blocks, dim=1)  # [P,NQ]

    def forward(self, y_p: torch.Tensor, phi: torch.Tensor, noise_var: torch.Tensor) -> PosteriorOutput:
        """Condition on pilots.

        Args:
            y_p: [B, RX, P]
            phi: [N, P]
            noise_var: scalar real tensor
        """
        bsz, n_rx, p = y_p.shape
        device = y_p.device
        ab = self._observation_basis(phi).to(device)  # [P,NQ]
        nq = ab.shape[1]
        sigma2 = noise_var * torch.exp(self.log_noise_scale).clamp(0.25, 4.0) + 1e-6
        eye_p = torch.eye(p, dtype=torch.complex64, device=device)
        c = ab @ ab.conj().transpose(0, 1) + sigma2.to(torch.complex64) * eye_p
        # Cholesky solve is stable and differentiable.
        chol = torch.linalg.cholesky(c + 1e-5 * eye_p)
        y_flat = y_p.reshape(bsz * n_rx, p).T.contiguous()  # [P,BRX]
        alpha = torch.cholesky_solve(y_flat, chol)          # [P,BRX]
        latent_mu = ab.conj().transpose(0, 1) @ alpha       # [NQ,BRX]
        latent_mu = latent_mu.T.reshape(bsz, n_rx, self.n_layers, self.rank)
        u = self.weighted_features().to(device)             # [R,Q]
        h_mu = torch.einsum("rq,bxnq->bnxr", u, latent_mu).contiguous()

        # Latent posterior covariance: I - AB^H C^{-1} AB.
        cinv_ab = torch.cholesky_solve(ab, chol)            # [P,NQ]
        latent_cov = torch.eye(nq, dtype=torch.complex64, device=device) - ab.conj().transpose(0, 1) @ cinv_ab
        latent_cov = 0.5 * (latent_cov + latent_cov.conj().transpose(0, 1))

        # Per-layer posterior variance diagonal at each RE.
        vars_layers = []
        for n in range(self.n_layers):
            m_nn = latent_cov[n*self.rank:(n+1)*self.rank, n*self.rank:(n+1)*self.rank]
            v = torch.einsum("rq,qs,rs->r", u.conj(), m_nn, u).real.clamp_min(1e-8)
            vars_layers.append(v)
        var_diag = torch.stack(vars_layers, dim=0)  # [N,R]
        return PosteriorOutput(mean=h_mu, var_diag=var_diag, latent_cov=latent_cov, effective_noise=sigma2.detach())

    def channel_nmse(self, posterior: PosteriorOutput, h_true: torch.Tensor, idx: torch.Tensor | None = None) -> torch.Tensor:
        mu = posterior.mean if idx is None else posterior.mean[..., idx]
        h = h_true if idx is None else h_true[..., idx]
        return (torch.mean(torch.abs(mu - h) ** 2) / torch.mean(torch.abs(h) ** 2).clamp_min(1e-8)).real


class BayesRouteDetector(nn.Module):
    """Uncertainty-aware soft detector.

    The detector uses posterior channel means and variances. It is deliberately
    compact, so ablations are easy to interpret.
    """

    def __init__(self, bits_per_symbol: int, n_iter: int = 3, use_uncertainty: bool = True):
        super().__init__()
        self.bits_per_symbol = int(bits_per_symbol)
        self.n_iter = int(n_iter)
        self.use_uncertainty = bool(use_uncertainty)

    def forward(self, y: torch.Tensor, h_mean: torch.Tensor, h_var: torch.Tensor,
                data_idx: torch.Tensor, noise_var: torch.Tensor, edge_mass: float = 1.0):
        device = y.device
        const, bit_table = make_qam_constellation(self.bits_per_symbol, device=device)
        m = int(const.numel())
        y_d = y[..., data_idx]                           # [B,RX,D]
        mu = h_mean[..., data_idx]                       # [B,N,RX,D]
        var = h_var[:, data_idx].to(device)              # [N,D]
        bsz, n_layers, n_rx, d = mu.shape
        x_mean = torch.zeros((bsz, n_layers, d), dtype=torch.complex64, device=device)
        x_var = torch.ones((bsz, n_layers, d), dtype=torch.float32, device=device)
        symbol_logits = None

        for _ in range(self.n_iter):
            total_mean = torch.sum(mu * x_mean[:, :, None, :], dim=1)  # [B,RX,D]
            layer_power = torch.abs(mu) ** 2 + (var[None, :, None, :] if self.use_uncertainty else 0.0)
            total_var = noise_var + torch.sum(x_var[:, :, None, :] * layer_power, dim=1)  # [B,RX,D]
            logits_n = []
            for n in range(n_layers):
                mu_n = mu[:, n, :, :]                   # [B,RX,D]
                var_n = var[n, :].view(1, 1, d) if self.use_uncertainty else 0.0
                mean_without_n = total_mean - mu_n * x_mean[:, n, None, :]
                var_without_n = total_var - x_var[:, n, None, :] * layer_power[:, n, :, :]
                cand_logits = []
                for a in const:
                    res = y_d - mean_without_n - mu_n * a
                    cvar = (var_without_n + (torch.abs(a) ** 2) * var_n).real.clamp_min(1e-6)
                    ll = -torch.sum((torch.abs(res) ** 2) / cvar + torch.log(cvar), dim=1)  # [B,D]
                    cand_logits.append(ll)
                logits_n.append(torch.stack(cand_logits, dim=-1))
            symbol_logits = torch.stack(logits_n, dim=1)  # [B,N,D,M]
            probs = torch.softmax(symbol_logits, dim=-1)
            x_mean, x_var = symbol_probs_to_mean_var(probs, const)
        bit_logits = symbol_logits_to_bit_logits(symbol_logits, bit_table)  # [B,N,D,Q]
        return bit_logits, symbol_logits, x_mean, x_var


def coupling_matrix(h_mean: torch.Tensor, h_var: torch.Tensor, data_idx: torch.Tensor,
                    noise_var: torch.Tensor) -> torch.Tensor:
    """Posterior interference coupling averaged over data REs.

    Returns [B, D, N, N] with zero diagonal. It approximates the expected squared
    off-diagonal whitened Gram entry under diagonal RX uncertainty.
    """
    mu = h_mean[..., data_idx]  # [B,N,RX,D]
    var = h_var[:, data_idx].to(mu.device)  # [N,D]
    bsz, n_layers, n_rx, d = mu.shape
    inv_noise = 1.0 / noise_var.clamp_min(1e-6)
    k = torch.zeros((bsz, d, n_layers, n_layers), dtype=torch.float32, device=mu.device)
    for n in range(n_layers):
        for m in range(n_layers):
            if n == m:
                continue
            coherent = torch.sum(mu[:, n].conj() * mu[:, m] * inv_noise, dim=1)  # [B,D]
            term0 = torch.abs(coherent) ** 2
            term1 = torch.sum(torch.abs(mu[:, n]) ** 2 * var[m].view(1, 1, d) * (inv_noise ** 2), dim=1)
            term2 = torch.sum(torch.abs(mu[:, m]) ** 2 * var[n].view(1, 1, d) * (inv_noise ** 2), dim=1)
            term3 = torch.sum(var[n].view(1, 1, d) * var[m].view(1, 1, d) * (inv_noise ** 2), dim=1)
            k[:, :, n, m] = (term0 + term1 + term2 + term3).real
    return k


class BayesRouteReceiver(nn.Module):
    def __init__(self, cfg, coords: torch.Tensor, pilot_idx: torch.Tensor):
        super().__init__()
        self.cfg = cfg
        sys = cfg.system
        mdl = cfg.model
        self.posterior = LowRankPosteriorOperator(
            coords=coords,
            pilot_idx=pilot_idx,
            n_layers=int(sys.n_layers),
            rank=int(mdl.rank),
            length_f=float(sys.channel_length_f),
            length_t=float(sys.channel_length_t),
            seed=int(cfg.seed) + 17,
        )
        self.detector = BayesRouteDetector(
            bits_per_symbol=int(sys.bits_per_symbol),
            n_iter=int(mdl.detector_iterations),
            use_uncertainty=bool(mdl.get("use_uncertainty", True)),
        )

    def forward(self, batch, use_uncertainty: bool | None = None):
        old = self.detector.use_uncertainty
        if use_uncertainty is not None:
            self.detector.use_uncertainty = bool(use_uncertainty)
        y_p = batch.y[..., batch.pilot_idx]
        post = self.posterior(y_p, batch.phi, batch.noise_var)
        bit_logits, symbol_logits, x_mean, x_var = self.detector(
            batch.y, post.mean, post.var_diag, batch.data_idx, batch.noise_var,
            edge_mass=float(self.cfg.model.get("edge_mass", 1.0)),
        )
        kappa = coupling_matrix(post.mean.detach(), post.var_diag.detach(), batch.data_idx, batch.noise_var.detach())
        self.detector.use_uncertainty = old
        return {
            "bit_logits": bit_logits,
            "symbol_logits": symbol_logits,
            "x_mean": x_mean,
            "x_var": x_var,
            "posterior": post,
            "kappa": kappa,
        }


class LSReceiver(nn.Module):
    """Simple orthogonal-pilot point-channel baseline."""
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.detector = BayesRouteDetector(int(cfg.system.bits_per_symbol), n_iter=int(cfg.model.detector_iterations), use_uncertainty=False)

    def forward(self, batch):
        y_p = batch.y[..., batch.pilot_idx]           # [B,RX,P]
        phi = batch.phi                               # [N,P]
        p = phi.shape[-1]
        h_avg = torch.einsum("np,bxp->bnx", phi.conj(), y_p) / float(p)  # [B,N,RX]
        h_mean = h_avg[..., None].expand(-1, -1, -1, batch.y.shape[-1]).contiguous()
        h_var = torch.zeros((phi.shape[0], batch.y.shape[-1]), dtype=torch.float32, device=batch.y.device)
        bit_logits, symbol_logits, x_mean, x_var = self.detector(batch.y, h_mean, h_var, batch.data_idx, batch.noise_var)
        return {"bit_logits": bit_logits, "symbol_logits": symbol_logits, "x_mean": x_mean, "x_var": x_var,
                "posterior": None, "kappa": None}


class OracleReceiver(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.detector = BayesRouteDetector(int(cfg.system.bits_per_symbol), n_iter=int(cfg.model.detector_iterations), use_uncertainty=False)

    def forward(self, batch):
        h_var = torch.zeros((batch.h.shape[1], batch.h.shape[-1]), dtype=torch.float32, device=batch.y.device)
        bit_logits, symbol_logits, x_mean, x_var = self.detector(batch.y, batch.h, h_var, batch.data_idx, batch.noise_var)
        return {"bit_logits": bit_logits, "symbol_logits": symbol_logits, "x_mean": x_mean, "x_var": x_var,
                "posterior": None, "kappa": None}
