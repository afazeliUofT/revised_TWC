from __future__ import annotations
import math
import torch
import torch.nn.functional as F


def bit_bce_loss(bit_logits: torch.Tensor, bits: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(bit_logits, bits.float())


def bit_metrics(bit_logits: torch.Tensor, bits: torch.Tensor) -> dict:
    with torch.no_grad():
        pred = (torch.sigmoid(bit_logits) >= 0.5).float()
        err = (pred != bits.float()).float()
        ber = err.mean().item()
        sample_error = (
            err.reshape(err.shape[0], -1).sum(dim=1) > 0
        ).float().mean().item()
        bce = F.binary_cross_entropy_with_logits(
            bit_logits, bits.float()
        ).item()
        probs = torch.sigmoid(bit_logits).detach()
        brier = torch.mean((probs - bits.float()) ** 2).item()
    return {
        "ber": float(ber),
        "tblER_proxy": float(sample_error),
        "bit_nll": float(bce),
        "brier": float(brier),
    }


def calibration_ece(bit_logits: torch.Tensor, bits: torch.Tensor,
                    n_bins: int = 10) -> float:
    with torch.no_grad():
        p1 = torch.sigmoid(bit_logits).flatten()
        y = bits.float().flatten()
        conf = torch.maximum(p1, 1.0 - p1)
        correct = ((p1 >= 0.5).float() == y).float()
        ece = torch.tensor(0.0, device=bit_logits.device)
        edges = torch.linspace(0.5, 1.0, n_bins + 1, device=bit_logits.device)
        for i in range(n_bins):
            upper = conf < edges[i + 1] if i < n_bins - 1 else conf <= edges[i + 1]
            mask = (conf >= edges[i]) & upper
            if mask.any():
                ece = ece + mask.float().mean() * torch.abs(
                    conf[mask].mean() - correct[mask].mean()
                )
        return float(ece.item())


def channel_nmse(mu: torch.Tensor, h: torch.Tensor) -> float:
    with torch.no_grad():
        return float((
            torch.mean(torch.abs(mu - h) ** 2)
            / torch.mean(torch.abs(h) ** 2).clamp_min(1e-8)
        ).real.item())


def channel_marginal_nll(
    mu: torch.Tensor, var_diag: torch.Tensor, h: torch.Tensor
) -> torch.Tensor:
    """Proper-complex marginal Gaussian NLL, up to the constant log(pi).

    `mu` and `h` have shape [B,N,RX,D]. `var_diag` has shape [N,D]
    and is broadcast across batch and receive antennas.
    """
    var = var_diag[None, :, None, :].to(mu.device).clamp_min(1e-8)
    error_power = torch.abs(mu - h) ** 2
    return torch.mean(error_power / var + torch.log(var)).real


def channel_coverage95(
    mu: torch.Tensor, var_diag: torch.Tensor, h: torch.Tensor
) -> float:
    """Empirical coverage of the 95% circular complex-Gaussian marginal region."""
    with torch.no_grad():
        var = var_diag[None, :, None, :].to(mu.device).clamp_min(1e-8)
        threshold = -math.log(0.05) * var
        covered = (torch.abs(mu - h) ** 2 <= threshold).float().mean()
        return float(covered.item())
