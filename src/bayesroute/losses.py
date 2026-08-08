from __future__ import annotations
import torch
import torch.nn.functional as F


def bit_bce_loss(bit_logits: torch.Tensor, bits: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(bit_logits, bits.float())


def bit_metrics(bit_logits: torch.Tensor, bits: torch.Tensor) -> dict:
    with torch.no_grad():
        pred = (torch.sigmoid(bit_logits) >= 0.5).float()
        err = (pred != bits.float()).float()
        ber = err.mean().item()
        block_err = (err.reshape(err.shape[0], -1).sum(dim=1) > 0).float().mean().item()
        bce = F.binary_cross_entropy_with_logits(bit_logits, bits.float()).item()
        probs = torch.sigmoid(bit_logits).detach()
        brier = torch.mean((probs - bits.float()) ** 2).item()
    return {"ber": float(ber), "tblER_proxy": float(block_err), "bit_nll": float(bce), "brier": float(brier)}


def calibration_ece(bit_logits: torch.Tensor, bits: torch.Tensor, n_bins: int = 10) -> float:
    with torch.no_grad():
        p1 = torch.sigmoid(bit_logits).flatten()
        y = bits.float().flatten()
        conf = torch.maximum(p1, 1.0 - p1)
        correct = ((p1 >= 0.5).float() == y).float()
        ece = torch.tensor(0.0, device=bit_logits.device)
        edges = torch.linspace(0.5, 1.0, n_bins + 1, device=bit_logits.device)
        for i in range(n_bins):
            mask = (conf >= edges[i]) & (conf < edges[i+1] if i < n_bins-1 else conf <= edges[i+1])
            if mask.any():
                ece = ece + mask.float().mean() * torch.abs(conf[mask].mean() - correct[mask].mean())
        return float(ece.item())


def channel_nmse(mu: torch.Tensor, h: torch.Tensor) -> float:
    with torch.no_grad():
        return float((torch.mean(torch.abs(mu - h) ** 2) / torch.mean(torch.abs(h) ** 2).clamp_min(1e-8)).real.item())
