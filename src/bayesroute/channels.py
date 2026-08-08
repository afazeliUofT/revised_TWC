from __future__ import annotations
import math
import torch


def complex_normal(shape, device=None, scale: float = 1.0,
                   dtype=torch.complex64) -> torch.Tensor:
    re = torch.randn(shape, device=device) * (scale / math.sqrt(2.0))
    im = torch.randn(shape, device=device) * (scale / math.sqrt(2.0))
    return torch.complex(re, im).to(dtype)


def rff_bank(coords: torch.Tensor, rank: int, length_f: float,
             length_t: float, seed: int = 0) -> torch.Tensor:
    """Fixed complex random Fourier feature bank [R, rank]."""
    if int(rank) <= 0:
        raise ValueError("rank must be positive")
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))
    wf = torch.randn(rank, generator=gen) / max(float(length_f), 1e-3)
    wt = torch.randn(rank, generator=gen) / max(float(length_t), 1e-3)
    phase0 = 2.0 * math.pi * torch.rand(rank, generator=gen)
    omega = torch.stack([wf, wt], dim=-1).to(coords.device, coords.dtype)
    phase0 = phase0.to(coords.device, coords.dtype)
    phase = 2.0 * math.pi * (coords @ omega.T) + phase0.view(1, -1)
    # Every coordinate has exactly unit feature energy. Therefore E|h[r]|^2=1
    # for unit-variance latent coefficients, without per-realization normalization.
    return torch.exp(1j * phase).to(torch.complex64) / math.sqrt(float(rank))


def generate_low_rank_channel(
    batch_size: int,
    n_layers: int,
    n_rx: int,
    coords: torch.Tensor,
    true_rank: int,
    length_f: float,
    length_t: float,
    seed: int = 777,
    layer_power_spread_db: float = 0.0,
) -> torch.Tensor:
    """Generate a proper-complex low-rank channel field [B,N,RX,R].

    The expected power is one before the optional deterministic layer-power
    spread. No sample-wise normalization is applied because it would break the
    Gaussian prior used by the posterior operator.
    """
    device = coords.device
    basis = rff_bank(coords, true_rank, length_f, length_t, seed=seed)
    latent = complex_normal(
        (batch_size, n_layers, n_rx, true_rank), device=device
    )
    h = torch.einsum("rq,bnxq->bnxr", basis, latent)
    spread = max(float(layer_power_spread_db), 0.0)
    if spread > 0.0 and n_layers > 1:
        layer_gains_db = torch.linspace(
            0.0, -spread, n_layers, device=device
        ).view(1, n_layers, 1, 1)
        h = h * (10.0 ** (layer_gains_db / 20.0))
    return h.to(torch.complex64)
