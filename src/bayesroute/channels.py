from __future__ import annotations

import math
import torch


def complex_normal(
    shape,
    device=None,
    scale: float = 1.0,
    dtype=torch.complex64,
) -> torch.Tensor:
    """Draw a proper complex Gaussian tensor with E|z|^2=scale^2."""
    re = torch.randn(shape, device=device) * (scale / math.sqrt(2.0))
    im = torch.randn(shape, device=device) * (scale / math.sqrt(2.0))
    return torch.complex(re, im).to(dtype)


def rff_bank(
    coords: torch.Tensor,
    rank: int,
    length_f: float,
    length_t: float,
    seed: int = 0,
    bank_rank: int | None = None,
) -> torch.Tensor:
    """Return a fixed complex random-Fourier-feature bank [R, rank].

    ``bank_rank`` fixes the largest generated bank. Candidate ranks obtained with
    the same seed and bank_rank are nested: a rank-q bank uses the first q modes
    of the common bank. This prevents Optuna from comparing different random
    basis realizations when it changes the model rank.
    """
    rank = int(rank)
    max_rank = rank if bank_rank is None else int(bank_rank)
    if rank <= 0:
        raise ValueError("rank must be positive")
    if max_rank < rank:
        raise ValueError(f"bank_rank={max_rank} must be at least rank={rank}")

    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))
    wf_all = torch.randn(max_rank, generator=gen)
    wt_all = torch.randn(max_rank, generator=gen)
    phase_all = 2.0 * math.pi * torch.rand(max_rank, generator=gen)

    wf = wf_all[:rank] / max(float(length_f), 1e-3)
    wt = wt_all[:rank] / max(float(length_t), 1e-3)
    phase0 = phase_all[:rank]
    omega = torch.stack([wf, wt], dim=-1).to(coords.device, coords.dtype)
    phase0 = phase0.to(coords.device, coords.dtype)
    phase = 2.0 * math.pi * (coords @ omega.T) + phase0.view(1, -1)

    # Each coordinate has unit total feature energy for every candidate rank.
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
    basis = rff_bank(
        coords,
        true_rank,
        length_f,
        length_t,
        seed=seed,
        bank_rank=true_rank,
    )
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
