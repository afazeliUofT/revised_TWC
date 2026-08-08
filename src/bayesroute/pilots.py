from __future__ import annotations
import math
import torch


def make_grid(n_subcarriers: int, n_symbols: int, device=None) -> torch.Tensor:
    """Return normalized [R, 2] coordinates [frequency, time]."""
    f = torch.arange(n_subcarriers, dtype=torch.float32, device=device)
    t = torch.arange(n_symbols, dtype=torch.float32, device=device)
    tt, ff = torch.meshgrid(t, f, indexing="ij")
    coords = torch.stack([ff.reshape(-1), tt.reshape(-1)], dim=-1)
    coords[:, 0] = (coords[:, 0] - coords[:, 0].mean()) / max(float(n_subcarriers - 1), 1.0)
    coords[:, 1] = (coords[:, 1] - coords[:, 1].mean()) / max(float(n_symbols - 1), 1.0)
    return coords


def pilot_indices(n_subcarriers: int, n_symbols: int, dmrs_symbols: list[int], device=None) -> torch.Tensor:
    idx = []
    for s in dmrs_symbols:
        if s < 0 or s >= n_symbols:
            raise ValueError(f"DMRS symbol index {s} outside 0..{n_symbols-1}")
        for f in range(n_subcarriers):
            idx.append(s * n_subcarriers + f)
    return torch.tensor(idx, dtype=torch.long, device=device)


def data_indices(n_subcarriers: int, n_symbols: int, dmrs_symbols: list[int], device=None) -> torch.Tensor:
    p = set(int(x) for x in pilot_indices(n_subcarriers, n_symbols, dmrs_symbols).cpu().tolist())
    idx = [r for r in range(n_subcarriers * n_symbols) if r not in p]
    return torch.tensor(idx, dtype=torch.long, device=device)


def make_orthogonal_dmrs(n_layers: int, n_pilots: int, device=None) -> torch.Tensor:
    """Create unit-modulus orthogonal pilot sequences for layer/port separation.

    Each layer uses one DFT row across all pilot REs. This is an abstract NR-like
    DMRS port/OCC/CDM model. It gives exact port orthogonality when n_pilots >= n_layers.
    """
    if n_pilots < n_layers:
        raise ValueError("Number of pilot REs must be at least number of layers for orthogonal ports.")
    j = torch.arange(n_pilots, dtype=torch.float32, device=device).view(1, -1)
    n = torch.arange(n_layers, dtype=torch.float32, device=device).view(-1, 1)
    phase = 2.0 * math.pi * n * j / float(n_pilots)
    return torch.exp(1j * phase).to(torch.complex64)


def port_metadata(n_layers: int) -> list[dict]:
    meta = []
    for n in range(n_layers):
        meta.append({
            "layer_index": n,
            "dmrs_port": n,
            "cdm_group": n // 2,
            "occ_index": n % 2,
            "comment": "Abstract orthogonal DMRS port assignment used by this simulator."
        })
    return meta


def pilot_orthogonality_report(phi: torch.Tensor) -> dict:
    gram = phi @ phi.conj().transpose(-1, -2)
    p = phi.shape[-1]
    eye = torch.eye(phi.shape[0], dtype=gram.dtype, device=gram.device) * p
    err = gram - eye
    max_offdiag = torch.max(torch.abs(err)).item()
    norm_err = torch.linalg.norm(err).item() / max(float(p), 1.0)
    return {
        "n_layers": int(phi.shape[0]),
        "n_pilots": int(phi.shape[1]),
        "max_abs_gram_error": float(max_offdiag),
        "relative_frobenius_error": float(norm_err),
        "passed": bool(max_offdiag < 1e-4)
    }
