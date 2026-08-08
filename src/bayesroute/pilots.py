from __future__ import annotations
import math
import torch

PILOT_MODEL_SCOPE = "gate0_abstract_orthogonal_codebook_not_3gpp_pusch"


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
    if len(dmrs_symbols) != len(set(int(s) for s in dmrs_symbols)):
        raise ValueError("DMRS symbol indices must be unique.")
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
    """Create a unit-modulus orthogonal pilot codebook for the Gate-0 simulator.

    This is intentionally not presented as the exact 3GPP PUSCH DMRS mapping.
    Each layer uses one row of a DFT codebook across the pilot observations.
    """
    if n_pilots < n_layers:
        raise ValueError("Number of pilot observations must be at least the number of layers.")
    j = torch.arange(n_pilots, dtype=torch.float32, device=device).view(1, -1)
    n = torch.arange(n_layers, dtype=torch.float32, device=device).view(-1, 1)
    phase = 2.0 * math.pi * n * j / float(n_pilots)
    return torch.exp(1j * phase).to(torch.complex64)


def port_metadata(n_layers: int) -> list[dict]:
    """Return explicit layer/port/CDM/OCC labels for the Gate-0 abstraction."""
    meta = []
    for n in range(n_layers):
        meta.append({
            "layer_index": n,
            "user_index": n,  # Gate-0 uses one single-layer user per graph node.
            "within_user_layer_index": 0,
            "dmrs_port": n,
            "cdm_group": n // 2,
            "occ_index": n % 2,
            "pilot_model_scope": PILOT_MODEL_SCOPE,
        })
    return meta


def pilot_orthogonality_report(phi: torch.Tensor) -> dict:
    gram = phi @ phi.conj().transpose(-1, -2)
    p = phi.shape[-1]
    eye = torch.eye(phi.shape[0], dtype=gram.dtype, device=gram.device) * p
    err = gram - eye
    max_abs = torch.max(torch.abs(err)).item()
    rel = torch.linalg.norm(err).item() / max(torch.linalg.norm(eye).item(), 1e-12)
    return {
        "n_layers": int(phi.shape[0]),
        "n_pilots": int(phi.shape[1]),
        "max_abs_gram_error": float(max_abs),
        "relative_frobenius_error": float(rel),
        "passed": bool(max_abs < 1e-4 and rel < 1e-5),
    }


def pilot_separation_report(phi: torch.Tensor) -> dict:
    """Verify exact noiseless layer separation under the declared code model."""
    n, p = phi.shape
    # Deterministic complex scalar channel per layer for this algebraic test.
    h = torch.arange(1, n + 1, device=phi.device, dtype=torch.float32)
    h = torch.complex(h, torch.flip(h, dims=[0])) / max(float(n), 1.0)
    y = torch.sum(h[:, None] * phi, dim=0)
    h_hat = torch.einsum("np,p->n", phi.conj(), y) / float(p)
    max_error = float(torch.max(torch.abs(h_hat - h)).item())
    return {"max_abs_recovery_error": max_error, "passed": bool(max_error < 1e-4)}


def port_metadata_report(meta: list[dict], n_layers: int) -> dict:
    layer_ids = [int(x["layer_index"]) for x in meta]
    port_ids = [int(x["dmrs_port"]) for x in meta]
    scopes = {str(x.get("pilot_model_scope", "")) for x in meta}
    passed = (
        len(meta) == int(n_layers)
        and sorted(layer_ids) == list(range(int(n_layers)))
        and len(set(port_ids)) == int(n_layers)
        and scopes == {PILOT_MODEL_SCOPE}
    )
    return {
        "n_entries": len(meta),
        "unique_ports": len(set(port_ids)),
        "scope": sorted(scopes),
        "passed": bool(passed),
    }


def resource_partition_report(pilot_idx: torch.Tensor, data_idx: torch.Tensor, n_re: int) -> dict:
    p = set(int(x) for x in pilot_idx.detach().cpu().tolist())
    d = set(int(x) for x in data_idx.detach().cpu().tolist())
    overlap = p.intersection(d)
    union = p.union(d)
    return {
        "pilot_count": len(p),
        "data_count": len(d),
        "overlap_count": len(overlap),
        "covered_count": len(union),
        "n_re": int(n_re),
        "passed": bool(len(overlap) == 0 and len(union) == int(n_re)),
    }
