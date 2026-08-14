from __future__ import annotations

"""Deterministic localized delay--Doppler basis for a hard oracle ceiling test.

The basis is a compact basis-expansion model (BEM).  Each atom combines

* a smooth, overlapping frequency window,
* a physical delay phase term, and
* a discrete Doppler/time mode over one NR slot.

The module is intentionally training-free.  It is used only to test whether a
bounded-rank localized physical subspace can, even with oracle coefficients,
improve an LS channel estimate enough to beat LS+LMMSE.  If this ceiling fails,
training a receiver in the same family cannot repair the gap.
"""

from dataclasses import dataclass
import math
from typing import Any

import torch


LOCALIZED_DD_VERSION = "localized_delay_doppler_oracle_ceiling_v1"


@dataclass(frozen=True)
class LocalizedDelayDopplerSpec:
    name: str
    frequency_windows: int
    delay_bins: int
    doppler_bins: int
    max_delay_s: float
    window_overlap: float = 1.5

    @property
    def nominal_rank(self) -> int:
        return int(self.frequency_windows * self.delay_bins * self.doppler_bins)

    def validate(self) -> None:
        if not self.name:
            raise ValueError("Localized basis name must be nonempty")
        if int(self.frequency_windows) <= 0:
            raise ValueError("frequency_windows must be positive")
        if int(self.delay_bins) <= 0:
            raise ValueError("delay_bins must be positive")
        if int(self.doppler_bins) <= 0:
            raise ValueError("doppler_bins must be positive")
        if float(self.max_delay_s) <= 0.0:
            raise ValueError("max_delay_s must be positive")
        if float(self.window_overlap) <= 0.0:
            raise ValueError("window_overlap must be positive")
        if self.nominal_rank > 128:
            raise ValueError("The decisive ceiling gate caps nominal rank at 128")

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> "LocalizedDelayDopplerSpec":
        result = cls(
            name=str(value["name"]),
            frequency_windows=int(value["frequency_windows"]),
            delay_bins=int(value["delay_bins"]),
            doppler_bins=int(value["doppler_bins"]),
            max_delay_s=float(value["max_delay_s"]),
            window_overlap=float(value.get("window_overlap", 1.5)),
        )
        result.validate()
        return result

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "frequency_windows": self.frequency_windows,
            "delay_bins": self.delay_bins,
            "doppler_bins": self.doppler_bins,
            "max_delay_s": self.max_delay_s,
            "window_overlap": self.window_overlap,
            "nominal_rank": self.nominal_rank,
        }


def _doppler_mode_indices(count: int, device: torch.device) -> torch.Tensor:
    values = [0]
    order = 1
    while len(values) < int(count):
        values.append(order)
        if len(values) < int(count):
            values.append(-order)
        order += 1
    return torch.tensor(values[: int(count)], dtype=torch.float32, device=device)


def localized_delay_doppler_features(
    *,
    num_symbols: int,
    num_subcarriers: int,
    subcarrier_spacing_khz: float,
    spec: LocalizedDelayDopplerSpec,
    device: torch.device | str,
    dtype: torch.dtype = torch.complex64,
    orthonormalize: bool = True,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Return a time-major [S*F,Q] localized delay--Doppler basis."""
    spec.validate()
    dev = torch.device(device)
    s = int(num_symbols)
    f = int(num_subcarriers)
    if s <= 0 or f <= 0:
        raise ValueError("Grid dimensions must be positive")

    scs_hz = float(subcarrier_spacing_khz) * 1e3
    # Normal-CP NR slot duration is 1 ms / 2^mu and 2^mu = SCS/15 kHz.
    slot_duration_s = 1e-3 * 15.0 / float(subcarrier_spacing_khz)
    time_s = (
        torch.arange(s, dtype=torch.float32, device=dev) - 0.5 * (s - 1)
    ) * (slot_duration_s / float(s))
    frequency_hz = (
        torch.arange(f, dtype=torch.float32, device=dev) - 0.5 * (f - 1)
    ) * scs_hz
    frequency_unit = (
        torch.arange(f, dtype=torch.float32, device=dev) - 0.5 * (f - 1)
    ) / float(max(f - 1, 1))

    centers = torch.linspace(
        -0.5, 0.5, int(spec.frequency_windows), device=dev
    )
    if int(spec.frequency_windows) == 1:
        windows = torch.ones(1, f, dtype=torch.float32, device=dev)
    else:
        spacing = 1.0 / float(spec.frequency_windows - 1)
        sigma = float(spec.window_overlap) * spacing
        windows = torch.exp(
            -0.5 * ((frequency_unit[None, :] - centers[:, None]) / sigma) ** 2
        )
        # Partition-of-unity normalization avoids edge attenuation.
        windows = windows / windows.square().sum(dim=0, keepdim=True).sqrt().clamp_min(1e-8)

    delays = torch.linspace(
        0.0, float(spec.max_delay_s), int(spec.delay_bins), device=dev
    )
    delay_atoms = torch.exp(
        -1j * 2.0 * math.pi * frequency_hz[:, None] * delays[None, :]
    ).to(dtype)

    mode_indices = _doppler_mode_indices(int(spec.doppler_bins), dev)
    doppler_hz = mode_indices / float(slot_duration_s)
    time_atoms = torch.exp(
        1j * 2.0 * math.pi * time_s[:, None] * doppler_hz[None, :]
    ).to(dtype)

    # [S,F,W,L,D] -> [S*F,W*L*D], matching NR time-major flattening.
    raw = torch.einsum(
        "fw,fl,sd->sfwld",
        windows.T.to(dtype),
        delay_atoms,
        time_atoms,
    ).reshape(s * f, spec.nominal_rank)
    norms = torch.linalg.vector_norm(raw, dim=0).clamp_min(1e-8)
    raw = raw / norms[None, :]

    if orthonormalize:
        # SVD is used instead of QR-diagonal thresholding because the localized
        # atoms can be strongly correlated while still spanning useful,
        # numerically distinct directions.  The relative singular-value test is
        # deterministic and keeps all directions above double-precision noise.
        u, singular_values, _ = torch.linalg.svd(
            raw.to(torch.complex128),
            full_matrices=False,
        )
        tolerance = 1e-10 * singular_values.max().clamp_min(1e-15)
        active = singular_values > tolerance
        features = u[:, active].to(dtype).contiguous()
    else:
        singular_values = torch.linalg.svdvals(raw.to(torch.complex128))
        tolerance = torch.tensor(0.0, dtype=singular_values.dtype, device=dev)
        active = torch.ones(raw.shape[1], dtype=torch.bool, device=dev)
        features = raw.contiguous()

    if int(features.shape[1]) <= 0:
        raise RuntimeError("Localized delay--Doppler basis is numerically empty")
    gram = features.conj().T @ features
    eye = torch.eye(features.shape[1], dtype=features.dtype, device=dev)
    orthogonality_error = float(torch.max(torch.abs(gram - eye)).item())
    report = {
        "version": LOCALIZED_DD_VERSION,
        "spec": spec.as_dict(),
        "grid": {
            "num_symbols": s,
            "num_subcarriers": f,
            "subcarrier_spacing_khz": float(subcarrier_spacing_khz),
            "slot_duration_s": float(slot_duration_s),
        },
        "nominal_rank": int(spec.nominal_rank),
        "effective_rank": int(features.shape[1]),
        "orthonormalized": bool(orthonormalize),
        "singular_value_relative_min_kept": float(
            (singular_values[active].min() / singular_values.max()).item()
        ),
        "singular_value_relative_threshold": float(
            (tolerance / singular_values.max().clamp_min(1e-15)).item()
        ),
        "orthogonality_max_abs_error": orthogonality_error,
        "finite": bool(torch.isfinite(features).all().item()),
    }
    return features, report


def project_tensor_to_basis(
    target: torch.Tensor,
    features: torch.Tensor,
    *,
    ridge: float = 1e-6,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Least-squares project [...,R] complex values onto [R,Q]."""
    if target.ndim < 2 or features.ndim != 2:
        raise ValueError("target must end in R and features must have shape [R,Q]")
    if int(target.shape[-1]) != int(features.shape[0]):
        raise ValueError("Target and basis resource-grid lengths disagree")
    r, q = features.shape
    flat = target.reshape(-1, r).T.contiguous()
    gram = features.conj().T @ features
    gram = gram + float(ridge) * torch.eye(q, dtype=features.dtype, device=features.device)
    coefficients = torch.linalg.solve(gram, features.conj().T @ flat)
    projected = (features @ coefficients).T.reshape(target.shape).contiguous()
    residual = target - projected
    nmse = residual.abs().square().mean() / target.abs().square().mean().clamp_min(1e-12)
    return projected, {
        "projection_nmse": float(nmse.real.item()),
        "projection_residual_power": float(residual.abs().square().mean().real.item()),
        "target_power": float(target.abs().square().mean().real.item()),
        "effective_rank": int(q),
    }


def mathematical_self_test(device: torch.device | str = "cpu") -> dict[str, Any]:
    dev = torch.device(device)
    spec = LocalizedDelayDopplerSpec(
        name="self_test",
        frequency_windows=2,
        delay_bins=4,
        doppler_bins=3,
        max_delay_s=1.0e-6,
    )
    features, basis = localized_delay_doppler_features(
        num_symbols=14,
        num_subcarriers=48,
        subcarrier_spacing_khz=30.0,
        spec=spec,
        device=dev,
    )
    torch.manual_seed(19031)
    coeff = (
        torch.randn(2, 3, features.shape[1], device=dev)
        + 1j * torch.randn(2, 3, features.shape[1], device=dev)
    ).to(torch.complex64) / math.sqrt(2.0)
    target = torch.einsum("rq,bxq->bxr", features, coeff)
    projected, projection = project_tensor_to_basis(target, features, ridge=1e-8)
    error = float(torch.max(torch.abs(projected - target)).item())
    checks = {
        "basis_finite": basis["finite"],
        "basis_rank_positive": basis["effective_rank"] > 0,
        "basis_rank_capped": basis["nominal_rank"] <= 128,
        "orthogonality": basis["orthogonality_max_abs_error"] < 2e-4,
        "exact_in_span_projection": error < 2e-4,
        "projection_nmse_small": projection["projection_nmse"] < 1e-8,
    }
    return {
        "version": LOCALIZED_DD_VERSION,
        "passed": all(checks.values()),
        "checks": checks,
        "basis": basis,
        "in_span_max_abs_error": error,
        "projection": projection,
    }
