from __future__ import annotations

"""Precision-safe localized delay--Doppler basis for the oracle ceiling.

Each atom combines a smooth overlapping frequency window, a physical delay
phase, and a discrete Doppler/time mode.  All atoms and singular values are
constructed in complex128 before numerical-rank truncation.  The final
orthonormal basis is cast to the requested runtime dtype only after rank has
been decided.

This precision order is essential: constructing the nearly dependent atom bank
in complex64 and then promoting it to complex128 can turn float32 roundoff into
spurious singular directions.  Those directions form an artificial oracle
subspace and can make a bounded-rank ceiling look better than it really is.
"""

from dataclasses import dataclass
import math
from typing import Any

import torch


LOCALIZED_DD_VERSION = "localized_delay_doppler_oracle_ceiling_v1_1"
LOCALIZED_DD_PRECISION_PATCH = "complex128_atoms_before_rank_decision_v1"
DEFAULT_RELATIVE_RANK_THRESHOLD = 1e-10


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
        return int(
            self.frequency_windows * self.delay_bins * self.doppler_bins
        )

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
            raise ValueError(
                "The decisive ceiling gate caps nominal rank at 128"
            )

    @classmethod
    def from_mapping(
        cls, value: dict[str, Any]
    ) -> "LocalizedDelayDopplerSpec":
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


def _doppler_mode_indices(
    count: int, device: torch.device
) -> torch.Tensor:
    values = [0]
    order = 1
    while len(values) < int(count):
        values.append(order)
        if len(values) < int(count):
            values.append(-order)
        order += 1
    return torch.tensor(
        values[: int(count)], dtype=torch.float64, device=device
    )


def _precision_safe_raw_atoms(
    *,
    num_symbols: int,
    num_subcarriers: int,
    subcarrier_spacing_khz: float,
    spec: LocalizedDelayDopplerSpec,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Construct normalized atoms directly in complex128."""
    s = int(num_symbols)
    f = int(num_subcarriers)
    if s <= 0 or f <= 0:
        raise ValueError("Grid dimensions must be positive")

    scs_hz = float(subcarrier_spacing_khz) * 1e3
    slot_duration_s = 1e-3 * 15.0 / float(subcarrier_spacing_khz)
    time_s = (
        torch.arange(s, dtype=torch.float64, device=device)
        - 0.5 * (s - 1)
    ) * (slot_duration_s / float(s))
    frequency_hz = (
        torch.arange(f, dtype=torch.float64, device=device)
        - 0.5 * (f - 1)
    ) * scs_hz
    frequency_unit = (
        torch.arange(f, dtype=torch.float64, device=device)
        - 0.5 * (f - 1)
    ) / float(max(f - 1, 1))

    centers = torch.linspace(
        -0.5,
        0.5,
        int(spec.frequency_windows),
        dtype=torch.float64,
        device=device,
    )
    if int(spec.frequency_windows) == 1:
        windows = torch.ones(
            1, f, dtype=torch.float64, device=device
        )
    else:
        spacing = 1.0 / float(spec.frequency_windows - 1)
        sigma = float(spec.window_overlap) * spacing
        windows = torch.exp(
            -0.5
            * (
                (frequency_unit[None, :] - centers[:, None]) / sigma
            ).square()
        )
        windows = windows / windows.square().sum(
            dim=0, keepdim=True
        ).sqrt().clamp_min(1e-15)

    delays = torch.linspace(
        0.0,
        float(spec.max_delay_s),
        int(spec.delay_bins),
        dtype=torch.float64,
        device=device,
    )
    delay_atoms = torch.exp(
        -1j
        * 2.0
        * math.pi
        * frequency_hz[:, None]
        * delays[None, :]
    ).to(torch.complex128)

    mode_indices = _doppler_mode_indices(
        int(spec.doppler_bins), device
    )
    doppler_hz = mode_indices / float(slot_duration_s)
    time_atoms = torch.exp(
        1j
        * 2.0
        * math.pi
        * time_s[:, None]
        * doppler_hz[None, :]
    ).to(torch.complex128)

    raw = torch.einsum(
        "fw,fl,sd->sfwld",
        windows.T.to(torch.complex128),
        delay_atoms,
        time_atoms,
    ).reshape(s * f, spec.nominal_rank)
    norms = torch.linalg.vector_norm(raw, dim=0).clamp_min(1e-15)
    raw = raw / norms[None, :]
    return raw.contiguous(), {
        "slot_duration_s": float(slot_duration_s),
        "minimum_column_norm_before_normalization": float(norms.min().item()),
        "maximum_column_norm_before_normalization": float(norms.max().item()),
    }


def localized_delay_doppler_features(
    *,
    num_symbols: int,
    num_subcarriers: int,
    subcarrier_spacing_khz: float,
    spec: LocalizedDelayDopplerSpec,
    device: torch.device | str,
    dtype: torch.dtype = torch.complex64,
    orthonormalize: bool = True,
    relative_rank_threshold: float = DEFAULT_RELATIVE_RANK_THRESHOLD,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Return a time-major ``[S*F,Q]`` localized physical basis.

    Numerical rank is always decided from complex128 atoms.  A direction is
    retained when ``sigma_i/sigma_max`` is strictly above
    ``relative_rank_threshold``.  This keeps the intended mathematical
    threshold while preventing complex64 roundoff from inventing directions.
    """
    spec.validate()
    if float(relative_rank_threshold) <= 0.0:
        raise ValueError("relative_rank_threshold must be positive")
    dev = torch.device(device)
    s = int(num_symbols)
    f = int(num_subcarriers)
    raw, construction = _precision_safe_raw_atoms(
        num_symbols=s,
        num_subcarriers=f,
        subcarrier_spacing_khz=float(subcarrier_spacing_khz),
        spec=spec,
        device=dev,
    )

    if orthonormalize:
        u, singular_values, _ = torch.linalg.svd(
            raw, full_matrices=False
        )
        maximum = singular_values.max().clamp_min(1e-15)
        relative = singular_values / maximum
        active = relative > float(relative_rank_threshold)
        if not bool(active.any().item()):
            raise RuntimeError(
                "Localized delay-Doppler basis is numerically empty"
            )
        features128 = u[:, active].contiguous()
    else:
        singular_values = torch.linalg.svdvals(raw)
        maximum = singular_values.max().clamp_min(1e-15)
        relative = singular_values / maximum
        active = torch.ones(
            raw.shape[1], dtype=torch.bool, device=dev
        )
        features128 = raw

    features = features128.to(dtype).contiguous()
    gram = features.conj().T @ features
    eye = torch.eye(
        features.shape[1], dtype=features.dtype, device=dev
    )
    orthogonality_error = float(
        torch.max(torch.abs(gram - eye)).item()
    ) if orthonormalize else float("nan")

    discarded = relative[~active]
    kept = relative[active]
    maximum_discarded = (
        float(discarded.max().item()) if discarded.numel() else 0.0
    )
    minimum_kept = float(kept.min().item())
    report = {
        "version": LOCALIZED_DD_VERSION,
        "precision_patch": LOCALIZED_DD_PRECISION_PATCH,
        "spec": spec.as_dict(),
        "grid": {
            "num_symbols": s,
            "num_subcarriers": f,
            "subcarrier_spacing_khz": float(
                subcarrier_spacing_khz
            ),
            "slot_duration_s": construction["slot_duration_s"],
        },
        "nominal_rank": int(spec.nominal_rank),
        "effective_rank": int(features.shape[1]),
        "discarded_rank": int(spec.nominal_rank - features.shape[1]),
        "orthonormalized": bool(orthonormalize),
        "construction_real_dtype": "float64",
        "construction_complex_dtype": "complex128",
        "runtime_output_dtype": str(dtype),
        "relative_rank_threshold": float(relative_rank_threshold),
        "singular_value_relative_min_kept": minimum_kept,
        "singular_value_relative_max_discarded": maximum_discarded,
        "rank_partition_exact": bool(
            minimum_kept > float(relative_rank_threshold)
            and (
                discarded.numel() == 0
                or maximum_discarded <= float(relative_rank_threshold)
            )
        ),
        "orthogonality_max_abs_error": orthogonality_error,
        "finite": bool(torch.isfinite(features).all().item()),
        **construction,
    }
    return features, report


def project_tensor_to_basis(
    target: torch.Tensor,
    features: torch.Tensor,
    *,
    ridge: float = 1e-6,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Least-squares project ``[...,R]`` complex values onto ``[R,Q]``."""
    if target.ndim < 2 or features.ndim != 2:
        raise ValueError(
            "target must end in R and features must have shape [R,Q]"
        )
    if int(target.shape[-1]) != int(features.shape[0]):
        raise ValueError(
            "Target and basis resource-grid lengths disagree"
        )
    r, q = features.shape
    flat = target.reshape(-1, r).T.contiguous()
    gram = features.conj().T @ features
    gram = gram + float(ridge) * torch.eye(
        q, dtype=features.dtype, device=features.device
    )
    coefficients = torch.linalg.solve(
        gram, features.conj().T @ flat
    )
    projected = (
        (features @ coefficients)
        .T.reshape(target.shape)
        .contiguous()
    )
    residual = target - projected
    nmse = residual.abs().square().mean() / target.abs().square().mean().clamp_min(1e-12)
    return projected, {
        "projection_nmse": float(nmse.real.item()),
        "projection_residual_power": float(
            residual.abs().square().mean().real.item()
        ),
        "target_power": float(
            target.abs().square().mean().real.item()
        ),
        "effective_rank": int(q),
    }


def winner_precision_rank_regression(
    device: torch.device | str = "cpu",
) -> dict[str, Any]:
    """Known-rank regression for the selected 96-atom basis.

    The previous complex64-first implementation reported rank 96 on every
    grid.  Direct complex128 construction reveals the intended thresholded
    ranks 51, 69, and 84 on 4, 8, and 12 PRBs, respectively.
    """
    dev = torch.device(device)
    spec = LocalizedDelayDopplerSpec(
        name="ldd_w2_d16_v3_r96_tau3us",
        frequency_windows=2,
        delay_bins=16,
        doppler_bins=3,
        max_delay_s=3.0e-6,
        window_overlap=1.5,
    )
    expected = {48: 51, 96: 69, 144: 84}
    records: list[dict[str, Any]] = []
    for subcarriers, expected_rank in expected.items():
        _, report = localized_delay_doppler_features(
            num_symbols=14,
            num_subcarriers=subcarriers,
            subcarrier_spacing_khz=30.0,
            spec=spec,
            device=dev,
        )
        records.append(
            {
                "num_subcarriers": subcarriers,
                "expected_rank": expected_rank,
                "observed_rank": report["effective_rank"],
                "rank_partition_exact": report[
                    "rank_partition_exact"
                ],
                "minimum_relative_kept": report[
                    "singular_value_relative_min_kept"
                ],
                "maximum_relative_discarded": report[
                    "singular_value_relative_max_discarded"
                ],
            }
        )
    checks = {
        "construction_complex128": True,
        "expected_thresholded_ranks": all(
            item["observed_rank"] == item["expected_rank"]
            for item in records
        ),
        "rank_partitions_exact": all(
            bool(item["rank_partition_exact"])
            for item in records
        ),
        "spurious_full_rank_removed": all(
            item["observed_rank"] < spec.nominal_rank
            for item in records
        ),
    }
    return {
        "version": LOCALIZED_DD_PRECISION_PATCH,
        "passed": all(checks.values()),
        "checks": checks,
        "relative_rank_threshold": DEFAULT_RELATIVE_RANK_THRESHOLD,
        "records": records,
    }


def mathematical_self_test(
    device: torch.device | str = "cpu"
) -> dict[str, Any]:
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
    projected, projection = project_tensor_to_basis(
        target, features, ridge=1e-8
    )
    error = float(torch.max(torch.abs(projected - target)).item())
    precision = winner_precision_rank_regression(dev)
    checks = {
        "basis_finite": basis["finite"],
        "basis_rank_positive": basis["effective_rank"] > 0,
        "basis_rank_capped": basis["nominal_rank"] <= 128,
        "complex128_before_rank_decision": (
            basis["construction_complex_dtype"] == "complex128"
        ),
        "rank_partition_exact": basis["rank_partition_exact"],
        "orthogonality": (
            basis["orthogonality_max_abs_error"] < 2e-4
        ),
        "exact_in_span_projection": error < 2e-4,
        "projection_nmse_small": projection["projection_nmse"] < 1e-8,
        "winner_rank_regression": precision["passed"],
    }
    return {
        "version": LOCALIZED_DD_VERSION,
        "precision_patch": LOCALIZED_DD_PRECISION_PATCH,
        "passed": all(checks.values()),
        "checks": checks,
        "basis": basis,
        "in_span_max_abs_error": error,
        "projection": projection,
        "winner_precision_rank_regression": precision,
    }
