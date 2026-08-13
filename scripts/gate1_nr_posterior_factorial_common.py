#!/usr/bin/env python3
from __future__ import annotations

"""Common components for the fair coordinate-by-kernel posterior screen."""

from dataclasses import dataclass, replace
from types import SimpleNamespace
import hashlib
import json
import math
import random
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch

from bayesroute.models import (
    LowRankPosteriorOperator,
    PosteriorOutput,
    coupling_matrix,
    coupling_selection_mask,
)
from bayesroute.multiscale_posterior import (
    MULTISCALE_POSTERIOR_VERSION,
    MultiScalePosteriorOperator,
    RFFScale,
    bind_shared_multiscale_parameters,
)
from bayesroute.nr_gate1 import (
    NRBayesRouteBridge,
    NRCase,
    NRGridDescription,
    diagonal_covariance,
)
from gate1_nr_joint_operator_common import (
    SELECTED_COVARIANCE_MODE,
    SELECTED_DETECTOR_DAMPING,
    SELECTED_DETECTOR_ITERATIONS,
    SELECTED_EDGE_MASS,
    make_repaired_detector,
    posterior_graph,
    repaired_forward,
    unique_parameters,
)


POSTERIOR_FACTORIAL_VERSION = "gate1_nr_posterior_factorial_v1"
LS_ALIGNMENT_PATCH_VERSION = "gate1_nr_posterior_factorial_ls_alignment_v1"
REFERENCE_NUM_SUBCARRIERS = 48
REFERENCE_SCS_KHZ = 30.0


@dataclass(frozen=True)
class FactorialCandidate:
    name: str
    coordinate_mode: str
    model_type: str
    rank: int
    bank_rank: int
    scales: tuple[RFFScale, ...]
    context_conditioned: bool
    learning_rate: float
    channel_loss_weight: float
    calibration_loss_weight: float
    steps: int

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> "FactorialCandidate":
        scales = tuple(
            RFFScale(
                rank=int(item["rank"]),
                length_f=float(item["length_f"]),
                length_t=float(item["length_t"]),
            )
            for item in value.get("scales", [])
        )
        result = cls(
            name=str(value["name"]),
            coordinate_mode=str(value["coordinate_mode"]),
            model_type=str(value["model_type"]),
            rank=int(value.get("rank", sum(item.rank for item in scales))),
            bank_rank=int(value.get("bank_rank", value.get("rank", 0))),
            scales=scales,
            context_conditioned=bool(value.get("context_conditioned", False)),
            learning_rate=float(value["learning_rate"]),
            channel_loss_weight=float(value["channel_loss_weight"]),
            calibration_loss_weight=float(value["calibration_loss_weight"]),
            steps=int(value["steps"]),
        )
        result.validate()
        return result

    def validate(self) -> None:
        if self.coordinate_mode not in {"allocation_normalized", "reference_physical"}:
            raise ValueError(f"Invalid coordinate mode: {self.coordinate_mode}")
        if self.model_type not in {"single", "multiscale"}:
            raise ValueError(f"Invalid model type: {self.model_type}")
        if self.model_type == "single":
            if self.rank <= 0 or self.bank_rank < self.rank:
                raise ValueError("Single-scale candidate rank contract is invalid")
            if self.scales:
                raise ValueError("Single-scale candidate must not define scale blocks")
            if self.context_conditioned:
                raise ValueError("Single-scale control is not context-conditioned")
        else:
            if not self.scales:
                raise ValueError("Multi-scale candidate must define scales")
            if sum(item.rank for item in self.scales) != self.rank:
                raise ValueError("Multi-scale rank must equal sum of scale ranks")
        if self.steps <= 0:
            raise ValueError("Candidate steps must be positive")

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "coordinate_mode": self.coordinate_mode,
            "model_type": self.model_type,
            "rank": self.rank,
            "bank_rank": self.bank_rank,
            "scales": [item.__dict__ for item in self.scales],
            "context_conditioned": self.context_conditioned,
            "learning_rate": self.learning_rate,
            "channel_loss_weight": self.channel_loss_weight,
            "calibration_loss_weight": self.calibration_loss_weight,
            "steps": self.steps,
        }


def set_all_seeds(seed: int) -> None:
    try:
        import sionna.phy
        sionna.phy.config.seed = int(seed)
    except Exception:
        pass
    random.seed(int(seed))
    np.random.seed(int(seed) % (2**32 - 1))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package_signature(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def save_json(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def atomic_torch_save(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def reference_physical_coords(
    grid: NRGridDescription,
    *,
    subcarrier_spacing_khz: float,
) -> torch.Tensor:
    device = grid.coords.device
    num_symbols = int(grid.num_ofdm_symbols)
    num_subcarriers = int(grid.num_effective_subcarriers)
    time_axis = torch.arange(num_symbols, dtype=torch.float32, device=device)
    freq_axis = torch.arange(num_subcarriers, dtype=torch.float32, device=device)
    tt, ff = torch.meshgrid(time_axis, freq_axis, indexing="ij")
    coords = torch.stack([ff.reshape(-1), tt.reshape(-1)], dim=-1)
    coords[:, 0] = (
        (coords[:, 0] - coords[:, 0].mean())
        * float(subcarrier_spacing_khz)
        / REFERENCE_SCS_KHZ
        / float(REFERENCE_NUM_SUBCARRIERS - 1)
    )
    if num_symbols > 1:
        coords[:, 1] = (coords[:, 1] - coords[:, 1].mean()) / float(
            num_symbols - 1
        )
    return coords


def grid_for_mode(
    case: NRCase,
    grid: NRGridDescription,
    mode: str,
) -> NRGridDescription:
    if mode == "allocation_normalized":
        return grid
    if mode == "reference_physical":
        return replace(
            grid,
            coords=reference_physical_coords(
                grid,
                subcarrier_spacing_khz=float(case.subcarrier_spacing_khz),
            ),
        )
    raise ValueError(f"Unknown coordinate mode: {mode}")


def context_vector(case: NRCase, device: torch.device) -> torch.Tensor:
    scenario_order = ["umi", "uma", "cdl-a", "cdl-c", "cdl-d"]
    scenario = case.scenario.lower()
    one_hot = [1.0 if scenario == item else 0.0 for item in scenario_order]
    values = [
        math.log2(max(float(case.num_prb), 1.0) / 4.0),
        math.log2(max(float(case.subcarrier_spacing_khz), 1.0) / 30.0),
        float(case.dmrs_config_type - 1),
        float(case.dmrs_length - 1),
        float(case.dmrs_additional_position) / 3.0,
        math.log2(max(float(case.num_streams), 1.0)),
        math.log2(max(float(case.num_rx_ant), 1.0)),
        float(case.speed_mps) / 30.0,
        *one_hot,
    ]
    return torch.tensor(values, dtype=torch.float32, device=device)


def build_candidate_bridge(
    case: NRCase,
    context: Any,
    spec: FactorialCandidate,
    *,
    operator_seed: int,
) -> NRBayesRouteBridge:
    grid = grid_for_mode(case, context.grid, spec.coordinate_mode)
    bridge = NRBayesRouteBridge(
        grid,
        num_streams=case.num_streams,
        rank=max(spec.rank, 1),
        bank_rank=max(spec.bank_rank, spec.rank, 1),
        detector_iterations=SELECTED_DETECTOR_ITERATIONS,
        edge_mass=SELECTED_EDGE_MASS,
        length_f=1.0,
        length_t=0.5,
        operator_seed=int(operator_seed),
    ).to(context.device)
    if spec.model_type == "single":
        bridge.posterior = LowRankPosteriorOperator(
            coords=grid.coords,
            pilot_idx=grid.pilot_idx,
            n_layers=case.num_streams,
            rank=spec.rank,
            length_f=1.0,
            length_t=0.5,
            seed=int(operator_seed),
            bank_rank=spec.bank_rank,
        ).to(context.device)
    else:
        bridge.posterior = MultiScalePosteriorOperator(
            coords=grid.coords,
            pilot_idx=grid.pilot_idx,
            n_layers=case.num_streams,
            scales=spec.scales,
            seed=int(operator_seed),
            context=context_vector(case, context.device),
            context_conditioned=spec.context_conditioned,
        ).to(context.device)
    return bridge


def bind_candidate_parameters(
    spec: FactorialCandidate,
    bridges: Sequence[NRBayesRouteBridge],
) -> None:
    if not bridges:
        raise ValueError("At least one bridge is required")
    if spec.model_type == "single":
        master = bridges[0].posterior
        for bridge in bridges[1:]:
            bridge.posterior.raw_weights = master.raw_weights
            bridge.posterior.log_noise_scale = master.log_noise_scale
    else:
        bind_shared_multiscale_parameters(
            [bridge.posterior for bridge in bridges]  # type: ignore[arg-type]
        )


def candidate_parameter_tensors(
    bridges: Sequence[NRBayesRouteBridge],
) -> list[torch.nn.Parameter]:
    return unique_parameters(bridges)


def extract_candidate_state(
    spec: FactorialCandidate,
    bridge: NRBayesRouteBridge,
) -> dict[str, Any]:
    operator = bridge.posterior
    if spec.model_type == "single":
        return {
            "model_type": "single",
            "raw_weights": operator.raw_weights.detach().cpu().clone(),
            "log_noise_scale": operator.log_noise_scale.detach().cpu().clone(),
        }
    assert isinstance(operator, MultiScalePosteriorOperator)
    result: dict[str, Any] = {
        "model_type": "multiscale",
        "raw_feature_weights": operator.raw_feature_weights.detach().cpu().clone(),
        "log_noise_scale": operator.log_noise_scale.detach().cpu().clone(),
    }
    if operator.context_conditioned:
        result["raw_scale_bias"] = operator.raw_scale_bias.detach().cpu().clone()
        result["context_to_scale"] = operator.context_to_scale.detach().cpu().clone()
    return result


def load_candidate_state(
    spec: FactorialCandidate,
    bridge: NRBayesRouteBridge,
    state: dict[str, Any],
) -> None:
    operator = bridge.posterior
    if state.get("model_type") != spec.model_type:
        raise RuntimeError("Candidate state model type mismatch")
    with torch.no_grad():
        if spec.model_type == "single":
            raw = torch.as_tensor(
                state["raw_weights"],
                dtype=operator.raw_weights.dtype,
                device=operator.raw_weights.device,
            )
            if raw.shape != operator.raw_weights.shape:
                raise RuntimeError("Single-scale checkpoint rank mismatch")
            operator.raw_weights.copy_(raw)
            operator.log_noise_scale.copy_(
                torch.as_tensor(
                    state["log_noise_scale"],
                    dtype=operator.log_noise_scale.dtype,
                    device=operator.log_noise_scale.device,
                )
            )
            return
        assert isinstance(operator, MultiScalePosteriorOperator)
        raw = torch.as_tensor(
            state["raw_feature_weights"],
            dtype=operator.raw_feature_weights.dtype,
            device=operator.raw_feature_weights.device,
        )
        if raw.shape != operator.raw_feature_weights.shape:
            raise RuntimeError("Multi-scale checkpoint rank mismatch")
        operator.raw_feature_weights.copy_(raw)
        operator.log_noise_scale.copy_(
            torch.as_tensor(
                state["log_noise_scale"],
                dtype=operator.log_noise_scale.dtype,
                device=operator.log_noise_scale.device,
            )
        )
        if operator.context_conditioned:
            operator.raw_scale_bias.copy_(
                torch.as_tensor(
                    state["raw_scale_bias"],
                    dtype=operator.raw_scale_bias.dtype,
                    device=operator.raw_scale_bias.device,
                )
            )
            operator.context_to_scale.copy_(
                torch.as_tensor(
                    state["context_to_scale"],
                    dtype=operator.context_to_scale.dtype,
                    device=operator.context_to_scale.device,
                )
            )


def model_report(spec: FactorialCandidate, bridge: NRBayesRouteBridge) -> dict[str, Any]:
    operator = bridge.posterior
    if spec.model_type == "single":
        return {
            "model_type": "single",
            "coordinate_mode": spec.coordinate_mode,
            "rank": spec.rank,
            "context_conditioned": False,
            "trainable_parameters": int(
                sum(p.numel() for p in operator.parameters() if p.requires_grad)
            ),
        }
    assert isinstance(operator, MultiScalePosteriorOperator)
    return {
        **operator.parameter_report(),
        "model_type": "multiscale",
        "coordinate_mode": spec.coordinate_mode,
    }



def _cpu_index_bounds(
    index: torch.Tensor,
    upper: int,
    *,
    name: str,
) -> tuple[int, int]:
    """Validate an index tensor on CPU before any CUDA gather is launched."""
    values = torch.as_tensor(index.detach().cpu(), dtype=torch.long).reshape(-1)
    if values.numel() == 0:
        raise RuntimeError(f"{name} must not be empty")
    lower = int(values.min().item())
    maximum = int(values.max().item())
    if lower < 0 or maximum >= int(upper):
        raise RuntimeError(
            f"{name} out of range: min={lower}, max={maximum}, upper={upper}"
        )
    return lower, maximum


def _broadcast_ls_error_variance(
    err_var: torch.Tensor,
    target_shape: torch.Size,
    *,
    device: torch.device,
) -> torch.Tensor:
    value = torch.as_tensor(err_var, device=device).real.to(torch.float32)
    try:
        return torch.broadcast_to(value, target_shape).clone()
    except RuntimeError as exc:
        raise RuntimeError(
            "Sionna LS error variance is not broadcast-compatible with the "
            f"channel estimate: err_var={tuple(value.shape)}, h_hat={tuple(target_shape)}"
        ) from exc


def _align_sionna_ls_grid(
    h_hat: torch.Tensor,
    err_var: torch.Tensor,
    context: Any,
    data_idx: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Align Sionna's LS output with BayesRoute's exact effective grid.

    Sionna's OFDM channel estimator normally returns the *effective* grid
    after removing nulled subcarriers.  Some estimator implementations may
    instead return the full FFT grid.  The previous control checked the FFT
    width first, which could re-index an already-effective tensor and launch
    an asynchronous CUDA out-of-bounds gather.  This function identifies the
    effective layout first, validates every index on CPU, and records the
    complete shape contract before any detector call.
    """
    if h_hat.ndim != 7 or int(h_hat.shape[1]) != 1:
        raise RuntimeError(f"Unexpected LS estimate shape: {tuple(h_hat.shape)}")
    if not torch.is_complex(h_hat):
        raise RuntimeError("Sionna LS channel estimate must be complex")

    raw_h_shape = [int(x) for x in h_hat.shape]
    raw_err_shape = [int(x) for x in torch.as_tensor(err_var).shape]
    batch_size = int(h_hat.shape[0])
    num_rx = int(h_hat.shape[2])
    num_users = int(h_hat.shape[3])
    num_layers = int(h_hat.shape[4])
    num_symbols = int(h_hat.shape[5])
    observed_width = int(h_hat.shape[6])
    expected_width = int(context.grid.num_effective_subcarriers)
    fft_size = int(context.grid.fft_size)
    expected_symbols = int(context.grid.num_ofdm_symbols)
    expected_streams = int(context.grid.num_streams)

    if num_symbols != expected_symbols:
        raise RuntimeError(
            f"LS OFDM-symbol count {num_symbols} != bridge count {expected_symbols}"
        )
    if num_users * num_layers != expected_streams:
        raise RuntimeError(
            "LS user/layer dimensions disagree with the bridge: "
            f"users={num_users}, layers={num_layers}, streams={expected_streams}"
        )

    err_var = _broadcast_ls_error_variance(
        err_var, h_hat.shape, device=h_hat.device
    )
    layout: str
    effective_bounds: tuple[int, int] | None = None

    # Effective-grid output is the documented Sionna estimator behavior and
    # must be checked before the full-FFT alternative, especially when the two
    # widths happen to be numerically equal.
    if observed_width == expected_width:
        layout = "effective_grid_as_returned"
    elif observed_width == fft_size:
        effective = torch.as_tensor(
            context.grid.effective_subcarrier_ind,
            dtype=torch.long,
        ).reshape(-1)
        if int(effective.numel()) != expected_width:
            raise RuntimeError(
                "effective_subcarrier_ind length does not match the bridge grid"
            )
        effective_bounds = _cpu_index_bounds(
            effective, observed_width, name="effective_subcarrier_ind"
        )
        effective_device = effective.to(h_hat.device)
        h_hat = torch.index_select(h_hat, -1, effective_device)
        err_var = torch.index_select(err_var, -1, effective_device)
        layout = "full_fft_selected_to_effective_grid"
    else:
        raise RuntimeError(
            "Unexpected Sionna LS frequency width: "
            f"observed={observed_width}, effective={expected_width}, fft={fft_size}"
        )

    if int(h_hat.shape[-1]) != expected_width:
        raise RuntimeError("LS effective-grid alignment produced the wrong width")
    expected_grid_length = expected_symbols * expected_width
    data_bounds = _cpu_index_bounds(
        data_idx, expected_grid_length, name="data_idx"
    )

    mean = h_hat[:, 0].permute(0, 2, 3, 1, 4, 5).reshape(
        batch_size,
        expected_streams,
        num_rx,
        expected_grid_length,
    ).contiguous()
    variance = err_var[:, 0].permute(0, 2, 3, 1, 4, 5).reshape(
        batch_size,
        expected_streams,
        num_rx,
        expected_grid_length,
    ).contiguous()
    if mean.shape[-1] != expected_grid_length or variance.shape != mean.shape:
        raise RuntimeError("LS flattening contract failed")
    if not torch.isfinite(mean).all().item():
        raise RuntimeError("LS channel estimate contains non-finite values")
    if not torch.isfinite(variance).all().item():
        raise RuntimeError("LS error variance contains non-finite values")

    report = {
        "version": LS_ALIGNMENT_PATCH_VERSION,
        "passed": True,
        "layout": layout,
        "raw_h_hat_shape": raw_h_shape,
        "raw_err_var_shape": raw_err_shape,
        "broadcast_err_var_shape": [int(x) for x in err_var.shape],
        "aligned_mean_shape": [int(x) for x in mean.shape],
        "expected_grid_length": int(expected_grid_length),
        "observed_frequency_width": int(observed_width),
        "effective_frequency_width": int(expected_width),
        "fft_size": int(fft_size),
        "effective_index_bounds": (
            list(effective_bounds) if effective_bounds is not None else None
        ),
        "data_index_bounds": list(data_bounds),
        "effective_grid_checked_before_fft": True,
        "cpu_index_validation": True,
    }
    return mean.to(torch.complex64), variance.real.clamp_min(1e-8), report


def ls_posterior_from_receiver(
    receiver: Any,
    context: Any,
    batch: Any,
) -> tuple[PosteriorOutput, dict[str, Any]]:
    """Expose and safely align Sionna's LS estimate for factorization tests."""
    estimator = getattr(receiver, "_channel_estimator", None)
    if estimator is None:
        raise RuntimeError("Standard LS receiver has no channel estimator")
    h_hat, err_var = estimator(batch.raw_y, batch.noise_var)
    mean, variance, report = _align_sionna_ls_grid(
        h_hat, err_var, context, batch.data_idx
    )
    var_diag = variance.mean(dim=(0, 2)).real.clamp_min(1e-8)
    # Avoid device-side advanced-index assignment.  [N,R] -> [R,N,N] -> [N,N,R].
    local_cov = torch.diag_embed(var_diag.transpose(0, 1).to(torch.complex64))
    local_cov = local_cov.permute(1, 2, 0).contiguous()
    posterior = PosteriorOutput(
        mean=mean,
        var_diag=var_diag,
        local_cov=local_cov,
        latent_cov=torch.zeros(1, 1, dtype=torch.complex64, device=mean.device),
        effective_noise=torch.as_tensor(
            batch.noise_var, dtype=torch.float32, device=mean.device
        ),
    )
    return posterior, report


def ls_repaired_forward(
    receiver: Any,
    context: Any,
    detector: torch.nn.Module,
    batch: Any,
) -> dict[str, Any]:
    """Run the repaired detector on Sionna LS estimates with local data indexing."""
    posterior, alignment = ls_posterior_from_receiver(
        receiver, context, batch
    )
    data_idx = torch.as_tensor(
        batch.data_idx, dtype=torch.long, device=posterior.mean.device
    ).reshape(-1)
    # Bounds were checked on CPU by _align_sionna_ls_grid.  Slice once and let
    # both graph construction and detection use a local [0,D) index.  This
    # removes the old double-indexing ambiguity while remaining mathematically
    # identical on the data REs.
    mean_data = torch.index_select(posterior.mean, -1, data_idx)
    covariance_data = torch.index_select(posterior.local_cov, -1, data_idx)
    y_data = torch.index_select(batch.y, -1, data_idx)
    local_idx = torch.arange(
        int(data_idx.numel()), dtype=torch.long, device=posterior.mean.device
    )
    kappa = coupling_matrix(
        mean_data.detach(),
        covariance_data.detach(),
        local_idx,
        batch.noise_var.detach(),
    )
    graph = coupling_selection_mask(kappa, float(SELECTED_EDGE_MASS))
    output = detector(
        y_data,
        mean_data,
        covariance_data,
        local_idx,
        batch.noise_var,
        graph,
        covariance_mode=SELECTED_COVARIANCE_MODE,
    )
    result = dict(output)
    result["posterior"] = posterior
    result["reference_graph_mask"] = graph
    result["graph_mask"] = graph
    result["edge_density"] = graph.float().mean()
    result["graph_mode"] = "ls_posterior_data_local"
    result["ls_grid_alignment_report"] = alignment
    result["ls_detector_local_data_indexing"] = True
    return result


def ls_alignment_self_test() -> dict[str, Any]:
    """Pure-Torch tests for effective/full-grid alignment and range rejection."""
    class DummyEstimator:
        def __init__(self, h_hat: torch.Tensor, err_var: torch.Tensor) -> None:
            self.h_hat = h_hat
            self.err_var = err_var

        def __call__(self, _y: torch.Tensor, _no: torch.Tensor):
            return self.h_hat, self.err_var

    def make_context(*, fft_size: int, effective_width: int, effective: list[int]):
        return SimpleNamespace(
            grid=SimpleNamespace(
                num_effective_subcarriers=effective_width,
                fft_size=fft_size,
                num_ofdm_symbols=2,
                num_streams=2,
                effective_subcarrier_ind=torch.tensor(effective, dtype=torch.long),
            )
        )

    batch = SimpleNamespace(
        raw_y=torch.zeros(1),
        noise_var=torch.tensor(0.2),
        data_idx=torch.tensor([0, 3, 7], dtype=torch.long),
    )
    effective_h = torch.randn(2, 1, 3, 2, 1, 2, 4, dtype=torch.complex64)
    # Deliberately invalid full-grid indices prove that effective-width output
    # is accepted before any attempt to apply effective_subcarrier_ind.
    context_effective = make_context(
        fft_size=4, effective_width=4, effective=[10, 11, 12, 13]
    )
    posterior_effective, report_effective = ls_posterior_from_receiver(
        SimpleNamespace(
            _channel_estimator=DummyEstimator(effective_h, torch.tensor(0.1))
        ),
        context_effective,
        batch,
    )

    full_h = torch.randn(2, 1, 3, 2, 1, 2, 6, dtype=torch.complex64)
    full_err = torch.full_like(full_h.real, 0.15)
    context_full = make_context(
        fft_size=6, effective_width=4, effective=[1, 2, 4, 5]
    )
    posterior_full, report_full = ls_posterior_from_receiver(
        SimpleNamespace(_channel_estimator=DummyEstimator(full_h, full_err)),
        context_full,
        batch,
    )

    range_rejected = False
    bad_batch = SimpleNamespace(
        raw_y=batch.raw_y,
        noise_var=batch.noise_var,
        data_idx=torch.tensor([0, 8], dtype=torch.long),
    )
    try:
        ls_posterior_from_receiver(
            SimpleNamespace(
                _channel_estimator=DummyEstimator(effective_h, torch.tensor(0.1))
            ),
            context_effective,
            bad_batch,
        )
    except RuntimeError as exc:
        range_rejected = "data_idx out of range" in str(exc)

    checks = {
        "effective_layout_first": (
            report_effective["layout"] == "effective_grid_as_returned"
        ),
        "scalar_error_variance_broadcast": (
            report_effective["broadcast_err_var_shape"]
            == list(effective_h.shape)
        ),
        "full_fft_selection": (
            report_full["layout"] == "full_fft_selected_to_effective_grid"
        ),
        "aligned_grid_lengths": (
            posterior_effective.mean.shape[-1] == 8
            and posterior_full.mean.shape[-1] == 8
        ),
        "local_covariance_shapes": (
            posterior_effective.local_cov.shape == (2, 2, 8)
            and posterior_full.local_cov.shape == (2, 2, 8)
        ),
        "out_of_range_data_rejected": range_rejected,
    }
    return {
        "version": LS_ALIGNMENT_PATCH_VERSION,
        "checks": checks,
        "effective_report": report_effective,
        "full_fft_report": report_full,
        "passed": all(checks.values()),
    }


def pure_torch_multiscale_self_test() -> dict[str, Any]:
    device = torch.device("cpu")
    coords4 = torch.stack(
        torch.meshgrid(
            torch.linspace(-0.5, 0.5, 8),
            torch.linspace(-0.5, 0.5, 4),
            indexing="ij",
        ),
        dim=-1,
    ).reshape(-1, 2)
    coords8 = coords4.clone()
    pilot_idx = torch.tensor([0, 3, 9, 15, 23, 31], dtype=torch.long)
    scales = (
        RFFScale(4, 0.5, 0.25),
        RFFScale(4, 1.0, 0.5),
        RFFScale(4, 2.0, 1.0),
        RFFScale(4, 4.0, 2.0),
    )
    first = MultiScalePosteriorOperator(
        coords4,
        pilot_idx,
        2,
        scales,
        seed=77,
        context=torch.zeros(13),
        context_conditioned=True,
    ).to(device)
    second = MultiScalePosteriorOperator(
        coords8,
        pilot_idx,
        2,
        scales,
        seed=77,
        context=torch.tensor([1.0] + [0.0] * 12),
        context_conditioned=True,
    ).to(device)
    bind_shared_multiscale_parameters([first, second])
    phi = torch.zeros(2, pilot_idx.numel(), dtype=torch.complex64)
    phi[0, ::2] = 1.0
    phi[1, 1::2] = 1.0
    y = torch.randn(3, 2, pilot_idx.numel(), dtype=torch.complex64)
    out1 = first(y, phi, torch.tensor(0.2))
    out2 = second(y, phi, torch.tensor(0.2))
    loss = out1.mean.abs().square().mean() + out2.mean.abs().square().mean()
    loss = loss + out1.var_diag.mean() + out2.var_diag.mean()
    loss.backward()
    params = unique_parameters([first, second])
    gradients = [
        p.grad is not None and torch.isfinite(p.grad).all().item() and p.grad.norm().item() > 0
        for p in params
    ]
    eig1 = torch.linalg.eigvalsh(out1.latent_cov.to(torch.complex128)).real
    eig2 = torch.linalg.eigvalsh(out2.latent_cov.to(torch.complex128)).real
    shared = {
        "feature_weights": id(first.raw_feature_weights) == id(second.raw_feature_weights),
        "noise": id(first.log_noise_scale) == id(second.log_noise_scale),
        "scale_bias": id(first.raw_scale_bias) == id(second.raw_scale_bias),
        "context_map": id(first.context_to_scale) == id(second.context_to_scale),
    }
    result = {
        "shared_parameters": shared,
        "unique_parameter_tensors": len(params),
        "gradients_nonzero_finite": bool(all(gradients)),
        "posterior_finite": bool(
            torch.isfinite(out1.mean).all().item()
            and torch.isfinite(out2.mean).all().item()
            and torch.isfinite(out1.var_diag).all().item()
            and torch.isfinite(out2.var_diag).all().item()
        ),
        "posterior_psd": bool(eig1.min().item() >= -1e-7 and eig2.min().item() >= -1e-7),
        "context_changes_scale_gains": bool(
            not torch.allclose(first.scale_gains(), second.scale_gains(), atol=1e-8)
        ),
        "first_report": first.parameter_report(),
        "second_report": second.parameter_report(),
    }
    result["passed"] = bool(
        all(shared.values())
        and result["gradients_nonzero_finite"]
        and result["posterior_finite"]
        and result["posterior_psd"]
        and result["context_changes_scale_gains"]
    )
    return result
