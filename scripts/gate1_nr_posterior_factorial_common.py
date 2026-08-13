#!/usr/bin/env python3
from __future__ import annotations

"""Common components for the fair coordinate-by-kernel posterior screen."""

from dataclasses import dataclass, replace
import hashlib
import json
import math
import random
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch

from bayesroute.models import LowRankPosteriorOperator, PosteriorOutput
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


def ls_posterior_from_receiver(
    receiver: Any,
    context: Any,
    batch: Any,
) -> PosteriorOutput:
    """Expose Sionna's LS estimate for a detector-factorization control."""
    estimator = getattr(receiver, "_channel_estimator", None)
    if estimator is None:
        raise RuntimeError("Standard LS receiver has no channel estimator")
    h_hat, err_var = estimator(batch.raw_y, batch.noise_var)
    effective = context.grid.effective_subcarrier_ind
    if int(h_hat.shape[-1]) == int(context.grid.fft_size):
        h_hat = torch.index_select(h_hat, -1, effective)
        err_var = torch.index_select(err_var, -1, effective)
    elif int(h_hat.shape[-1]) != int(context.grid.num_effective_subcarriers):
        raise RuntimeError(f"Unexpected LS channel width: {tuple(h_hat.shape)}")
    if h_hat.ndim != 7 or int(h_hat.shape[1]) != 1:
        raise RuntimeError(f"Unexpected LS estimate shape: {tuple(h_hat.shape)}")
    batch_size = int(h_hat.shape[0])
    num_rx = int(h_hat.shape[2])
    num_users = int(h_hat.shape[3])
    num_layers = int(h_hat.shape[4])
    mean = h_hat[:, 0].permute(0, 2, 3, 1, 4, 5).reshape(
        batch_size,
        num_users * num_layers,
        num_rx,
        -1,
    )
    variance = err_var[:, 0].permute(0, 2, 3, 1, 4, 5).reshape(
        batch_size,
        num_users * num_layers,
        num_rx,
        -1,
    )
    var_diag = variance.mean(dim=(0, 2)).real.clamp_min(1e-8)
    n = int(var_diag.shape[0])
    local_cov = torch.zeros(
        n,
        n,
        int(var_diag.shape[-1]),
        dtype=torch.complex64,
        device=mean.device,
    )
    index = torch.arange(n, device=mean.device)
    local_cov[index, index] = var_diag.to(torch.complex64)
    return PosteriorOutput(
        mean=mean.to(torch.complex64),
        var_diag=var_diag,
        local_cov=local_cov,
        latent_cov=torch.zeros(1, 1, dtype=torch.complex64, device=mean.device),
        effective_noise=torch.as_tensor(
            batch.noise_var, dtype=torch.float32, device=mean.device
        ),
    )


def ls_repaired_forward(
    receiver: Any,
    context: Any,
    detector: torch.nn.Module,
    batch: Any,
) -> dict[str, Any]:
    posterior = ls_posterior_from_receiver(receiver, context, batch)
    _, graph = posterior_graph(posterior, batch)
    output = detector(
        batch.y,
        posterior.mean,
        posterior.local_cov,
        batch.data_idx,
        batch.noise_var,
        graph,
        covariance_mode=SELECTED_COVARIANCE_MODE,
    )
    result = dict(output)
    result["posterior"] = posterior
    result["reference_graph_mask"] = graph
    result["graph_mask"] = graph
    result["edge_density"] = graph.float().mean()
    result["graph_mode"] = "ls_posterior"
    return result


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
