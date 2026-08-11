#!/usr/bin/env python3
from __future__ import annotations

"""Shared helpers for Gate-1 joint posterior-operator training.

This gate keeps the selected damped-extrinsic spatial-LMMSE detector fixed and
trains the stochastic channel operator through that detector.  It also supports
a case-specific operator as a diagnostic upper bound.  The case-specific model
is not a publication candidate; it is used only to decide whether the remaining
gap is caused by global parameter sharing or by the kernel basis itself.
"""

import hashlib
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]

from bayesroute.lmmse_ep import (  # noqa: E402
    DETECTOR_VERSION,
    DampedExtrinsicLMMSEDetector,
    full_directed_graph,
)
from bayesroute.models import (  # noqa: E402
    PosteriorOutput,
    coupling_matrix,
    coupling_selection_mask,
    edge_density,
)
from bayesroute.nr_gate1 import (  # noqa: E402
    NRBayesRouteBridge,
    NRCase,
    fixed_cardinality_mask,
)

JOINT_OPERATOR_VERSION = "gate1_nr_joint_operator_v1"
SELECTED_DETECTOR_ITERATIONS = 4
SELECTED_DETECTOR_DAMPING = 0.7
SELECTED_EDGE_MASS = 0.8
SELECTED_COVARIANCE_MODE = "diagonal"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_json(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def atomic_torch_save(value: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def set_all_seeds(seed: int) -> None:
    import sionna.phy

    random.seed(int(seed))
    np.random.seed(int(seed) % (2**32 - 1))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    sionna.phy.config.seed = int(seed)


def package_signature(payload: Any) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def unique_parameters(modules: Iterable[torch.nn.Module]) -> list[torch.nn.Parameter]:
    result: list[torch.nn.Parameter] = []
    seen: set[int] = set()
    for module in modules:
        for parameter in module.parameters():
            if not parameter.requires_grad or id(parameter) in seen:
                continue
            result.append(parameter)
            seen.add(id(parameter))
    return result


def bind_shared_operator(bridges: Sequence[NRBayesRouteBridge]) -> None:
    """Make all bridges use the same trainable operator parameters.

    Grid-dependent Fourier features and pilot indices remain local buffers.  Only
    the nonnegative spectral weights and effective-noise scale are shared.
    """
    if not bridges:
        raise ValueError("At least one bridge is required")
    master = bridges[0].posterior
    for bridge in bridges[1:]:
        if bridge.posterior.raw_weights.shape != master.raw_weights.shape:
            raise ValueError("Cannot share operators with different active ranks")
        bridge.posterior.raw_weights = master.raw_weights
        bridge.posterior.log_noise_scale = master.log_noise_scale


def shared_parameter_report(bridges: Sequence[NRBayesRouteBridge]) -> dict[str, Any]:
    raw_ids = [id(item.posterior.raw_weights) for item in bridges]
    noise_ids = [id(item.posterior.log_noise_scale) for item in bridges]
    return {
        "raw_weight_ids_identical": len(set(raw_ids)) == 1,
        "noise_scale_ids_identical": len(set(noise_ids)) == 1,
        "raw_weight_shape": list(bridges[0].posterior.raw_weights.shape),
        "unique_trainable_parameters": int(
            sum(p.numel() for p in unique_parameters(bridges))
        ),
    }


def extract_operator_state(bridge: NRBayesRouteBridge) -> dict[str, torch.Tensor]:
    return {
        "raw_weights": bridge.posterior.raw_weights.detach().cpu().clone(),
        "log_noise_scale": bridge.posterior.log_noise_scale.detach().cpu().clone(),
    }


def load_operator_state(
    bridge: NRBayesRouteBridge,
    state: dict[str, torch.Tensor],
) -> None:
    raw = torch.as_tensor(
        state["raw_weights"],
        dtype=bridge.posterior.raw_weights.dtype,
        device=bridge.posterior.raw_weights.device,
    )
    noise = torch.as_tensor(
        state["log_noise_scale"],
        dtype=bridge.posterior.log_noise_scale.dtype,
        device=bridge.posterior.log_noise_scale.device,
    )
    if raw.shape != bridge.posterior.raw_weights.shape:
        raise ValueError(
            f"Operator rank mismatch: checkpoint={tuple(raw.shape)}, "
            f"model={tuple(bridge.posterior.raw_weights.shape)}"
        )
    with torch.no_grad():
        bridge.posterior.raw_weights.copy_(raw)
        bridge.posterior.log_noise_scale.copy_(noise)


def copy_old_operator_if_compatible(
    bridge: NRBayesRouteBridge,
    old_state_dict: dict[str, torch.Tensor],
) -> bool:
    raw_key = "posterior.raw_weights"
    noise_key = "posterior.log_noise_scale"
    if raw_key not in old_state_dict or noise_key not in old_state_dict:
        raise RuntimeError("Old checkpoint has no posterior operator parameters")
    if old_state_dict[raw_key].shape != bridge.posterior.raw_weights.shape:
        return False
    load_operator_state(
        bridge,
        {
            "raw_weights": old_state_dict[raw_key],
            "log_noise_scale": old_state_dict[noise_key],
        },
    )
    return True


def make_repaired_detector(bits_per_symbol: int) -> DampedExtrinsicLMMSEDetector:
    return DampedExtrinsicLMMSEDetector(
        int(bits_per_symbol),
        n_iter=SELECTED_DETECTOR_ITERATIONS,
        damping=SELECTED_DETECTOR_DAMPING,
        covariance_mode=SELECTED_COVARIANCE_MODE,
    )


def posterior_for_batch(
    bridge: NRBayesRouteBridge,
    batch: Any,
) -> PosteriorOutput:
    return bridge.posterior(
        batch.y[..., batch.pilot_idx],
        batch.phi,
        batch.noise_var,
    )


def posterior_graph(
    posterior: PosteriorOutput,
    batch: Any,
    *,
    edge_mass: float = SELECTED_EDGE_MASS,
) -> tuple[torch.Tensor, torch.Tensor]:
    kappa = coupling_matrix(
        posterior.mean.detach(),
        posterior.local_cov.detach(),
        batch.data_idx,
        batch.noise_var.detach(),
    )
    graph = coupling_selection_mask(kappa, float(edge_mass))
    return kappa, graph


def random_fixed_cardinality_graph(
    reference: torch.Tensor,
    seed: int,
) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    scores = torch.rand(reference.shape, generator=generator, dtype=torch.float32)
    return fixed_cardinality_mask(scores.to(reference.device), reference)


def repaired_forward(
    bridge: NRBayesRouteBridge,
    detector: DampedExtrinsicLMMSEDetector,
    batch: Any,
    *,
    graph_mode: str = "posterior",
    covariance_mode: str = SELECTED_COVARIANCE_MODE,
    random_seed: int = 0,
    posterior: PosteriorOutput | None = None,
    reference_graph: torch.Tensor | None = None,
) -> dict[str, Any]:
    if posterior is None:
        posterior = posterior_for_batch(bridge, batch)
    if reference_graph is None:
        _, reference_graph = posterior_graph(posterior, batch)

    graph = reference_graph
    if graph_mode == "posterior":
        pass
    elif graph_mode == "full":
        graph = full_directed_graph(
            int(batch.y.shape[0]),
            int(batch.data_idx.numel()),
            int(batch.h.shape[1]),
            device=batch.y.device,
        )
    elif graph_mode == "off":
        graph = torch.zeros_like(reference_graph)
    elif graph_mode == "random":
        graph = random_fixed_cardinality_graph(reference_graph, int(random_seed))
    else:
        raise ValueError(f"Unknown graph_mode: {graph_mode}")

    output = detector(
        batch.y,
        posterior.mean,
        posterior.local_cov,
        batch.data_idx,
        batch.noise_var,
        graph,
        covariance_mode=str(covariance_mode),
    )
    result = dict(output)
    result["posterior"] = posterior
    result["reference_graph_mask"] = reference_graph
    result["graph_mask"] = graph
    result["edge_density"] = edge_density(graph)
    result["graph_mode"] = graph_mode
    return result


def differentiable_channel_nll(
    posterior: PosteriorOutput,
    batch: Any,
) -> torch.Tensor:
    mean = posterior.mean[..., batch.data_idx]
    truth = batch.h[..., batch.data_idx]
    var = posterior.var_diag[None, :, None, batch.data_idx].to(mean.device)
    var = var.clamp_min(1e-8)
    return torch.mean(torch.abs(mean - truth) ** 2 / var + torch.log(var)).real


def differentiable_calibration_penalty(
    posterior: PosteriorOutput,
    batch: Any,
) -> torch.Tensor:
    """Penalize global variance-scale mismatch on the current supervised batch."""
    mean = posterior.mean[..., batch.data_idx]
    truth = batch.h[..., batch.data_idx]
    var = posterior.var_diag[None, :, None, batch.data_idx].to(mean.device)
    normalized = torch.mean(torch.abs(mean - truth) ** 2 / var.clamp_min(1e-8)).real
    return torch.square(torch.log(normalized.clamp(0.05, 20.0)))


def differentiable_loss(
    output: dict[str, Any],
    batch: Any,
    *,
    channel_loss_weight: float,
    calibration_loss_weight: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    bit_nll = F.binary_cross_entropy_with_logits(
        output["bit_logits"], batch.coded_bits.float()
    )
    channel_nll = differentiable_channel_nll(output["posterior"], batch)
    calibration = differentiable_calibration_penalty(output["posterior"], batch)
    loss = (
        bit_nll
        + float(channel_loss_weight) * channel_nll
        + float(calibration_loss_weight) * calibration
    )
    return loss, {
        "bit_nll": bit_nll,
        "channel_nll": channel_nll,
        "calibration_penalty": calibration,
    }


def posterior_metrics(output: dict[str, Any], batch: Any) -> dict[str, float]:
    posterior = output["posterior"]
    mean = posterior.mean[..., batch.data_idx]
    truth = batch.h[..., batch.data_idx]
    var = posterior.var_diag[None, :, None, batch.data_idx].to(mean.device)
    error = torch.abs(mean - truth) ** 2
    nmse = error.mean() / torch.abs(truth).square().mean().clamp_min(1e-8)
    normalized = error / var.clamp_min(1e-8)
    threshold = -math.log(0.05) * var
    return {
        "channel_nmse": float(nmse.real.item()),
        "normalized_error_mean": float(normalized.real.mean().item()),
        "coverage95": float((error <= threshold).float().mean().item()),
    }


def coded_metrics(output: dict[str, Any], batch: Any) -> dict[str, float]:
    logits = output["bit_logits"]
    bits = batch.coded_bits.float()
    hard = logits >= 0
    probabilities = torch.sigmoid(logits)
    return {
        "coded_ber": float((hard != bits.bool()).float().mean().item()),
        "coded_bit_nll": float(
            F.binary_cross_entropy_with_logits(logits, bits).item()
        ),
        "coded_brier": float(torch.mean((probabilities - bits) ** 2).item()),
    }


def gradient_report(parameters: Sequence[torch.nn.Parameter]) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    total_sq = 0.0
    for index, parameter in enumerate(parameters):
        gradient = parameter.grad
        present = gradient is not None
        finite = bool(present and torch.isfinite(gradient).all().item())
        norm = float(gradient.detach().norm().item()) if present else 0.0
        total_sq += norm * norm
        entries.append(
            {
                "index": index,
                "shape": list(parameter.shape),
                "present": present,
                "finite": finite,
                "norm": norm,
            }
        )
    return {
        "entries": entries,
        "all_present": all(item["present"] for item in entries),
        "all_finite": all(item["finite"] for item in entries),
        "total_norm": math.sqrt(total_sq),
    }


def pure_torch_shared_parameter_self_test() -> dict[str, Any]:
    """Small login-node test independent of Sionna and NR channel objects."""
    class Holder(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.raw_weights = torch.nn.Parameter(torch.zeros(4))
            self.log_noise_scale = torch.nn.Parameter(torch.tensor(0.0))

    first = Holder()
    second = Holder()
    second.raw_weights = first.raw_weights
    second.log_noise_scale = first.log_noise_scale
    optimizer = torch.optim.SGD(unique_parameters([first, second]), lr=0.1)
    optimizer.zero_grad(set_to_none=True)
    loss = torch.square(first.raw_weights - 1.0).sum()
    loss = loss + 0.5 * torch.square(second.raw_weights + 0.5).sum()
    loss = loss + torch.square(first.log_noise_scale - 0.2)
    loss.backward()
    before = first.raw_weights.detach().clone()
    optimizer.step()
    changed = not torch.equal(before, first.raw_weights.detach())
    result = {
        "raw_identity": id(first.raw_weights) == id(second.raw_weights),
        "noise_identity": id(first.log_noise_scale) == id(second.log_noise_scale),
        "unique_parameter_count": len(unique_parameters([first, second])),
        "gradient_finite": bool(torch.isfinite(first.raw_weights.grad).all().item()),
        "optimizer_changed_parameter": changed,
    }
    result["passed"] = bool(
        result["raw_identity"]
        and result["noise_identity"]
        and result["unique_parameter_count"] == 2
        and result["gradient_finite"]
        and result["optimizer_changed_parameter"]
    )
    return result


@dataclass(frozen=True)
class CandidateSpec:
    name: str
    mode: str
    rank: int
    bank_rank: int
    length_f: float
    length_t: float
    initialization: str
    learning_rate: float
    channel_loss_weight: float
    calibration_loss_weight: float
    steps: int

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> "CandidateSpec":
        return cls(
            name=str(value["name"]),
            mode=str(value["mode"]),
            rank=int(value["rank"]),
            bank_rank=int(value["bank_rank"]),
            length_f=float(value["length_f"]),
            length_t=float(value["length_t"]),
            initialization=str(value["initialization"]),
            learning_rate=float(value["learning_rate"]),
            channel_loss_weight=float(value["channel_loss_weight"]),
            calibration_loss_weight=float(value["calibration_loss_weight"]),
            steps=int(value["steps"]),
        )

    def validate(self) -> None:
        if self.mode not in {"global", "case_specific"}:
            raise ValueError(f"Unsupported candidate mode: {self.mode}")
        if self.initialization not in {"old_checkpoint", "cold"}:
            raise ValueError(f"Unsupported initialization: {self.initialization}")
        if self.bank_rank < self.rank:
            raise ValueError("bank_rank must be at least rank")
        if self.steps <= 0:
            raise ValueError("steps must be positive")


def parse_cases(raw_cases: Sequence[dict[str, Any]]) -> list[NRCase]:
    return [NRCase.from_mapping(dict(item)) for item in raw_cases]
