#!/usr/bin/env python3
from __future__ import annotations

"""Shared helpers for the final implementable localized receiver gate."""

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

from bayesroute.ls_anchored_localized_posterior import (
    IMPLEMENTABLE_LOCALIZED_VERSION,
    LSAnchoredLocalizedResidualPosterior,
    bind_shared_localized_parameters,
    load_shared_localized_state,
    shared_localized_state,
    unique_localized_parameters,
)
from bayesroute.localized_delay_doppler import (
    LocalizedDelayDopplerSpec,
    localized_delay_doppler_features,
)
from bayesroute.models import PosteriorOutput
from bayesroute.nr_gate1 import (
    NRCase,
    build_nr_context,
    decode_bridge,
    normalize_device,
    run_standard_receiver,
    standard_receiver,
)
from gate1_nr_joint_operator_common import (
    coded_metrics,
    make_repaired_detector,
    posterior_graph,
    posterior_metrics,
    repaired_forward,
)
from gate1_nr_posterior_factorial_common import (
    ls_posterior_from_receiver,
    ls_repaired_forward,
)
from gate1_nr_turbo_posterior_common import build_loaded_bridge


GATE_VERSION = "gate1_nr_implementable_localized_v1"
REQUIRED_CEILING_CLASSIFICATION = "GATE1_LOCALIZED_CEILING_BEATS_LS"
REQUIRED_CEILING_VERSION = "gate1_nr_localized_ceiling_v1_1"
REQUIRED_BASIS_PRECISION_PATCH = "complex128_atoms_before_rank_decision_v1"
REQUIRED_WINNER = "ldd_w3_d10_v3_r90_tau2us"
REQUIRED_CEILING_ROWS = 540
CEILING_REPORT = ROOT / "outputs/reports/gate1_nr_localized_ceiling.json"

SOURCE_FILES = [
    "configs/gate1_nr_implementable_localized_smoke.yaml",
    "configs/gate1_nr_implementable_localized.yaml",
    "src/bayesroute/ls_anchored_localized_posterior.py",
    "src/bayesroute/localized_delay_doppler.py",
    "scripts/gate1_nr_implementable_localized_common.py",
    "scripts/gate1_nr_implementable_localized_smoke.py",
    "scripts/gate1_nr_implementable_localized.py",
    "scripts/gate1_nr_joint_operator_common.py",
    "scripts/gate1_nr_posterior_factorial_common.py",
    "scripts/gate1_nr_turbo_posterior_common.py",
    "src/bayesroute/multiscale_posterior.py",
    "src/bayesroute/lmmse_ep.py",
    "src/bayesroute/models.py",
    "src/bayesroute/nr_gate1.py",
    "src/bayesroute/qam.py",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_hashes() -> dict[str, str]:
    return {name: sha256_file(ROOT / name) for name in SOURCE_FILES}


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


def package_signature(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


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


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected JSON object: {path}")
    return value


def localized_ceiling_winner_spec(report: dict[str, Any]) -> dict[str, Any]:
    """Return the full selected-basis mapping across report schema versions.

    The precision-corrected ceiling report stores the human-readable winner at
    top level as a string, while the complete basis specification is stored in
    ``holdout.contract.winner``.  Older reports may store the complete mapping
    directly at top level.  Downstream training requires the complete mapping,
    not only the name.
    """
    holdout_winner = (
        report.get("holdout", {})
        .get("contract", {})
        .get("winner")
    )
    if isinstance(holdout_winner, dict) and holdout_winner.get("name"):
        return dict(holdout_winner)

    top_level_winner = report.get("winner")
    if isinstance(top_level_winner, dict) and top_level_winner.get("name"):
        return dict(top_level_winner)

    raise RuntimeError(
        "Localized ceiling report does not contain a complete winner "
        "specification in holdout.contract.winner or winner"
    )


def localized_ceiling_preconditions() -> dict[str, Any]:
    if not CEILING_REPORT.is_file():
        raise RuntimeError(f"Missing localized ceiling report: {CEILING_REPORT}")
    report = load_json(CEILING_REPORT)
    winner = localized_ceiling_winner_spec(report)
    checks = {
        "complete": report.get("complete") is True,
        "version": report.get("version") == REQUIRED_CEILING_VERSION,
        "classification": report.get("classification") == REQUIRED_CEILING_CLASSIFICATION,
        "rows": report.get("evaluation", {}).get("rows") == REQUIRED_CEILING_ROWS,
        "unique_rows": report.get("evaluation", {}).get("unique_rows") == REQUIRED_CEILING_ROWS,
        "winner": winner.get("name") == REQUIRED_WINNER,
        "precision_patch": report.get("basis_precision_patch") == REQUIRED_BASIS_PRECISION_PATCH,
        "holdout_unused": report.get("holdout", {}).get("contract", {}).get("holdout_used_for_selection") is False,
        "oracle_beats_ls": report.get("scientific_checks", {}).get("localized_oracle_statistically_beats_ls") is True,
        "true_channel_control": report.get("scientific_checks", {}).get("true_channel_repaired_matches_perfect") is True,
    }
    if not all(checks.values()):
        raise RuntimeError(f"Implementable-localized preconditions failed: {checks}")
    return {
        "passed": True,
        "checks": checks,
        "report": str(CEILING_REPORT.relative_to(ROOT)),
        "classification": report["classification"],
        "winner": winner,
        "ceiling_metrics": report.get("metrics", {}),
    }


def selected_basis_spec(preconditions: dict[str, Any] | None = None) -> LocalizedDelayDopplerSpec:
    pre = localized_ceiling_preconditions() if preconditions is None else preconditions
    spec = LocalizedDelayDopplerSpec.from_mapping(pre["winner"])
    if spec.name != REQUIRED_WINNER:
        raise RuntimeError("Localized winner identity mismatch")
    return spec


def context_vector(case: NRCase, device: torch.device) -> torch.Tensor:
    return torch.tensor(
        [
            math.log2(max(float(case.num_prb), 1.0) / 4.0),
            float(case.speed_mps) / 30.0,
        ],
        dtype=torch.float32,
        device=device,
    )


@dataclass
class LocalizedStackItem:
    case: NRCase
    context: Any
    operator: LSAnchoredLocalizedResidualPosterior
    detector: torch.nn.Module
    ls_receiver: Any
    basis_report: dict[str, Any]


def build_stack_item(
    raw_case: dict[str, Any],
    device: torch.device,
    spec: LocalizedDelayDopplerSpec,
    *,
    num_knots: int,
) -> LocalizedStackItem:
    case = NRCase.from_mapping(raw_case)
    case.validate()
    context = build_nr_context(case, device)
    features, basis_report = localized_delay_doppler_features(
        num_symbols=int(context.grid.num_ofdm_symbols),
        num_subcarriers=int(context.grid.num_effective_subcarriers),
        subcarrier_spacing_khz=float(case.subcarrier_spacing_khz),
        spec=spec,
        device=device,
    )
    operator = LSAnchoredLocalizedResidualPosterior(
        features=features,
        pilot_idx=context.grid.pilot_idx,
        n_layers=case.num_streams,
        nominal_rank=spec.nominal_rank,
        context=context_vector(case, device),
        num_knots=int(num_knots),
    ).to(device)
    detector = make_repaired_detector(int(context.grid.bits_per_symbol)).to(device)
    ls_receiver = standard_receiver(context, perfect_csi=False, return_crc=True)
    return LocalizedStackItem(
        case=case,
        context=context,
        operator=operator,
        detector=detector,
        ls_receiver=ls_receiver,
        basis_report=basis_report,
    )


def build_shared_stack(
    raw_cases: Sequence[dict[str, Any]],
    device: torch.device,
    spec: LocalizedDelayDopplerSpec,
    *,
    num_knots: int,
) -> list[LocalizedStackItem]:
    items = [
        build_stack_item(raw, device, spec, num_knots=num_knots)
        for raw in raw_cases
    ]
    bind_shared_localized_parameters([item.operator for item in items])
    return items


def observable_forward(
    item: LocalizedStackItem,
    batch: Any,
) -> dict[str, Any]:
    ls_posterior, alignment = ls_posterior_from_receiver(
        item.ls_receiver, item.context, batch
    )
    result = item.operator(
        y_p=batch.y[..., batch.pilot_idx],
        phi=batch.phi,
        noise_var=batch.noise_var,
        ls_mean=ls_posterior.mean,
        ls_var_diag=ls_posterior.var_diag,
    )
    _, graph = posterior_graph(result.posterior, batch)
    output = repaired_forward(
        None,
        item.detector,
        batch,
        posterior=result.posterior,
        reference_graph=graph,
    )
    output["localized_result"] = result
    output["ls_posterior"] = ls_posterior
    output["ls_grid_alignment_report"] = alignment
    output["basis_report"] = item.basis_report
    output["inference_uses_true_channel"] = False
    return output


def uncertainty_off_forward(
    item: LocalizedStackItem,
    batch: Any,
    trained_output: dict[str, Any],
) -> dict[str, Any]:
    posterior = trained_output["posterior"]
    graph = trained_output["reference_graph_mask"]
    raw = item.detector(
        batch.y,
        posterior.mean,
        posterior.local_cov,
        batch.data_idx,
        batch.noise_var,
        graph,
        covariance_mode="none",
    )
    output = dict(raw)
    output["posterior"] = posterior
    output["reference_graph_mask"] = graph
    output["graph_mask"] = graph
    output["edge_density"] = graph.float().mean()
    output["localized_result"] = trained_output["localized_result"]
    return output


def mean_only_forward(
    item: LocalizedStackItem,
    batch: Any,
    trained_output: dict[str, Any],
) -> dict[str, Any]:
    trained = trained_output["posterior"]
    ls = trained_output["ls_posterior"]
    mean_only = PosteriorOutput(
        mean=trained.mean,
        var_diag=ls.var_diag,
        local_cov=ls.local_cov,
        latent_cov=trained.latent_cov,
        effective_noise=trained.effective_noise,
    )
    graph = trained_output["reference_graph_mask"]
    output = repaired_forward(
        None,
        item.detector,
        batch,
        posterior=mean_only,
        reference_graph=graph,
    )
    output["localized_result"] = trained_output["localized_result"]
    return output


def channel_mse(posterior: PosteriorOutput, batch: Any) -> torch.Tensor:
    mean = posterior.mean[..., batch.data_idx]
    truth = batch.h[..., batch.data_idx]
    return torch.mean(torch.abs(mean - truth) ** 2).real


def channel_nll(posterior: PosteriorOutput, batch: Any) -> torch.Tensor:
    mean = posterior.mean[..., batch.data_idx]
    truth = batch.h[..., batch.data_idx]
    var = posterior.var_diag[None, :, None, batch.data_idx].to(mean.device)
    var = var.clamp_min(1e-8)
    return torch.mean(torch.abs(mean - truth) ** 2 / var + torch.log(var)).real


def calibration_penalty(posterior: PosteriorOutput, batch: Any) -> torch.Tensor:
    mean = posterior.mean[..., batch.data_idx]
    truth = batch.h[..., batch.data_idx]
    var = posterior.var_diag[None, :, None, batch.data_idx].to(mean.device)
    normalized = torch.mean(torch.abs(mean - truth) ** 2 / var.clamp_min(1e-8)).real
    return torch.square(torch.log(normalized.clamp(0.05, 20.0)))


def differentiable_training_loss(
    output: dict[str, Any],
    batch: Any,
    *,
    channel_loss_weight: float,
    calibration_loss_weight: float,
    ls_gain_loss_weight: float,
    ls_gain_target_ratio: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    bit_nll = F.binary_cross_entropy_with_logits(
        output["bit_logits"], batch.coded_bits.float()
    )
    posterior = output["posterior"]
    ch_nll = channel_nll(posterior, batch)
    calibration = calibration_penalty(posterior, batch)
    fused_mse = channel_mse(posterior, batch)
    ls_mse = channel_mse(output["ls_posterior"], batch).detach().clamp_min(1e-8)
    gain_hinge = F.relu(fused_mse / ls_mse - float(ls_gain_target_ratio))
    loss = (
        bit_nll
        + float(channel_loss_weight) * ch_nll
        + float(calibration_loss_weight) * calibration
        + float(ls_gain_loss_weight) * gain_hinge
    )
    return loss, {
        "bit_nll": bit_nll,
        "channel_nll": ch_nll,
        "calibration_penalty": calibration,
        "fused_channel_mse": fused_mse,
        "ls_channel_mse": ls_mse,
        "ls_gain_hinge": gain_hinge,
    }


def observable_metrics(output: dict[str, Any], batch: Any) -> dict[str, float]:
    base = {**coded_metrics(output, batch), **posterior_metrics(output, batch)}
    result = output.get("localized_result")
    if result is not None:
        base.update(
            {
                "residual_gate": float(result.residual_gate.detach().item()),
                "correction_power": float(result.correction_power.detach().mean().item()),
                "effective_rank": float(result.diagnostics["effective_rank"]),
                "delta_gain_mean_abs": float(
                    result.diagnostics["delta_gain_mean_abs"].detach().item()
                ),
            }
        )
    return base


def gradient_report(parameters: Sequence[torch.nn.Parameter]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for index, parameter in enumerate(parameters):
        gradient = parameter.grad
        rows.append(
            {
                "index": index,
                "shape": list(parameter.shape),
                "present": gradient is not None,
                "finite": bool(
                    gradient is not None and torch.isfinite(gradient).all().item()
                ),
                "norm": float(gradient.norm().item()) if gradient is not None else 0.0,
            }
        )
    return {
        "entries": rows,
        "all_present": all(item["present"] for item in rows),
        "all_finite": all(item["finite"] for item in rows),
        "any_nonzero": any(item["norm"] > 0.0 for item in rows),
        "total_norm": math.sqrt(sum(item["norm"] ** 2 for item in rows)),
    }


def decode_outputs(
    context: Any,
    batch: Any,
    outputs: dict[str, dict[str, Any]],
    *,
    bp_iterations: int,
    device: torch.device,
) -> dict[str, dict[str, Any]]:
    from sionna.phy.nr import LayerDemapper, TBDecoder

    decoder = TBDecoder(
        context.transmitter._tb_encoder,
        num_bp_iter=int(bp_iterations),
        device=str(device),
    )
    demapper = LayerDemapper(
        context.transmitter._layer_mapper,
        num_bits_per_symbol=int(context.grid.bits_per_symbol),
        device=str(device),
    )
    return {
        name: decode_bridge(
            context.transmitter,
            output,
            batch.information_bits,
            num_bp_iter=int(bp_iterations),
            device=device,
            decoder=decoder,
            layer_demapper=demapper,
        )
        for name, output in outputs.items()
    }


def crc_disagreement(decoded: dict[str, Any], bits: torch.Tensor) -> float:
    block_error = (decoded["b_hat"] != bits).reshape(
        bits.shape[0], bits.shape[1], -1
    ).any(-1)
    crc_failure = ~decoded["crc"].bool()
    if crc_failure.shape != block_error.shape:
        crc_failure = crc_failure.reshape(block_error.shape)
    return float((block_error != crc_failure).float().mean().item())


def custom_row(
    *,
    case: NRCase,
    group: str,
    variant: str,
    snr: float,
    rep: int,
    seed: int,
    output: dict[str, Any],
    batch: Any,
    decoded: dict[str, Any],
    signature: str,
) -> dict[str, Any]:
    metrics = observable_metrics(output, batch)
    return {
        "case": case.name,
        "group": group,
        "scenario": case.scenario,
        "num_prb": int(case.num_prb),
        "num_streams": int(case.num_streams),
        "num_rx_ant": int(case.num_rx_ant),
        "variant": variant,
        "ebno_db": float(snr),
        "rep": int(rep),
        "eval_seed": int(seed),
        "information_ber": float(decoded["information_ber"]),
        "tbler": float(decoded["tbler"]),
        "crc_failure_rate": float(decoded["crc_failure_rate"]),
        "crc_block_disagreement_rate": crc_disagreement(
            decoded, batch.information_bits
        ),
        **metrics,
        "edge_density": float(output["edge_density"].item()),
        "inference_uses_true_channel": False,
        "contract_signature": signature,
    }


def standard_row(
    *,
    case: NRCase,
    group: str,
    variant: str,
    snr: float,
    rep: int,
    seed: int,
    metrics: dict[str, Any],
    signature: str,
) -> dict[str, Any]:
    return {
        "case": case.name,
        "group": group,
        "scenario": case.scenario,
        "num_prb": int(case.num_prb),
        "num_streams": int(case.num_streams),
        "num_rx_ant": int(case.num_rx_ant),
        "variant": variant,
        "ebno_db": float(snr),
        "rep": int(rep),
        "eval_seed": int(seed),
        "information_ber": float(metrics["information_ber"]),
        "tbler": float(metrics["tbler"]),
        "crc_failure_rate": float(metrics["crc_failure_rate"]),
        "crc_block_disagreement_rate": float("nan"),
        "coded_ber": float("nan"),
        "coded_bit_nll": float("nan"),
        "coded_brier": float("nan"),
        "channel_nmse": float("nan"),
        "normalized_error_mean": float("nan"),
        "coverage95": float("nan"),
        "residual_gate": float("nan"),
        "correction_power": float("nan"),
        "effective_rank": float("nan"),
        "delta_gain_mean_abs": float("nan"),
        "edge_density": float("nan"),
        "inference_uses_true_channel": False,
        "contract_signature": signature,
    }
