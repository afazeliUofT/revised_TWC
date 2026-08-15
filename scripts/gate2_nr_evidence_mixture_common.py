#!/usr/bin/env python3
from __future__ import annotations

"""Shared NR helpers for the evidence-mixture LMMSE estimator gate."""

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

from bayesroute.evidence_mixture_lmmse import (
    EVIDENCE_MIXTURE_LMMSE_VERSION,
    EvidenceMixtureLMMSEPosterior,
    EvidenceMixtureResult,
    bind_shared_evidence_parameters,
    evidence_state_to_jsonable,
    load_shared_evidence_state,
    shared_evidence_state,
    unique_evidence_parameters,
)
from bayesroute.localized_delay_doppler import (
    LocalizedDelayDopplerSpec,
    localized_delay_doppler_features,
)
from bayesroute.models import (
    PosteriorOutput,
    coupling_selection_mask,
)
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
    repaired_forward,
)
from gate1_nr_posterior_factorial_common import (
    ls_posterior_from_receiver,
    ls_repaired_forward,
)


GATE2_VERSION = "gate2_nr_evidence_mixture_lmmse_v1"
REQUIRED_SOURCE_CLASSIFICATION = "GATE1_IMPLEMENTABLE_LOCALIZED_FINAL_PARITY_WITH_LS"
REQUIRED_SOURCE_ROWS = 2304
SOURCE_REPORT = ROOT / "outputs/reports/gate1_nr_implementable_localized_confirmation.json"
SOURCE_CHECKPOINT_SHA256 = (
    "cfb1afd665d6df89e6590b611badd68a77b9ada43de07daeee7b8d8a27dd70aa"
)

SOURCE_FILES = [
    "configs/gate2_nr_evidence_mixture_smoke.yaml",
    "configs/gate2_nr_evidence_mixture_screen.yaml",
    "src/bayesroute/evidence_mixture_lmmse.py",
    "src/bayesroute/localized_delay_doppler.py",
    "scripts/gate2_nr_evidence_mixture_common.py",
    "scripts/gate2_nr_evidence_mixture_smoke.py",
    "scripts/gate2_nr_evidence_mixture_screen.py",
    "scripts/gate1_nr_joint_operator_common.py",
    "scripts/gate1_nr_posterior_factorial_common.py",
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


def source_result_preconditions() -> dict[str, Any]:
    if not SOURCE_REPORT.is_file():
        raise RuntimeError(f"Missing synchronized Gate-1 result: {SOURCE_REPORT}")
    report = load_json(SOURCE_REPORT)
    checks = {
        "complete": report.get("complete") is True,
        "classification": report.get("classification") == REQUIRED_SOURCE_CLASSIFICATION,
        "rows": report.get("evaluation", {}).get("rows") == REQUIRED_SOURCE_ROWS,
        "unique_rows": report.get("evaluation", {}).get("unique_rows") == REQUIRED_SOURCE_ROWS,
        "no_training": report.get("preconditions", {}).get("no_retraining") is True,
        "no_retuning": report.get("preconditions", {}).get("no_retuning") is True,
        "frozen_checkpoint": (
            report.get("preconditions", {}).get("checkpoint_sha256")
            == SOURCE_CHECKPOINT_SHA256
        ),
        "inference_no_truth": (
            report.get("evaluation", {}).get("contract", {}).get("policy", {}).get(
                "inference_uses_true_channel"
            )
            is False
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(f"Gate-2 source-result preconditions failed: {checks}")
    return {
        "passed": True,
        "checks": checks,
        "report": str(SOURCE_REPORT.relative_to(ROOT)),
        "classification": report["classification"],
        "metrics": report.get("metrics", {}),
        "next_action": report.get("next_action"),
    }


def estimator_context_vector(case: NRCase, device: torch.device) -> torch.Tensor:
    """Known NR configuration only; no true channel or scenario label."""
    return torch.tensor(
        [
            math.log2(max(float(case.num_prb), 1.0) / 4.0),
            math.log2(max(float(case.subcarrier_spacing_khz), 1.0) / 30.0),
            float(case.dmrs_config_type - 1),
            float(case.dmrs_length - 1),
            float(case.dmrs_additional_position) / 3.0,
            math.log2(max(float(case.num_streams), 1.0)),
            math.log2(max(float(case.num_rx_ant), 1.0)),
            float(case.mcs_index) / 28.0,
        ],
        dtype=torch.float32,
        device=device,
    )


@dataclass
class EvidenceStackItem:
    case: NRCase
    context: Any
    operator: EvidenceMixtureLMMSEPosterior
    detector: torch.nn.Module
    ls_receiver: Any
    basis_report: dict[str, Any]


def basis_spec_from_config(config: dict[str, Any]) -> LocalizedDelayDopplerSpec:
    return LocalizedDelayDopplerSpec.from_mapping(config["basis"])


def build_stack_item(
    raw_case: dict[str, Any],
    device: torch.device,
    spec: LocalizedDelayDopplerSpec,
    *,
    num_components: int,
    num_knots: int,
) -> EvidenceStackItem:
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
    operator = EvidenceMixtureLMMSEPosterior(
        features=features,
        pilot_idx=context.grid.pilot_idx,
        n_layers=case.num_streams,
        nominal_rank=spec.nominal_rank,
        context=estimator_context_vector(case, device),
        num_components=int(num_components),
        num_knots=int(num_knots),
    ).to(device)
    detector = make_repaired_detector(int(context.grid.bits_per_symbol)).to(device)
    ls_receiver = standard_receiver(context, perfect_csi=False, return_crc=True)
    return EvidenceStackItem(
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
    num_components: int,
    num_knots: int,
) -> list[EvidenceStackItem]:
    items = [
        build_stack_item(
            raw,
            device,
            spec,
            num_components=num_components,
            num_knots=num_knots,
        )
        for raw in raw_cases
    ]
    bind_shared_evidence_parameters([item.operator for item in items])
    return items


def _variance_for_mean(
    posterior: PosteriorOutput,
    mean: torch.Tensor,
    data_idx: torch.Tensor,
) -> torch.Tensor:
    variance = posterior.var_diag
    if variance.ndim == 2:
        return variance[None, :, None, data_idx].to(mean.device)
    if variance.ndim == 3:
        if int(variance.shape[0]) != int(mean.shape[0]):
            raise ValueError("Batch-dependent variance has wrong batch size")
        return variance[:, :, None, data_idx].to(mean.device)
    raise ValueError("var_diag must have shape [N,R] or [B,N,R]")


def batch_posterior_graph(
    posterior: PosteriorOutput,
    batch: Any,
    *,
    edge_mass: float = 0.8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Expected interference coupling for shared or batch-dependent covariance."""
    mean = posterior.mean.detach()[..., batch.data_idx]
    covariance = posterior.local_cov.detach()
    if covariance.ndim == 3:
        covariance = covariance[None, ...].expand(int(mean.shape[0]), -1, -1, -1)
    elif covariance.ndim != 4:
        raise ValueError("local_cov must have shape [N,N,R] or [B,N,N,R]")
    if int(covariance.shape[0]) != int(mean.shape[0]):
        raise ValueError("Batch-dependent covariance has wrong batch size")
    covariance = covariance[..., batch.data_idx].to(mean.device)
    diagonal = torch.diagonal(covariance, dim1=1, dim2=2).permute(0, 2, 1)
    diagonal = diagonal.real.clamp_min(0.0)  # [B,N,D]
    batch_size, n_layers, n_rx, n_data = mean.shape
    noise = torch.as_tensor(batch.noise_var, device=mean.device).real.to(torch.float32)
    if noise.ndim == 0:
        noise = noise.expand(batch_size)
    elif noise.ndim != 1 or int(noise.shape[0]) != batch_size:
        noise = noise.reshape(batch_size, -1).mean(dim=1)
    inv_noise = 1.0 / noise.clamp_min(1e-6)
    inv_noise_sq = inv_noise.square()
    coupling = torch.zeros(
        batch_size, n_data, n_layers, n_layers,
        dtype=torch.float32, device=mean.device,
    )
    for n in range(n_layers):
        for m in range(n + 1, n_layers):
            cross_trace = (
                float(n_rx)
                * covariance[:, m, n]
                * inv_noise[:, None].to(covariance.dtype)
            )
            coherent = (
                torch.sum(mean[:, n].conj() * mean[:, m], dim=1)
                * inv_noise[:, None].to(mean.dtype)
                + cross_trace
            )
            term0 = torch.abs(coherent).square()
            term1 = torch.sum(
                torch.abs(mean[:, n]).square()
                * diagonal[:, m, None, :]
                * inv_noise_sq[:, None, None],
                dim=1,
            )
            term2 = torch.sum(
                torch.abs(mean[:, m]).square()
                * diagonal[:, n, None, :]
                * inv_noise_sq[:, None, None],
                dim=1,
            )
            term3 = (
                float(n_rx)
                * diagonal[:, n]
                * diagonal[:, m]
                * inv_noise_sq[:, None]
            )
            value = (term0 + term1 + term2 + term3).real.clamp_min(0.0)
            coupling[:, :, n, m] = value
            coupling[:, :, m, n] = value
    graph = coupling_selection_mask(coupling, float(edge_mass))
    return coupling, graph


def evidence_forward(
    item: EvidenceStackItem,
    batch: Any,
    *,
    mode: str = "mixture",
) -> dict[str, Any]:
    result = item.operator(
        y_p=batch.y[..., batch.pilot_idx],
        phi=batch.phi,
        noise_var=batch.noise_var,
        mode=mode,
    )
    _, graph = batch_posterior_graph(result.posterior, batch)
    output = repaired_forward(
        None,
        item.detector,
        batch,
        posterior=result.posterior,
        reference_graph=graph,
    )
    output["evidence_result"] = result
    output["basis_report"] = item.basis_report
    output["inference_uses_true_channel"] = False
    return output


def perfect_channel_forward(item: EvidenceStackItem, batch: Any) -> dict[str, Any]:
    n_layers = int(batch.h.shape[1])
    n_re = int(batch.h.shape[-1])
    var_diag = torch.full(
        (n_layers, n_re),
        1e-8,
        dtype=torch.float32,
        device=batch.h.device,
    )
    local_cov = torch.diag_embed(var_diag.transpose(0, 1).to(torch.complex64))
    local_cov = local_cov.permute(1, 2, 0).contiguous()
    posterior = PosteriorOutput(
        mean=batch.h,
        var_diag=var_diag,
        local_cov=local_cov,
        latent_cov=torch.zeros(1, 1, dtype=torch.complex64, device=batch.h.device),
        effective_noise=batch.noise_var,
    )
    _, graph = batch_posterior_graph(posterior, batch)
    output = repaired_forward(
        None,
        item.detector,
        batch,
        posterior=posterior,
        reference_graph=graph,
    )
    output["inference_uses_true_channel"] = True
    return output


def channel_mse(posterior: PosteriorOutput, batch: Any) -> torch.Tensor:
    mean = posterior.mean[..., batch.data_idx]
    truth = batch.h[..., batch.data_idx]
    return torch.mean(torch.abs(mean - truth) ** 2).real


def channel_nmse_tensor(posterior: PosteriorOutput, batch: Any) -> torch.Tensor:
    mean = posterior.mean[..., batch.data_idx]
    truth = batch.h[..., batch.data_idx]
    return (
        torch.mean(torch.abs(mean - truth) ** 2)
        / torch.mean(torch.abs(truth) ** 2).clamp_min(1e-8)
    ).real


def channel_nll(posterior: PosteriorOutput, batch: Any) -> torch.Tensor:
    mean = posterior.mean[..., batch.data_idx]
    truth = batch.h[..., batch.data_idx]
    var = _variance_for_mean(posterior, mean, batch.data_idx)
    var = var.clamp_min(1e-8)
    return torch.mean(torch.abs(mean - truth) ** 2 / var + torch.log(var)).real


def calibration_penalty(posterior: PosteriorOutput, batch: Any) -> torch.Tensor:
    mean = posterior.mean[..., batch.data_idx]
    truth = batch.h[..., batch.data_idx]
    var = _variance_for_mean(posterior, mean, batch.data_idx)
    normalized = torch.mean(
        torch.abs(mean - truth) ** 2 / var.clamp_min(1e-8)
    ).real
    return torch.square(torch.log(normalized.clamp(0.05, 20.0)))


def training_loss(
    item: EvidenceStackItem,
    output: dict[str, Any],
    batch: Any,
    training: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    result: EvidenceMixtureResult = output["evidence_result"]
    bit_nll = F.binary_cross_entropy_with_logits(
        output["bit_logits"], batch.coded_bits.float()
    )
    ch_nll = channel_nll(output["posterior"], batch)
    calibration = calibration_penalty(output["posterior"], batch)
    prior_nll = item.operator.prior_coefficient_nll(batch.h, batch.noise_var)
    evidence_nll = result.normalized_negative_log_evidence.mean()
    diversity = item.operator.diversity_penalty()
    nmse = channel_nmse_tensor(output["posterior"], batch)
    loss = (
        bit_nll
        + float(training["channel_nll_weight"]) * ch_nll
        + float(training["calibration_weight"]) * calibration
        + float(training["prior_nll_weight"]) * prior_nll
        + float(training["evidence_nll_weight"]) * evidence_nll
        + float(training["diversity_weight"]) * diversity
    )
    return loss, {
        "bit_nll": bit_nll,
        "channel_nll": ch_nll,
        "calibration_penalty": calibration,
        "prior_nll": prior_nll,
        "evidence_nll": evidence_nll,
        "diversity_penalty": diversity,
        "channel_nmse": nmse,
    }


def evidence_metrics(output: dict[str, Any], batch: Any) -> dict[str, float]:
    posterior = output["posterior"]
    mean = posterior.mean[..., batch.data_idx]
    truth = batch.h[..., batch.data_idx]
    variance = _variance_for_mean(posterior, mean, batch.data_idx).clamp_min(1e-8)
    error = torch.abs(mean - truth).square()
    nmse = error.mean() / torch.abs(truth).square().mean().clamp_min(1e-8)
    normalized = error / variance
    threshold = -math.log(0.05) * variance
    base = {
        **coded_metrics(output, batch),
        "channel_nmse": float(nmse.real.item()),
        "normalized_error_mean": float(normalized.real.mean().item()),
        "coverage95": float((error <= threshold).float().mean().item()),
    }
    result: EvidenceMixtureResult | None = output.get("evidence_result")
    if result is not None:
        mean_weights = result.weights.detach().mean(dim=0)
        base.update(
            {
                "evidence_entropy": float(result.evidence_entropy.detach().mean().item()),
                "effective_component_count": float(
                    result.effective_component_count.detach().mean().item()
                ),
                "negative_log_evidence": float(
                    result.normalized_negative_log_evidence.detach().mean().item()
                ),
                "maximum_component_weight": float(mean_weights.max().item()),
                "minimum_component_weight": float(mean_weights.min().item()),
                "component_weight_vector": json.dumps(
                    mean_weights.cpu().tolist(), separators=(",", ":")
                ),
                "inference_mode": result.mode,
            }
        )
    return base


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


def block_error_count(decoded: dict[str, Any], bits: torch.Tensor) -> tuple[int, int]:
    block_error = (decoded["b_hat"] != bits).reshape(
        bits.shape[0], bits.shape[1], -1
    ).any(-1)
    return int(block_error.sum().item()), int(block_error.numel())


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
    metrics = evidence_metrics(output, batch)
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
        "block_errors": block_error_count(decoded, batch.information_bits)[0],
        "transport_blocks": block_error_count(decoded, batch.information_bits)[1],
        **metrics,
        "edge_density": float(output["edge_density"].item()),
        "inference_uses_true_channel": bool(
            output.get("inference_uses_true_channel", False)
        ),
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
        "block_errors": int(metrics.get("block_errors", round(float(metrics["tbler"]) * int(metrics.get("transport_blocks", 0))))),
        "transport_blocks": int(metrics.get("transport_blocks", 0)),
        "coded_ber": float("nan"),
        "coded_bit_nll": float("nan"),
        "coded_brier": float("nan"),
        "channel_nmse": float("nan"),
        "normalized_error_mean": float("nan"),
        "coverage95": float("nan"),
        "evidence_entropy": float("nan"),
        "effective_component_count": float("nan"),
        "negative_log_evidence": float("nan"),
        "maximum_component_weight": float("nan"),
        "minimum_component_weight": float("nan"),
        "component_weight_vector": "",
        "inference_mode": "",
        "edge_density": float("nan"),
        "inference_uses_true_channel": False,
        "contract_signature": signature,
    }


def gradient_report(parameters: Sequence[torch.nn.Parameter]) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    for index, parameter in enumerate(parameters):
        gradient = parameter.grad
        entries.append(
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
        "entries": entries,
        "all_present": all(item["present"] for item in entries),
        "all_finite": all(item["finite"] for item in entries),
        "any_nonzero": any(item["norm"] > 0.0 for item in entries),
        "total_norm": math.sqrt(sum(item["norm"] ** 2 for item in entries)),
    }


def audit_state_json(operator: EvidenceMixtureLMMSEPosterior) -> dict[str, Any]:
    return {
        "version": EVIDENCE_MIXTURE_LMMSE_VERSION,
        "parameter_report": operator.parameter_report(),
        "state": evidence_state_to_jsonable(shared_evidence_state(operator)),
        "component_variance_profiles": operator.component_variance_profiles()
        .detach()
        .cpu()
        .tolist(),
        "ordered_log_scales": operator.ordered_log_scales().detach().cpu().tolist(),
    }
