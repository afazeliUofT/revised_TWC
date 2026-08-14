#!/usr/bin/env python3
from __future__ import annotations

"""Gate-1 turbo posterior basis/update audit.

This gate performs no training. It separates four explanations for the failed
one-step turbo screen:
  1) a received-signal / true-symbol alignment defect;
  2) arbitrary tied reliability selection or inconsistent observation noise;
  3) covariance over-contraction despite a useful mean update; and
  4) irreducible approximation error of the global Fourier feature basis.
"""

import argparse
import json
import math
import os
import platform
import time
from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]

from bayesroute.models import PosteriorOutput
from bayesroute.nr_gate1 import (
    decode_bridge,
    normalize_device,
    run_standard_receiver,
    standard_receiver,
)
from bayesroute.turbo_posterior import (
    fractional_gaussian_condition,
    mathematical_self_test,
    posterior_batch_metrics,
    project_batched_latent_covariance,
)
from gate1_nr_joint_operator_common import coded_metrics, make_repaired_detector, repaired_forward
from gate1_nr_posterior_factorial_common import ls_repaired_forward, save_json, sha256_file
from gate1_nr_turbo_posterior_common import (
    TurboSetting,
    build_loaded_bridge,
    experiment_signature,
    initial_detector_output,
    make_case_context,
    pilot_state_and_reference,
    set_all_seeds,
    source_hashes,
    true_data_symbols,
    turbo_forward,
)

VERSION = "gate1_nr_turbo_basis_audit_v1"
REQUIRED_CLASSIFICATION = "GATE1_POSTERIOR_BASIS_LOCALIZATION_REQUIRED"
REQUIRED_NEXT_ACTION = "REPLACE_GLOBAL_RFF_WITH_LOCALIZED_DELAY_DOPPLER_POSTERIOR"
REQUIRED_TURBO_ROWS = 492
TURBO_REPORT = ROOT / "outputs/reports/gate1_nr_turbo_posterior.json"
RAW_PATH = ROOT / "outputs/eval/gate1_nr_turbo_basis_audit.csv"
CONTRACT_PATH = ROOT / "outputs/eval/gate1_nr_turbo_basis_audit_contract.json"
AGGREGATE_PATH = ROOT / "outputs/eval/gate1_nr_turbo_basis_audit_aggregate.csv"
REPORT_PATH = ROOT / "outputs/reports/gate1_nr_turbo_basis_audit.json"
GATE_JSON = ROOT / "outputs/gates/GATE1_NR_TURBO_BASIS_AUDIT.json"
GATE_TXT = ROOT / "outputs/gates/GATE1_NR_TURBO_BASIS_AUDIT.txt"
EXPECTED_ROWS = 396
VARIANTS = [
    "pilot_only",
    "current_oracle_topk_raw_rho1",
    "spread_oracle_raw_rho1",
    "spread_oracle_effective_rho1",
    "spread_oracle_effective_rho0125",
    "spread_oracle_effective_mean_only",
    "spread_oracle_effective_cov_only",
    "best_in_global_basis_calibrated",
    "true_channel_repaired",
    "ls_lmmse",
    "perfect_csi_lmmse",
]
SOURCE_FILES = [
    "configs/gate1_nr_turbo_basis_audit.yaml",
    "scripts/gate1_nr_turbo_basis_audit.py",
    "scripts/gate1_nr_turbo_posterior_common.py",
    "src/bayesroute/turbo_posterior.py",
    "src/bayesroute/multiscale_posterior.py",
    "src/bayesroute/lmmse_ep.py",
    "src/bayesroute/nr_gate1.py",
    "src/bayesroute/qam.py",
]


def load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected YAML mapping: {path}")
    return value


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected JSON object: {path}")
    return value


def preconditions() -> dict[str, Any]:
    if not TURBO_REPORT.is_file():
        raise RuntimeError(f"Missing turbo-screen report: {TURBO_REPORT}")
    report = load_json(TURBO_REPORT)
    checks = {
        "turbo_complete": report.get("complete") is True,
        "turbo_rows": report.get("evaluation", {}).get("rows") == REQUIRED_TURBO_ROWS,
        "turbo_unique_rows": report.get("evaluation", {}).get("unique_rows") == REQUIRED_TURBO_ROWS,
        "turbo_classification": report.get("classification") == REQUIRED_CLASSIFICATION,
        "turbo_next_action": report.get("next_action") == REQUIRED_NEXT_ACTION,
        "winner": report.get("winner") == "turbo_f05_r0125",
        "holdout_unused_for_selection": report.get("holdout", {}).get("contract", {}).get("holdout_used_for_selection") is False,
    }
    if not all(checks.values()):
        raise RuntimeError(f"Turbo basis-audit preconditions failed: {checks}")
    return {
        "passed": True,
        "checks": checks,
        "report": str(TURBO_REPORT.relative_to(ROOT)),
        "classification": report["classification"],
        "next_action": report["next_action"],
        "winner": report["winner"],
        "checkpoint_sha256": report["holdout"]["contract"]["checkpoint_sha256"],
        "metrics": report.get("metrics", {}),
    }


def append_rows_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    new = pd.DataFrame(rows)
    if path.is_file():
        old = pd.read_csv(path)
        if list(old.columns) != list(new.columns):
            raise RuntimeError(f"CSV column contract mismatch: {path}")
        combined = pd.concat([old, new], ignore_index=True)
    else:
        combined = new
    temporary = path.with_suffix(path.suffix + ".tmp")
    combined.to_csv(temporary, index=False)
    temporary.replace(path)


def ensure_contract(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    payload = dict(payload)
    payload["signature"] = experiment_signature(payload)
    if path.is_file():
        existing = load_json(path)
        if existing.get("signature") != payload["signature"]:
            raise RuntimeError(f"Evaluation contract mismatch: {path}")
    else:
        save_json(payload, path)
    return payload


def completed_batches(path: Path, variants: Sequence[str]) -> set[tuple[str, float, int]]:
    if not path.is_file():
        return set()
    frame = pd.read_csv(path)
    keys = ["case", "variant", "ebno_db", "rep"]
    if frame[keys].duplicated().any():
        raise RuntimeError(f"Duplicate audit rows: {path}")
    counts = frame.groupby(["case", "ebno_db", "rep"])["variant"].nunique()
    partial = counts[counts != len(variants)]
    if len(partial):
        raise RuntimeError(f"Partial paired audit batch: {partial.index.tolist()}")
    return {(str(a), float(b), int(c)) for a, b, c in counts.index}


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
    block_error = (decoded["b_hat"] != bits).reshape(bits.shape[0], bits.shape[1], -1).any(-1)
    crc_failure = ~decoded["crc"].bool()
    if crc_failure.shape != block_error.shape:
        crc_failure = crc_failure.reshape(block_error.shape)
    return float((block_error != crc_failure).float().mean().item())


def _noise_grid(value: torch.Tensor, batch: int, n_data: int, device: torch.device) -> torch.Tensor:
    tensor = torch.as_tensor(value, device=device).real.float()
    if tensor.ndim == 0:
        return tensor.expand(batch, n_data)
    if tensor.ndim == 1 and tensor.numel() == batch:
        return tensor[:, None].expand(batch, n_data)
    if tensor.ndim == 2 and tensor.shape == (batch, n_data):
        return tensor
    return tensor.mean().expand(batch, n_data)


def desired_count(n_data: int, audit: dict[str, Any]) -> int:
    desired = int(math.ceil(float(audit["data_fraction"]) * int(n_data)))
    count = min(int(n_data), int(audit["max_observations"]))
    return min(count, max(int(audit["min_observations"]), desired))


def spread_indices(batch: int, n_data: int, count: int, device: torch.device) -> torch.Tensor:
    if count > n_data:
        raise ValueError("count cannot exceed n_data")
    if count == n_data:
        base = torch.arange(n_data, device=device)
    else:
        base = torch.linspace(0, n_data - 1, count, device=device).round().long()
        base = torch.unique_consecutive(base)
        if int(base.numel()) != count:
            chosen = set(int(x) for x in base.detach().cpu().tolist())
            fill = [i for i in range(n_data) if i not in chosen][: count - len(chosen)]
            base = torch.cat([base, torch.tensor(fill, device=device, dtype=torch.long)]).sort().values
    return base[None, :].expand(batch, -1).contiguous()


def selection_geometry(selected: torch.Tensor, data_idx: torch.Tensor, fft_size: int, n_symbols: int) -> dict[str, float]:
    actual = torch.as_tensor(data_idx, device=selected.device)[selected]
    time_index = torch.div(actual, int(fft_size), rounding_mode="floor")
    freq_index = torch.remainder(actual, int(fft_size))
    unique_time = []
    time_span = []
    freq_span = []
    for b in range(int(selected.shape[0])):
        t = time_index[b]
        f = freq_index[b]
        unique_time.append(float(torch.unique(t).numel()) / float(max(n_symbols, 1)))
        time_span.append(float((t.max() - t.min() + 1).item()) / float(max(n_symbols, 1)))
        freq_span.append(float((f.max() - f.min() + 1).item()) / float(max(fft_size, 1)))
    return {
        "selected_time_unique_fraction": float(np.mean(unique_time)),
        "selected_time_span_fraction": float(np.mean(time_span)),
        "selected_frequency_span_fraction": float(np.mean(freq_span)),
    }


def observation_equation_diagnostics(batch: Any) -> dict[str, float]:
    x = true_data_symbols(batch)
    data_idx = torch.as_tensor(batch.data_idx, dtype=torch.long, device=batch.y.device)
    h = batch.h[..., data_idx]
    signal = torch.sum(h * x[:, :, None, :], dim=1)
    y_data = batch.y[..., data_idx]
    residual = y_data - signal
    raw = _noise_grid(batch.noise_var, int(y_data.shape[0]), int(y_data.shape[-1]), y_data.device)
    residual_power = residual.abs().square().mean()
    raw_power = raw.mean().clamp_min(1e-12)
    return {
        "observation_noise_ratio": float((residual_power / raw_power).real.item()),
        "observation_residual_power": float(residual_power.real.item()),
        "declared_noise_power": float(raw_power.real.item()),
    }


def build_update_matrices(
    state: Any,
    batch: Any,
    selected: torch.Tensor,
    *,
    use_effective_noise: bool,
) -> dict[str, torch.Tensor]:
    symbol_mean = true_data_symbols(batch)
    batch_size, n_layers, n_data = symbol_mean.shape
    count = int(selected.shape[1])
    rank = int(state.latent_mean.shape[-1])
    data_idx = torch.as_tensor(batch.data_idx, dtype=torch.long, device=batch.y.device)
    features_data = state.features[data_idx]
    selected_features = features_data[selected]
    gather = selected[:, None, :].expand(batch_size, n_layers, count)
    x_mean = torch.gather(symbol_mean, 2, gather)
    matrix = torch.einsum("bnk,bkq->bknq", x_mean, selected_features).reshape(batch_size, count, n_layers * rank)
    y_data = batch.y[..., data_idx]
    observations = torch.gather(y_data, 2, selected[:, None, :].expand(batch_size, y_data.shape[1], count))
    noise_source = state.posterior.effective_noise if use_effective_noise else batch.noise_var
    noise = _noise_grid(noise_source, batch_size, n_data, batch.y.device)
    noise = torch.gather(noise, 1, selected).clamp_min(1e-8)
    return {"matrix": matrix, "observations": observations, "noise": noise}


def innovation_nis(state: Any, matrices: dict[str, torch.Tensor], rho: float) -> float:
    matrix = matrices["matrix"].to(state.latent_mean.dtype)
    observations = matrices["observations"]
    noise = matrices["noise"] / float(rho)
    batch = int(matrix.shape[0])
    covariance = state.latent_cov.unsqueeze(0).expand(batch, -1, -1)
    latent_mean = state.latent_mean.reshape(batch, observations.shape[1], -1)
    prediction = torch.einsum("bkl,brl->brk", matrix, latent_mean)
    innovation = observations - prediction
    a_cov = matrix @ covariance
    s = a_cov @ matrix.conj().transpose(-1, -2) + torch.diag_embed(noise).to(matrix.dtype)
    s = 0.5 * (s + s.conj().transpose(-1, -2))
    eye = torch.eye(s.shape[-1], dtype=s.dtype, device=s.device)
    s = s + 1e-6 * torch.diagonal(s, dim1=-2, dim2=-1).real.mean(-1).clamp_min(1.0)[..., None, None] * eye
    chol = torch.linalg.cholesky(s)
    rhs = innovation.transpose(1, 2).contiguous()
    solved = torch.cholesky_solve(rhs, chol)
    quadratic = torch.sum(rhs.conj() * solved, dim=1).real / float(max(s.shape[-1], 1))
    return float(quadratic.mean().item())


def posterior_from_latent(
    state: Any,
    latent_mean_flat: torch.Tensor,
    latent_cov: torch.Tensor,
) -> PosteriorOutput:
    batch, n_rx, _ = latent_mean_flat.shape
    n_layers = int(state.latent_mean.shape[2])
    rank = int(state.latent_mean.shape[-1])
    latent_mean = latent_mean_flat.reshape(batch, n_rx, n_layers, rank)
    channel_mean = torch.einsum("rq,bxnq->bnxr", state.features, latent_mean).contiguous()
    local_cov, var_diag = project_batched_latent_covariance(state.features, latent_cov, n_layers, rank)
    return PosteriorOutput(
        mean=channel_mean,
        var_diag=var_diag,
        local_cov=local_cov,
        latent_cov=latent_cov,
        effective_noise=state.posterior.effective_noise,
    )


def custom_oracle_posteriors(
    state: Any,
    batch: Any,
    selected: torch.Tensor,
    *,
    rho: float,
    use_effective_noise: bool,
) -> tuple[PosteriorOutput, dict[str, Any]]:
    matrices = build_update_matrices(state, batch, selected, use_effective_noise=use_effective_noise)
    batch_size = int(batch.y.shape[0])
    latent_mean_flat = state.latent_mean.reshape(batch_size, batch.y.shape[1], -1)
    updated_mean, updated_cov = fractional_gaussian_condition(
        latent_mean_flat,
        state.latent_cov,
        matrices["matrix"],
        matrices["observations"],
        matrices["noise"],
        information_damping=float(rho),
    )
    posterior = posterior_from_latent(state, updated_mean, updated_cov)
    before = torch.diagonal(state.latent_cov).real.sum()
    after = torch.diagonal(updated_cov, dim1=-2, dim2=-1).real.sum(-1).mean()
    raw = _noise_grid(batch.noise_var, batch_size, int(true_data_symbols(batch).shape[-1]), batch.y.device)
    effective = _noise_grid(state.posterior.effective_noise, batch_size, int(true_data_symbols(batch).shape[-1]), batch.y.device)
    diagnostics = {
        "information_damping": float(rho),
        "selected_observations": int(selected.shape[1]),
        "latent_trace_reduction_fraction": float(((before - after) / before.clamp_min(1e-8)).item()),
        "innovation_nis": innovation_nis(state, matrices, float(rho)),
        "effective_to_raw_noise_ratio": float((effective.mean() / raw.mean().clamp_min(1e-12)).item()),
        "used_effective_noise": bool(use_effective_noise),
    }
    return posterior, diagnostics


def diagonal_local_covariance(var_diag: torch.Tensor) -> torch.Tensor:
    if var_diag.ndim != 3:
        raise ValueError("Expected batch-dependent var_diag [B,N,R]")
    batch, n_layers, n_re = var_diag.shape
    local = torch.zeros(batch, n_layers, n_layers, n_re, dtype=torch.complex64, device=var_diag.device)
    index = torch.arange(n_layers, device=var_diag.device)
    local[:, index, index, :] = var_diag.to(torch.complex64)
    return local


def best_in_basis_posterior(state: Any, batch: Any, ridge: float) -> tuple[PosteriorOutput, dict[str, float]]:
    features = state.features.to(batch.h.dtype)
    n_re, rank = features.shape
    h = batch.h
    if int(h.shape[-1]) != int(n_re):
        raise RuntimeError("Feature-grid and channel-grid lengths disagree")
    gram = features.conj().T @ features + float(ridge) * torch.eye(rank, dtype=features.dtype, device=features.device)
    rhs = features.conj().T @ h.permute(3, 0, 1, 2).reshape(n_re, -1)
    latent = torch.linalg.solve(gram, rhs)
    projected = (features @ latent).reshape(n_re, h.shape[0], h.shape[1], h.shape[2]).permute(1, 2, 3, 0).contiguous()
    residual = h - projected
    residual_var = residual.abs().square().mean(dim=2).clamp_min(1e-8)
    local_cov = diagonal_local_covariance(residual_var)
    posterior = PosteriorOutput(
        mean=projected,
        var_diag=residual_var,
        local_cov=local_cov,
        latent_cov=state.latent_cov,
        effective_noise=state.posterior.effective_noise,
    )
    x = true_data_symbols(batch)
    data_idx = torch.as_tensor(batch.data_idx, dtype=torch.long, device=h.device)
    true_signal = torch.sum(h[..., data_idx] * x[:, :, None, :], dim=1)
    projected_signal = torch.sum(projected[..., data_idx] * x[:, :, None, :], dim=1)
    nmse = residual.abs().square().mean() / h.abs().square().mean().clamp_min(1e-8)
    signal_ratio = (true_signal - projected_signal).abs().square().mean() / true_signal.abs().square().mean().clamp_min(1e-8)
    return posterior, {
        "basis_projection_nmse": float(nmse.real.item()),
        "basis_signal_residual_ratio": float(signal_ratio.real.item()),
    }


def true_channel_posterior(state: Any, batch: Any) -> PosteriorOutput:
    batch_size, n_layers, _, n_re = batch.h.shape
    var = torch.full((batch_size, n_layers, n_re), 1e-8, device=batch.h.device)
    return PosteriorOutput(
        mean=batch.h,
        var_diag=var,
        local_cov=diagonal_local_covariance(var),
        latent_cov=state.latent_cov,
        effective_noise=state.posterior.effective_noise,
    )


def output_from_posterior(bridge: Any, detector: Any, batch: Any, posterior: PosteriorOutput, graph: torch.Tensor, diagnostics: dict[str, Any]) -> dict[str, Any]:
    output = repaired_forward(bridge, detector, batch, posterior=posterior, reference_graph=graph)
    output["posterior_metrics"] = posterior_batch_metrics(posterior, batch.h, batch.data_idx)
    output["audit_diagnostics"] = diagnostics
    return output


def custom_row(
    *,
    case: Any,
    variant: str,
    snr: float,
    rep: int,
    seed: int,
    output: dict[str, Any],
    batch: Any,
    decoded: dict[str, Any],
    signature: str,
    shared: dict[str, float],
) -> dict[str, Any]:
    posterior_metrics = output.get("posterior_metrics") or posterior_batch_metrics(output["posterior"], batch.h, batch.data_idx)
    diagnostics = dict(output.get("audit_diagnostics", {}))
    return {
        "case": case.name,
        "group": getattr(case, "group", "audit"),
        "num_prb": int(case.num_prb),
        "variant": variant,
        "ebno_db": float(snr),
        "rep": int(rep),
        "eval_seed": int(seed),
        "information_ber": float(decoded["information_ber"]),
        "tbler": float(decoded["tbler"]),
        "crc_failure_rate": float(decoded["crc_failure_rate"]),
        "crc_block_disagreement_rate": crc_disagreement(decoded, batch.information_bits),
        **coded_metrics(output, batch),
        **posterior_metrics,
        "edge_density": float(output["edge_density"].item()),
        "selected_observations": int(diagnostics.get("selected_observations", 0)),
        "information_damping": float(diagnostics.get("information_damping", 0.0)),
        "latent_trace_reduction_fraction": float(diagnostics.get("latent_trace_reduction_fraction", 0.0)),
        "innovation_nis": float(diagnostics.get("innovation_nis", float("nan"))),
        "effective_to_raw_noise_ratio": float(diagnostics.get("effective_to_raw_noise_ratio", shared["effective_to_raw_noise_ratio"])),
        "selected_time_unique_fraction": float(diagnostics.get("selected_time_unique_fraction", float("nan"))),
        "selected_time_span_fraction": float(diagnostics.get("selected_time_span_fraction", float("nan"))),
        "selected_frequency_span_fraction": float(diagnostics.get("selected_frequency_span_fraction", float("nan"))),
        "basis_projection_nmse": float(shared["basis_projection_nmse"]),
        "basis_signal_residual_ratio": float(shared["basis_signal_residual_ratio"]),
        "observation_noise_ratio": float(shared["observation_noise_ratio"]),
        "contract_signature": signature,
    }


def standard_row(case: Any, variant: str, snr: float, rep: int, seed: int, metrics: dict[str, Any], signature: str, shared: dict[str, float]) -> dict[str, Any]:
    return {
        "case": case.name,
        "group": getattr(case, "group", "audit"),
        "num_prb": int(case.num_prb),
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
        "edge_density": float("nan"),
        "selected_observations": 0,
        "information_damping": 0.0,
        "latent_trace_reduction_fraction": 0.0,
        "innovation_nis": float("nan"),
        "effective_to_raw_noise_ratio": float(shared["effective_to_raw_noise_ratio"]),
        "selected_time_unique_fraction": float("nan"),
        "selected_time_span_fraction": float("nan"),
        "selected_frequency_span_fraction": float("nan"),
        "basis_projection_nmse": float(shared["basis_projection_nmse"]),
        "basis_signal_residual_ratio": float(shared["basis_signal_residual_ratio"]),
        "observation_noise_ratio": float(shared["observation_noise_ratio"]),
        "contract_signature": signature,
    }


def paired_delta(
    frame: pd.DataFrame,
    reference: str,
    comparator: str,
    metric: str,
    *,
    prb: int | None = None,
) -> dict[str, float]:
    keys = ["case", "ebno_db", "rep", "eval_seed"]
    work = frame
    if prb is not None:
        work = work[work["num_prb"] == int(prb)]
    a = work[work["variant"] == reference]
    b = work[work["variant"] == comparator]
    merged = a.merge(b, on=keys, suffixes=("_a", "_b"))
    values = (pd.to_numeric(merged[f"{metric}_a"], errors="coerce") - pd.to_numeric(merged[f"{metric}_b"], errors="coerce")).dropna()
    mean = float(values.mean())
    std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    if len(values) > 1:
        try:
            from scipy.stats import t as student_t
            critical = float(student_t.ppf(0.975, len(values) - 1))
        except Exception:
            critical = 1.96
    else:
        critical = 0.0
    half = critical * std / math.sqrt(max(len(values), 1))
    return {"pairs": int(len(values)), "mean": mean, "ci95_low": mean - half, "ci95_high": mean + half}


def mean_metric(frame: pd.DataFrame, variant: str, metric: str, *, prb: int | None = None) -> float:
    sub = frame[frame["variant"] == variant]
    if prb is not None:
        sub = sub[sub["num_prb"] == int(prb)]
    values = pd.to_numeric(sub[metric], errors="coerce").dropna()
    return float(values.mean()) if len(values) else float("nan")


def aggregate(frame: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "information_ber", "tbler", "crc_failure_rate", "coded_ber", "coded_bit_nll",
        "channel_nmse", "normalized_error_mean", "coverage95", "edge_density",
        "selected_observations", "latent_trace_reduction_fraction", "innovation_nis",
        "effective_to_raw_noise_ratio", "selected_time_unique_fraction",
        "selected_time_span_fraction", "selected_frequency_span_fraction",
        "basis_projection_nmse", "basis_signal_residual_ratio", "observation_noise_ratio",
    ]
    return frame.groupby(["case", "num_prb", "variant", "ebno_db"], dropna=False)[metrics].agg(["mean", "std", "count"]).reset_index()


def classify(frame: pd.DataFrame) -> dict[str, Any]:
    # The architectural decision is made on the fresh 12-PRB holdout.  The
    # 4-/8-PRB rows remain useful controls but cannot mask a wide-grid failure.
    comparisons = {
        "current_oracle_minus_pilot_12prb_tbler": paired_delta(
            frame, "current_oracle_topk_raw_rho1", "pilot_only", "tbler", prb=12
        ),
        "spread_raw_minus_pilot_12prb_tbler": paired_delta(
            frame, "spread_oracle_raw_rho1", "pilot_only", "tbler", prb=12
        ),
        "spread_effective_minus_pilot_12prb_tbler": paired_delta(
            frame, "spread_oracle_effective_rho1", "pilot_only", "tbler", prb=12
        ),
        "spread_effective_weak_minus_pilot_12prb_tbler": paired_delta(
            frame, "spread_oracle_effective_rho0125", "pilot_only", "tbler", prb=12
        ),
        "mean_only_minus_pilot_12prb_tbler": paired_delta(
            frame, "spread_oracle_effective_mean_only", "pilot_only", "tbler", prb=12
        ),
        "cov_only_minus_pilot_12prb_tbler": paired_delta(
            frame, "spread_oracle_effective_cov_only", "pilot_only", "tbler", prb=12
        ),
        "spread_minus_current_12prb_tbler": paired_delta(
            frame, "spread_oracle_raw_rho1", "current_oracle_topk_raw_rho1", "tbler", prb=12
        ),
        "effective_minus_raw_12prb_tbler": paired_delta(
            frame, "spread_oracle_effective_rho1", "spread_oracle_raw_rho1", "tbler", prb=12
        ),
        "mean_only_minus_full_update_12prb_tbler": paired_delta(
            frame, "spread_oracle_effective_mean_only", "spread_oracle_effective_rho1", "tbler", prb=12
        ),
        "best_basis_minus_ls_12prb_tbler": paired_delta(
            frame, "best_in_global_basis_calibrated", "ls_lmmse", "tbler", prb=12
        ),
        "true_channel_minus_perfect_12prb_tbler": paired_delta(
            frame, "true_channel_repaired", "perfect_csi_lmmse", "tbler", prb=12
        ),
    }
    projection_nmse_12 = mean_metric(
        frame, "best_in_global_basis_calibrated", "basis_projection_nmse", prb=12
    )
    projection_signal_12 = mean_metric(
        frame, "best_in_global_basis_calibrated", "basis_signal_residual_ratio", prb=12
    )
    observation_ratio = mean_metric(frame, "pilot_only", "observation_noise_ratio")
    effective_ratio = mean_metric(frame, "pilot_only", "effective_to_raw_noise_ratio")

    corrected_candidates = [
        comparisons["spread_effective_minus_pilot_12prb_tbler"],
        comparisons["spread_effective_weak_minus_pilot_12prb_tbler"],
        comparisons["mean_only_minus_pilot_12prb_tbler"],
    ]
    corrected_oracle_helps = any(item["ci95_high"] < 0.0 for item in corrected_candidates)
    spread_selection_helps = comparisons["spread_minus_current_12prb_tbler"]["ci95_high"] < 0.0
    effective_noise_helps = comparisons["effective_minus_raw_12prb_tbler"]["ci95_high"] < 0.0
    covariance_overcontracts = (
        comparisons["cov_only_minus_pilot_12prb_tbler"]["ci95_low"] > 0.0
        or comparisons["mean_only_minus_full_update_12prb_tbler"]["ci95_high"] < 0.0
    )
    best_basis_near_ls = comparisons["best_basis_minus_ls_12prb_tbler"]["mean"] <= 0.01
    true_channel_matches = abs(
        comparisons["true_channel_minus_perfect_12prb_tbler"]["mean"]
    ) <= 0.01
    observation_consistent = 0.5 <= observation_ratio <= 1.5
    basis_small_error = projection_nmse_12 <= 0.03 and projection_signal_12 <= 0.03

    checks = {
        "observation_equation_consistent": observation_consistent,
        "true_channel_repaired_matches_perfect": true_channel_matches,
        "corrected_oracle_update_improves_pilot_12prb": corrected_oracle_helps,
        "space_filling_selection_improves_current_oracle_12prb": spread_selection_helps,
        "effective_noise_improves_raw_noise_update_12prb": effective_noise_helps,
        "covariance_overcontraction_detected_12prb": covariance_overcontracts,
        "best_global_basis_within_0p01_of_ls_12prb": best_basis_near_ls,
        "global_basis_projection_nmse_12prb_le_0p03": projection_nmse_12 <= 0.03,
        "global_basis_signal_residual_12prb_le_0p03": projection_signal_12 <= 0.03,
    }

    if not observation_consistent or not true_channel_matches:
        classification = "GATE1_TURBO_UPDATE_CONTRACT_REPAIR_REQUIRED"
        next_action = "REPAIR_OBSERVATION_OR_TRUE_CHANNEL_CONTROL_BEFORE_POSTERIOR_REDESIGN"
    elif best_basis_near_ls and corrected_oracle_helps:
        classification = "GATE1_TURBO_NOISE_SELECTION_REPAIR_REQUIRED"
        next_action = "PATCH_EFFECTIVE_NOISE_SPREAD_SELECTION_AND_COVARIANCE_DAMPING"
    elif best_basis_near_ls or basis_small_error:
        classification = "GATE1_GLOBAL_BASIS_ADEQUATE_INFERENCE_REPAIR_REQUIRED"
        next_action = "REPAIR_POSTERIOR_INFERENCE_OR_DECODER_EXTRINSIC_UPDATE_WITHOUT_CHANGING_BASIS"
    else:
        classification = "GATE1_LOCALIZED_DELAY_DOPPLER_POSTERIOR_JUSTIFIED"
        next_action = "IMPLEMENT_LOCALIZED_DELAY_DOPPLER_POSTERIOR_AND_RETRAIN_ON_4_8_PRB_HOLD_OUT_12_PRB"

    per_prb: dict[str, dict[str, float]] = {}
    for prb in sorted(int(x) for x in frame["num_prb"].unique()):
        per_prb[str(prb)] = {
            "pilot_tbler": mean_metric(frame, "pilot_only", "tbler", prb=prb),
            "corrected_oracle_weak_tbler": mean_metric(
                frame, "spread_oracle_effective_rho0125", "tbler", prb=prb
            ),
            "best_global_basis_tbler": mean_metric(
                frame, "best_in_global_basis_calibrated", "tbler", prb=prb
            ),
            "ls_tbler": mean_metric(frame, "ls_lmmse", "tbler", prb=prb),
            "perfect_tbler": mean_metric(frame, "perfect_csi_lmmse", "tbler", prb=prb),
            "basis_projection_nmse": mean_metric(
                frame, "best_in_global_basis_calibrated", "basis_projection_nmse", prb=prb
            ),
            "basis_signal_residual_ratio": mean_metric(
                frame, "best_in_global_basis_calibrated", "basis_signal_residual_ratio", prb=prb
            ),
        }

    return {
        "classification": classification,
        "next_action": next_action,
        "scientific_checks": checks,
        "paired_comparisons": comparisons,
        "metrics": {
            "pilot_12prb_tbler": mean_metric(frame, "pilot_only", "tbler", prb=12),
            "current_oracle_12prb_tbler": mean_metric(
                frame, "current_oracle_topk_raw_rho1", "tbler", prb=12
            ),
            "spread_effective_12prb_tbler": mean_metric(
                frame, "spread_oracle_effective_rho1", "tbler", prb=12
            ),
            "spread_effective_weak_12prb_tbler": mean_metric(
                frame, "spread_oracle_effective_rho0125", "tbler", prb=12
            ),
            "mean_only_12prb_tbler": mean_metric(
                frame, "spread_oracle_effective_mean_only", "tbler", prb=12
            ),
            "cov_only_12prb_tbler": mean_metric(
                frame, "spread_oracle_effective_cov_only", "tbler", prb=12
            ),
            "best_global_basis_12prb_tbler": mean_metric(
                frame, "best_in_global_basis_calibrated", "tbler", prb=12
            ),
            "ls_12prb_tbler": mean_metric(frame, "ls_lmmse", "tbler", prb=12),
            "perfect_12prb_tbler": mean_metric(
                frame, "perfect_csi_lmmse", "tbler", prb=12
            ),
            "projection_nmse_12prb": projection_nmse_12,
            "projection_signal_residual_12prb": projection_signal_12,
            "observation_noise_ratio": observation_ratio,
            "effective_to_raw_noise_ratio": effective_ratio,
        },
        "per_prb_metrics": per_prb,
    }


def make_plots(frame: pd.DataFrame) -> list[str]:
    out = ROOT / "outputs/plots"
    out.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    selected = [
        "pilot_only", "spread_oracle_effective_rho0125",
        "best_in_global_basis_calibrated", "ls_lmmse", "perfect_csi_lmmse",
    ]
    for prb in sorted(frame["num_prb"].unique()):
        plt.figure(figsize=(7.4, 4.8))
        for variant in selected:
            sub = frame[(frame["num_prb"] == prb) & (frame["variant"] == variant)]
            grouped = sub.groupby("ebno_db")["tbler"].mean().sort_index()
            plt.plot(grouped.index, grouped.values, marker="o", label=variant)
        plt.xlabel("$E_b/N_0$ (dB)")
        plt.ylabel("TBLER")
        plt.ylim(-0.01, 0.25)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=7)
        plt.tight_layout()
        path = out / f"gate1_turbo_basis_audit_{int(prb)}prb_tbler.png"
        plt.savefig(path, dpi=180)
        plt.close()
        paths.append(str(path.relative_to(ROOT)))
    plt.figure(figsize=(6.6, 4.6))
    basis = frame[frame["variant"] == "best_in_global_basis_calibrated"].groupby("num_prb")["basis_projection_nmse"].mean().sort_index()
    plt.plot(basis.index, basis.values, marker="o")
    plt.xlabel("PRBs")
    plt.ylabel("Oracle projection NMSE")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    path = out / "gate1_turbo_basis_projection_nmse.png"
    plt.savefig(path, dpi=180)
    plt.close()
    paths.append(str(path.relative_to(ROOT)))
    return paths


def write_incomplete(rows: int, contract: dict[str, Any]) -> None:
    report = {
        "version": VERSION,
        "complete": False,
        "evaluation": {"rows": rows, "expected_rows": EXPECTED_ROWS, "complete": False, "contract": contract},
        "classification": "GATE1_NR_TURBO_BASIS_AUDIT_INCOMPLETE",
        "next_action": "RESUBMIT_SAME_COMMAND",
        "publication_nr_ready": False,
    }
    save_json(report, REPORT_PATH)
    save_json(report, GATE_JSON)
    GATE_TXT.parent.mkdir(parents=True, exist_ok=True)
    GATE_TXT.write_text("CLASSIFICATION: GATE1_NR_TURBO_BASIS_AUDIT_INCOMPLETE\nNEXT_ACTION: RESUBMIT_SAME_COMMAND\nPUBLICATION_NR_READY: NO\n", encoding="utf-8")
    print("GATE1_NR_TURBO_BASIS_AUDIT_INCOMPLETE: RESUBMIT")


def real_pipeline_preflight(config: dict[str, Any], device: torch.device) -> dict[str, Any]:
    case, context = make_case_context(config["cases"][0], device)
    bridge = build_loaded_bridge(case, context, operator_seed=int(config["operator_seed"]))
    detector = make_repaired_detector(int(context.grid.bits_per_symbol)).to(device)
    set_all_seeds(int(config["seed"]) + 999)
    batch = context.sample(2, float(config["ebno_db"][0]))
    with torch.inference_mode():
        state, graph, _ = pilot_state_and_reference(bridge, batch)
        initial = initial_detector_output(bridge, detector, batch, state, graph)
        projection, projection_diag = best_in_basis_posterior(state, batch, float(config["audit"]["projection_ridge"]))
        projection_output = output_from_posterior(bridge, detector, batch, projection, graph, projection_diag)
        observation = observation_equation_diagnostics(batch)
    checks = {
        "initial_finite": bool(torch.isfinite(initial["bit_logits"]).all().item()),
        "projection_finite": bool(torch.isfinite(projection_output["bit_logits"]).all().item()),
        "projection_shape": projection.mean.shape == batch.h.shape,
        "observation_ratio_finite": math.isfinite(observation["observation_noise_ratio"]),
    }
    if not all(checks.values()):
        raise RuntimeError(f"Real pipeline preflight failed: {checks}")
    return {"passed": True, "checks": checks, "observation": observation, "projection": projection_diag}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/gate1_nr_turbo_basis_audit.yaml")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--deadline-minutes", type=float, default=25.0)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    config_path = ROOT / args.config
    config = load_yaml(config_path)
    pre = preconditions()
    expected = len(config["cases"]) * len(config["ebno_db"]) * int(config["repetitions"]) * len(VARIANTS)
    if expected != EXPECTED_ROWS:
        raise RuntimeError((expected, EXPECTED_ROWS))
    math_report = mathematical_self_test("cpu")
    if math_report.get("passed") is not True:
        raise RuntimeError(f"Fractional Gaussian self-test failed: {math_report}")
    if args.preflight_only:
        device = normalize_device(args.device)
        pipeline = real_pipeline_preflight(config, device)
        print("GATE1_NR_TURBO_BASIS_AUDIT_PREFLIGHT_PASS")
        print("TURBO_RESULT", pre["classification"])
        print("MATH_SELF_TEST", math_report["passed"])
        print("REAL_PIPELINE_PREFLIGHT", pipeline["passed"])
        print("CASES", len(config["cases"]))
        print("VARIANTS", len(VARIANTS))
        print("EXPECTED_ROWS", EXPECTED_ROWS)
        print("TRAINING_REQUIRED NO")
        return

    device = normalize_device(args.device)
    payload = {
        "version": VERSION,
        "config_sha256": sha256_file(config_path),
        "source_sha256": source_hashes(SOURCE_FILES),
        "checkpoint_sha256": pre["checkpoint_sha256"],
        "cases": config["cases"],
        "ebno_db": config["ebno_db"],
        "repetitions": config["repetitions"],
        "variants": VARIANTS,
        "training_required": False,
    }
    contract = ensure_contract(CONTRACT_PATH, payload)
    done = completed_batches(RAW_PATH, VARIANTS)
    deadline = time.time() + 60.0 * float(args.deadline_minutes)
    audit = config["audit"]

    for case_index, raw_case in enumerate(config["cases"]):
        case, context = make_case_context(raw_case, device)
        bridge = build_loaded_bridge(case, context, operator_seed=int(config["operator_seed"]))
        detector = make_repaired_detector(int(context.grid.bits_per_symbol)).to(device)
        ls_receiver = standard_receiver(context, perfect_csi=False, return_crc=True)
        perfect_receiver = standard_receiver(context, perfect_csi=True, return_crc=True)
        for snr_index, raw_snr in enumerate(config["ebno_db"]):
            snr = float(raw_snr)
            for rep in range(int(config["repetitions"])):
                key = (case.name, snr, rep)
                if key in done:
                    continue
                seed = int(config["seed"]) + case_index * 100_000 + snr_index * 1_000 + rep
                set_all_seeds(seed)
                batch = context.sample(int(config["batch_size"]), snr)
                with torch.inference_mode():
                    state, graph, _ = pilot_state_and_reference(bridge, batch)
                    initial = initial_detector_output(bridge, detector, batch, state, graph)
                    initial["posterior_metrics"] = posterior_batch_metrics(state.posterior, batch.h, batch.data_idx)
                    initial["audit_diagnostics"] = {}

                    oracle_setting = TurboSetting(
                        name="current_oracle_topk_raw_rho1",
                        information_damping=float(audit["full_information_damping"]),
                        data_fraction=float(audit["data_fraction"]),
                        min_observations=int(audit["min_observations"]),
                        max_observations=int(audit["max_observations"]),
                    )
                    current = turbo_forward(
                        bridge, detector, batch, oracle_setting,
                        state=state, reference_graph=graph,
                        initial_output=initial, oracle_symbols=True,
                    )
                    current_geometry = selection_geometry(
                        current["selected_data_indices"], batch.data_idx,
                        int(context.grid.fft_size), int(context.grid.num_ofdm_symbols),
                    )
                    current_diag = dict(current["turbo_diagnostics"])
                    raw_grid = _noise_grid(batch.noise_var, int(batch.y.shape[0]), int(true_data_symbols(batch).shape[-1]), batch.y.device)
                    eff_grid = _noise_grid(state.posterior.effective_noise, int(batch.y.shape[0]), int(true_data_symbols(batch).shape[-1]), batch.y.device)
                    current_diag.update(current_geometry)
                    current_diag["effective_to_raw_noise_ratio"] = float((eff_grid.mean() / raw_grid.mean().clamp_min(1e-12)).item())
                    current["audit_diagnostics"] = current_diag

                    count = desired_count(int(true_data_symbols(batch).shape[-1]), audit)
                    spread = spread_indices(int(batch.y.shape[0]), int(true_data_symbols(batch).shape[-1]), count, batch.y.device)
                    spread_geometry = selection_geometry(spread, batch.data_idx, int(context.grid.fft_size), int(context.grid.num_ofdm_symbols))
                    raw_posterior, raw_diag = custom_oracle_posteriors(state, batch, spread, rho=1.0, use_effective_noise=False)
                    raw_diag.update(spread_geometry)
                    raw_output = output_from_posterior(bridge, detector, batch, raw_posterior, graph, raw_diag)

                    effective_posterior, effective_diag = custom_oracle_posteriors(state, batch, spread, rho=1.0, use_effective_noise=True)
                    effective_diag.update(spread_geometry)
                    effective_output = output_from_posterior(bridge, detector, batch, effective_posterior, graph, effective_diag)

                    weak_posterior, weak_diag = custom_oracle_posteriors(state, batch, spread, rho=float(audit["weak_information_damping"]), use_effective_noise=True)
                    weak_diag.update(spread_geometry)
                    weak_output = output_from_posterior(bridge, detector, batch, weak_posterior, graph, weak_diag)

                    mean_only = PosteriorOutput(
                        mean=effective_posterior.mean,
                        var_diag=state.posterior.var_diag,
                        local_cov=state.posterior.local_cov,
                        latent_cov=state.posterior.latent_cov,
                        effective_noise=state.posterior.effective_noise,
                    )
                    mean_only_output = output_from_posterior(bridge, detector, batch, mean_only, graph, {**effective_diag, "decomposition": "updated_mean_initial_covariance"})

                    cov_only = PosteriorOutput(
                        mean=state.posterior.mean,
                        var_diag=effective_posterior.var_diag,
                        local_cov=effective_posterior.local_cov,
                        latent_cov=effective_posterior.latent_cov,
                        effective_noise=state.posterior.effective_noise,
                    )
                    cov_only_output = output_from_posterior(bridge, detector, batch, cov_only, graph, {**effective_diag, "decomposition": "initial_mean_updated_covariance"})

                    basis_posterior, basis_diag = best_in_basis_posterior(state, batch, float(audit["projection_ridge"]))
                    basis_output = output_from_posterior(bridge, detector, batch, basis_posterior, graph, basis_diag)
                    true_output = output_from_posterior(bridge, detector, batch, true_channel_posterior(state, batch), graph, {})

                    outputs = {
                        "pilot_only": initial,
                        "current_oracle_topk_raw_rho1": current,
                        "spread_oracle_raw_rho1": raw_output,
                        "spread_oracle_effective_rho1": effective_output,
                        "spread_oracle_effective_rho0125": weak_output,
                        "spread_oracle_effective_mean_only": mean_only_output,
                        "spread_oracle_effective_cov_only": cov_only_output,
                        "best_in_global_basis_calibrated": basis_output,
                        "true_channel_repaired": true_output,
                    }
                    decoded = decode_outputs(context, batch, outputs, bp_iterations=int(config["bp_iterations"]), device=device)
                    ls_metrics = run_standard_receiver(ls_receiver, batch, batch.information_bits, perfect_csi=False)
                    perfect_metrics = run_standard_receiver(perfect_receiver, batch, batch.information_bits, perfect_csi=True)
                    shared = {
                        **observation_equation_diagnostics(batch),
                        **basis_diag,
                        "effective_to_raw_noise_ratio": float((eff_grid.mean() / raw_grid.mean().clamp_min(1e-12)).item()),
                    }

                rows = [
                    custom_row(
                        case=case, variant=name, snr=snr, rep=rep, seed=seed,
                        output=output, batch=batch, decoded=decoded[name],
                        signature=contract["signature"], shared=shared,
                    )
                    for name, output in outputs.items()
                ]
                rows.extend([
                    standard_row(case, "ls_lmmse", snr, rep, seed, ls_metrics, contract["signature"], shared),
                    standard_row(case, "perfect_csi_lmmse", snr, rep, seed, perfect_metrics, contract["signature"], shared),
                ])
                if {row["variant"] for row in rows} != set(VARIANTS):
                    raise RuntimeError("Audit paired variant set mismatch")
                append_rows_atomic(RAW_PATH, rows)
                done.add(key)
                print(json.dumps({
                    "case": case.name, "num_prb": int(case.num_prb), "ebno_db": snr,
                    "rep": rep, "rows_committed": len(rows),
                    "completed_rows": len(done) * len(VARIANTS), "expected_rows": EXPECTED_ROWS,
                }), flush=True)
                if time.time() >= deadline:
                    frame = pd.read_csv(RAW_PATH)
                    write_incomplete(len(frame), contract)
                    return
        del bridge, detector, ls_receiver, perfect_receiver
        if device.type == "cuda":
            torch.cuda.empty_cache()

    frame = pd.read_csv(RAW_PATH)
    unique = len(frame.drop_duplicates(["case", "variant", "ebno_db", "rep"]))
    if len(frame) != EXPECTED_ROWS or unique != EXPECTED_ROWS:
        raise RuntimeError(f"Audit incomplete: rows={len(frame)}, unique={unique}")
    core = frame[~frame["variant"].isin(["ls_lmmse", "perfect_csi_lmmse"])]
    required_numeric = ["tbler", "coded_bit_nll", "channel_nmse", "basis_projection_nmse", "observation_noise_ratio"]
    if not np.isfinite(core[required_numeric].to_numpy(dtype=float)).all():
        raise RuntimeError("Non-finite core audit metric")
    aggregate(frame).to_csv(AGGREGATE_PATH, index=False)
    scientific = classify(frame)
    plots = make_plots(frame)
    report = {
        "version": VERSION,
        "complete": True,
        "classification": scientific["classification"],
        "next_action": scientific["next_action"],
        "publication_nr_ready": False,
        "preconditions": pre,
        "math_self_test": math_report,
        "evaluation": {
            "complete": True,
            "rows": len(frame),
            "unique_rows": unique,
            "expected_rows": EXPECTED_ROWS,
            "raw_csv": str(RAW_PATH.relative_to(ROOT)),
            "aggregate_csv": str(AGGREGATE_PATH.relative_to(ROOT)),
            "contract": contract,
        },
        "environment": {
            "device": str(device),
            "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "torch": torch.__version__,
            "python": platform.python_version(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        },
        "plots": plots,
        **scientific,
    }
    save_json(report, REPORT_PATH)
    save_json(report, GATE_JSON)
    software_checks = {
        "complete_rows": len(frame) == EXPECTED_ROWS,
        "unique_rows": unique == EXPECTED_ROWS,
        "all_variants_present": set(frame["variant"].unique()) == set(VARIANTS),
        "all_core_metrics_finite": True,
        "training_required": False,
    }
    lines = [
        *(f"{name}: {'PASS' if value else 'FAIL'}" for name, value in software_checks.items()),
        *(f"{name}: {'PASS' if value else 'FAIL'}" for name, value in report["scientific_checks"].items()),
        f"CLASSIFICATION: {report['classification']}",
        f"NEXT_ACTION: {report['next_action']}",
        "PUBLICATION_NR_READY: NO",
    ]
    GATE_TXT.parent.mkdir(parents=True, exist_ok=True)
    GATE_TXT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
