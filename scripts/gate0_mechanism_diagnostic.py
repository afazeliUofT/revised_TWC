#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bayesroute.config import apply_optuna_best, get_device, load_config, save_json, set_seed
from bayesroute.losses import bit_metrics, calibration_ece, channel_coverage95, channel_marginal_nll, channel_nmse
from bayesroute.models import (
    BayesRouteDetector,
    BayesRouteReceiver,
    LowRankPosteriorOperator,
    coupling_matrix,
    edge_density,
)
from bayesroute.qam import make_qam_constellation, symbol_logits_to_bit_logits
from bayesroute.simulator import UplinkToySimulator

DIAGNOSTIC_VERSION = "gate0_mechanism_diagnostic_v1"


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise RuntimeError(f"Expected a mapping in {path}")
    return data


def _load_trained_model(cfg, simulator, checkpoint: Path) -> BayesRouteReceiver:
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Missing trained checkpoint: {checkpoint}")
    state = torch.load(checkpoint, map_location=simulator.device, weights_only=False)
    saved_cfg = state.get("config")
    if not isinstance(saved_cfg, dict):
        raise RuntimeError("Checkpoint has no saved config")
    expected = {
        "package_revision": cfg.get("package_revision"),
        "system": cfg.system.to_dict(),
        "model": cfg.model.to_dict(),
    }
    observed = {
        "package_revision": saved_cfg.get("package_revision"),
        "system": saved_cfg.get("system"),
        "model": saved_cfg.get("model"),
    }
    if observed != expected:
        raise RuntimeError("Diagnostic checkpoint/config contract mismatch")
    model = BayesRouteReceiver(cfg, simulator.coords, simulator.pilot_idx).to(simulator.device)
    model.load_state_dict(state["model"], strict=True)
    return model.eval()


def _zero_covariance(local_cov: torch.Tensor) -> torch.Tensor:
    return torch.zeros_like(local_cov)


def _diagonal_covariance(local_cov: torch.Tensor) -> torch.Tensor:
    n_layers, _, n_re = local_cov.shape
    out = torch.zeros_like(local_cov)
    idx = torch.arange(n_layers, device=local_cov.device)
    out[idx, idx, :] = local_cov[idx, idx, :]
    return out


def _homoscedastic_covariance(local_cov: torch.Tensor) -> torch.Tensor:
    n_layers, _, n_re = local_cov.shape
    diagonal = torch.stack([local_cov[n, n].real for n in range(n_layers)], dim=0)
    scalar = diagonal.mean().clamp_min(1e-8)
    out = torch.zeros_like(local_cov)
    idx = torch.arange(n_layers, device=local_cov.device)
    out[idx, idx, :] = scalar.to(out.dtype)
    return out


def _random_kappa_like(reference: torch.Tensor, seed: int) -> torch.Tensor:
    """Symmetric almost-equal random scores; edge_mass=0.8 keeps 4/5 edges for N=6."""
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    shape = reference.shape
    raw = torch.rand(shape, generator=generator, dtype=torch.float32).to(reference.device)
    raw = 1.0 + 1e-3 * raw
    raw = 0.5 * (raw + raw.transpose(-1, -2))
    n_layers = shape[-1]
    diagonal = torch.eye(n_layers, dtype=torch.bool, device=reference.device).view(1, 1, n_layers, n_layers)
    return raw.masked_fill(diagonal, 0.0)


def _detector_forward(
    detector: BayesRouteDetector,
    batch,
    mean: torch.Tensor,
    covariance: torch.Tensor,
    kappa: torch.Tensor | None,
    edge_mass_value: float,
    use_uncertainty: bool,
) -> dict[str, Any]:
    bit_logits, symbol_logits, x_mean, x_var, graph_mask = detector(
        batch.y,
        mean,
        covariance,
        batch.data_idx,
        batch.noise_var,
        kappa=kappa,
        edge_mass=float(edge_mass_value),
        use_uncertainty=bool(use_uncertainty),
    )
    return {
        "bit_logits": bit_logits,
        "symbol_logits": symbol_logits,
        "x_mean": x_mean,
        "x_var": x_var,
        "graph_mask": graph_mask,
        "edge_density": float(edge_density(graph_mask).item()),
    }


def _perfect_csi_lmmse(batch, bits_per_symbol: int) -> dict[str, Any]:
    """Gaussian-prior perfect-CSI LMMSE marginal detector.

    This is a strong linear reference, not exact discrete-input ML detection.
    """
    device = batch.y.device
    h = batch.h[..., batch.data_idx].permute(0, 3, 2, 1).contiguous()  # [B,D,RX,N]
    y = batch.y[..., batch.data_idx].permute(0, 2, 1).unsqueeze(-1).contiguous()  # [B,D,RX,1]
    hh = h.conj().transpose(-1, -2)
    n_layers = h.shape[-1]
    eye = torch.eye(n_layers, dtype=h.dtype, device=device).view(1, 1, n_layers, n_layers)
    gram = hh @ h
    system = gram + batch.noise_var.to(h.dtype) * eye
    rhs = hh @ y
    mean = torch.linalg.solve(system, rhs).squeeze(-1)  # [B,D,N]
    posterior_cov = batch.noise_var.to(h.dtype) * torch.linalg.inv(system)
    variance = torch.diagonal(posterior_cov, dim1=-2, dim2=-1).real.clamp_min(1e-6)  # [B,D,N]
    mean = mean.permute(0, 2, 1).contiguous()  # [B,N,D]
    variance = variance.permute(0, 2, 1).contiguous()
    constellation, bit_table = make_qam_constellation(bits_per_symbol, device=device)
    candidate = constellation.view(1, 1, 1, -1)
    logits = -torch.abs(mean.unsqueeze(-1) - candidate) ** 2 / variance.unsqueeze(-1)
    bit_logits = symbol_logits_to_bit_logits(logits, bit_table)
    return {
        "bit_logits": bit_logits,
        "symbol_logits": logits,
        "edge_density": float("nan"),
    }


def _confidence_diagnostics(bit_logits: torch.Tensor, bits: torch.Tensor) -> dict[str, float]:
    probs = torch.sigmoid(bit_logits)
    prediction = probs >= 0.5
    error = prediction != bits.bool()
    abs_logits = bit_logits.abs()
    wrong = abs_logits[error]
    return {
        "mean_abs_logit": float(abs_logits.mean().item()),
        "max_abs_logit": float(abs_logits.max().item()),
        "wrong_mean_abs_logit": float(wrong.mean().item()) if wrong.numel() else 0.0,
        "wrong_max_abs_logit": float(wrong.max().item()) if wrong.numel() else 0.0,
    }


def _temperature_scale(
    logits: torch.Tensor,
    bits: torch.Tensor,
    lower: float,
    upper: float,
    points: int,
) -> tuple[float, float]:
    scales = torch.logspace(
        math.log10(float(lower)),
        math.log10(float(upper)),
        int(points),
        device=logits.device,
    )
    best_scale = 1.0
    best_nll = float("inf")
    with torch.no_grad():
        for scale in scales:
            value = F.binary_cross_entropy_with_logits(logits * scale, bits.float()).item()
            if value < best_nll:
                best_nll = float(value)
                best_scale = float(scale.item())
    return best_scale, best_nll


def _row(
    variant: str,
    output: dict[str, Any],
    batch,
    *,
    snr_db: float,
    rep: int,
    eval_seed: int,
    posterior_mean: torch.Tensor | None = None,
    posterior_var: torch.Tensor | None = None,
    temperature: float = 1.0,
) -> dict[str, Any]:
    logits = output["bit_logits"] * float(temperature)
    metrics = bit_metrics(logits, batch.data_bits)
    metrics["ece"] = calibration_ece(logits, batch.data_bits)
    metrics.update(_confidence_diagnostics(logits, batch.data_bits))
    if posterior_mean is not None and posterior_var is not None:
        metrics["channel_nmse"] = channel_nmse(
            posterior_mean[..., batch.data_idx], batch.h[..., batch.data_idx]
        )
        metrics["channel_marginal_nll"] = float(
            channel_marginal_nll(
                posterior_mean[..., batch.data_idx],
                posterior_var[:, batch.data_idx],
                batch.h[..., batch.data_idx],
            ).item()
        )
        metrics["channel_coverage95"] = channel_coverage95(
            posterior_mean[..., batch.data_idx],
            posterior_var[:, batch.data_idx],
            batch.h[..., batch.data_idx],
        )
    else:
        metrics["channel_nmse"] = float("nan")
        metrics["channel_marginal_nll"] = float("nan")
        metrics["channel_coverage95"] = float("nan")
    metrics.update(
        {
            "variant": variant,
            "snr_db": float(snr_db),
            "rep": int(rep),
            "eval_seed": int(eval_seed),
            "temperature": float(temperature),
            "edge_density": float(output.get("edge_density", float("nan"))),
        }
    )
    return metrics


def _aggregate(df: pd.DataFrame) -> pd.DataFrame:
    numeric = [
        "ber", "bit_nll", "brier", "ece", "mean_abs_logit", "max_abs_logit",
        "wrong_mean_abs_logit", "wrong_max_abs_logit", "channel_nmse",
        "channel_marginal_nll", "channel_coverage95", "temperature", "edge_density",
    ]
    records: list[dict[str, Any]] = []
    for (variant, snr), sub in df.groupby(["variant", "snr_db"], sort=True):
        record: dict[str, Any] = {"variant": variant, "snr_db": float(snr), "reps": int(len(sub))}
        for metric in numeric:
            values = pd.to_numeric(sub[metric], errors="coerce").dropna()
            record[f"{metric}_mean"] = float(values.mean()) if len(values) else float("nan")
            record[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        records.append(record)
    return pd.DataFrame(records)


def _paired(df: pd.DataFrame, reference: str) -> pd.DataFrame:
    reference_df = df[df["variant"] == reference]
    metrics = ["ber", "bit_nll", "brier", "ece", "wrong_mean_abs_logit"]
    rows: list[dict[str, Any]] = []
    keys = ["snr_db", "rep", "eval_seed"]
    for variant in sorted(set(df["variant"]) - {reference}):
        other = df[df["variant"] == variant]
        merged = reference_df.merge(other, on=keys, suffixes=("_reference", "_other"))
        for _, item in merged.iterrows():
            row: dict[str, Any] = {
                "reference": reference,
                "comparator": variant,
                "snr_db": float(item["snr_db"]),
                "rep": int(item["rep"]),
                "eval_seed": int(item["eval_seed"]),
            }
            for metric in metrics:
                row[f"{metric}_delta_reference_minus_comparator"] = float(
                    item[f"{metric}_reference"] - item[f"{metric}_other"]
                )
            rows.append(row)
    return pd.DataFrame(rows)


def _paired_summary(paired: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    metric_columns = [column for column in paired.columns if column.endswith("_delta_reference_minus_comparator")]
    for comparator, sub in paired.groupby("comparator", sort=True):
        row: dict[str, Any] = {"comparator": comparator, "pairs": int(len(sub))}
        for metric in metric_columns:
            values = sub[metric].astype(float)
            mean = float(values.mean())
            std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
            half = 1.96 * std / math.sqrt(max(len(values), 1))
            row[f"{metric}_mean"] = mean
            row[f"{metric}_ci95_low"] = mean - half
            row[f"{metric}_ci95_high"] = mean + half
        rows.append(row)
    return pd.DataFrame(rows)


def _plot_curves(aggregate: pd.DataFrame, variants: list[str], metric: str, ylabel: str, path: Path, log_y: bool = False) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    for variant in variants:
        sub = aggregate[aggregate["variant"] == variant].sort_values("snr_db")
        if sub.empty:
            continue
        y = sub[f"{metric}_mean"]
        ax.plot(sub["snr_db"], y, marker="o", label=variant)
    if log_y:
        ax.set_yscale("log")
    ax.set_xlabel("SNR [dB]")
    ax.set_ylabel(ylabel)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _lookup_mean(paired_summary: pd.DataFrame, comparator: str, metric: str) -> float:
    row = paired_summary[paired_summary["comparator"] == comparator]
    if row.empty:
        return float("nan")
    return float(row.iloc[0][f"{metric}_delta_reference_minus_comparator_mean"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--diagnostic-config", default="configs/gate0_mechanism_diagnostic.yaml")
    parser.add_argument("--resume", action="store_true", default=True)
    args = parser.parse_args()

    diagnostic_path = Path(args.diagnostic_config)
    diagnostic = _load_yaml(diagnostic_path)
    source_config_path = Path(str(diagnostic["source_config"]))
    cfg = load_config(source_config_path)
    cfg, optuna_meta = apply_optuna_best(cfg)
    if not optuna_meta.get("applied", False):
        raise RuntimeError("Completed Optuna result is required")
    if str(cfg.get("package_revision")) != str(diagnostic["package_revision"]):
        raise RuntimeError("Diagnostic/source package revision mismatch")

    device = get_device(cfg)
    simulator = UplinkToySimulator(cfg, device)
    checkpoint = Path(str(diagnostic["checkpoint"]))
    trained = _load_trained_model(cfg, simulator, checkpoint)
    untrained = BayesRouteReceiver(cfg, simulator.coords, simulator.pilot_idx).to(device).eval()
    matched_operator = LowRankPosteriorOperator(
        coords=simulator.coords,
        pilot_idx=simulator.pilot_idx,
        n_layers=int(cfg.system.n_layers),
        rank=int(cfg.system.channel_true_rank),
        length_f=float(cfg.system.channel_length_f),
        length_t=float(cfg.system.channel_length_t),
        seed=int(cfg.system.channel_seed),
        bank_rank=int(cfg.system.channel_true_rank),
    ).to(device).eval()
    matched_detector = BayesRouteDetector(
        int(cfg.system.bits_per_symbol),
        n_iter=int(cfg.model.detector_iterations),
        use_uncertainty=True,
    ).to(device).eval()
    pic_detectors = {
        iteration: BayesRouteDetector(
            int(cfg.system.bits_per_symbol), n_iter=iteration, use_uncertainty=False
        ).to(device).eval()
        for iteration in (1, 2, 3, 4)
    }

    prefix = str(diagnostic["output_prefix"])
    eval_dir = Path("outputs/eval")
    report_dir = Path("outputs/reports")
    plot_dir = Path("outputs/plots")
    gate_dir = Path("outputs/gates")
    for directory in (eval_dir, report_dir, plot_dir, gate_dir):
        directory.mkdir(parents=True, exist_ok=True)
    raw_path = eval_dir / f"{prefix}_eval.csv"
    contract_path = eval_dir / f"{prefix}_contract.json"
    aggregate_path = eval_dir / f"{prefix}_aggregate.csv"
    paired_path = eval_dir / f"{prefix}_paired.csv"
    paired_summary_path = report_dir / f"{prefix}_paired_summary.csv"

    contract = {
        "diagnostic_version": DIAGNOSTIC_VERSION,
        "diagnostic_config": diagnostic,
        "diagnostic_config_sha256": _sha256_file(diagnostic_path),
        "source_config_sha256": _sha256_file(source_config_path),
        "checkpoint_sha256": _sha256_file(checkpoint),
        "effective_model": cfg.model.to_dict(),
        "effective_system": cfg.system.to_dict(),
        "optuna": optuna_meta,
    }
    contract["signature"] = hashlib.sha256(_canonical_json(contract).encode("utf-8")).hexdigest()
    if raw_path.exists():
        if not contract_path.exists():
            raise RuntimeError("Diagnostic CSV exists without a contract")
        saved = json.loads(contract_path.read_text(encoding="utf-8"))
        if saved.get("signature") != contract["signature"]:
            raise RuntimeError("Diagnostic resume contract mismatch")
    else:
        save_json(contract, contract_path)

    done: set[tuple[str, float, int]] = set()
    if raw_path.exists():
        old = pd.read_csv(raw_path)
        for _, item in old.iterrows():
            done.add((str(item["variant"]), float(item["snr_db"]), int(item["rep"])))

    snrs = [float(value) for value in diagnostic["snr_grid_db"]]
    repetitions = int(diagnostic["repetitions"])
    batch_size = int(diagnostic["batch_size"])
    edge_mass_value = float(diagnostic["edge_mass"])
    required_variants = [str(value) for value in diagnostic["required_variants"]]
    expected_rows = len(required_variants) * len(snrs) * repetitions

    # Held-out, SNR-specific scalar calibration for two deliberately overconfident controls.
    temperature_by_snr: dict[str, dict[str, float]] = {}
    for snr_index, snr in enumerate(snrs):
        calibration_seed = int(diagnostic["base_seed"]) + int(diagnostic["calibration_seed_offset"]) + 1000 * snr_index
        set_seed(calibration_seed)
        calibration_batch = simulator.sample(
            batch_size=int(diagnostic["calibration_batch_size"]), snr_db=snr
        )
        with torch.no_grad():
            posterior = trained.posterior(
                calibration_batch.y[..., calibration_batch.pilot_idx],
                calibration_batch.phi,
                calibration_batch.noise_var,
            )
            full_kappa = coupling_matrix(
                posterior.mean,
                posterior.local_cov,
                calibration_batch.data_idx,
                calibration_batch.noise_var,
            )
            off = _detector_forward(
                trained.detector,
                calibration_batch,
                posterior.mean,
                posterior.local_cov,
                full_kappa,
                edge_mass_value,
                False,
            )
            pic4 = _detector_forward(
                pic_detectors[4],
                calibration_batch,
                calibration_batch.h,
                torch.zeros(
                    (int(cfg.system.n_layers), int(cfg.system.n_layers), calibration_batch.h.shape[-1]),
                    dtype=torch.complex64,
                    device=device,
                ),
                None,
                1.0,
                False,
            )
        off_scale, off_nll = _temperature_scale(
            off["bit_logits"], calibration_batch.data_bits,
            float(diagnostic["temperature_grid_min"]),
            float(diagnostic["temperature_grid_max"]),
            int(diagnostic["temperature_grid_points"]),
        )
        pic_scale, pic_nll = _temperature_scale(
            pic4["bit_logits"], calibration_batch.data_bits,
            float(diagnostic["temperature_grid_min"]),
            float(diagnostic["temperature_grid_max"]),
            int(diagnostic["temperature_grid_points"]),
        )
        temperature_by_snr[str(snr)] = {
            "uncertainty_detector_off_fixed_graph": off_scale,
            "uncertainty_detector_off_fixed_graph_calibration_nll": off_nll,
            "perfect_csi_pic_iter4": pic_scale,
            "perfect_csi_pic_iter4_calibration_nll": pic_nll,
        }

    for snr_index, snr in enumerate(snrs):
        for rep in range(repetitions):
            missing = [
                variant for variant in required_variants
                if (variant, snr, rep) not in done
            ]
            if not missing:
                continue
            eval_seed = int(diagnostic["base_seed"]) + 100000 * snr_index + rep
            set_seed(eval_seed)
            batch = simulator.sample(batch_size=batch_size, snr_db=snr)
            with torch.no_grad():
                trained_posterior = trained.posterior(
                    batch.y[..., batch.pilot_idx], batch.phi, batch.noise_var
                )
                full_cov = trained_posterior.local_cov
                zero_cov = _zero_covariance(full_cov)
                diagonal_cov = _diagonal_covariance(full_cov)
                scalar_cov = _homoscedastic_covariance(full_cov)
                full_kappa = coupling_matrix(
                    trained_posterior.mean, full_cov, batch.data_idx, batch.noise_var
                )
                mean_kappa = coupling_matrix(
                    trained_posterior.mean, zero_cov, batch.data_idx, batch.noise_var
                )
                diagonal_kappa = coupling_matrix(
                    trained_posterior.mean, diagonal_cov, batch.data_idx, batch.noise_var
                )
                scalar_kappa = coupling_matrix(
                    trained_posterior.mean, scalar_cov, batch.data_idx, batch.noise_var
                )
                random_kappa = _random_kappa_like(full_kappa, eval_seed + 900000)

                outputs: dict[str, tuple[dict[str, Any], torch.Tensor | None, torch.Tensor | None, float]] = {}
                outputs["trained_proposed"] = (
                    _detector_forward(trained.detector, batch, trained_posterior.mean, full_cov, full_kappa, edge_mass_value, True),
                    trained_posterior.mean, trained_posterior.var_diag, 1.0,
                )
                off_fixed = _detector_forward(
                    trained.detector, batch, trained_posterior.mean, full_cov, full_kappa, edge_mass_value, False
                )
                outputs["uncertainty_detector_off_fixed_graph"] = (off_fixed, None, None, 1.0)
                outputs["uncertainty_detector_off_fixed_graph_calibrated"] = (
                    off_fixed, None, None,
                    float(temperature_by_snr[str(snr)]["uncertainty_detector_off_fixed_graph"]),
                )
                outputs["uncertainty_all_off"] = (
                    _detector_forward(trained.detector, batch, trained_posterior.mean, zero_cov, mean_kappa, edge_mass_value, False),
                    None, None, 1.0,
                )
                outputs["diagonal_posterior"] = (
                    _detector_forward(trained.detector, batch, trained_posterior.mean, diagonal_cov, diagonal_kappa, edge_mass_value, True),
                    trained_posterior.mean,
                    torch.stack([diagonal_cov[n, n].real for n in range(diagonal_cov.shape[0])]),
                    1.0,
                )
                outputs["homoscedastic_posterior"] = (
                    _detector_forward(trained.detector, batch, trained_posterior.mean, scalar_cov, scalar_kappa, edge_mass_value, True),
                    trained_posterior.mean,
                    torch.stack([scalar_cov[n, n].real for n in range(scalar_cov.shape[0])]),
                    1.0,
                )
                outputs["graph_off"] = (
                    _detector_forward(trained.detector, batch, trained_posterior.mean, full_cov, full_kappa, 0.0, True),
                    trained_posterior.mean, trained_posterior.var_diag, 1.0,
                )
                outputs["mean_only_graph"] = (
                    _detector_forward(trained.detector, batch, trained_posterior.mean, full_cov, mean_kappa, edge_mass_value, True),
                    trained_posterior.mean, trained_posterior.var_diag, 1.0,
                )
                outputs["random_graph_same_density"] = (
                    _detector_forward(trained.detector, batch, trained_posterior.mean, full_cov, random_kappa, edge_mass_value, True),
                    trained_posterior.mean, trained_posterior.var_diag, 1.0,
                )
                outputs["full_graph"] = (
                    _detector_forward(trained.detector, batch, trained_posterior.mean, full_cov, full_kappa, 1.0, True),
                    trained_posterior.mean, trained_posterior.var_diag, 1.0,
                )

                untrained_posterior = untrained.posterior(
                    batch.y[..., batch.pilot_idx], batch.phi, batch.noise_var
                )
                untrained_kappa = coupling_matrix(
                    untrained_posterior.mean, untrained_posterior.local_cov, batch.data_idx, batch.noise_var
                )
                outputs["untrained_proposed"] = (
                    _detector_forward(untrained.detector, batch, untrained_posterior.mean, untrained_posterior.local_cov, untrained_kappa, edge_mass_value, True),
                    untrained_posterior.mean, untrained_posterior.var_diag, 1.0,
                )

                matched_posterior = matched_operator(
                    batch.y[..., batch.pilot_idx], batch.phi, batch.noise_var
                )
                matched_kappa = coupling_matrix(
                    matched_posterior.mean, matched_posterior.local_cov, batch.data_idx, batch.noise_var
                )
                outputs["matched_posterior"] = (
                    _detector_forward(matched_detector, batch, matched_posterior.mean, matched_posterior.local_cov, matched_kappa, edge_mass_value, True),
                    matched_posterior.mean, matched_posterior.var_diag, 1.0,
                )

                perfect_cov = torch.zeros(
                    (int(cfg.system.n_layers), int(cfg.system.n_layers), batch.h.shape[-1]),
                    dtype=torch.complex64,
                    device=device,
                )
                for iteration in (1, 2, 3, 4):
                    name = f"perfect_csi_pic_iter{iteration}"
                    outputs[name] = (
                        _detector_forward(pic_detectors[iteration], batch, batch.h, perfect_cov, None, 1.0, False),
                        None, None, 1.0,
                    )
                outputs["perfect_csi_pic_iter4_calibrated"] = (
                    outputs["perfect_csi_pic_iter4"][0], None, None,
                    float(temperature_by_snr[str(snr)]["perfect_csi_pic_iter4"]),
                )
                outputs["perfect_csi_lmmse"] = (
                    _perfect_csi_lmmse(batch, int(cfg.system.bits_per_symbol)), None, None, 1.0
                )

            rows = []
            for variant in missing:
                if variant not in outputs:
                    raise RuntimeError(f"Required diagnostic variant was not produced: {variant}")
                output, posterior_mean, posterior_var, temperature = outputs[variant]
                rows.append(
                    _row(
                        variant,
                        output,
                        batch,
                        snr_db=snr,
                        rep=rep,
                        eval_seed=eval_seed,
                        posterior_mean=posterior_mean,
                        posterior_var=posterior_var,
                        temperature=temperature,
                    )
                )
            pd.DataFrame(rows).to_csv(
                raw_path, mode="a", header=not raw_path.exists(), index=False
            )
            for variant in missing:
                done.add((variant, snr, rep))
            print(json.dumps({"snr_db": snr, "rep": rep, "completed_variants": len(missing)}), flush=True)

    df = pd.read_csv(raw_path)
    unique_rows = len(df.drop_duplicates(["variant", "snr_db", "rep"]))
    complete = len(df) == expected_rows and unique_rows == expected_rows
    if not complete:
        raise RuntimeError(
            f"Diagnostic evaluation incomplete: rows={len(df)}, unique={unique_rows}, expected={expected_rows}"
        )
    aggregate = _aggregate(df)
    aggregate.to_csv(aggregate_path, index=False)
    paired = _paired(df, "trained_proposed")
    paired.to_csv(paired_path, index=False)
    paired_summary = _paired_summary(paired)
    paired_summary.to_csv(paired_summary_path, index=False)

    plot_variants = [
        "trained_proposed",
        "uncertainty_detector_off_fixed_graph",
        "uncertainty_detector_off_fixed_graph_calibrated",
        "graph_off",
        "random_graph_same_density",
        "full_graph",
        "matched_posterior",
        "perfect_csi_pic_iter4",
        "perfect_csi_pic_iter4_calibrated",
        "perfect_csi_lmmse",
    ]
    _plot_curves(aggregate, plot_variants, "ber", "Uncoded BER", plot_dir / f"{prefix}_ber.png", log_y=True)
    _plot_curves(aggregate, plot_variants, "bit_nll", "Bit NLL", plot_dir / f"{prefix}_bit_nll.png")
    _plot_curves(
        aggregate,
        [f"perfect_csi_pic_iter{i}" for i in (1, 2, 3, 4)] + ["perfect_csi_pic_iter4_calibrated", "perfect_csi_lmmse"],
        "wrong_mean_abs_logit",
        "Mean |LLR| on wrong bits",
        plot_dir / f"{prefix}_wrong_confidence.png",
        log_y=True,
    )

    proposed_vs_fixed_off_ber = _lookup_mean(
        paired_summary, "uncertainty_detector_off_fixed_graph", "ber"
    )
    proposed_vs_fixed_off_nll = _lookup_mean(
        paired_summary, "uncertainty_detector_off_fixed_graph", "bit_nll"
    )
    proposed_vs_calibrated_off_nll = _lookup_mean(
        paired_summary, "uncertainty_detector_off_fixed_graph_calibrated", "bit_nll"
    )
    proposed_vs_graph_off_ber = _lookup_mean(paired_summary, "graph_off", "ber")
    proposed_vs_random_ber = _lookup_mean(paired_summary, "random_graph_same_density", "ber")
    proposed_vs_full_ber = _lookup_mean(paired_summary, "full_graph", "ber")
    proposed_vs_untrained_nll = _lookup_mean(paired_summary, "untrained_proposed", "bit_nll")

    pic4 = aggregate[aggregate["variant"] == "perfect_csi_pic_iter4"].sort_values("snr_db")
    pic4_nll_slope = float(pic4.iloc[-1]["bit_nll_mean"] - pic4.iloc[0]["bit_nll_mean"])
    pic4_ber_slope = float(pic4.iloc[-1]["ber_mean"] - pic4.iloc[0]["ber_mean"])
    lmmse = aggregate[aggregate["variant"] == "perfect_csi_lmmse"].sort_values("snr_db")
    lmmse_ber_slope = float(lmmse.iloc[-1]["ber_mean"] - lmmse.iloc[0]["ber_mean"])
    matched = aggregate[aggregate["variant"] == "matched_posterior"]
    matched_coverage = float(matched["channel_coverage95_mean"].mean())

    checks = {
        "complete_rows": complete,
        "fixed_graph_uncertainty_ber_gain": bool(proposed_vs_fixed_off_ber < 0.0),
        "fixed_graph_uncertainty_nll_gain": bool(proposed_vs_fixed_off_nll < 0.0),
        "uncertainty_not_only_temperature": bool(proposed_vs_fixed_off_ber < 0.0),
        "coupling_graph_beats_graph_off": bool(proposed_vs_graph_off_ber < 0.0),
        "coupling_graph_beats_random_same_density": bool(proposed_vs_random_ber < 0.0),
        "sparse_graph_not_worse_than_full_by_0p001": bool(proposed_vs_full_ber <= 0.001),
        "training_improves_nll": bool(proposed_vs_untrained_nll < 0.0),
        "matched_posterior_coverage_near_95": bool(0.92 <= matched_coverage <= 0.98),
        "perfect_csi_lmmse_ber_improves_with_snr": bool(lmmse_ber_slope < 0.0),
    }
    classification = "GATE0_MECHANISM_SUPPORTED" if all(checks.values()) else "GATE0_MECHANISM_MIXED_OR_BLOCKED"
    summary = {
        "classification": classification,
        "diagnostic_version": DIAGNOSTIC_VERSION,
        "contract_signature": contract["signature"],
        "complete": complete,
        "rows": int(len(df)),
        "expected_rows": int(expected_rows),
        "variants": required_variants,
        "snr_grid_db": snrs,
        "repetitions": repetitions,
        "checkpoint_sha256": contract["checkpoint_sha256"],
        "temperatures": temperature_by_snr,
        "key_paired_deltas_reference_minus_comparator": {
            "ber_vs_uncertainty_detector_off_fixed_graph": proposed_vs_fixed_off_ber,
            "bit_nll_vs_uncertainty_detector_off_fixed_graph": proposed_vs_fixed_off_nll,
            "bit_nll_vs_temperature_calibrated_off": proposed_vs_calibrated_off_nll,
            "ber_vs_graph_off": proposed_vs_graph_off_ber,
            "ber_vs_random_graph_same_density": proposed_vs_random_ber,
            "ber_vs_full_graph": proposed_vs_full_ber,
            "bit_nll_vs_untrained_proposed": proposed_vs_untrained_nll,
        },
        "perfect_csi_diagnostics": {
            "pic4_nll_change_high_minus_low_snr": pic4_nll_slope,
            "pic4_ber_change_high_minus_low_snr": pic4_ber_slope,
            "lmmse_ber_change_high_minus_low_snr": lmmse_ber_slope,
            "pic4_overconfidence_flag": bool(pic4_nll_slope > 0.05 and pic4_ber_slope < 0.0),
        },
        "matched_posterior_mean_coverage95": matched_coverage,
        "checks": checks,
        "raw_csv": str(raw_path),
        "aggregate_csv": str(aggregate_path),
        "paired_csv": str(paired_path),
        "paired_summary_csv": str(paired_summary_path),
    }
    save_json(summary, report_dir / f"{prefix}_summary.json")
    save_json(summary, gate_dir / "GATE0_MECHANISM_DIAGNOSTIC.json")
    lines = [f"{name}: {'PASS' if passed else 'FAIL'}" for name, passed in checks.items()]
    lines.append(f"CLASSIFICATION: {classification}")
    lines.append("PUBLICATION_NR_READY: NO")
    (gate_dir / "GATE0_MECHANISM_DIAGNOSTIC.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
