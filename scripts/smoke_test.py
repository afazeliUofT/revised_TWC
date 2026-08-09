#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import socket
import subprocess
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bayesroute.channels import complex_normal, rff_bank
from bayesroute.config import (
    count_parameters,
    get_device,
    load_config,
    save_json,
    set_seed,
)
from bayesroute.losses import (
    bit_bce_loss,
    bit_metrics,
    channel_coverage95,
    channel_marginal_nll,
    channel_nmse,
)
from bayesroute.models import (
    BayesRouteReceiver,
    LSReceiver,
    LowRankPosteriorOperator,
    OracleReceiver,
    coupling_matrix,
    diagonal_interference_moments,
)
from bayesroute.pilots import (
    PILOT_MODEL_SCOPE,
    pilot_orthogonality_report,
    pilot_separation_report,
    port_metadata_report,
    resource_partition_report,
)
from bayesroute.simulator import UplinkToySimulator
from bayesroute.sionna_check import check_sionna
import importlib.util

EXPECTED_REVISION = "gate0_v2_4_20260809"



def _optuna_workflow_report() -> dict:
    module_path = ROOT / "scripts" / "optuna_tune.py"
    spec = importlib.util.spec_from_file_location("bayesroute_gate0_optuna", module_path)
    if spec is None or spec.loader is None:
        return {"passed": False, "error": "could not load scripts/optuna_tune.py"}
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    report = module._workflow_self_test()
    report["search_space_version"] = module.SEARCH_SPACE_VERSION
    report["design_name"] = module.DESIGN_NAME
    report["design_signature"] = module.DESIGN_SIGNATURE
    return report

def _git_value(*args: str) -> str | None:
    try:
        proc = subprocess.run(
            ["git", *args],
            cwd=ROOT,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        return proc.stdout.strip()
    except Exception:
        return None


def _max_abs_difference(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.max(torch.abs(a - b)).item())


def _verify_manifest(root: Path) -> dict:
    manifest = root / "MANIFEST.sha256"
    if not manifest.exists():
        return {"passed": False, "error": "MANIFEST.sha256 is missing"}
    checked = 0
    failures = []
    for raw in manifest.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        expected, rel = raw.split(None, 1)
        rel = rel.strip().lstrip("*")
        path = root / rel
        if not path.is_file():
            failures.append({"path": rel, "reason": "missing"})
            continue
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        checked += 1
        if actual != expected:
            failures.append(
                {"path": rel, "reason": "hash_mismatch", "actual": actual}
            )
    return {
        "passed": not failures and checked > 0,
        "checked_files": checked,
        "failures": failures,
    }


def _rank_bank_nesting_report(cfg, coords: torch.Tensor) -> dict:
    active_rank = int(cfg.model.rank)
    bank_rank = int(cfg.model.get("operator_bank_rank", active_rank))
    small_rank = min(active_rank, 8)
    seed = int(cfg.model.operator_seed)
    small = rff_bank(
        coords,
        small_rank,
        float(cfg.system.channel_length_f),
        float(cfg.system.channel_length_t),
        seed=seed,
        bank_rank=bank_rank,
    )
    large = rff_bank(
        coords,
        bank_rank,
        float(cfg.system.channel_length_f),
        float(cfg.system.channel_length_t),
        seed=seed,
        bank_rank=bank_rank,
    )
    # Remove rank-dependent unit-power normalization before comparing modes.
    difference = torch.max(
        torch.abs(
            small * math.sqrt(float(small_rank))
            - large[:, :small_rank] * math.sqrt(float(bank_rank))
        )
    ).item()
    return {
        "active_rank": active_rank,
        "bank_rank": bank_rank,
        "small_rank": small_rank,
        "max_abs_common_mode_difference": float(difference),
        "passed": bool(bank_rank >= active_rank and difference < 2e-6),
    }


def _posterior_projection_report(model: BayesRouteReceiver, posterior) -> dict:
    latent_cov = posterior.latent_cov
    local_cov = posterior.local_cov
    features = model.posterior.weighted_features()
    n_layers = model.posterior.n_layers
    rank = model.posterior.rank

    hermitian_error = torch.max(
        torch.abs(local_cov - local_cov.transpose(0, 1).conj())
    ).item()
    diagonal = torch.stack(
        [local_cov[n, n].real for n in range(n_layers)], dim=0
    )
    diagonal_error = torch.max(torch.abs(diagonal - posterior.var_diag)).item()

    explicit_errors = []
    test_res = sorted({0, int(features.shape[0] // 2), int(features.shape[0] - 1)})
    for n in range(n_layers):
        block = latent_cov[
            n * rank:(n + 1) * rank,
            n * rank:(n + 1) * rank,
        ]
        for r in test_res:
            row = features[r:r + 1]
            explicit = (row @ block @ row.conj().T).real.squeeze()
            explicit_errors.append(
                torch.abs(explicit - posterior.var_diag[n, r])
            )
    explicit_max_error = float(torch.stack(explicit_errors).max().item())
    return {
        "local_cov_shape": list(local_cov.shape),
        "max_layer_hermitian_error": float(hermitian_error),
        "max_diagonal_consistency_error": float(diagonal_error),
        "max_explicit_u_C_uH_error": explicit_max_error,
        "minimum_variance": float(posterior.var_diag.min().item()),
        "passed": bool(
            hermitian_error < 2e-5
            and diagonal_error < 2e-6
            and explicit_max_error < 2e-5
            and posterior.var_diag.min().item() > 0.0
        ),
    }


def _matched_posterior_calibration_report(cfg, simulator, device) -> dict:
    """Validate posterior variance under data drawn from the operator's own prior."""
    rank = int(cfg.model.rank)
    operator = LowRankPosteriorOperator(
        coords=simulator.coords,
        pilot_idx=simulator.pilot_idx,
        n_layers=simulator.n_layers,
        rank=rank,
        length_f=float(cfg.system.channel_length_f),
        length_t=float(cfg.system.channel_length_t),
        seed=int(cfg.model.operator_seed),
        bank_rank=int(cfg.model.operator_bank_rank),
    ).to(device).eval()

    n_samples = 512
    n_rx = 2
    noise_var = torch.tensor(0.1, dtype=torch.float32, device=device)
    set_seed(int(cfg.seed) + 4001)
    with torch.no_grad():
        features = operator.weighted_features()
        latent = complex_normal(
            (n_samples, n_rx, simulator.n_layers, rank), device=device
        )
        channel = torch.einsum("rq,bxnq->bnxr", features, latent)
        y_pilot = torch.sum(
            channel[..., simulator.pilot_idx]
            * simulator.phi[None, :, None, :],
            dim=1,
        )
        y_pilot = y_pilot + complex_normal(
            y_pilot.shape,
            device=device,
            scale=math.sqrt(float(noise_var.item())),
        )
        posterior = operator(y_pilot, simulator.phi, noise_var)
        variance = posterior.var_diag[None, :, None, :]
        squared_error = torch.abs(channel - posterior.mean) ** 2
        normalized_error_mean = float(
            torch.mean(squared_error / variance).item()
        )
        coverage95 = float(
            torch.mean(
                (
                    squared_error
                    <= (-math.log(0.05)) * variance
                ).float()
            ).item()
        )
        # Cross-layer covariance is also part of the receiver contract. Compare
        # the conditional residual covariance at one representative RE against
        # the projected analytical posterior covariance.
        test_re = int(simulator.data_idx[len(simulator.data_idx) // 2].item())
        residual = (channel - posterior.mean)[..., test_re]
        residual_samples = residual.permute(0, 2, 1).reshape(-1, simulator.n_layers)
        empirical_cov = (
            residual_samples.T @ residual_samples.conj()
            / float(residual_samples.shape[0])
        )
        predicted_cov = posterior.local_cov[:, :, test_re]
        cross_cov_relative_error = float(
            (
                torch.linalg.matrix_norm(empirical_cov - predicted_cov)
                / torch.linalg.matrix_norm(predicted_cov).clamp_min(1e-8)
            ).item()
        )
    return {
        "independent_channel_draws": n_samples,
        "receive_antennas_per_draw": n_rx,
        "normalized_squared_error_mean": normalized_error_mean,
        "coverage95": coverage95,
        "cross_layer_covariance_test_re": test_re,
        "cross_layer_covariance_relative_error": cross_cov_relative_error,
        "passed": bool(
            0.85 <= normalized_error_mean <= 1.15
            and 0.92 <= coverage95 <= 0.98
            and cross_cov_relative_error < 0.15
        ),
    }


def _detector_moment_formula_report(device) -> dict:
    """Check soft cancellation and full cross-layer channel uncertainty."""
    mu = torch.tensor(
        [
            [
                [[1.2 + 0.4j]],
                [[-0.7 + 0.9j]],
                [[0.3 - 0.5j]],
            ]
        ],
        dtype=torch.complex64,
        device=device,
    )
    factor = torch.tensor(
        [
            [0.50 + 0.00j, 0.00 + 0.00j, 0.00 + 0.00j],
            [0.10 + 0.05j, 0.40 + 0.00j, 0.00 + 0.00j],
            [-0.08 + 0.03j, 0.06 - 0.02j, 0.35 + 0.00j],
        ],
        dtype=torch.complex64,
        device=device,
    )
    covariance = factor @ factor.conj().T
    local_cov = covariance[:, :, None]
    x_mean = torch.tensor(
        [[[0.0 + 0.0j], [0.8 - 0.1j], [-0.3 + 0.4j]]],
        dtype=torch.complex64,
        device=device,
    )
    x_var = torch.tensor(
        [[[1.0], [0.2], [0.3]]], dtype=torch.float32, device=device
    )
    noise = torch.tensor(0.1, dtype=torch.float32, device=device)
    strong = torch.tensor(
        [[[False], [True], [False]]], dtype=torch.bool, device=device
    )

    (
        interference_mean,
        variance_without_target,
        target_cross,
        target_variance,
    ) = diagonal_interference_moments(
        mu,
        local_cov,
        x_mean,
        x_var,
        strong,
        target_layer=0,
        noise_var=noise,
        use_uncertainty=True,
    )

    second = x_var + torch.abs(x_mean) ** 2
    expected_mean = mu[0, 1, :, 0] * x_mean[0, 1, 0]
    expected_mean_channel_variance = (
        x_var[0, 1, 0] * torch.abs(mu[0, 1, :, 0]) ** 2
        + second[0, 2, 0] * torch.abs(mu[0, 2, :, 0]) ** 2
    )
    means_other = x_mean[0, :, 0].clone()
    means_other[0] = 0.0
    expected_channel_error = torch.einsum(
        "i,ij,j->",
        means_other,
        covariance,
        means_other.conj(),
    ).real
    expected_channel_error = expected_channel_error + torch.sum(
        x_var[0, 1:, 0]
        * torch.diagonal(covariance, dim1=0, dim2=1)[1:].real
    )
    expected_base = (
        noise + expected_mean_channel_variance + expected_channel_error
    )
    expected_cross = torch.sum(
        covariance[0, 1:] * x_mean[0, 1:, 0].conj()
    )
    expected_target_variance = covariance[0, 0].real

    candidate = torch.tensor(0.7 + 0.2j, dtype=torch.complex64, device=device)
    candidate_variance = (
        variance_without_target
        + torch.abs(candidate) ** 2 * target_variance.view(1, 1, 1)
        + 2.0 * torch.real(candidate * target_cross).view(1, 1, 1)
    )
    expected_candidate = (
        expected_base
        + torch.abs(candidate) ** 2 * expected_target_variance
        + 2.0 * torch.real(candidate * expected_cross)
    )

    mean_error = float(
        torch.max(torch.abs(interference_mean.squeeze() - expected_mean)).item()
    )
    base_error = float(
        torch.max(
            torch.abs(variance_without_target.squeeze() - expected_base)
        ).item()
    )
    cross_error = float(torch.abs(target_cross.squeeze() - expected_cross).item())
    target_variance_error = float(
        torch.abs(target_variance.squeeze() - expected_target_variance).item()
    )
    candidate_error = float(
        torch.max(torch.abs(candidate_variance.squeeze() - expected_candidate)).item()
    )

    # Independent Monte Carlo check of the complete scalar residual moment for
    # one strong and one weak interferer. With one weak interferer, the declared
    # zero-mean Gaussianization has no omitted weak-weak coherent cross term.
    n_draws = 32768
    set_seed(654321)
    with torch.no_grad():
        chol = torch.linalg.cholesky(covariance)
        channel_white = complex_normal((n_draws, 3), device=device)
        channel_error = torch.einsum("ij,bj->bi", chol, channel_white)
        symbol_white = complex_normal((n_draws, 2), device=device)
        x1 = x_mean[0, 1, 0] + torch.sqrt(x_var[0, 1, 0]) * symbol_white[:, 0]
        x2 = x_mean[0, 2, 0] + torch.sqrt(x_var[0, 2, 0]) * symbol_white[:, 1]
        noise_draw = math.sqrt(float(noise.item())) * complex_normal(
            (n_draws,), device=device
        )
        residual = (
            mu[0, 1, 0, 0] * (x1 - x_mean[0, 1, 0])
            + mu[0, 2, 0, 0] * x2
            + channel_error[:, 0] * candidate
            + channel_error[:, 1] * x1
            + channel_error[:, 2] * x2
            + noise_draw
        )
        empirical_candidate_second_moment = torch.mean(torch.abs(residual) ** 2)
    monte_carlo_relative_error = float(
        torch.abs(empirical_candidate_second_moment - expected_candidate).item()
        / max(float(expected_candidate.item()), 1e-12)
    )
    return {
        "interference_mean_error": mean_error,
        "variance_without_target_error": base_error,
        "target_cross_covariance_error": cross_error,
        "target_variance_error": target_variance_error,
        "candidate_variance_error": candidate_error,
        "monte_carlo_draws": n_draws,
        "predicted_candidate_second_moment": float(expected_candidate.item()),
        "empirical_candidate_second_moment": float(
            empirical_candidate_second_moment.item()
        ),
        "monte_carlo_relative_error": monte_carlo_relative_error,
        "passed": bool(
            mean_error < 2e-6
            and base_error < 2e-6
            and cross_error < 2e-6
            and target_variance_error < 2e-6
            and candidate_error < 2e-6
            and monte_carlo_relative_error < 0.04
        ),
    }


def _coupling_monte_carlo_report(device) -> dict:
    """Compare closed-form posterior coupling against direct Monte Carlo."""
    n_draws = 32768
    n_rx = 3
    noise_var = torch.tensor(0.7, dtype=torch.float32, device=device)
    means = torch.tensor(
        [
            [0.5 + 0.2j, -0.3 + 0.1j, 0.7 - 0.4j],
            [-0.2 + 0.6j, 0.4 - 0.5j, 0.1 + 0.3j],
        ],
        dtype=torch.complex64,
        device=device,
    )
    layer_cov = torch.tensor(
        [
            [0.40 + 0.00j, 0.12 + 0.07j],
            [0.12 - 0.07j, 0.30 + 0.00j],
        ],
        dtype=torch.complex64,
        device=device,
    )
    local_cov = layer_cov[:, :, None]
    formula = coupling_matrix(
        means[None, :, :, None],
        local_cov,
        torch.tensor([0], dtype=torch.long, device=device),
        noise_var,
    )[0, 0, 0, 1]

    set_seed(123456)
    with torch.no_grad():
        chol = torch.linalg.cholesky(layer_cov)
        white = complex_normal((n_draws, n_rx, 2), device=device)
        error = torch.einsum("ij,brj->bri", chol, white)
        h0 = means[0][None, :] + error[:, :, 0]
        h1 = means[1][None, :] + error[:, :, 1]
        sample_coupling = torch.abs(
            torch.sum(h0.conj() * h1, dim=1) / noise_var
        ) ** 2
        empirical = torch.mean(sample_coupling)
    relative_error = float(
        torch.abs(empirical - formula).item()
        / max(float(formula.item()), 1e-12)
    )
    return {
        "draws": n_draws,
        "closed_form": float(formula.item()),
        "monte_carlo": float(empirical.item()),
        "relative_error": relative_error,
        "passed": bool(relative_error < 0.04),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/smoke.yaml")
    parser.add_argument("--out", default="outputs/smoke")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(int(cfg.seed))
    device = get_device(cfg)
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    report: dict = {
        "time": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "gate": "GATE0_PRINCIPLE_IMPLEMENTATION",
        "optuna_scope": "short_hyperparameter_search_for_gate0_only",
        "publication_nr_ready": False,
        "package_revision": str(cfg.get("package_revision", "unknown")),
        "device": str(device),
        "provenance": {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "python": sys.version,
            "torch": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_version": torch.version.cuda,
            "gpu_name": (
                torch.cuda.get_device_name(0)
                if torch.cuda.is_available()
                else None
            ),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "git_commit": _git_value("rev-parse", "HEAD"),
            "git_status_porcelain": _git_value("status", "--porcelain"),
        },
    }

    report["manifest"] = _verify_manifest(ROOT)
    report["optuna_workflow"] = _optuna_workflow_report()
    report["sionna"] = check_sionna(
        device=device,
        bits_per_symbol=int(cfg.system.bits_per_symbol),
    )
    simulator = UplinkToySimulator(cfg, device)
    report["pilot_model_scope"] = simulator.pilot_model_scope
    report["port_metadata"] = simulator.port_meta
    report["pilot_orthogonality"] = pilot_orthogonality_report(simulator.phi)
    report["pilot_separation"] = pilot_separation_report(simulator.phi)
    report["port_metadata_check"] = port_metadata_report(
        simulator.port_meta, simulator.n_layers
    )
    report["resource_partition"] = resource_partition_report(
        simulator.pilot_idx,
        simulator.data_idx,
        simulator.n_resource_elements,
    )
    report["rank_bank_nesting"] = _rank_bank_nesting_report(
        cfg, simulator.coords
    )

    operator_seed = int(cfg.model.operator_seed)
    report["seed_separation"] = {
        "channel_seed": int(simulator.channel_seed),
        "operator_seed": operator_seed,
        "passed": bool(int(simulator.channel_seed) != operator_seed),
    }

    set_seed(int(cfg.seed) + 77)
    repeat_a = simulator.sample(batch_size=2, snr_db=10.0)
    set_seed(int(cfg.seed) + 77)
    repeat_b = simulator.sample(batch_size=2, snr_db=10.0)
    report["deterministic_repeat"] = {
        "max_y_difference": _max_abs_difference(repeat_a.y, repeat_b.y),
        "max_h_difference": _max_abs_difference(repeat_a.h, repeat_b.h),
        "bit_mismatch_count": int(
            torch.sum(repeat_a.data_bits != repeat_b.data_bits).item()
        ),
    }
    report["deterministic_repeat"]["passed"] = bool(
        report["deterministic_repeat"]["max_y_difference"] == 0.0
        and report["deterministic_repeat"]["max_h_difference"] == 0.0
        and report["deterministic_repeat"]["bit_mismatch_count"] == 0
    )

    set_seed(int(cfg.seed))
    batch = simulator.sample(
        batch_size=int(cfg.training.batch_size), snr_db=10.0
    )
    expected = {
        "y": [
            int(cfg.training.batch_size),
            simulator.n_rx,
            simulator.n_resource_elements,
        ],
        "h": [
            int(cfg.training.batch_size),
            simulator.n_layers,
            simulator.n_rx,
            simulator.n_resource_elements,
        ],
        "x": [
            int(cfg.training.batch_size),
            simulator.n_layers,
            simulator.n_resource_elements,
        ],
        "data_bits": [
            int(cfg.training.batch_size),
            simulator.n_layers,
            int(simulator.data_idx.numel()),
            int(cfg.system.bits_per_symbol),
        ],
        "phi": [simulator.n_layers, int(simulator.pilot_idx.numel())],
        "pilot_idx": [int(simulator.pilot_idx.numel())],
        "data_idx": [int(simulator.data_idx.numel())],
    }
    actual = {
        "y": list(batch.y.shape),
        "h": list(batch.h.shape),
        "x": list(batch.x.shape),
        "data_bits": list(batch.data_bits.shape),
        "phi": list(batch.phi.shape),
        "pilot_idx": list(batch.pilot_idx.shape),
        "data_idx": list(batch.data_idx.shape),
    }
    report["shapes"] = {
        "expected": expected,
        "actual": actual,
        "passed": actual == expected,
    }
    report["dtypes"] = {
        "y_complex": bool(torch.is_complex(batch.y)),
        "h_complex": bool(torch.is_complex(batch.h)),
        "x_complex": bool(torch.is_complex(batch.x)),
        "bits_real": bool(not torch.is_complex(batch.data_bits)),
    }
    report["dtypes"]["passed"] = bool(all(report["dtypes"].values()))

    model = BayesRouteReceiver(
        cfg, simulator.coords, simulator.pilot_idx
    ).to(device)
    model.train()
    before = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
    }
    output = model(batch)
    bce = bit_bce_loss(output["bit_logits"], batch.data_bits)
    channel_nll_loss = channel_marginal_nll(
        output["posterior"].mean[..., batch.data_idx],
        output["posterior"].var_diag[:, batch.data_idx],
        batch.h[..., batch.data_idx],
    )
    loss = bce + float(cfg.training.channel_loss_weight) * channel_nll_loss
    loss.backward()

    gradient_details = {}
    gradient_norm_sq = 0.0
    all_finite = True
    any_nonzero = False
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            gradient_details[name] = {
                "present": False,
                "finite": False,
                "norm": 0.0,
            }
            all_finite = False
            continue
        finite = bool(torch.isfinite(parameter.grad).all().item())
        norm = float(torch.linalg.vector_norm(parameter.grad.detach()).item())
        gradient_details[name] = {
            "present": True,
            "finite": finite,
            "norm": norm,
        }
        gradient_norm_sq += norm * norm
        all_finite = all_finite and finite
        any_nonzero = any_nonzero or norm > 0.0
    report["gradient_flow"] = {
        "loss": float(loss.item()),
        "bce": float(bce.item()),
        "channel_marginal_nll_loss": float(channel_nll_loss.item()),
        "global_norm": math.sqrt(gradient_norm_sq),
        "all_finite": all_finite,
        "any_nonzero": any_nonzero,
        "per_parameter": gradient_details,
    }
    report["trainable_params"] = count_parameters(model)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(cfg.training.lr)
    )
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    parameter_change = {
        name: float(
            torch.max(torch.abs(parameter.detach() - before[name])).item()
        )
        for name, parameter in model.named_parameters()
    }
    report["optimizer_step"] = {
        "max_change_by_parameter": parameter_change,
        "max_change": (
            max(parameter_change.values()) if parameter_change else 0.0
        ),
    }
    report["optimizer_step"]["passed"] = bool(
        report["optimizer_step"]["max_change"] > 0.0
    )

    model.eval()
    with torch.no_grad():
        output_default = model(batch)
        output_mean = model(batch, use_uncertainty=False)
        output_uncertain = model(batch, use_uncertainty=True)
        output_sparse = model(batch, edge_mass=0.0)
        output_full = model(batch, edge_mass=1.0)

    latent_cov = output_default["posterior"].latent_cov
    eigenvalues = torch.linalg.eigvalsh(latent_cov).real
    report["posterior_projection"] = _posterior_projection_report(
        model, output_default["posterior"]
    )
    report["posterior_validity"] = {
        "min_latent_cov_eigenvalue": float(eigenvalues.min().item()),
        "max_latent_cov_hermitian_error": float(
            torch.max(
                torch.abs(latent_cov - latent_cov.conj().transpose(0, 1))
            ).item()
        ),
        "minimum_variance": float(
            output_default["posterior"].var_diag.min().item()
        ),
        "all_finite": bool(
            torch.isfinite(output_default["posterior"].mean).all().item()
            and torch.isfinite(output_default["posterior"].var_diag).all().item()
            and torch.isfinite(output_default["posterior"].local_cov).all().item()
            and torch.isfinite(latent_cov).all().item()
        ),
    }
    report["posterior_validity"]["passed"] = bool(
        report["posterior_validity"]["all_finite"]
        and report["posterior_validity"]["min_latent_cov_eigenvalue"] > -2e-4
        and report["posterior_validity"]["max_latent_cov_hermitian_error"] < 2e-5
        and report["posterior_validity"]["minimum_variance"] > 0.0
        and report["posterior_projection"]["passed"]
    )
    report["posterior_matched_calibration"] = (
        _matched_posterior_calibration_report(cfg, simulator, device)
    )
    report["detector_moment_formula"] = _detector_moment_formula_report(
        device
    )
    report["coupling_monte_carlo"] = _coupling_monte_carlo_report(device)

    kappa = output_default["kappa"]
    kappa_asymmetry = torch.max(
        torch.abs(kappa - kappa.transpose(-1, -2))
    ).item()
    diagonal = torch.diagonal(kappa, dim1=-2, dim2=-1)
    report["coupling"] = {
        "shape": list(kappa.shape),
        "finite": bool(torch.isfinite(kappa).all().item()),
        "nonnegative": bool((kappa >= -1e-7).all().item()),
        "max_asymmetry": float(kappa_asymmetry),
        "max_abs_diagonal": float(torch.max(torch.abs(diagonal)).item()),
        "mean": float(kappa.mean().item()),
        "max": float(kappa.max().item()),
    }
    report["coupling"]["passed"] = bool(
        report["coupling"]["finite"]
        and report["coupling"]["nonnegative"]
        and report["coupling"]["max_asymmetry"] < 2e-4
        and report["coupling"]["max_abs_diagonal"] < 1e-7
        and report["coupling_monte_carlo"]["passed"]
    )

    uncertainty_delta = float(
        torch.mean(
            torch.abs(
                output_uncertain["bit_logits"]
                - output_mean["bit_logits"]
            )
        ).item()
    )
    routing_delta = float(
        torch.mean(
            torch.abs(
                output_full["bit_logits"]
                - output_sparse["bit_logits"]
            )
        ).item()
    )
    report["ablation_activation"] = {
        "uncertainty_logit_mean_abs_delta": uncertainty_delta,
        "routing_logit_mean_abs_delta": routing_delta,
        "edge_density_mass_0": float(output_sparse["edge_density"]),
        "edge_density_mass_1": float(output_full["edge_density"]),
        "edge_density_default": float(output_default["edge_density"]),
    }
    report["ablation_activation"]["passed"] = bool(
        uncertainty_delta > 1e-7
        and routing_delta > 1e-7
        and report["ablation_activation"]["edge_density_mass_0"] == 0.0
        and report["ablation_activation"]["edge_density_mass_1"] > 0.99
        and 0.0 < report["ablation_activation"]["edge_density_default"] < 0.99
    )

    checkpoint_path = outdir / "_roundtrip_checkpoint.pt"
    torch.save({"model": model.state_dict()}, checkpoint_path)
    clone = BayesRouteReceiver(
        cfg, simulator.coords, simulator.pilot_idx
    ).to(device).eval()
    state = torch.load(
        checkpoint_path, map_location=device, weights_only=False
    )
    clone.load_state_dict(state["model"], strict=True)
    with torch.no_grad():
        clone_output = clone(batch)
    roundtrip_delta = _max_abs_difference(
        output_default["bit_logits"], clone_output["bit_logits"]
    )
    checkpoint_path.unlink(missing_ok=True)
    report["checkpoint_roundtrip"] = {
        "max_abs_logit_difference": roundtrip_delta,
        "passed": bool(roundtrip_delta < 1e-6),
    }

    report["post_step_metrics"] = bit_metrics(
        output_default["bit_logits"], batch.data_bits
    )
    report["post_step_channel_nmse"] = channel_nmse(
        output_default["posterior"].mean[..., batch.data_idx],
        batch.h[..., batch.data_idx],
    )
    report["post_step_channel_coverage95_mismatched"] = channel_coverage95(
        output_default["posterior"].mean[..., batch.data_idx],
        output_default["posterior"].var_diag[:, batch.data_idx],
        batch.h[..., batch.data_idx],
    )
    for name, receiver_class in [
        ("ls", LSReceiver),
        ("oracle", OracleReceiver),
    ]:
        baseline = receiver_class(cfg).to(device).eval()
        with torch.no_grad():
            baseline_output = baseline(batch)
        report[f"{name}_metrics"] = bit_metrics(
            baseline_output["bit_logits"], batch.data_bits
        )
    report["baseline_sanity"] = {
        "oracle_ber_not_worse_than_ls": bool(
            report["oracle_metrics"]["ber"]
            <= report["ls_metrics"]["ber"] + 0.02
        )
    }
    report["baseline_sanity"]["passed"] = bool(
        all(report["baseline_sanity"].values())
    )

    checks = {
        "package_revision_v2_4": bool(
            str(cfg.get("package_revision", "")) == EXPECTED_REVISION
        ),
        "package_manifest_valid": bool(report["manifest"]["passed"]),
        "optuna_exact_design_resume_workflow": bool(
            report["optuna_workflow"].get("passed", False)
        ),
        "cuda_compute_node": bool(
            device.type == "cuda" and torch.cuda.is_available()
        ),
        "sionna_mapper_demapper_executed": bool(
            report["sionna"].get("passed", False)
        ),
        "pilot_scope_explicit": bool(
            simulator.pilot_model_scope == PILOT_MODEL_SCOPE
        ),
        "pilot_orthogonality": bool(
            report["pilot_orthogonality"]["passed"]
        ),
        "pilot_noiseless_separation": bool(
            report["pilot_separation"]["passed"]
        ),
        "port_metadata_consistent": bool(
            report["port_metadata_check"]["passed"]
        ),
        "pilot_data_partition": bool(
            report["resource_partition"]["passed"]
        ),
        "simulator_operator_seed_separation": bool(
            report["seed_separation"]["passed"]
        ),
        "rff_rank_bank_nested": bool(
            report["rank_bank_nesting"]["passed"]
        ),
        "deterministic_repeat": bool(
            report["deterministic_repeat"]["passed"]
        ),
        "tensor_shapes_exact": bool(report["shapes"]["passed"]),
        "tensor_dtypes": bool(report["dtypes"]["passed"]),
        "loss_finite": bool(math.isfinite(report["gradient_flow"]["loss"])),
        "gradient_nonzero_finite": bool(all_finite and any_nonzero),
        "optimizer_updates_parameters": bool(
            report["optimizer_step"]["passed"]
        ),
        "posterior_projection_u_C_uH": bool(
            report["posterior_projection"]["passed"]
        ),
        "posterior_psd_and_finite": bool(
            report["posterior_validity"]["passed"]
        ),
        "posterior_matched_calibration": bool(
            report["posterior_matched_calibration"]["passed"]
        ),
        "detector_moment_formula": bool(
            report["detector_moment_formula"]["passed"]
        ),
        "coupling_closed_form_monte_carlo": bool(
            report["coupling_monte_carlo"]["passed"]
        ),
        "coupling_valid": bool(report["coupling"]["passed"]),
        "uncertainty_and_routing_paths_active": bool(
            report["ablation_activation"]["passed"]
        ),
        "checkpoint_roundtrip": bool(
            report["checkpoint_roundtrip"]["passed"]
        ),
        "baseline_sanity": bool(report["baseline_sanity"]["passed"]),
    }
    report["checks"] = checks
    report["overall_pass"] = bool(all(checks.values()))
    report["optuna_ready"] = bool(report["overall_pass"])
    save_json(report, outdir / "SMOKE_HEALTH.json")

    lines = [
        f"{key}: {'PASS' if value else 'FAIL'}"
        for key, value in checks.items()
    ]
    lines.append(
        f"OVERALL: {'PASS' if report['overall_pass'] else 'FAIL'}"
    )
    lines.append(
        f"OPTUNA_READY: {'YES' if report['optuna_ready'] else 'NO'}"
    )
    lines.append(
        "PUBLICATION_NR_READY: NO "
        "(a separate 3GPP/Sionna NR integration gate is required)"
    )
    text = "\n".join(lines) + "\n"
    (outdir / "SMOKE_HEALTH.txt").write_text(text, encoding="utf-8")
    print(text)
    if not report["overall_pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
