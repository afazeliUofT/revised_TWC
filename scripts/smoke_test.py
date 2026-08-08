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

from bayesroute.config import load_config, set_seed, get_device, save_json, count_parameters
from bayesroute.simulator import UplinkToySimulator
from bayesroute.pilots import (
    PILOT_MODEL_SCOPE,
    pilot_orthogonality_report,
    pilot_separation_report,
    port_metadata_report,
    resource_partition_report,
)
from bayesroute.models import BayesRouteReceiver, LSReceiver, OracleReceiver
from bayesroute.losses import bit_bce_loss, bit_metrics, channel_nmse, channel_marginal_nll, channel_coverage95
from bayesroute.sionna_check import check_sionna


def _git_value(*args: str) -> str | None:
    try:
        p = subprocess.run(
            ["git", *args], cwd=ROOT, check=True, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        )
        return p.stdout.strip()
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
            failures.append({"path": rel, "reason": "hash_mismatch", "actual": actual})
    return {"passed": not failures and checked > 0, "checked_files": checked, "failures": failures}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/smoke.yaml")
    ap.add_argument("--out", default="outputs/smoke")
    args = ap.parse_args()
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
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "git_commit": _git_value("rev-parse", "HEAD"),
            "git_status_porcelain": _git_value("status", "--porcelain"),
        },
    }

    report["manifest"] = _verify_manifest(ROOT)
    report["sionna"] = check_sionna(device=device, bits_per_symbol=int(cfg.system.bits_per_symbol))
    sim = UplinkToySimulator(cfg, device)
    report["pilot_model_scope"] = sim.pilot_model_scope
    report["port_metadata"] = sim.port_meta
    report["pilot_orthogonality"] = pilot_orthogonality_report(sim.phi)
    report["pilot_separation"] = pilot_separation_report(sim.phi)
    report["port_metadata_check"] = port_metadata_report(sim.port_meta, sim.n_layers)
    report["resource_partition"] = resource_partition_report(
        sim.pilot_idx, sim.data_idx, sim.n_resource_elements
    )

    operator_seed = int(cfg.model.get("operator_seed", int(cfg.seed) + 1009))
    report["seed_separation"] = {
        "channel_seed": int(sim.channel_seed),
        "operator_seed": operator_seed,
        "passed": bool(int(sim.channel_seed) != operator_seed),
    }

    # Reproducibility check for one generated mini-batch.
    set_seed(int(cfg.seed) + 77)
    repeat_a = sim.sample(batch_size=2, snr_db=10.0)
    set_seed(int(cfg.seed) + 77)
    repeat_b = sim.sample(batch_size=2, snr_db=10.0)
    report["deterministic_repeat"] = {
        "max_y_difference": _max_abs_difference(repeat_a.y, repeat_b.y),
        "max_h_difference": _max_abs_difference(repeat_a.h, repeat_b.h),
        "bit_mismatch_count": int(torch.sum(repeat_a.data_bits != repeat_b.data_bits).item()),
    }
    report["deterministic_repeat"]["passed"] = bool(
        report["deterministic_repeat"]["max_y_difference"] == 0.0
        and report["deterministic_repeat"]["max_h_difference"] == 0.0
        and report["deterministic_repeat"]["bit_mismatch_count"] == 0
    )

    set_seed(int(cfg.seed))
    batch = sim.sample(batch_size=int(cfg.training.batch_size), snr_db=10.0)
    expected = {
        "y": [int(cfg.training.batch_size), sim.n_rx, sim.n_resource_elements],
        "h": [int(cfg.training.batch_size), sim.n_layers, sim.n_rx, sim.n_resource_elements],
        "x": [int(cfg.training.batch_size), sim.n_layers, sim.n_resource_elements],
        "data_bits": [
            int(cfg.training.batch_size), sim.n_layers,
            int(sim.data_idx.numel()), int(cfg.system.bits_per_symbol),
        ],
        "phi": [sim.n_layers, int(sim.pilot_idx.numel())],
        "pilot_idx": [int(sim.pilot_idx.numel())],
        "data_idx": [int(sim.data_idx.numel())],
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
    report["shapes"] = {"expected": expected, "actual": actual, "passed": actual == expected}
    report["dtypes"] = {
        "y_complex": bool(torch.is_complex(batch.y)),
        "h_complex": bool(torch.is_complex(batch.h)),
        "x_complex": bool(torch.is_complex(batch.x)),
        "bits_real": bool(not torch.is_complex(batch.data_bits)),
    }
    report["dtypes"]["passed"] = bool(all(report["dtypes"].values()))

    model = BayesRouteReceiver(cfg, sim.coords, sim.pilot_idx).to(device)
    model.train()
    before = {name: p.detach().clone() for name, p in model.named_parameters()}
    out = model(batch)
    bce = bit_bce_loss(out["bit_logits"], batch.data_bits)
    channel_nll_loss = channel_marginal_nll(
        out["posterior"].mean[..., batch.data_idx],
        out["posterior"].var_diag[:, batch.data_idx],
        batch.h[..., batch.data_idx],
    )
    loss = bce + float(cfg.training.channel_loss_weight) * channel_nll_loss
    loss.backward()

    grad_details = {}
    grad_norm_sq = 0.0
    all_finite = True
    any_nonzero = False
    for name, p in model.named_parameters():
        if p.grad is None:
            grad_details[name] = {"present": False, "finite": False, "norm": 0.0}
            all_finite = False
            continue
        finite = bool(torch.isfinite(p.grad).all().item())
        norm = float(torch.linalg.vector_norm(p.grad.detach()).item())
        grad_details[name] = {"present": True, "finite": finite, "norm": norm}
        grad_norm_sq += norm * norm
        all_finite = all_finite and finite
        any_nonzero = any_nonzero or norm > 0.0
    report["gradient_flow"] = {
        "loss": float(loss.item()),
        "bce": float(bce.item()),
        "channel_marginal_nll_loss": float(channel_nll_loss.item()),
        "global_norm": math.sqrt(grad_norm_sq),
        "all_finite": all_finite,
        "any_nonzero": any_nonzero,
        "per_parameter": grad_details,
    }
    report["trainable_params"] = count_parameters(model)

    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg.training.lr))
    opt.step()
    opt.zero_grad(set_to_none=True)
    param_change = {
        name: float(torch.max(torch.abs(p.detach() - before[name])).item())
        for name, p in model.named_parameters()
    }
    report["optimizer_step"] = {
        "max_change_by_parameter": param_change,
        "max_change": max(param_change.values()) if param_change else 0.0,
    }
    report["optimizer_step"]["passed"] = bool(report["optimizer_step"]["max_change"] > 0.0)

    model.eval()
    with torch.no_grad():
        out_default = model(batch)
        out_mean = model(batch, use_uncertainty=False)
        out_unc = model(batch, use_uncertainty=True)
        out_sparse = model(batch, edge_mass=0.0)
        out_full = model(batch, edge_mass=1.0)

    latent_cov = out_default["posterior"].latent_cov
    eigenvalues = torch.linalg.eigvalsh(latent_cov).real
    report["posterior_validity"] = {
        "min_latent_cov_eigenvalue": float(eigenvalues.min().item()),
        "max_latent_cov_hermitian_error": float(
            torch.max(torch.abs(latent_cov - latent_cov.conj().T)).item()
        ),
        "min_variance": float(out_default["posterior"].var_diag.min().item()),
        "all_finite": bool(
            torch.isfinite(out_default["posterior"].mean).all().item()
            and torch.isfinite(out_default["posterior"].var_diag).all().item()
            and torch.isfinite(latent_cov).all().item()
        ),
    }
    report["posterior_validity"]["passed"] = bool(
        report["posterior_validity"]["all_finite"]
        and report["posterior_validity"]["min_latent_cov_eigenvalue"] > -2e-4
        and report["posterior_validity"]["max_latent_cov_hermitian_error"] < 2e-5
        and report["posterior_validity"]["min_variance"] > 0.0
    )

    kappa = out_default["kappa"]
    kappa_asym = torch.max(torch.abs(kappa - kappa.transpose(-1, -2))).item()
    diagonal = torch.diagonal(kappa, dim1=-2, dim2=-1)
    report["coupling"] = {
        "shape": list(kappa.shape),
        "finite": bool(torch.isfinite(kappa).all().item()),
        "nonnegative": bool((kappa >= -1e-7).all().item()),
        "max_asymmetry": float(kappa_asym),
        "max_abs_diagonal": float(torch.max(torch.abs(diagonal)).item()),
        "mean": float(kappa.mean().item()),
        "max": float(kappa.max().item()),
    }
    report["coupling"]["passed"] = bool(
        report["coupling"]["finite"]
        and report["coupling"]["nonnegative"]
        and report["coupling"]["max_asymmetry"] < 2e-4
        and report["coupling"]["max_abs_diagonal"] < 1e-7
    )

    uncertainty_delta = float(
        torch.mean(torch.abs(out_unc["bit_logits"] - out_mean["bit_logits"])).item()
    )
    routing_delta = float(
        torch.mean(torch.abs(out_full["bit_logits"] - out_sparse["bit_logits"])).item()
    )
    report["ablation_activation"] = {
        "uncertainty_logit_mean_abs_delta": uncertainty_delta,
        "routing_logit_mean_abs_delta": routing_delta,
        "edge_density_mass_0": float(out_sparse["edge_density"]),
        "edge_density_mass_1": float(out_full["edge_density"]),
        "edge_density_default": float(out_default["edge_density"]),
    }
    report["ablation_activation"]["passed"] = bool(
        uncertainty_delta > 1e-7
        and routing_delta > 1e-7
        and report["ablation_activation"]["edge_density_mass_0"] == 0.0
        and report["ablation_activation"]["edge_density_mass_1"] > 0.99
        and 0.0 < report["ablation_activation"]["edge_density_default"] < 0.99
    )

    # Checkpoint round trip: this also verifies architecture/config consistency.
    ckpt_path = outdir / "_roundtrip_checkpoint.pt"
    torch.save({"model": model.state_dict()}, ckpt_path)
    clone = BayesRouteReceiver(cfg, sim.coords, sim.pilot_idx).to(device).eval()
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    clone.load_state_dict(state["model"], strict=True)
    with torch.no_grad():
        clone_out = clone(batch)
    roundtrip_delta = _max_abs_difference(
        out_default["bit_logits"], clone_out["bit_logits"]
    )
    ckpt_path.unlink(missing_ok=True)
    report["checkpoint_roundtrip"] = {
        "max_abs_logit_difference": roundtrip_delta,
        "passed": bool(roundtrip_delta < 1e-6),
    }

    report["post_step_metrics"] = bit_metrics(out_default["bit_logits"], batch.data_bits)
    report["post_step_channel_nmse"] = channel_nmse(
        out_default["posterior"].mean[..., batch.data_idx], batch.h[..., batch.data_idx]
    )
    report["post_step_channel_coverage95"] = channel_coverage95(
        out_default["posterior"].mean[..., batch.data_idx],
        out_default["posterior"].var_diag[:, batch.data_idx],
        batch.h[..., batch.data_idx],
    )
    for name, cls in [("ls", LSReceiver), ("oracle", OracleReceiver)]:
        baseline = cls(cfg).to(device).eval()
        with torch.no_grad():
            baseline_out = baseline(batch)
        report[f"{name}_metrics"] = bit_metrics(
            baseline_out["bit_logits"], batch.data_bits
        )
    report["baseline_sanity"] = {
        "oracle_ber_not_worse_than_ls": bool(
            report["oracle_metrics"]["ber"] <= report["ls_metrics"]["ber"] + 0.02
        )
    }
    report["baseline_sanity"]["passed"] = bool(
        all(report["baseline_sanity"].values())
    )

    checks = {
        "package_revision_v2": bool(str(cfg.get("package_revision", "")).startswith("gate0_v2_")),
        "package_manifest_valid": bool(report["manifest"]["passed"]),
        "cuda_compute_node": bool(device.type == "cuda" and torch.cuda.is_available()),
        "sionna_mapper_demapper_executed": bool(report["sionna"].get("passed", False)),
        "pilot_scope_explicit": bool(sim.pilot_model_scope == PILOT_MODEL_SCOPE),
        "pilot_orthogonality": bool(report["pilot_orthogonality"]["passed"]),
        "pilot_noiseless_separation": bool(report["pilot_separation"]["passed"]),
        "port_metadata_consistent": bool(report["port_metadata_check"]["passed"]),
        "pilot_data_partition": bool(report["resource_partition"]["passed"]),
        "simulator_operator_seed_separation": bool(report["seed_separation"]["passed"]),
        "deterministic_repeat": bool(report["deterministic_repeat"]["passed"]),
        "tensor_shapes_exact": bool(report["shapes"]["passed"]),
        "tensor_dtypes": bool(report["dtypes"]["passed"]),
        "loss_finite": bool(math.isfinite(report["gradient_flow"]["loss"])),
        "gradient_nonzero_finite": bool(all_finite and any_nonzero),
        "optimizer_updates_parameters": bool(report["optimizer_step"]["passed"]),
        "posterior_psd_and_finite": bool(report["posterior_validity"]["passed"]),
        "posterior_coverage_metric_valid": bool(
            math.isfinite(report["post_step_channel_coverage95"])
            and 0.0 <= report["post_step_channel_coverage95"] <= 1.0
        ),
        "coupling_valid": bool(report["coupling"]["passed"]),
        "uncertainty_and_routing_paths_active": bool(report["ablation_activation"]["passed"]),
        "checkpoint_roundtrip": bool(report["checkpoint_roundtrip"]["passed"]),
        "baseline_sanity": bool(report["baseline_sanity"]["passed"]),
    }
    report["checks"] = checks
    report["overall_pass"] = bool(all(checks.values()))
    report["optuna_ready"] = bool(report["overall_pass"])
    save_json(report, outdir / "SMOKE_HEALTH.json")

    lines = [f"{key}: {'PASS' if value else 'FAIL'}" for key, value in checks.items()]
    lines.append(f"OVERALL: {'PASS' if report['overall_pass'] else 'FAIL'}")
    lines.append(f"OPTUNA_READY: {'YES' if report['optuna_ready'] else 'NO'}")
    lines.append("PUBLICATION_NR_READY: NO (a separate 3GPP/Sionna NR integration gate is required)")
    text = "\n".join(lines) + "\n"
    (outdir / "SMOKE_HEALTH.txt").write_text(text, encoding="utf-8")
    print(text)
    if not report["overall_pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
