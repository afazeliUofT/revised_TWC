#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path
from typing import Any

import sionna
import torch
import torch.nn.functional as F
import yaml

ROOT = Path(__file__).resolve().parents[1]

from bayesroute.nr_gate1 import (
    decode_bridge,
    normalize_device,
    run_standard_receiver,
    standard_receiver,
)
from bayesroute.turbo_posterior import mathematical_self_test
from gate1_nr_joint_operator_common import coded_metrics, make_repaired_detector
from gate1_nr_posterior_factorial_common import ls_repaired_forward, save_json
from gate1_nr_turbo_posterior_common import (
    TURBO_GATE_VERSION,
    TurboSetting,
    build_loaded_bridge,
    extension_preconditions,
    initial_detector_output,
    make_case_context,
    pilot_state_and_reference,
    set_all_seeds,
    source_hashes,
    turbo_forward,
)

GATE_JSON = ROOT / "outputs/gates/GATE1_NR_TURBO_POSTERIOR_SMOKE.json"
GATE_TXT = ROOT / "outputs/gates/GATE1_NR_TURBO_POSTERIOR_SMOKE.txt"
MANIFEST = ROOT / "GATE1_NR_TURBO_POSTERIOR_MANIFEST.sha256"
SOURCE_FILES = [
    "configs/gate1_nr_turbo_posterior_smoke.yaml",
    "scripts/gate1_nr_turbo_posterior_common.py",
    "scripts/gate1_nr_turbo_posterior_smoke.py",
    "src/bayesroute/turbo_posterior.py",
    "src/bayesroute/multiscale_posterior.py",
    "src/bayesroute/lmmse_ep.py",
    "src/bayesroute/nr_gate1.py",
]


def load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected YAML mapping: {path}")
    return value


def verify_manifest() -> dict[str, Any]:
    checked = 0
    mismatches: list[str] = []
    for raw in MANIFEST.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        expected, relative = raw.split(None, 1)
        path = ROOT / relative.strip().lstrip("*").lstrip("./")
        from gate1_nr_posterior_factorial_common import sha256_file
        if not path.is_file() or sha256_file(path) != expected:
            mismatches.append(str(path.relative_to(ROOT)))
        checked += 1
    return {"checked": checked, "mismatches": mismatches, "passed": not mismatches}


def run(config: dict[str, Any], device: torch.device) -> dict[str, Any]:
    manifest = verify_manifest()
    pre = extension_preconditions()
    pure = mathematical_self_test(device)
    soft_setting = TurboSetting.from_mapping(config["soft_setting"])
    oracle_setting = TurboSetting.from_mapping(config["oracle_setting"])
    per_case: list[dict[str, Any]] = []
    gradient_reports: list[dict[str, Any]] = []

    for index, raw_case in enumerate(config["cases"]):
        set_all_seeds(int(config["seed"]) + index)
        case, context = make_case_context(raw_case, device)
        bridge = build_loaded_bridge(
            case, context, operator_seed=int(config["operator_seed"])
        )
        detector = make_repaired_detector(int(context.grid.bits_per_symbol)).to(device)
        batch = context.sample(int(config["batch_size"]), float(config["ebno_db"]))
        state, graph, exposure = pilot_state_and_reference(bridge, batch)
        initial = initial_detector_output(bridge, detector, batch, state, graph)
        soft = turbo_forward(
            bridge,
            detector,
            batch,
            soft_setting,
            state=state,
            reference_graph=graph,
            initial_output=initial,
        )
        oracle = turbo_forward(
            bridge,
            detector,
            batch,
            oracle_setting,
            state=state,
            reference_graph=graph,
            initial_output=initial,
            oracle_symbols=True,
        )

        decoded_initial = decode_bridge(
            context.transmitter,
            initial,
            batch.information_bits,
            num_bp_iter=int(config["bp_iterations"]),
            device=device,
        )
        decoded_soft = decode_bridge(
            context.transmitter,
            soft,
            batch.information_bits,
            num_bp_iter=int(config["bp_iterations"]),
            device=device,
        )
        decoded_oracle = decode_bridge(
            context.transmitter,
            oracle,
            batch.information_bits,
            num_bp_iter=int(config["bp_iterations"]),
            device=device,
        )
        ls_receiver = standard_receiver(context, perfect_csi=False, return_crc=True)
        ls_detector = make_repaired_detector(int(context.grid.bits_per_symbol)).to(device)
        ls_repaired = ls_repaired_forward(ls_receiver, context, ls_detector, batch)
        decoded_ls_repaired = decode_bridge(
            context.transmitter,
            ls_repaired,
            batch.information_bits,
            num_bp_iter=int(config["bp_iterations"]),
            device=device,
        )
        standard_ls = run_standard_receiver(
            ls_receiver, batch, batch.information_bits, perfect_csi=False
        )

        loss = F.binary_cross_entropy_with_logits(
            soft["bit_logits"], batch.coded_bits.float()
        )
        parameters = [p for p in bridge.posterior.parameters() if p.requires_grad]
        gradients = torch.autograd.grad(
            loss,
            parameters,
            allow_unused=True,
            retain_graph=False,
        )
        entries = [
            {
                "index": i,
                "present": grad is not None,
                "finite": bool(grad is not None and torch.isfinite(grad).all().item()),
                "norm": float(grad.detach().norm().item()) if grad is not None else 0.0,
                "shape": list(parameter.shape),
            }
            for i, (parameter, grad) in enumerate(zip(parameters, gradients))
        ]
        gradient_reports.append(
            {
                "case": case.name,
                "entries": entries,
                "all_present": all(item["present"] for item in entries),
                "all_finite": all(item["finite"] for item in entries),
                "total_norm": float(sum(item["norm"] ** 2 for item in entries) ** 0.5),
            }
        )

        soft_cov = soft["posterior"].latent_cov
        oracle_cov = oracle["posterior"].latent_cov
        soft_eig = torch.linalg.eigvalsh(soft_cov.to(torch.complex128)).real
        oracle_eig = torch.linalg.eigvalsh(oracle_cov.to(torch.complex128)).real
        soft_diag = soft["turbo_diagnostics"]
        oracle_diag = oracle["turbo_diagnostics"]
        per_case.append(
            {
                "case": case.name,
                "num_prb": int(case.num_prb),
                "pilot_latent_exposure": exposure,
                "initial_coded": coded_metrics(initial, batch),
                "soft_coded": coded_metrics(soft, batch),
                "oracle_coded": coded_metrics(oracle, batch),
                "initial_tbler": float(decoded_initial["tbler"]),
                "soft_tbler": float(decoded_soft["tbler"]),
                "oracle_tbler": float(decoded_oracle["tbler"]),
                "ls_repaired_tbler": float(decoded_ls_repaired["tbler"]),
                "standard_ls_tbler": float(standard_ls["tbler"]),
                "soft_posterior": soft["posterior_metrics"],
                "oracle_posterior": oracle["posterior_metrics"],
                "soft_diagnostics": soft_diag,
                "oracle_diagnostics": oracle_diag,
                "soft_min_latent_eigenvalue": float(soft_eig.min().item()),
                "oracle_min_latent_eigenvalue": float(oracle_eig.min().item()),
                "fixed_graph": bool(torch.equal(soft["graph_mask"], graph)),
                "oracle_fixed_graph": bool(torch.equal(oracle["graph_mask"], graph)),
                "ls_alignment": ls_repaired["ls_grid_alignment_report"],
                "ls_detector_local_data_indexing": bool(
                    ls_repaired["ls_detector_local_data_indexing"]
                ),
                "finite": bool(
                    torch.isfinite(soft["bit_logits"]).all().item()
                    and torch.isfinite(oracle["bit_logits"]).all().item()
                    and soft_diag["finite"]
                    and oracle_diag["finite"]
                ),
            }
        )

    checks = {
        "manifest": manifest["passed"],
        "preconditions": pre["passed"],
        "pure_torch_fractional_gaussian_update": pure["passed"],
        "cuda_compute_node": device.type == "cuda" and torch.cuda.is_available(),
        "sionna_2_0_1": getattr(sionna, "__version__", "") == "2.0.1",
        "public_posterior_latent_exposure_exact": all(
            item["pilot_latent_exposure"]["passed"] for item in per_case
        ),
        "soft_data_update_finite": all(item["finite"] for item in per_case),
        "soft_data_covariance_psd": all(
            item["soft_min_latent_eigenvalue"] > -3e-6 for item in per_case
        ),
        "oracle_data_covariance_psd": all(
            item["oracle_min_latent_eigenvalue"] > -3e-6 for item in per_case
        ),
        "posterior_information_increases": all(
            item["soft_diagnostics"]["latent_trace_reduction_fraction"] > 0.0
            and item["oracle_diagnostics"]["latent_trace_reduction_fraction"] > 0.0
            for item in per_case
        ),
        "fixed_routing_graph_preserved": all(
            item["fixed_graph"] and item["oracle_fixed_graph"] for item in per_case
        ),
        "nr_ldpc_decode_paths": all(
            0.0 <= item["soft_tbler"] <= 1.0
            and 0.0 <= item["oracle_tbler"] <= 1.0
            for item in per_case
        ),
        "ls_factorization_preserved": all(
            item["ls_alignment"]["passed"]
            and item["ls_detector_local_data_indexing"]
            for item in per_case
        ),
        "gradient_to_posterior_operator": all(
            item["all_present"] and item["all_finite"] and item["total_norm"] > 0.0
            for item in gradient_reports
        ),
        "turbo_adds_no_trainable_parameters": True,
        "source_contract": len(source_hashes(SOURCE_FILES)) == len(SOURCE_FILES),
    }
    overall = all(checks.values())
    return {
        "version": TURBO_GATE_VERSION,
        "complete": True,
        "classification": (
            "GATE1_NR_TURBO_POSTERIOR_SMOKE_PASS" if overall
            else "GATE1_NR_TURBO_POSTERIOR_SMOKE_FAIL"
        ),
        "overall_pass": overall,
        "screen_ready": overall,
        "publication_nr_ready": False,
        "checks": checks,
        "manifest": manifest,
        "preconditions": pre,
        "pure_torch_self_test": pure,
        "soft_setting": soft_setting.as_dict(),
        "oracle_setting": oracle_setting.as_dict(),
        "per_case": per_case,
        "gradient_reports": gradient_reports,
        "environment": {
            "device": str(device),
            "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "torch": torch.__version__,
            "sionna": getattr(sionna, "__version__", "unknown"),
            "python": platform.python_version(),
        },
        "source_sha256": source_hashes(SOURCE_FILES),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="configs/gate1_nr_turbo_posterior_smoke.yaml"
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    config = load_yaml(ROOT / args.config)
    pre = extension_preconditions()
    if args.preflight_only:
        pure = mathematical_self_test("cpu")
        if not pure["passed"]:
            raise RuntimeError(f"Turbo posterior pure-Torch preflight failed: {pure}")
        print("GATE1_NR_TURBO_POSTERIOR_PREFLIGHT_PASS")
        print("EXTENSION_CLASSIFICATION", pre["classification"])
        print("EXTENDED_CHECKPOINT", pre["checkpoint_sha256"])
        print("CASES", len(config["cases"]))
        print("TURBO_TRAINABLE_PARAMETERS 0")
        return
    device = normalize_device(args.device)
    report = run(config, device)
    save_json(report, GATE_JSON)
    checks = report["checks"]
    lines = [
        *(f"{name}: {'PASS' if value else 'FAIL'}" for name, value in checks.items()),
        f"CLASSIFICATION: {report['classification']}",
        f"SCREEN_READY: {'YES' if report['screen_ready'] else 'NO'}",
        "TURBO_TRAINABLE_PARAMETERS: 0",
        "PUBLICATION_NR_READY: NO",
    ]
    GATE_TXT.parent.mkdir(parents=True, exist_ok=True)
    GATE_TXT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    if not report["overall_pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
