#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
from pathlib import Path
from typing import Any

import torch
import yaml
import sionna

ROOT = Path(__file__).resolve().parents[1]

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
    posterior_metrics,
    repaired_forward,
    unique_parameters,
)
from gate1_nr_posterior_factorial_common import (
    POSTERIOR_FACTORIAL_VERSION,
    FactorialCandidate,
    atomic_torch_save,
    bind_candidate_parameters,
    build_candidate_bridge,
    candidate_parameter_tensors,
    extract_candidate_state,
    load_candidate_state,
    ls_repaired_forward,
    ls_alignment_self_test,
    model_report,
    package_signature,
    pure_torch_multiscale_self_test,
    save_json,
    set_all_seeds,
    sha256_file,
)

GATE_JSON = ROOT / "outputs/gates/GATE1_NR_POSTERIOR_FACTORIAL_SMOKE.json"
GATE_TXT = ROOT / "outputs/gates/GATE1_NR_POSTERIOR_FACTORIAL_SMOKE.txt"
REQUIRED_GRID_REPORT = ROOT / "outputs/reports/gate1_nr_grid_scale_audit.json"
REQUIRED_GRID_CLASSIFICATION = "GRID_SCALE_COORDINATE_HYPOTHESIS_NOT_SUPPORTED"
REQUIRED_GRID_ROWS = 360
REQUIRED_CHECKPOINT_SHA256 = (
    "4f71c7a0a925005d676687e90c5a241668cfcfed21503e2874c3528721c66980"
)


def load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected YAML mapping: {path}")
    return value


def manifest_report() -> dict[str, Any]:
    manifest = ROOT / "GATE1_NR_POSTERIOR_FACTORIAL_MANIFEST.sha256"
    checked = 0
    mismatches: list[str] = []
    for raw in manifest.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        expected, relative = raw.split(None, 1)
        path = ROOT / relative.strip().lstrip("*").lstrip("./")
        if not path.is_file() or sha256_file(path) != expected:
            mismatches.append(str(path.relative_to(ROOT)))
        checked += 1
    return {"checked": checked, "mismatches": mismatches, "passed": not mismatches}


def preconditions() -> dict[str, Any]:
    if not REQUIRED_GRID_REPORT.is_file():
        raise RuntimeError("Missing completed grid-scale audit")
    report = json.loads(REQUIRED_GRID_REPORT.read_text(encoding="utf-8"))
    checkpoint = (
        ROOT
        / "outputs/gate1_nr_joint_operator/checkpoints/"
        "global_r24_cold_lf1_lt0p5/best.pt"
    )
    checks = {
        "grid_audit_complete": report.get("complete") is True,
        "grid_audit_classification": (
            report.get("classification") == REQUIRED_GRID_CLASSIFICATION
        ),
        "grid_audit_rows": report.get("evaluation", {}).get("rows")
        == REQUIRED_GRID_ROWS,
        "checkpoint_present": checkpoint.is_file(),
        "checkpoint_hash": checkpoint.is_file()
        and sha256_file(checkpoint) == REQUIRED_CHECKPOINT_SHA256,
    }
    return {"checks": checks, "passed": all(checks.values())}


def checkpoint_roundtrip(
    spec: FactorialCandidate,
    bridge: Any,
    context: Any,
    batch: Any,
    detector: Any,
) -> dict[str, Any]:
    before = repaired_forward(bridge, detector, batch)["bit_logits"].detach().clone()
    state = extract_candidate_state(spec, bridge)
    path = ROOT / "outputs/gate1_nr_posterior_factorial/smoke_roundtrip.pt"
    atomic_torch_save({"state": state, "version": POSTERIOR_FACTORIAL_VERSION}, path)
    reloaded = build_candidate_bridge(
        context.case,
        context,
        spec,
        operator_seed=58001,
    )
    load_candidate_state(spec, reloaded, torch.load(path, weights_only=False)["state"])
    after = repaired_forward(
        reloaded,
        make_repaired_detector(int(context.grid.bits_per_symbol)).to(context.device),
        batch,
    )["bit_logits"]
    error = float(torch.max(torch.abs(before - after)).item())
    return {"passed": error <= 2e-6, "max_abs_logit_error": error}


def run(config: dict[str, Any], device: torch.device) -> dict[str, Any]:
    manifest = manifest_report()
    pre = preconditions()
    pure = pure_torch_multiscale_self_test()
    ls_alignment_self = ls_alignment_self_test()
    spec = FactorialCandidate.from_mapping(config["candidate"])
    cases = [NRCase.from_mapping(item) for item in config["cases"]]
    contexts = [build_nr_context(case, device) for case in cases]
    bridges = [
        build_candidate_bridge(
            case,
            context,
            spec,
            operator_seed=int(config["operator_seed"]),
        )
        for case, context in zip(cases, contexts)
    ]
    bind_candidate_parameters(spec, bridges)
    detectors = [
        make_repaired_detector(int(context.grid.bits_per_symbol)).to(device)
        for context in contexts
    ]
    parameters = candidate_parameter_tensors(bridges)
    optimizer = torch.optim.Adam(parameters, lr=float(spec.learning_rate))
    per_case: list[dict[str, Any]] = []
    mixed_loss = torch.tensor(0.0, device=device)
    batches: list[Any] = []
    per_case_gradients: list[dict[str, Any]] = []

    for index, (case, context, bridge, detector) in enumerate(
        zip(cases, contexts, bridges, detectors)
    ):
        set_all_seeds(int(config["seed"]) + index)
        batch = context.sample(
            int(config["batch_size"]), float(config["ebno_db"])
        )
        batches.append(batch)
        output = repaired_forward(bridge, detector, batch)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            output["bit_logits"], batch.coded_bits.float()
        )
        case_gradients = torch.autograd.grad(
            loss,
            parameters,
            retain_graph=True,
            allow_unused=True,
        )
        case_gradient_norms = [
            float(item.detach().norm().item()) if item is not None else 0.0
            for item in case_gradients
        ]
        per_case_gradients.append(
            {
                "case": case.name,
                "all_present": all(item is not None for item in case_gradients),
                "all_finite": all(
                    item is not None and torch.isfinite(item).all().item()
                    for item in case_gradients
                ),
                "total_norm": float(
                    math.sqrt(sum(value * value for value in case_gradient_norms))
                ),
                "parameter_norms": case_gradient_norms,
            }
        )
        mixed_loss = mixed_loss + loss
        decoded = decode_bridge(
            context.transmitter,
            output,
            batch.information_bits,
            num_bp_iter=int(config["bp_iterations"]),
            device=device,
        )
        ls_receiver = standard_receiver(context, perfect_csi=False, return_crc=True)
        ls_output = ls_repaired_forward(ls_receiver, context, detector, batch)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        ls_decoded = decode_bridge(
            context.transmitter,
            ls_output,
            batch.information_bits,
            num_bp_iter=int(config["bp_iterations"]),
            device=device,
        )
        standard_ls = run_standard_receiver(
            ls_receiver,
            batch,
            batch.information_bits,
            perfect_csi=False,
        )
        per_case.append(
            {
                "case": case.name,
                "num_prb": int(case.num_prb),
                "loss": float(loss.detach().item()),
                "posterior": posterior_metrics(output, batch),
                "coded": coded_metrics(output, batch),
                "decoded_tbler": float(decoded["tbler"]),
                "ls_estimate_repaired_tbler": float(ls_decoded["tbler"]),
                "standard_ls_lmmse_tbler": float(standard_ls["tbler"]),
                "posterior_finite": bool(
                    torch.isfinite(output["posterior"].mean).all().item()
                    and torch.isfinite(output["posterior"].var_diag).all().item()
                    and torch.isfinite(output["bit_logits"]).all().item()
                ),
                "positive_variance": bool(
                    torch.all(output["posterior"].var_diag > 0).item()
                ),
                "ls_control_shape": list(ls_output["posterior"].mean.shape),
                "ls_grid_alignment": ls_output["ls_grid_alignment_report"],
                "ls_detector_local_data_indexing": bool(
                    ls_output["ls_detector_local_data_indexing"]
                ),
                "model": model_report(spec, bridge),
            }
        )

    optimizer.zero_grad(set_to_none=True)
    mixed_loss.backward()
    gradient_entries = []
    for index, parameter in enumerate(parameters):
        gradient = parameter.grad
        gradient_entries.append(
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
    before = [parameter.detach().clone() for parameter in parameters]
    optimizer.step()
    updated = [
        not torch.equal(old, parameter.detach())
        for old, parameter in zip(before, parameters)
    ]
    roundtrip = checkpoint_roundtrip(
        spec, bridges[0], contexts[0], batches[0], detectors[0]
    )

    shared_ids: dict[str, bool] = {}
    first = bridges[0].posterior
    second = bridges[1].posterior
    for name in (
        "raw_feature_weights",
        "raw_weights",
        "log_noise_scale",
        "raw_scale_bias",
        "context_to_scale",
    ):
        if hasattr(first, name) and getattr(first, name) is not None:
            shared_ids[name] = id(getattr(first, name)) == id(getattr(second, name))

    checks = {
        "manifest": manifest["passed"],
        "preconditions": pre["passed"],
        "pure_torch_multiscale": pure["passed"],
        "ls_alignment_self_test": ls_alignment_self["passed"],
        "cuda_compute_node": device.type == "cuda" and torch.cuda.is_available(),
        "sionna_2_0_1": getattr(sionna, "__version__", "") == "2.0.1",
        "selected_detector_contract": all(
            int(detector.n_iter) == 4
            and abs(float(detector.damping) - 0.7) < 1e-12
            and str(detector.covariance_mode) == "diagonal"
            for detector in detectors
        ),
        "shared_parameter_identity": bool(shared_ids and all(shared_ids.values())),
        "each_grid_has_gradient": all(
            item["all_present"]
            and item["all_finite"]
            and item["total_norm"] > 0.0
            for item in per_case_gradients
        ),
        "each_case_posterior_finite": all(item["posterior_finite"] for item in per_case),
        "each_case_positive_variance": all(item["positive_variance"] for item in per_case),
        "ls_estimator_factorization_path": all(
            len(item["ls_control_shape"]) == 4 for item in per_case
        ),
        "ls_estimator_effective_grid_alignment": all(
            item["ls_grid_alignment"].get("passed") is True
            and item["ls_grid_alignment"].get(
                "effective_grid_checked_before_fft"
            ) is True
            and item["ls_detector_local_data_indexing"] is True
            for item in per_case
        ),
        "mixed_case_gradients": all(
            item["present"] and item["finite"] and item["norm"] > 0.0
            for item in gradient_entries
        ),
        "optimizer_updates_all_parameters": all(updated),
        "nr_ldpc_paths": all(math.isfinite(item["decoded_tbler"]) for item in per_case),
        "checkpoint_roundtrip": roundtrip["passed"],
    }
    overall = all(checks.values())
    return {
        "version": POSTERIOR_FACTORIAL_VERSION,
        "classification": (
            "GATE1_NR_POSTERIOR_FACTORIAL_SMOKE_PASS"
            if overall
            else "GATE1_NR_POSTERIOR_FACTORIAL_SMOKE_FAIL"
        ),
        "overall_pass": overall,
        "screen_ready": overall,
        "publication_nr_ready": False,
        "environment": {
            "device": str(device),
            "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "torch": torch.__version__,
            "sionna": getattr(sionna, "__version__", "unknown"),
            "python": platform.python_version(),
        },
        "checks": checks,
        "manifest": manifest,
        "preconditions": pre,
        "pure_torch_self_test": pure,
        "ls_alignment_self_test": ls_alignment_self,
        "candidate": spec.as_dict(),
        "shared_parameter_identity": shared_ids,
        "per_grid_gradient_report": per_case_gradients,
        "gradient_report": gradient_entries,
        "optimizer_updated": updated,
        "checkpoint_roundtrip": roundtrip,
        "per_case": per_case,
        "contract_signature": package_signature(
            {"config": config, "manifest": manifest, "candidate": spec.as_dict()}
        ),
    }


def write_gate(report: dict[str, Any]) -> None:
    checks = report["checks"]
    lines = [
        *(f"{name}: {'PASS' if value else 'FAIL'}" for name, value in checks.items()),
        f"CLASSIFICATION: {report['classification']}",
        f"SCREEN_READY: {'YES' if report['screen_ready'] else 'NO'}",
        "PUBLICATION_NR_READY: NO",
    ]
    save_json(report, GATE_JSON)
    save_json(report, ROOT / "outputs/reports/gate1_nr_posterior_factorial_smoke.json")
    GATE_TXT.parent.mkdir(parents=True, exist_ok=True)
    GATE_TXT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/gate1_nr_posterior_factorial_smoke.yaml",
    )
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    config = load_yaml(ROOT / args.config)
    if config.get("revision") != POSTERIOR_FACTORIAL_VERSION:
        raise RuntimeError("Posterior-factorial smoke revision mismatch")
    device = normalize_device(args.device)
    report = run(config, device)
    write_gate(report)
    if not report["overall_pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
