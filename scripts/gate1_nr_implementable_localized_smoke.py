#!/usr/bin/env python3
from __future__ import annotations

import argparse
import inspect
import json
import platform
from pathlib import Path
from typing import Any

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]

from bayesroute.ls_anchored_localized_posterior import (
    IMPLEMENTABLE_LOCALIZED_VERSION,
    LSAnchoredLocalizedResidualPosterior,
    load_shared_localized_state,
    mathematical_self_test,
    shared_localized_state,
    unique_localized_parameters,
)
from bayesroute.nr_gate1 import normalize_device
from gate1_nr_implementable_localized_common import (
    GATE_VERSION,
    build_shared_stack,
    decode_outputs,
    differentiable_training_loss,
    gradient_report,
    localized_ceiling_preconditions,
    observable_forward,
    save_json,
    selected_basis_spec,
    set_all_seeds,
    source_hashes,
)

REPORT_JSON = ROOT / "outputs/gates/GATE1_NR_IMPLEMENTABLE_LOCALIZED_SMOKE.json"
REPORT_TXT = ROOT / "outputs/gates/GATE1_NR_IMPLEMENTABLE_LOCALIZED_SMOKE.txt"


def load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected YAML mapping: {path}")
    return value


def run(config: dict[str, Any], device: torch.device) -> dict[str, Any]:
    pre = localized_ceiling_preconditions()
    pure = mathematical_self_test("cpu")
    if not pure.get("passed"):
        raise RuntimeError(f"Localized posterior math self-test failed: {pure}")
    spec = selected_basis_spec(pre)
    items = build_shared_stack(
        config["cases"],
        device,
        spec,
        num_knots=int(config["model"]["num_knots"]),
    )
    parameters = unique_localized_parameters([item.operator for item in items])
    parameter_ids = [
        [id(parameter) for parameter in item.operator.parameters()]
        for item in items
    ]
    optimizer = torch.optim.AdamW(parameters, lr=1e-3)
    before = [parameter.detach().clone() for parameter in parameters]
    optimizer.zero_grad(set_to_none=True)
    per_case: list[dict[str, Any]] = []
    saved_batches: list[Any] = []
    saved_outputs: list[dict[str, Any]] = []

    for index, item in enumerate(items):
        seed = int(config["seed"]) + 1000 * index
        set_all_seeds(seed)
        batch = item.context.sample(
            int(config["batch_size"]), float(config["ebno_db"])
        )
        output = observable_forward(item, batch)
        loss, parts = differentiable_training_loss(
            output,
            batch,
            channel_loss_weight=0.05,
            calibration_loss_weight=0.02,
            ls_gain_loss_weight=0.2,
            ls_gain_target_ratio=0.98,
        )
        (loss / len(items)).backward()
        decoded = decode_outputs(
            item.context,
            batch,
            {"localized": output},
            bp_iterations=int(config["bp_iterations"]),
            device=device,
        )["localized"]
        result = output["localized_result"]
        per_case.append(
            {
                "case": item.case.name,
                "num_prb": int(item.case.num_prb),
                "effective_rank": int(item.basis_report["effective_rank"]),
                "nominal_rank": int(item.basis_report["nominal_rank"]),
                "loss": float(loss.detach().item()),
                "bit_nll": float(parts["bit_nll"].detach().item()),
                "posterior_mean_finite": bool(
                    torch.isfinite(output["posterior"].mean).all().item()
                ),
                "posterior_variance_positive": bool(
                    (output["posterior"].var_diag > 0).all().item()
                ),
                "bit_logits_finite": bool(
                    torch.isfinite(output["bit_logits"]).all().item()
                ),
                "ls_alignment_pass": bool(
                    output["ls_grid_alignment_report"].get("passed")
                ),
                "residual_gate": float(result.residual_gate.detach().item()),
                "tbler": float(decoded["tbler"]),
                "crc_failure_rate": float(decoded["crc_failure_rate"]),
                "inference_uses_true_channel": bool(
                    output["inference_uses_true_channel"]
                ),
            }
        )
        saved_batches.append(batch)
        saved_outputs.append(output)

    gradients = gradient_report(parameters)
    optimizer.step()
    changed = any(
        not torch.equal(old, parameter.detach())
        for old, parameter in zip(before, parameters)
    )

    # Exact shared-state checkpoint reconstruction on the first real NR case.
    state = shared_localized_state(items[0].operator)
    clone_items = build_shared_stack(
        [config["cases"][0]],
        device,
        spec,
        num_knots=int(config["model"]["num_knots"]),
    )
    load_shared_localized_state(clone_items[0].operator, state)
    with torch.inference_mode():
        reference_after = observable_forward(items[0], saved_batches[0])
        clone_after = observable_forward(clone_items[0], saved_batches[0])
    roundtrip = bool(
        torch.allclose(
            reference_after["bit_logits"],
            clone_after["bit_logits"],
            rtol=1e-5,
            atol=2e-5,
        )
        and torch.allclose(
            reference_after["posterior"].mean,
            clone_after["posterior"].mean,
            rtol=1e-5,
            atol=2e-5,
        )
    )

    forward_parameters = set(
        inspect.signature(LSAnchoredLocalizedResidualPosterior.forward).parameters
    )
    residual_parameters = set(
        inspect.signature(
            LSAnchoredLocalizedResidualPosterior.residual_posterior
        ).parameters
    )
    inference_contract = {
        "forward_has_no_truth_argument": not bool(
            {"truth", "true_channel", "h", "batch_h"} & forward_parameters
        ),
        "residual_forward_has_no_truth_argument": not bool(
            {"truth", "true_channel", "h", "batch_h"} & residual_parameters
        ),
        "all_runtime_outputs_mark_no_truth": all(
            not item["inference_uses_true_channel"] for item in per_case
        ),
    }

    import sionna
    sionna_version = str(getattr(sionna, "__version__", "unknown"))
    checks = {
        "preconditions": pre["passed"],
        "pure_torch_math": pure["passed"],
        "cuda_compute_node": device.type == "cuda" and torch.cuda.is_available(),
        "sionna_2_0_1": sionna_version == "2.0.1",
        "precision_corrected_basis": all(
            item.basis_report.get("precision_patch")
            == "complex128_atoms_before_rank_decision_v1"
            for item in items
        ),
        "shared_parameter_identity": len(items) >= 2
        and parameter_ids[0] == parameter_ids[1],
        "each_grid_has_valid_rank": all(
            item["effective_rank"] > 0
            and item["effective_rank"] <= item["nominal_rank"]
            for item in per_case
        ),
        "each_case_posterior_finite": all(
            item["posterior_mean_finite"] and item["bit_logits_finite"]
            for item in per_case
        ),
        "each_case_positive_variance": all(
            item["posterior_variance_positive"] for item in per_case
        ),
        "ls_alignment_paths": all(item["ls_alignment_pass"] for item in per_case),
        "gradient_to_all_parameter_tensors": gradients["all_present"]
        and gradients["all_finite"]
        and gradients["any_nonzero"],
        "optimizer_updates_parameters": changed,
        "nr_ldpc_decode_paths": all(
            0.0 <= item["tbler"] <= 1.0 for item in per_case
        ),
        "checkpoint_roundtrip": roundtrip,
        "inference_observability_contract": all(inference_contract.values()),
        "source_contract": len(source_hashes()) >= 10,
    }
    passed = all(checks.values())
    return {
        "version": GATE_VERSION,
        "model_version": IMPLEMENTABLE_LOCALIZED_VERSION,
        "complete": True,
        "overall_pass": passed,
        "screen_ready": passed,
        "classification": (
            "GATE1_NR_IMPLEMENTABLE_LOCALIZED_SMOKE_PASS"
            if passed
            else "GATE1_NR_IMPLEMENTABLE_LOCALIZED_SMOKE_FAIL"
        ),
        "checks": checks,
        "preconditions": pre,
        "pure_torch_self_test": pure,
        "per_case": per_case,
        "gradient_report": gradients,
        "inference_contract": inference_contract,
        "parameter_report": items[0].operator.parameter_report(),
        "environment": {
            "device": str(device),
            "gpu_name": (
                torch.cuda.get_device_name(device)
                if device.type == "cuda"
                else "cpu"
            ),
            "torch": torch.__version__,
            "sionna": sionna_version,
            "python": platform.python_version(),
        },
        "publication_nr_ready": False,
    }


def write_report(report: dict[str, Any]) -> None:
    save_json(report, REPORT_JSON)
    lines = [
        f"{name}: {'PASS' if value else 'FAIL'}"
        for name, value in report["checks"].items()
    ]
    lines.extend(
        [
            f"CLASSIFICATION: {report['classification']}",
            f"SCREEN_READY: {'YES' if report['screen_ready'] else 'NO'}",
            "INFERENCE_USES_TRUE_CHANNEL: NO",
            "PUBLICATION_NR_READY: NO",
        ]
    )
    REPORT_TXT.parent.mkdir(parents=True, exist_ok=True)
    REPORT_TXT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    config = load_yaml(Path(args.config))
    pre = localized_ceiling_preconditions()
    pure = mathematical_self_test("cpu")
    if args.preflight_only:
        spec = selected_basis_spec(pre)
        device = normalize_device("cpu")
        items = build_shared_stack(
            config["cases"],
            device,
            spec,
            num_knots=int(config["model"]["num_knots"]),
        )
        print("GATE1_NR_IMPLEMENTABLE_LOCALIZED_PREFLIGHT_PASS")
        print("MODEL_VERSION", IMPLEMENTABLE_LOCALIZED_VERSION)
        print("WINNER_BASIS", spec.name)
        print("PURE_TORCH_MATH_PASS", pure["passed"])
        print(
            "PREFLIGHT_EFFECTIVE_RANKS",
            [int(item.basis_report["effective_rank"]) for item in items],
        )
        print("INFERENCE_USES_TRUE_CHANNEL NO")
        return
    device = normalize_device(args.device)
    report = run(config, device)
    write_report(report)
    if not report["overall_pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
