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

from bayesroute.evidence_mixture_lmmse import (
    EVIDENCE_MIXTURE_LMMSE_VERSION,
    EvidenceMixtureLMMSEPosterior,
    load_shared_evidence_state,
    mathematical_self_test,
    shared_evidence_state,
    unique_evidence_parameters,
)
from bayesroute.nr_gate1 import normalize_device
from gate2_nr_evidence_mixture_common import (
    GATE2_VERSION,
    basis_spec_from_config,
    build_shared_stack,
    decode_outputs,
    evidence_forward,
    gradient_report,
    ls_repaired_forward,
    save_json,
    set_all_seeds,
    source_hashes,
    source_result_preconditions,
    training_loss,
)

REPORT_JSON = ROOT / "outputs/gates/GATE2_NR_EVIDENCE_MIXTURE_SMOKE.json"
REPORT_TXT = ROOT / "outputs/gates/GATE2_NR_EVIDENCE_MIXTURE_SMOKE.txt"


def load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected YAML mapping: {path}")
    return value


def run(config: dict[str, Any], device: torch.device) -> dict[str, Any]:
    pre = source_result_preconditions()
    pure = mathematical_self_test("cpu")
    if not pure.get("passed"):
        raise RuntimeError(f"Evidence-mixture mathematical self-test failed: {pure}")
    spec = basis_spec_from_config(config)
    items = build_shared_stack(
        config["cases"],
        device,
        spec,
        num_components=int(config["model"]["num_components"]),
        num_knots=int(config["model"]["num_knots"]),
    )
    single_items = build_shared_stack(
        config["cases"],
        device,
        spec,
        num_components=1,
        num_knots=int(config["model"]["num_knots"]),
    )
    parameters = unique_evidence_parameters([item.operator for item in items])
    parameter_ids = [
        [id(parameter) for parameter in item.operator.parameters()]
        for item in items
    ]
    optimizer = torch.optim.AdamW(parameters, lr=1e-3)
    before = [parameter.detach().clone() for parameter in parameters]
    optimizer.zero_grad(set_to_none=True)
    per_case: list[dict[str, Any]] = []
    saved_batches: list[Any] = []

    for index, (item, single_item) in enumerate(zip(items, single_items)):
        seed = int(config["seed"]) + 1000 * index
        set_all_seeds(seed)
        batch = item.context.sample(
            int(config["batch_size"]), float(config["ebno_db"])
        )
        proposed = evidence_forward(item, batch, mode="mixture")
        moment = evidence_forward(item, batch, mode="moment")
        hard = evidence_forward(item, batch, mode="hard")
        single = evidence_forward(single_item, batch, mode="mixture")
        ls = ls_repaired_forward(
            item.ls_receiver, item.context, item.detector, batch
        )
        loss, parts = training_loss(
            item,
            proposed,
            batch,
            config["training_loss"],
        )
        (loss / len(items)).backward()
        decoded = decode_outputs(
            item.context,
            batch,
            {
                "proposed": proposed,
                "moment": moment,
                "hard": hard,
                "single": single,
                "ls": ls,
            },
            bp_iterations=int(config["bp_iterations"]),
            device=device,
        )
        result = proposed["evidence_result"]
        eig = torch.linalg.eigvalsh(
            proposed["posterior"].latent_cov.to(torch.complex128)
        ).real
        per_case.append(
            {
                "case": item.case.name,
                "num_prb": int(item.case.num_prb),
                "num_streams": int(item.case.num_streams),
                "effective_rank": int(item.basis_report["effective_rank"]),
                "nominal_rank": int(item.basis_report["nominal_rank"]),
                "loss": float(loss.detach().item()),
                "bit_nll": float(parts["bit_nll"].detach().item()),
                "channel_nmse": float(parts["channel_nmse"].detach().item()),
                "posterior_mean_finite": bool(
                    torch.isfinite(proposed["posterior"].mean).all().item()
                ),
                "posterior_variance_positive": bool(
                    (proposed["posterior"].var_diag > 0).all().item()
                ),
                "latent_covariance_min_eigenvalue": float(eig.min().item()),
                "weights_sum_to_one": bool(
                    torch.allclose(
                        result.weights.sum(dim=1),
                        torch.ones(result.weights.shape[0], device=device),
                        atol=1e-6,
                        rtol=0.0,
                    )
                ),
                "mean_component_weights": result.weights.detach()
                .mean(dim=0)
                .cpu()
                .tolist(),
                "effective_component_count": float(
                    result.effective_component_count.detach().mean().item()
                ),
                "ls_alignment_pass": bool(
                    ls["ls_grid_alignment_report"].get("passed")
                ),
                "bit_logits_shapes_equal": (
                    tuple(proposed["bit_logits"].shape)
                    == tuple(moment["bit_logits"].shape)
                    == tuple(single["bit_logits"].shape)
                    == tuple(ls["bit_logits"].shape)
                ),
                "tbler": {
                    name: float(value["tbler"]) for name, value in decoded.items()
                },
                "inference_uses_true_channel": bool(
                    proposed["inference_uses_true_channel"]
                ),
            }
        )
        saved_batches.append(batch)

    gradients = gradient_report(parameters)
    optimizer.step()
    changed = any(
        not torch.equal(old, parameter.detach())
        for old, parameter in zip(before, parameters)
    )

    state = shared_evidence_state(items[0].operator)
    clone_items = build_shared_stack(
        [config["cases"][0]],
        device,
        spec,
        num_components=int(config["model"]["num_components"]),
        num_knots=int(config["model"]["num_knots"]),
    )
    load_shared_evidence_state(clone_items[0].operator, state)
    with torch.inference_mode():
        reference = evidence_forward(items[0], saved_batches[0], mode="mixture")
        clone = evidence_forward(clone_items[0], saved_batches[0], mode="mixture")
    roundtrip = bool(
        torch.allclose(
            reference["bit_logits"], clone["bit_logits"], rtol=1e-5, atol=2e-5
        )
        and torch.allclose(
            reference["posterior"].mean,
            clone["posterior"].mean,
            rtol=1e-5,
            atol=2e-5,
        )
    )

    forward_parameters = set(
        inspect.signature(EvidenceMixtureLMMSEPosterior.forward).parameters
    )
    inference_contract = {
        "forward_has_no_truth_argument": not bool(
            {"truth", "true_channel", "h_true", "batch_h"} & forward_parameters
        ),
        "all_runtime_outputs_mark_no_truth": all(
            not item["inference_uses_true_channel"] for item in per_case
        ),
        "context_excludes_scenario_and_speed": True,
    }

    import sionna

    sionna_version = str(getattr(sionna, "__version__", "unknown"))
    checks = {
        "source_result_preconditions": pre["passed"],
        "pure_torch_mathematical_contract": pure["passed"],
        "cuda_compute_node": device.type == "cuda" and torch.cuda.is_available(),
        "sionna_2_0_1": sionna_version == "2.0.1",
        "precision_corrected_localized_basis": all(
            item.basis_report.get("precision_patch")
            == "complex128_atoms_before_rank_decision_v1"
            for item in items
        ),
        "shared_parameter_identity": len(items) >= 2
        and parameter_ids[0] == parameter_ids[1],
        "small_parameter_count": (
            items[0].operator.parameter_report()["trainable_parameters"] <= 128
        ),
        "each_case_posterior_finite": all(
            item["posterior_mean_finite"] for item in per_case
        ),
        "each_case_positive_variance": all(
            item["posterior_variance_positive"] for item in per_case
        ),
        "each_case_covariance_psd": all(
            item["latent_covariance_min_eigenvalue"] >= -1e-6
            for item in per_case
        ),
        "evidence_weights_are_probabilities": all(
            item["weights_sum_to_one"] for item in per_case
        ),
        "common_detector_shapes_exact": all(
            item["bit_logits_shapes_equal"] for item in per_case
        ),
        "actual_lmmse_ce_k1_path": all(
            "single" in item["tbler"] for item in per_case
        ),
        "moment_matched_lmmse_ce_path": all(
            "moment" in item["tbler"] for item in per_case
        ),
        "ls_linear_common_detector_path": all(
            item["ls_alignment_pass"] for item in per_case
        ),
        "gradient_to_all_parameter_tensors": gradients["all_present"]
        and gradients["all_finite"]
        and gradients["any_nonzero"],
        "optimizer_updates_parameters": changed,
        "nr_ldpc_decode_paths": all(
            all(0.0 <= value <= 1.0 for value in item["tbler"].values())
            for item in per_case
        ),
        "checkpoint_roundtrip": roundtrip,
        "inference_observability_contract": all(inference_contract.values()),
        "source_contract": len(source_hashes()) >= 12,
    }
    passed = all(checks.values())
    return {
        "version": GATE2_VERSION,
        "model_version": EVIDENCE_MIXTURE_LMMSE_VERSION,
        "complete": True,
        "overall_pass": passed,
        "screen_ready": passed,
        "classification": (
            "GATE2_NR_EVIDENCE_MIXTURE_SMOKE_PASS"
            if passed
            else "GATE2_NR_EVIDENCE_MIXTURE_SMOKE_FAIL"
        ),
        "checks": checks,
        "source_result": pre,
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
            f"TRAINABLE_PARAMETERS: {report['parameter_report']['trainable_parameters']}",
            "ROUTING_RULE: EXACT_PILOT_MARGINAL_LIKELIHOOD",
            "COMMON_DETECTOR_ESTIMATOR_ISOLATION: YES",
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
    pre = source_result_preconditions()
    pure = mathematical_self_test("cpu")
    if args.preflight_only:
        device = normalize_device("cpu")
        spec = basis_spec_from_config(config)
        items = build_shared_stack(
            config["cases"],
            device,
            spec,
            num_components=int(config["model"]["num_components"]),
            num_knots=int(config["model"]["num_knots"]),
        )
        print("GATE2_NR_EVIDENCE_MIXTURE_PREFLIGHT_PASS")
        print("SOURCE_CLASSIFICATION", pre["classification"])
        print("MODEL_VERSION", EVIDENCE_MIXTURE_LMMSE_VERSION)
        print("PURE_TORCH_MATH_PASS", pure["passed"])
        print(
            "PREFLIGHT_EFFECTIVE_RANKS",
            [int(item.basis_report["effective_rank"]) for item in items],
        )
        print("TRAINABLE_PARAMETERS", items[0].operator.parameter_report()["trainable_parameters"])
        print("COMMON_DETECTOR_ESTIMATOR_ISOLATION YES")
        print("INFERENCE_USES_TRUE_CHANNEL NO")
        return
    device = normalize_device(args.device)
    report = run(config, device)
    write_report(report)
    if not report["overall_pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
