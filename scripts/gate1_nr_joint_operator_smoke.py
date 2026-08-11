#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path
from typing import Any

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from bayesroute.nr_gate1 import (  # noqa: E402
    GATE1_NR_VERSION,
    NRBayesRouteBridge,
    NRCase,
    build_nr_context,
    decode_bridge,
)
from gate1_nr_joint_operator_common import (  # noqa: E402
    JOINT_OPERATOR_VERSION,
    SELECTED_DETECTOR_DAMPING,
    SELECTED_DETECTOR_ITERATIONS,
    SELECTED_EDGE_MASS,
    atomic_torch_save,
    bind_shared_operator,
    coded_metrics,
    copy_old_operator_if_compatible,
    differentiable_loss,
    extract_operator_state,
    gradient_report,
    load_operator_state,
    make_repaired_detector,
    package_signature,
    posterior_metrics,
    pure_torch_shared_parameter_self_test,
    repaired_forward,
    save_json,
    set_all_seeds,
    sha256_file,
    shared_parameter_report,
    unique_parameters,
)

SMOKE_VERSION = "gate1_nr_joint_operator_smoke_v1"
REQUIRED_SCREEN_CLASSIFICATION = "GATE1_DETECTOR_REPAIR_PARTIAL"
REQUIRED_SELECTED_VARIANT = "delmmse_sparse_i4_d0p7"
REQUIRED_CHECKPOINT_SHA256 = (
    "ca3243386e3d0511236a3c2c68f0396df9d05b7dc7f4118a8748d150613d3576"
)
SOURCE_CONTRACT_FILES = (
    "scripts/gate1_nr_joint_operator_common.py",
    "scripts/gate1_nr_joint_operator_smoke.py",
    "configs/gate1_nr_joint_operator_smoke.yaml",
    "src/bayesroute/lmmse_ep.py",
    "src/bayesroute/nr_gate1.py",
    "src/bayesroute/models.py",
    "src/bayesroute/qam.py",
)


def load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected a mapping in {path}")
    return value


def source_hashes() -> dict[str, str]:
    result: dict[str, str] = {}
    for relative in SOURCE_CONTRACT_FILES:
        path = ROOT / relative
        if not path.is_file():
            raise RuntimeError(f"Missing source-contract file: {relative}")
        result[relative] = sha256_file(path)
    return result


def verify_preconditions(config: dict[str, Any]) -> dict[str, Any]:
    screen_path = ROOT / "outputs/reports/gate1_nr_detector_repair_screen.json"
    checkpoint_path = ROOT / str(config["checkpoint_path"])
    repair_revision_path = ROOT / "GATE1_NR_DETECTOR_REPAIR_REVISION.json"
    joint_revision_path = ROOT / "GATE1_NR_JOINT_OPERATOR_REVISION.json"
    for path in (
        screen_path,
        checkpoint_path,
        repair_revision_path,
        joint_revision_path,
    ):
        if not path.is_file():
            raise RuntimeError(f"Missing precondition file: {path}")

    screen = json.loads(screen_path.read_text(encoding="utf-8"))
    repair_revision = json.loads(repair_revision_path.read_text(encoding="utf-8"))
    joint_revision = json.loads(joint_revision_path.read_text(encoding="utf-8"))
    checkpoint_sha = sha256_file(checkpoint_path)
    checks = {
        "screen_complete": screen.get("complete") is True,
        "screen_rows": screen.get("evaluation", {}).get("rows") == 2688,
        "screen_classification": (
            screen.get("classification") == REQUIRED_SCREEN_CLASSIFICATION
        ),
        "selected_variant": (
            screen.get("selected_variant") == REQUIRED_SELECTED_VARIANT
        ),
        "screen_next_action": (
            screen.get("next_action")
            == "JOINTLY_TUNE_POSTERIOR_OPERATOR_AND_LMMSE_MESSAGE_DETECTOR"
        ),
        "checkpoint_sha256": checkpoint_sha == REQUIRED_CHECKPOINT_SHA256,
        "repair_revision": (
            repair_revision.get("revision") == "gate1_nr_detector_repair_v1"
        ),
        "detector_version": (
            repair_revision.get("detector_version")
            == "damped_extrinsic_lmmse_v1"
        ),
        "joint_revision": (
            joint_revision.get("revision") == JOINT_OPERATOR_VERSION
        ),
        "gate1_revision": (
            joint_revision.get("gate1_revision") == GATE1_NR_VERSION
        ),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "checkpoint_sha256": checkpoint_sha,
        "screen_classification": screen.get("classification"),
        "selected_variant": screen.get("selected_variant"),
    }


def build_bridge(
    case: NRCase,
    context: Any,
    config: dict[str, Any],
) -> NRBayesRouteBridge:
    operator = config["operator"]
    return NRBayesRouteBridge(
        context.grid,
        num_streams=case.num_streams,
        rank=int(operator["rank"]),
        bank_rank=int(operator["bank_rank"]),
        detector_iterations=4,
        edge_mass=SELECTED_EDGE_MASS,
        length_f=float(operator["length_f"]),
        length_t=float(operator["length_t"]),
        operator_seed=int(operator["operator_seed"]),
    ).to(context.device)


def decode_one(context: Any, batch: Any, output: dict[str, Any], device: torch.device) -> dict[str, float]:
    decoded = decode_bridge(
        context.transmitter,
        output,
        batch.information_bits,
        num_bp_iter=int(8),
        device=device,
    )
    return {
        "information_ber": float(decoded["information_ber"]),
        "tbler": float(decoded["tbler"]),
        "crc_failure_rate": float(decoded["crc_failure_rate"]),
    }


def run_case_loss(
    bridge: NRBayesRouteBridge,
    detector: torch.nn.Module,
    context: Any,
    *,
    seed: int,
    batch_size: int,
    ebno_db: float,
    config: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, Any], Any]:
    set_all_seeds(seed)
    batch = context.sample(batch_size=batch_size, ebno_db=float(ebno_db))
    output = repaired_forward(bridge, detector, batch)
    loss, parts = differentiable_loss(
        output,
        batch,
        channel_loss_weight=float(config["training"]["channel_loss_weight"]),
        calibration_loss_weight=float(
            config["training"]["calibration_loss_weight"]
        ),
    )
    return loss, {"output": output, "parts": parts}, batch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/gate1_nr_joint_operator_smoke.yaml",
    )
    args = parser.parse_args()
    config_path = ROOT / args.config
    config = load_yaml(config_path)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    preconditions = verify_preconditions(config)
    if not preconditions["passed"]:
        raise RuntimeError(f"Joint-operator smoke preconditions failed: {preconditions}")

    pure_test = pure_torch_shared_parameter_self_test()
    if not pure_test["passed"]:
        raise RuntimeError(f"Shared-parameter self-test failed: {pure_test}")

    cases = [NRCase.from_mapping(item) for item in config["cases"]]
    contexts = [build_nr_context(case, device) for case in cases]
    bridges = [
        build_bridge(case, context, config)
        for case, context in zip(cases, contexts)
    ]
    bind_shared_operator(bridges)
    shared_report = shared_parameter_report(bridges)

    checkpoint = torch.load(
        ROOT / str(config["checkpoint_path"]),
        map_location=device,
        weights_only=False,
    )
    if "model" not in checkpoint:
        raise RuntimeError("Preliminary checkpoint has no model state")
    warm_started = copy_old_operator_if_compatible(
        bridges[0], checkpoint["model"]
    )
    if not warm_started:
        raise RuntimeError("Smoke operator rank must permit old-checkpoint warm start")

    detectors = [
        make_repaired_detector(int(context.grid.bits_per_symbol)).to(device)
        for context in contexts
    ]
    parameters = unique_parameters(bridges)
    if len(parameters) != 2:
        raise RuntimeError(f"Expected two shared parameter tensors, found {len(parameters)}")
    optimizer = torch.optim.Adam(parameters, lr=float(config["training"]["lr"]))

    per_case_gradients: list[dict[str, Any]] = []
    for index, (bridge, detector, context) in enumerate(
        zip(bridges, detectors, contexts)
    ):
        optimizer.zero_grad(set_to_none=True)
        loss, _, _ = run_case_loss(
            bridge,
            detector,
            context,
            seed=int(config["seed"]) + 1000 * index,
            batch_size=int(config["batch_size"]),
            ebno_db=float(config["ebno_db"]),
            config=config,
        )
        loss.backward()
        report = gradient_report(parameters)
        report["case"] = cases[index].name
        report["loss"] = float(loss.detach().item())
        per_case_gradients.append(report)

    optimizer.zero_grad(set_to_none=True)
    combined_loss = torch.zeros((), dtype=torch.float32, device=device)
    combined_outputs: list[dict[str, Any]] = []
    combined_batches: list[Any] = []
    for index, (bridge, detector, context) in enumerate(
        zip(bridges, detectors, contexts)
    ):
        loss, detail, batch = run_case_loss(
            bridge,
            detector,
            context,
            seed=int(config["seed"]) + 100_000 + 1000 * index,
            batch_size=int(config["batch_size"]),
            ebno_db=float(config["ebno_db"]),
            config=config,
        )
        combined_loss = combined_loss + loss / len(bridges)
        combined_outputs.append(detail["output"])
        combined_batches.append(batch)
    combined_loss.backward()
    mixed_gradient = gradient_report(parameters)
    before = [item.detach().clone() for item in parameters]
    optimizer.step()
    changed = [
        not torch.equal(old, new.detach())
        for old, new in zip(before, parameters)
    ]

    post_update_records: list[dict[str, Any]] = []
    fixed_batches: list[Any] = []
    fixed_outputs: list[dict[str, Any]] = []
    for index, (bridge, detector, context) in enumerate(
        zip(bridges, detectors, contexts)
    ):
        set_all_seeds(int(config["seed"]) + 200_000 + index)
        batch = context.sample(
            batch_size=int(config["batch_size"]),
            ebno_db=float(config["ebno_db"]),
        )
        with torch.no_grad():
            output = repaired_forward(bridge, detector, batch)
            decoded = decode_one(context, batch, output, device)
        fixed_batches.append(batch)
        fixed_outputs.append(output)
        post_update_records.append(
            {
                "case": cases[index].name,
                "coded": coded_metrics(output, batch),
                "posterior": posterior_metrics(output, batch),
                "decoded": decoded,
                "edge_density": float(output["edge_density"].item()),
                "finite": bool(
                    torch.isfinite(output["bit_logits"]).all().item()
                    and torch.isfinite(output["posterior"].mean).all().item()
                    and torch.isfinite(output["posterior"].local_cov).all().item()
                ),
            }
        )

    checkpoint_path = ROOT / "outputs/gate1_nr_joint_operator/smoke_checkpoint.pt"
    checkpoint_payload = {
        "version": SMOKE_VERSION,
        "operator": extract_operator_state(bridges[0]),
        "optimizer": optimizer.state_dict(),
        "config_sha256": sha256_file(config_path),
    }
    atomic_torch_save(checkpoint_payload, checkpoint_path)

    reloaded_bridges = [
        build_bridge(case, context, config)
        for case, context in zip(cases, contexts)
    ]
    bind_shared_operator(reloaded_bridges)
    reloaded = torch.load(checkpoint_path, map_location=device, weights_only=False)
    load_operator_state(reloaded_bridges[0], reloaded["operator"])
    reloaded_detectors = [
        make_repaired_detector(int(context.grid.bits_per_symbol)).to(device)
        for context in contexts
    ]
    max_reload_error = 0.0
    with torch.no_grad():
        for bridge, detector, batch, expected in zip(
            reloaded_bridges,
            reloaded_detectors,
            fixed_batches,
            fixed_outputs,
        ):
            actual = repaired_forward(bridge, detector, batch)
            max_reload_error = max(
                max_reload_error,
                float(
                    torch.max(
                        torch.abs(actual["bit_logits"] - expected["bit_logits"])
                    ).item()
                ),
            )

    contract = {
        "version": SMOKE_VERSION,
        "joint_operator_version": JOINT_OPERATOR_VERSION,
        "config_sha256": sha256_file(config_path),
        "source_sha256": source_hashes(),
        "checkpoint_sha256": preconditions["checkpoint_sha256"],
        "cases": [case.__dict__ for case in cases],
        "detector": {
            "iterations": SELECTED_DETECTOR_ITERATIONS,
            "damping": SELECTED_DETECTOR_DAMPING,
            "edge_mass": SELECTED_EDGE_MASS,
        },
    }
    contract["signature"] = package_signature(contract)

    checks = {
        "preconditions": bool(preconditions["passed"]),
        "pure_torch_shared_parameter_self_test": bool(pure_test["passed"]),
        "cuda_compute_node": device.type == "cuda",
        "sionna_2_0_1": __import__("sionna").__version__ == "2.0.1",
        "selected_detector_contract": bool(
            SELECTED_DETECTOR_ITERATIONS == 4
            and abs(SELECTED_DETECTOR_DAMPING - 0.7) < 1e-12
            and abs(SELECTED_EDGE_MASS - 0.8) < 1e-12
        ),
        "shared_parameter_identity": bool(
            shared_report["raw_weight_ids_identical"]
            and shared_report["noise_scale_ids_identical"]
        ),
        "each_case_has_gradient": bool(
            all(
                item["all_present"]
                and item["all_finite"]
                and item["total_norm"] > 0.0
                for item in per_case_gradients
            )
        ),
        "mixed_case_gradient": bool(
            mixed_gradient["all_present"]
            and mixed_gradient["all_finite"]
            and mixed_gradient["total_norm"] > 0.0
        ),
        "optimizer_updates_shared_operator": all(changed),
        "posterior_outputs_finite": all(item["finite"] for item in post_update_records),
        "nr_ldpc_paths": all(
            0.0 <= item["decoded"]["tbler"] <= 1.0
            and 0.0 <= item["decoded"]["information_ber"] <= 1.0
            for item in post_update_records
        ),
        "checkpoint_roundtrip": max_reload_error <= 1e-6,
        "source_contract": len(contract["source_sha256"]) == len(SOURCE_CONTRACT_FILES),
    }
    overall = all(checks.values())
    report = {
        "version": SMOKE_VERSION,
        "joint_operator_version": JOINT_OPERATOR_VERSION,
        "timestamp": __import__("datetime").datetime.now().astimezone().strftime(
            "%Y-%m-%d %H:%M:%S %Z"
        ),
        "environment": {
            "device": str(device),
            "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "sionna": __import__("sionna").__version__,
        },
        "preconditions": preconditions,
        "pure_torch_self_test": pure_test,
        "shared_parameter_report": shared_report,
        "per_case_gradients": per_case_gradients,
        "mixed_gradient": mixed_gradient,
        "combined_loss": float(combined_loss.detach().item()),
        "parameters_changed": changed,
        "post_update_records": post_update_records,
        "checkpoint_roundtrip_max_abs_error": max_reload_error,
        "contract": contract,
        "checks": checks,
        "overall_pass": overall,
        "classification": (
            "GATE1_NR_JOINT_OPERATOR_SMOKE_PASS"
            if overall
            else "GATE1_NR_JOINT_OPERATOR_SMOKE_FAIL"
        ),
        "capacity_diagnostic_ready": overall,
        "publication_nr_ready": False,
    }
    gate_dir = ROOT / "outputs/gates"
    save_json(report, gate_dir / "GATE1_NR_JOINT_OPERATOR_SMOKE.json")
    lines = [
        *(f"{key}: {'PASS' if value else 'FAIL'}" for key, value in checks.items()),
        f"CLASSIFICATION: {report['classification']}",
        f"CAPACITY_DIAGNOSTIC_READY: {'YES' if overall else 'NO'}",
        "PUBLICATION_NR_READY: NO",
    ]
    (gate_dir / "GATE1_NR_JOINT_OPERATOR_SMOKE.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print("\n".join(lines), flush=True)
    if not overall:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
