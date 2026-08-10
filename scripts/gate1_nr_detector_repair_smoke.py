#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bayesroute.lmmse_ep import (  # noqa: E402
    DETECTOR_VERSION,
    DampedExtrinsicLMMSEDetector,
    mathematical_self_test,
)
from bayesroute.models import (  # noqa: E402
    BayesRouteDetector,
    coupling_matrix,
    coupling_selection_mask,
    edge_density,
)
from bayesroute.nr_gate1 import (  # noqa: E402
    GATE1_NR_VERSION,
    NRBayesRouteBridge,
    NRCase,
    build_nr_context,
    coded_bit_metrics,
    decode_bridge,
    package_contract_signature,
    run_standard_receiver,
    standard_receiver,
    transfer_operator_parameters,
)

SMOKE_VERSION = "gate1_nr_detector_repair_smoke_v1"
REPAIR_REVISION = "gate1_nr_detector_repair_v1"
REQUIRED_FAILURE_CLASSIFICATION = "GATE1_DETECTOR_REPAIR_REQUIRED"
REQUIRED_CHECKPOINT_SHA256 = (
    "ca3243386e3d0511236a3c2c68f0396df9d05b7dc7f4118a8748d150613d3576"
)
SOURCE_CONTRACT_FILES = (
    "src/bayesroute/lmmse_ep.py",
    "scripts/gate1_nr_detector_repair_smoke.py",
    "configs/gate1_nr_detector_repair_smoke.yaml",
    "src/bayesroute/nr_gate1.py",
    "src/bayesroute/models.py",
    "src/bayesroute/qam.py",
)


def load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected mapping in {path}")
    return value


def save_json(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_hashes() -> dict[str, str]:
    result: dict[str, str] = {}
    for relative in SOURCE_CONTRACT_FILES:
        path = ROOT / relative
        if not path.is_file():
            raise RuntimeError(f"Missing source-contract file: {relative}")
        result[relative] = sha256_file(path)
    return result


def verify_preconditions(config: dict[str, Any]) -> dict[str, Any]:
    revision_path = ROOT / "GATE1_NR_DETECTOR_REPAIR_REVISION.json"
    failure_path = ROOT / "outputs/gates/GATE1_NR_FAILURE_MODE_DIAGNOSTIC.json"
    train_path = ROOT / "outputs/reports/gate1_nr_preliminary_train_summary.json"
    checkpoint_path = ROOT / str(config["checkpoint_path"])
    for path in (revision_path, failure_path, train_path, checkpoint_path):
        if not path.is_file():
            raise RuntimeError(f"Missing required file: {path}")
    revision = json.loads(revision_path.read_text(encoding="utf-8"))
    failure = json.loads(failure_path.read_text(encoding="utf-8"))
    training = json.loads(train_path.read_text(encoding="utf-8"))
    checkpoint_sha = sha256_file(checkpoint_path)
    checks = {
        "repair_revision": revision.get("revision") == REPAIR_REVISION,
        "gate1_revision": revision.get("gate1_revision") == GATE1_NR_VERSION,
        "failure_diagnostic_complete": failure.get("complete") is True,
        "failure_classification": (
            failure.get("classification") == REQUIRED_FAILURE_CLASSIFICATION
        ),
        "failure_next_action": (
            failure.get("next_action")
            == "REPLACE_SOFT_PIC_WITH_DAMPED_EXTRINSIC_EP_OR_LMMSE_EP"
        ),
        "checkpoint_record": (
            training.get("best_checkpoint_sha256") == REQUIRED_CHECKPOINT_SHA256
        ),
        "checkpoint_file": checkpoint_sha == REQUIRED_CHECKPOINT_SHA256,
        "detector_version": DETECTOR_VERSION == "damped_extrinsic_lmmse_v1",
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "checkpoint_sha256": checkpoint_sha,
        "failure_classification": failure.get("classification"),
        "repair_revision": revision.get("revision"),
    }


def bridge_from_config(
    case: NRCase,
    context: Any,
    config: dict[str, Any],
) -> NRBayesRouteBridge:
    bridge = config["bridge"]
    return NRBayesRouteBridge(
        context.grid,
        num_streams=case.num_streams,
        rank=int(bridge["rank"]),
        bank_rank=int(bridge["bank_rank"]),
        detector_iterations=4,
        edge_mass=float(bridge["edge_mass"]),
        length_f=float(bridge["length_f"]),
        length_t=float(bridge["length_t"]),
        operator_seed=int(bridge["operator_seed"]),
    ).to(context.device)


def load_source_bridge(
    config: dict[str, Any], device: torch.device
) -> tuple[NRBayesRouteBridge, Any]:
    summary = json.loads(
        (ROOT / "outputs/reports/gate1_nr_preliminary_train_summary.json").read_text(
            encoding="utf-8"
        )
    )
    source_case = NRCase.from_mapping(summary["source_case"])
    source_context = build_nr_context(source_case, device)
    source = bridge_from_config(source_case, source_context, config)
    checkpoint = torch.load(
        ROOT / str(config["checkpoint_path"]),
        map_location=device,
        weights_only=False,
    )
    source.load_state_dict(checkpoint["model"], strict=True)
    source.eval()
    return source, source_context


def posterior_and_graph(
    bridge: NRBayesRouteBridge,
    batch: Any,
    edge_mass: float,
) -> tuple[Any, torch.Tensor]:
    posterior = bridge.posterior(
        batch.y[..., batch.pilot_idx], batch.phi, batch.noise_var
    )
    coupling = coupling_matrix(
        posterior.mean.detach(),
        posterior.local_cov.detach(),
        batch.data_idx,
        batch.noise_var.detach(),
    )
    graph = coupling_selection_mask(coupling, float(edge_mass))
    return posterior, graph


def attach_posterior(
    output: dict[str, Any], posterior: Any, graph: torch.Tensor
) -> dict[str, Any]:
    result = dict(output)
    result["posterior"] = posterior
    result["reference_graph_mask"] = graph
    result["edge_density"] = edge_density(graph)
    return result


def true_channel_output(
    detector: nn.Module,
    batch: Any,
    graph: torch.Tensor,
) -> dict[str, Any]:
    streams = int(batch.h.shape[1])
    resource_elements = int(batch.h.shape[-1])
    zero_cov = torch.zeros(
        (streams, streams, resource_elements),
        dtype=torch.complex64,
        device=batch.h.device,
    )
    result = detector(
        batch.y,
        batch.h,
        zero_cov,
        batch.data_idx,
        batch.noise_var,
        graph,
        covariance_mode="none",
    )
    result["reference_graph_mask"] = graph
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="configs/gate1_nr_detector_repair_smoke.yaml"
    )
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    config_path = ROOT / args.config
    config = load_yaml(config_path)
    preconditions = verify_preconditions(config)
    if not preconditions["passed"]:
        raise SystemExit(f"BLOCKED: detector-repair preconditions failed: {preconditions}")

    requested_device = args.device or str(config["device"])
    if requested_device == "cuda" and torch.cuda.is_available():
        requested_device = "cuda:0"
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        if args.preflight_only:
            requested_device = "cpu"
        else:
            raise SystemExit("Gate-1 detector repair smoke requires a CUDA node")
    device = torch.device(requested_device)

    math_report = mathematical_self_test(device)
    if not math_report["passed"]:
        raise SystemExit(f"BLOCKED: LMMSE detector mathematical self-test failed: {math_report}")

    contract = {
        "version": SMOKE_VERSION,
        "repair_revision": REPAIR_REVISION,
        "checkpoint_sha256": preconditions["checkpoint_sha256"],
        "config_sha256": sha256_file(config_path),
        "source_sha256": source_hashes(),
        "detector": config["detector"],
        "case": config["case"],
    }
    contract["signature"] = package_contract_signature(contract)

    if args.preflight_only:
        print("GATE1_NR_DETECTOR_REPAIR_CPU_PREFLIGHT_PASS")
        print("DETECTOR_VERSION", DETECTOR_VERSION)
        print("SOURCE_CONTRACT_FILES", len(contract["source_sha256"]))
        return

    import sionna
    import sionna.phy

    if str(getattr(sionna, "__version__", "")) != "2.0.1":
        raise SystemExit("Gate-1 detector repair requires Sionna 2.0.1")
    sionna.phy.config.device = str(device)
    torch.manual_seed(int(config["seed"]))
    sionna.phy.config.seed = int(config["seed"])

    source, source_context = load_source_bridge(config, device)
    case = NRCase.from_mapping(config["case"])
    context = build_nr_context(case, device)
    bridge = bridge_from_config(case, context, config)
    transfer = transfer_operator_parameters(source, bridge)
    if not transfer["passed"]:
        raise RuntimeError("Could not transfer learned posterior operator")
    bridge.eval()

    batch = context.sample(
        batch_size=int(config["batch_size"]),
        ebno_db=float(config["ebno_db"]),
    )
    posterior, graph = posterior_and_graph(
        bridge, batch, float(config["bridge"]["edge_mass"])
    )
    detector_cfg = config["detector"]
    detector = DampedExtrinsicLMMSEDetector(
        int(context.grid.bits_per_symbol),
        n_iter=int(detector_cfg["iterations"]),
        damping=float(detector_cfg["damping"]),
        covariance_mode=str(detector_cfg["covariance_mode"]),
    ).to(device)
    output = attach_posterior(
        detector(
            batch.y,
            posterior.mean,
            posterior.local_cov,
            batch.data_idx,
            batch.noise_var,
            graph,
        ),
        posterior,
        graph,
    )
    decoded = decode_bridge(
        context.transmitter,
        output,
        batch.information_bits,
        num_bp_iter=int(config["bp_iterations"]),
        device=device,
    )

    # True-channel path isolates the detector from the learned posterior.
    zero_cov = torch.zeros(
        (case.num_streams, case.num_streams, batch.h.shape[-1]),
        dtype=torch.complex64,
        device=device,
    )
    true_coupling = coupling_matrix(
        batch.h.detach(), zero_cov, batch.data_idx, batch.noise_var.detach()
    )
    true_graph = coupling_selection_mask(
        true_coupling, float(config["bridge"]["edge_mass"])
    )
    true_output = true_channel_output(detector, batch, true_graph)
    true_decoded = decode_bridge(
        context.transmitter,
        true_output,
        batch.information_bits,
        num_bp_iter=int(config["bp_iterations"]),
        device=device,
    )

    # Existing true-channel soft-PIC is retained only as a diagnostic control.
    old_pic = BayesRouteDetector(
        int(context.grid.bits_per_symbol), n_iter=4, use_uncertainty=False
    ).to(device)
    old_logits, old_symbols, old_mean, old_var, old_mask = old_pic(
        batch.y,
        batch.h,
        zero_cov,
        batch.data_idx,
        batch.noise_var,
        kappa=true_coupling,
        edge_mass=float(config["bridge"]["edge_mass"]),
        use_uncertainty=False,
    )
    old_output = {
        "bit_logits": old_logits,
        "symbol_logits": old_symbols,
        "x_mean": old_mean,
        "x_var": old_var,
        "graph_mask": old_mask,
        "edge_density": edge_density(old_mask),
    }
    old_decoded = decode_bridge(
        context.transmitter,
        old_output,
        batch.information_bits,
        num_bp_iter=int(config["bp_iterations"]),
        device=device,
    )

    perfect = standard_receiver(context, perfect_csi=True, return_crc=True)
    perfect_metrics = run_standard_receiver(
        perfect, batch, batch.information_bits, perfect_csi=True
    )

    # Verify that gradients reach both learned posterior parameters through the
    # new detector. The graph is intentionally detached/hard-selected.
    bridge.train()
    bridge.zero_grad(set_to_none=True)
    posterior_grad, graph_grad = posterior_and_graph(
        bridge, batch, float(config["bridge"]["edge_mass"])
    )
    grad_output = detector(
        batch.y,
        posterior_grad.mean,
        posterior_grad.local_cov,
        batch.data_idx,
        batch.noise_var,
        graph_grad,
    )
    loss = F.binary_cross_entropy_with_logits(
        grad_output["bit_logits"], batch.coded_bits.float()
    )
    loss.backward()
    gradient_report: dict[str, Any] = {}
    for name, parameter in bridge.named_parameters():
        if not parameter.requires_grad:
            continue
        gradient = parameter.grad
        gradient_report[name] = {
            "present": gradient is not None,
            "finite": bool(gradient is not None and torch.isfinite(gradient).all().item()),
            "norm": float(gradient.norm().item()) if gradient is not None else 0.0,
        }
    gradient_pass = bool(
        gradient_report
        and all(
            item["present"] and item["finite"] and item["norm"] > 0.0
            for item in gradient_report.values()
        )
    )

    expected_shape = list(batch.coded_bits.shape)
    checks = {
        "preconditions": preconditions["passed"],
        "mathematical_self_test": math_report["passed"],
        "cuda_compute_node": device.type == "cuda",
        "sionna_2_0_1": str(getattr(sionna, "__version__", "")) == "2.0.1",
        "operator_transfer": transfer["passed"],
        "posterior_output_finite": bool(
            torch.isfinite(posterior.mean).all().item()
            and torch.isfinite(posterior.local_cov).all().item()
        ),
        "new_detector_finite": bool(output["diagnostics"]["finite"]),
        "new_detector_shape": list(output["bit_logits"].shape) == expected_shape,
        "positive_extrinsic_variance": bool(
            output["extrinsic_var"].min().item() > 0.0
        ),
        "graph_preserved": torch.equal(output["graph_mask"], graph),
        "nr_ldpc_decode_path": bool(
            decoded["decoded_shape"] == list(batch.information_bits.shape)
        ),
        "true_channel_path": bool(
            true_decoded["decoded_shape"] == list(batch.information_bits.shape)
        ),
        "perfect_csi_lmmse_path": bool(
            math.isfinite(float(perfect_metrics["tbler"]))
        ),
        "gradient_to_operator": gradient_pass,
        "source_contract": len(contract["source_sha256"]) == len(SOURCE_CONTRACT_FILES),
    }
    overall = all(checks.values())
    result = {
        "version": SMOKE_VERSION,
        "repair_revision": REPAIR_REVISION,
        "detector_version": DETECTOR_VERSION,
        "classification": (
            "GATE1_NR_DETECTOR_REPAIR_SMOKE_PASS"
            if overall
            else "GATE1_NR_DETECTOR_REPAIR_SMOKE_FAIL"
        ),
        "overall_pass": overall,
        "screen_ready": overall,
        "publication_nr_ready": False,
        "checks": checks,
        "preconditions": preconditions,
        "mathematical_self_test": math_report,
        "contract": contract,
        "case": case.__dict__,
        "graph_edge_density": float(edge_density(graph).item()),
        "detector_diagnostics": output["diagnostics"],
        "coded_metrics": coded_bit_metrics(output["bit_logits"], batch.coded_bits),
        "decoded": {
            "tbler": decoded["tbler"],
            "information_ber": decoded["information_ber"],
            "crc_failure_rate": decoded["crc_failure_rate"],
        },
        "true_channel_lmmse_message_passing": {
            "tbler": true_decoded["tbler"],
            "information_ber": true_decoded["information_ber"],
        },
        "old_true_channel_soft_pic": {
            "tbler": old_decoded["tbler"],
            "information_ber": old_decoded["information_ber"],
        },
        "perfect_csi_lmmse": perfect_metrics,
        "gradient_report": gradient_report,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "sionna": getattr(sionna, "__version__", "unknown"),
            "device": str(device),
            "gpu_name": torch.cuda.get_device_name(device),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "hostname": platform.node(),
        },
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
    }

    gate_dir = ROOT / "outputs/gates"
    gate_dir.mkdir(parents=True, exist_ok=True)
    save_json(result, gate_dir / "GATE1_NR_DETECTOR_REPAIR_SMOKE.json")
    lines = [
        *[f"{name}: {'PASS' if value else 'FAIL'}" for name, value in checks.items()],
        f"CLASSIFICATION: {result['classification']}",
        f"SCREEN_READY: {'YES' if overall else 'NO'}",
        "PUBLICATION_NR_READY: NO",
    ]
    (gate_dir / "GATE1_NR_DETECTOR_REPAIR_SMOKE.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print("\n".join(lines))
    if not overall:
        raise SystemExit(2)

    del source, source_context, bridge, context
    if device.type == "cuda":
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
