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
import torch.nn.functional as F
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bayesroute.nr_gate1 import (
    GATE1_NR_VERSION,
    NRBayesRouteBridge,
    NRCase,
    build_nr_context,
    build_transmitter,
    codec_roundtrip,
    coded_bit_metrics,
    decode_bridge,
    extract_nr_grid,
    graph_cardinality_report,
    pilot_grid_match_report,
    pilot_orthogonality_report,
    posterior_psd_report,
    run_standard_receiver,
    standard_receiver,
    transfer_operator_parameters,
)


def load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected mapping in {path}")
    return value


def save_json(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    tmp.replace(path)


def manifest_report(root: Path) -> dict[str, Any]:
    path = root / "GATE1_NR_MANIFEST.sha256"
    failures: list[str] = []
    checked = 0
    if not path.is_file():
        return {"passed": False, "checked": 0, "failures": ["missing manifest"]}
    for raw in path.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        expected, relative = raw.split(None, 1)
        relative = relative.strip().lstrip("*")
        target = root / relative
        if not target.is_file():
            failures.append(f"missing:{relative}")
        elif hashlib.sha256(target.read_bytes()).hexdigest() != expected:
            failures.append(f"hash:{relative}")
        checked += 1
    return {"passed": not failures and checked >= 12, "checked": checked, "failures": failures}


def gate0_report(root: Path, expected: str) -> dict[str, Any]:
    path = root / "outputs/gates/GATE0_MECHANISM_DIAGNOSTIC.txt"
    if not path.is_file():
        return {"passed": False, "path": str(path), "classification": None}
    classification = None
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("CLASSIFICATION:"):
            classification = line.split(":", 1)[1].strip()
    return {
        "passed": classification == expected,
        "path": str(path),
        "classification": classification,
    }


def metadata_report(case: NRCase, metadata: list[dict[str, Any]]) -> dict[str, Any]:
    ports = [int(item["dmrs_port"]) for item in metadata]
    flat = [int(item["layer_index"]) for item in metadata]
    user_layer = [
        (int(item["user_index"]), int(item["within_user_layer_index"]))
        for item in metadata
    ]
    expected_user_layer = [
        (user, layer)
        for user in range(case.num_users)
        for layer in range(case.num_layers_per_user)
    ]
    passed = bool(
        len(metadata) == case.num_streams
        and ports == list(case.dmrs_ports)
        and flat == list(range(case.num_streams))
        and user_layer == expected_user_layer
        and len(set(ports)) == len(ports)
        and all(int(item["dmrs_config_type"]) == case.dmrs_config_type for item in metadata)
        and all(int(item["dmrs_length"]) == case.dmrs_length for item in metadata)
        and all(len(item["w_f"]) == 2 and len(item["w_t"]) == 2 for item in metadata)
    )
    return {
        "passed": passed,
        "ports": ports,
        "flat_layer_indices": flat,
        "user_layer_map": user_layer,
        "expected_user_layer_map": expected_user_layer,
        "metadata": metadata,
    }


def mapping_case(case: NRCase, device: torch.device, batch_size: int) -> dict[str, Any]:
    transmitter, configs = build_transmitter(case, device)
    grid = extract_nr_grid(transmitter, configs, device)
    orthogonality = pilot_orthogonality_report(grid.phi)
    exact_grid = pilot_grid_match_report(transmitter, grid, batch_size=batch_size)
    codec = codec_roundtrip(transmitter, batch_size=batch_size, device=device)
    metadata = metadata_report(case, grid.port_metadata)
    partition = {
        "passed": bool(
            torch.all(grid.data_mask ^ grid.reserved_mask).item()
            and not torch.any(grid.data_mask & grid.reserved_mask).item()
            and torch.all(grid.reserved_zero_mask <= grid.reserved_mask).item()
            and torch.all(grid.reserved_mask.reshape(-1)[grid.pilot_idx]).item()
            and grid.data_idx.numel() > 0
            and grid.pilot_idx.numel() > 0
        ),
        "actual_pilot_observations": grid.num_pilot_observations,
        "data_symbols_per_stream": grid.num_data_symbols,
        "globally_unused_reserved_re": int(grid.reserved_zero_mask.sum().item()),
        "resource_elements": grid.num_resource_elements,
    }
    return {
        "passed": bool(
            orthogonality["passed"]
            and exact_grid["passed"]
            and codec["passed"]
            and metadata["passed"]
            and partition["passed"]
        ),
        "case": case.__dict__,
        "num_streams": case.num_streams,
        "orthogonality": orthogonality,
        "exact_grid": exact_grid,
        "codec_roundtrip": codec,
        "metadata": metadata,
        "partition": partition,
        "phi_shape": list(grid.phi.shape),
    }


def channel_case(
    case: NRCase,
    device: torch.device,
    batch_size: int,
    ebno_db: float,
) -> dict[str, Any]:
    context = build_nr_context(case, device)
    batch = context.sample(batch_size=batch_size, ebno_db=ebno_db)
    finite = bool(
        torch.isfinite(batch.y).all().item()
        and torch.isfinite(batch.h).all().item()
        and torch.isfinite(batch.raw_y).all().item()
        and torch.isfinite(batch.raw_h).all().item()
    )
    shape = bool(
        tuple(batch.y.shape[:2]) == (batch_size, case.num_rx_ant)
        and tuple(batch.h.shape[:3])
        == (batch_size, case.num_streams, case.num_rx_ant)
        and tuple(batch.coded_bits.shape[:2]) == (batch_size, case.num_streams)
        and tuple(batch.information_bits.shape[:2]) == (batch_size, case.num_users)
        and tuple(batch.x_grid.shape[:2]) == (batch_size, case.num_streams)
    )
    ls_receiver = standard_receiver(context, perfect_csi=False, return_crc=True)
    perfect_receiver = standard_receiver(context, perfect_csi=True, return_crc=True)
    with torch.no_grad():
        ls = run_standard_receiver(ls_receiver, batch, batch.information_bits, perfect_csi=False)
        perfect = run_standard_receiver(
            perfect_receiver, batch, batch.information_bits, perfect_csi=True
        )
    metrics_valid = all(
        0.0 <= float(result[key]) <= 1.0
        for result in (ls, perfect)
        for key in ("information_ber", "tbler", "crc_failure_rate")
    )
    return {
        "passed": bool(finite and shape and metrics_valid),
        "case": case.__dict__,
        "finite": finite,
        "shapes_valid": shape,
        "raw_y_shape": list(batch.raw_y.shape),
        "raw_h_shape": list(batch.raw_h.shape),
        "adapted_y_shape": list(batch.y.shape),
        "adapted_h_shape": list(batch.h.shape),
        "coded_bits_shape": list(batch.coded_bits.shape),
        "ls_lmmse": ls,
        "perfect_csi_lmmse": perfect,
    }


def kbest_check(
    case: NRCase,
    device: torch.device,
    batch_size: int,
    ebno_db: float,
    k: int,
) -> dict[str, Any]:
    context = build_nr_context(case, device)
    batch = context.sample(batch_size=batch_size, ebno_db=ebno_db)
    receiver = standard_receiver(
        context, perfect_csi=True, kbest_k=int(k), return_crc=True
    )
    with torch.no_grad():
        metrics = run_standard_receiver(
            receiver, batch, batch.information_bits, perfect_csi=True
        )
    valid = all(
        0.0 <= float(metrics[key]) <= 1.0
        for key in ("information_ber", "tbler", "crc_failure_rate")
    )
    return {
        "passed": bool(valid),
        "metrics": metrics,
        "case": case.__dict__,
        "num_streams": case.num_streams,
        "k": int(k),
    }


def differentiable_channel_nll(output: dict[str, Any], batch: Any) -> torch.Tensor:
    posterior = output["posterior"]
    mean = posterior.mean[..., batch.data_idx]
    truth = batch.h[..., batch.data_idx]
    var = posterior.var_diag[None, :, None, batch.data_idx].to(mean.device).clamp_min(1e-8)
    return torch.mean(torch.abs(mean - truth) ** 2 / var + torch.log(var)).real


def bridge_from_config(
    case: NRCase,
    context: Any,
    config: dict[str, Any],
) -> NRBayesRouteBridge:
    b = config["bridge"]
    return NRBayesRouteBridge(
        context.grid,
        num_streams=case.num_streams,
        rank=int(b["rank"]),
        bank_rank=int(b["bank_rank"]),
        detector_iterations=int(b["detector_iterations"]),
        edge_mass=float(b["edge_mass"]),
        length_f=float(b["length_f"]),
        length_t=float(b["length_t"]),
        operator_seed=int(b["operator_seed"]),
    ).to(device=context.device)


def bridge_check(
    case: NRCase,
    config: dict[str, Any],
    device: torch.device,
    batch_size: int,
    ebno_db: float,
) -> tuple[dict[str, Any], NRBayesRouteBridge, Any]:
    context = build_nr_context(case, device)
    batch = context.sample(batch_size=batch_size, ebno_db=ebno_db)
    bridge = bridge_from_config(case, context, config)
    variants = [
        "proposed",
        "uncertainty_off_fixed_graph",
        "diagonal_posterior_fixed_graph",
        "mean_only_graph_fixed_cardinality",
        "random_graph_fixed_cardinality",
        "full_graph",
        "graph_off",
    ]
    outputs = bridge.forward_variants(batch, variants, random_seed=int(config["seed"]))
    proposed = outputs["proposed"]
    graph = graph_cardinality_report(outputs)
    posterior = posterior_psd_report(proposed)
    shape_pass = tuple(proposed["bit_logits"].shape) == tuple(batch.coded_bits.shape)
    coded = coded_bit_metrics(proposed["bit_logits"], batch.coded_bits)

    bridge.zero_grad(set_to_none=True)
    bce = F.binary_cross_entropy_with_logits(proposed["bit_logits"], batch.coded_bits)
    channel_nll = differentiable_channel_nll(proposed, batch)
    loss = bce + 0.01 * channel_nll
    loss.backward()
    gradients = {
        name: {
            "present": parameter.grad is not None,
            "finite": bool(
                parameter.grad is not None and torch.isfinite(parameter.grad).all().item()
            ),
            "norm": float(parameter.grad.norm().item()) if parameter.grad is not None else 0.0,
        }
        for name, parameter in bridge.named_parameters()
    }
    gradient_pass = bool(
        gradients
        and all(item["present"] and item["finite"] for item in gradients.values())
        and any(item["norm"] > 0.0 for item in gradients.values())
    )

    with torch.no_grad():
        decoded = decode_bridge(
            context.transmitter,
            proposed,
            batch.information_bits,
            num_bp_iter=4,
            device=device,
        )
    decode_valid = all(
        0.0 <= float(decoded[key]) <= 1.0
        for key in ("information_ber", "tbler", "crc_failure_rate")
    )
    finite_metrics = all(math.isfinite(float(value)) for value in coded.values())
    report = {
        "passed": bool(
            graph["passed"]
            and posterior["passed"]
            and shape_pass
            and gradient_pass
            and decode_valid
            and finite_metrics
        ),
        "case": case.__dict__,
        "num_streams": case.num_streams,
        "posterior": posterior,
        "graph_cardinality": graph,
        "logit_shape_pass": shape_pass,
        "logit_shape": list(proposed["bit_logits"].shape),
        "coded_metrics": coded,
        "bce": float(bce.item()),
        "channel_nll": float(channel_nll.item()),
        "gradients": gradients,
        "gradient_pass": gradient_pass,
        "tb_decoder_accepts_layer_demapped_bridge_llrs": decode_valid,
        "decoded_metrics": {
            key: value for key, value in decoded.items() if key not in {"b_hat", "crc"}
        },
        "edge_densities": {
            name: float(output["edge_density"].item()) for name, output in outputs.items()
        },
    }
    return report, bridge, context


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/gate1_nr_smoke.yaml")
    parser.add_argument("--out", default="outputs/gates")
    args = parser.parse_args()

    config_path = Path(args.config)
    config = load_yaml(config_path)
    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)

    import sionna
    import sionna.phy

    seed = int(config["seed"])
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    sionna.phy.config.seed = seed
    device = torch.device(str(config["device"]))
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda:0")
    sionna.phy.config.device = str(device)

    manifest = manifest_report(ROOT)
    gate0 = gate0_report(ROOT, str(config["required_gate0_classification"]))
    version_ok = str(getattr(sionna, "__version__", "")) == "2.0.1"
    cuda_ok = bool(torch.cuda.is_available() and device.type == "cuda")

    mapping_results: dict[str, Any] = {}
    for raw in config["mapping_cases"]:
        case = NRCase.from_mapping(raw)
        mapping_results[case.name] = mapping_case(
            case, device, int(config["batch_size"])
        )

    channel_results: dict[str, Any] = {}
    channel_cases: list[tuple[dict[str, Any], NRCase]] = []
    for raw in config["channel_cases"]:
        case = NRCase.from_mapping(raw)
        channel_cases.append((raw, case))
        channel_results[case.name] = channel_case(
            case,
            device,
            int(config["batch_size"]),
            float(config["ebno_db"]),
        )

    kbest_raw, kbest_case = next(
        (raw, case) for raw, case in channel_cases if bool(raw.get("run_kbest_smoke"))
    )
    kbest = kbest_check(
        kbest_case,
        device,
        int(config["batch_size"]),
        float(config["ebno_db"]),
        int(kbest_raw.get("kbest_k", 16)),
    )

    bridge_case = next(
        case
        for _, case in channel_cases
        if case.scenario.lower() == "umi" and case.num_layers_per_user > 1
    )
    bridge_report, source_bridge, source_context = bridge_check(
        bridge_case,
        config,
        device,
        int(config["batch_size"]),
        float(config["ebno_db"]),
    )

    type2_case = NRCase.from_mapping(config["mapping_cases"][1])
    type2_context = build_nr_context(type2_case, device)
    target_bridge = bridge_from_config(type2_case, type2_context, config)
    transfer = transfer_operator_parameters(source_bridge, target_bridge)
    target_batch = type2_context.sample(
        batch_size=1, ebno_db=float(config["ebno_db"])
    )
    with torch.no_grad():
        target_output = target_bridge(target_batch)
    target_finite = bool(
        torch.isfinite(target_output["bit_logits"]).all().item()
        and tuple(target_output["bit_logits"].shape) == tuple(target_batch.coded_bits.shape)
    )
    transfer["target_forward_finite_and_shape_valid"] = target_finite
    transfer["source_streams"] = source_context.grid.num_streams
    transfer["target_streams"] = type2_context.grid.num_streams
    transfer["passed"] = bool(transfer["passed"] and target_finite)

    checks = {
        "gate1_manifest_valid": bool(manifest["passed"]),
        "gate0_mechanism_supported": bool(gate0["passed"]),
        "sionna_2_0_1": version_ok,
        "cuda_compute_node": cuda_ok,
        "all_dmrs_mapping_cases": all(item["passed"] for item in mapping_results.values()),
        "explicit_user_layer_port_mapping": all(
            item["metadata"]["passed"] for item in mapping_results.values()
        ),
        "exact_reserved_re_and_layer_mapping": all(
            item["exact_grid"]["passed"] for item in mapping_results.values()
        ),
        "nr_tb_ldpc_roundtrip": all(
            item["codec_roundtrip"]["passed"] for item in mapping_results.values()
        ),
        "all_38901_channel_cases": all(item["passed"] for item in channel_results.values()),
        "standard_ls_and_perfect_csi_lmmse_paths": all(
            item["passed"] for item in channel_results.values()
        ),
        "kbest_path": bool(kbest["passed"]),
        "bayesroute_nr_bridge": bool(bridge_report["passed"]),
        "learned_operator_transfer_across_dmrs": bool(transfer["passed"]),
    }
    overall = all(checks.values())
    classification = "GATE1_NR_SMOKE_PASS" if overall else "GATE1_NR_SMOKE_BLOCKED"
    report = {
        "classification": classification,
        "gate1_revision": GATE1_NR_VERSION,
        "complete": True,
        "overall_pass": overall,
        "evidence_ready": overall,
        "publication_nr_ready": False,
        "pgca_agmp_baseline_included": False,
        "checks": checks,
        "manifest": manifest,
        "gate0": gate0,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "sionna": getattr(sionna, "__version__", "unknown"),
            "cuda_available": torch.cuda.is_available(),
            "device": str(device),
            "gpu_name": torch.cuda.get_device_name(device) if torch.cuda.is_available() else None,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "hostname": platform.node(),
        },
        "mapping_cases": mapping_results,
        "channel_cases": channel_results,
        "kbest": kbest,
        "bridge": bridge_report,
        "operator_transfer": transfer,
        "source_grid": {
            "case": source_context.case.name,
            "num_users": source_context.grid.num_users,
            "num_layers_per_user": source_context.grid.num_layers_per_user,
            "num_streams": source_context.grid.num_streams,
            "pilot_observations": source_context.grid.num_pilot_observations,
            "data_symbols_per_stream": source_context.grid.num_data_symbols,
            "phi_shape": list(source_context.grid.phi.shape),
        },
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
    }
    save_json(report, output_dir / "GATE1_NR_SMOKE.json")
    lines = [f"{name}: {'PASS' if passed else 'FAIL'}" for name, passed in checks.items()]
    lines.extend(
        [
            f"CLASSIFICATION: {classification}",
            f"GATE1_NR_EVIDENCE_READY: {'YES' if overall else 'NO'}",
            "PGCA_AGMP_BASELINE_INCLUDED: NO",
            "PUBLICATION_NR_READY: NO",
        ]
    )
    (output_dir / "GATE1_NR_SMOKE.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print("\n".join(lines))
    if not overall:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
