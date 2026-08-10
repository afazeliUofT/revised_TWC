#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import random
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bayesroute.models import (  # noqa: E402
    BayesRouteDetector,
    PosteriorOutput,
    coupling_matrix,
    coupling_selection_mask,
    edge_density,
)
from bayesroute.nr_gate1 import (  # noqa: E402
    GATE1_NR_VERSION,
    NRBayesRouteBridge,
    NRCase,
    build_nr_context,
    channel_metrics,
    coded_bit_metrics,
    decode_bridge,
    mask_as_kappa,
    package_contract_signature,
    run_standard_receiver,
    standard_receiver,
    transfer_operator_parameters,
)

DIAGNOSTIC_VERSION = "gate1_nr_failure_mode_diagnostic_v1"
SOURCE_CONTRACT_FILES = (
    "scripts/gate1_nr_failure_mode_diagnostic.py",
    "src/bayesroute/nr_gate1.py",
    "src/bayesroute/models.py",
    "src/bayesroute/channels.py",
    "src/bayesroute/qam.py",
    "src/bayesroute/config.py",
    "src/bayesroute/sionna_kbest_compat.py",
)
EXISTING_VARIANTS = (
    "proposed",
    "graph_off",
    "uncertainty_off_fixed_graph",
    "diagonal_posterior_fixed_graph",
    "mean_only_graph_fixed_cardinality",
    "random_graph_fixed_cardinality",
    "full_graph",
)


def load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected a mapping in {path}")
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


def set_all_seeds(seed: int) -> None:
    import sionna.phy

    random.seed(int(seed))
    np.random.seed(int(seed) % (2**32 - 1))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    sionna.phy.config.seed = int(seed)


def source_hashes() -> dict[str, str]:
    result: dict[str, str] = {}
    for relative in SOURCE_CONTRACT_FILES:
        path = ROOT / relative
        if not path.is_file():
            raise RuntimeError(f"Missing source-contract file: {relative}")
        result[relative] = sha256_file(path)
    return result


def expected_rows(config: dict[str, Any]) -> int:
    evaluation = config["evaluation"]
    return int(
        len(evaluation["cases"])
        * len(evaluation["ebno_db"])
        * int(evaluation["repetitions"])
        * len(evaluation["variants"])
    )


def verify_preliminary_evidence(config: dict[str, Any]) -> dict[str, Any]:
    gate_path = ROOT / "outputs/gates/GATE1_NR_PRELIMINARY_EVIDENCE.json"
    if not gate_path.is_file():
        raise RuntimeError("Missing Gate-1 preliminary evidence report")
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    required_classification = str(config["required_preliminary_classification"])
    if gate.get("classification") != required_classification:
        raise RuntimeError(
            f"Expected {required_classification}, found {gate.get('classification')}"
        )
    if gate.get("complete") is not True:
        raise RuntimeError("Gate-1 preliminary evidence is not complete")

    summary_path = ROOT / str(config["checkpoint"]["summary_path"])
    checkpoint_path = ROOT / str(config["checkpoint"]["checkpoint_path"])
    if not summary_path.is_file() or not checkpoint_path.is_file():
        raise RuntimeError("Missing preliminary summary or checkpoint")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    expected_sha = str(config["checkpoint"]["expected_sha256"])
    observed_sha = sha256_file(checkpoint_path)
    if observed_sha != expected_sha:
        raise RuntimeError(
            f"Checkpoint SHA-256 mismatch: expected {expected_sha}, got {observed_sha}"
        )
    if summary.get("best_checkpoint_sha256") != expected_sha:
        raise RuntimeError("Preliminary summary/checkpoint SHA-256 mismatch")
    if summary.get("complete") is not True or int(summary.get("steps", -1)) != 500:
        raise RuntimeError("Preliminary training summary is incomplete")
    return {
        "gate_path": str(gate_path),
        "summary_path": str(summary_path),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": observed_sha,
        "preliminary_classification": gate.get("classification"),
        "summary": summary,
    }


def bridge_from_config(
    case: NRCase,
    context: Any,
    config: dict[str, Any],
    *,
    detector_iterations: int | None = None,
) -> NRBayesRouteBridge:
    bridge = config["bridge"]
    return NRBayesRouteBridge(
        context.grid,
        num_streams=case.num_streams,
        rank=int(bridge["rank"]),
        bank_rank=int(bridge["bank_rank"]),
        detector_iterations=(
            int(bridge["detector_iterations"])
            if detector_iterations is None
            else int(detector_iterations)
        ),
        edge_mass=float(bridge["edge_mass"]),
        length_f=float(bridge["length_f"]),
        length_t=float(bridge["length_t"]),
        operator_seed=int(bridge["operator_seed"]),
    ).to(context.device)


def load_source_bridge(
    config: dict[str, Any],
    preliminary: dict[str, Any],
    device: torch.device,
) -> tuple[NRBayesRouteBridge, Any, NRCase]:
    source_mapping = preliminary["summary"]["source_case"]
    source_case = NRCase.from_mapping(source_mapping)
    source_context = build_nr_context(source_case, device)
    source_bridge = bridge_from_config(source_case, source_context, config)
    state = torch.load(
        preliminary["checkpoint_path"], map_location=device, weights_only=False
    )
    if "model" not in state:
        raise RuntimeError("Preliminary checkpoint has no model state")
    source_bridge.load_state_dict(state["model"], strict=True)
    source_bridge.eval()
    return source_bridge, source_context, source_case


def fit_temperature(
    logits: torch.Tensor,
    bits: torch.Tensor,
    *,
    minimum: float,
    maximum: float,
    points: int,
    max_samples: int,
) -> dict[str, float]:
    logits = logits.detach().float().reshape(-1)
    bits = bits.detach().float().reshape(-1)
    if logits.numel() > int(max_samples):
        indices = torch.linspace(
            0, logits.numel() - 1, int(max_samples), dtype=torch.float64
        ).round().long()
        logits = logits[indices]
        bits = bits[indices]
    grid = torch.logspace(
        math.log10(float(minimum)),
        math.log10(float(maximum)),
        int(points),
        dtype=torch.float32,
    )
    losses = torch.stack(
        [F.binary_cross_entropy_with_logits(logits / value, bits) for value in grid]
    )
    index = int(torch.argmin(losses).item())
    return {
        "temperature": float(grid[index].item()),
        "calibration_nll": float(losses[index].item()),
        "uncalibrated_nll": float(
            F.binary_cross_entropy_with_logits(logits, bits).item()
        ),
    }


def calibration_contract(
    config: dict[str, Any],
    config_path: Path,
    preliminary: dict[str, Any],
) -> dict[str, Any]:
    payload = {
        "version": DIAGNOSTIC_VERSION,
        "kind": "calibration",
        "config_sha256": sha256_file(config_path),
        "checkpoint_sha256": preliminary["checkpoint_sha256"],
        "source_sha256": source_hashes(),
        "calibration": config["calibration"],
        "cases": config["evaluation"]["cases"],
        "ebno_db": config["evaluation"]["ebno_db"],
        "bridge": config["bridge"],
    }
    payload["signature"] = package_contract_signature(payload)
    return payload


def fit_case_calibration(
    case: NRCase,
    case_index: int,
    config: dict[str, Any],
    source_bridge: NRBayesRouteBridge,
    device: torch.device,
) -> dict[str, Any]:
    calibration = config["calibration"]
    context = build_nr_context(case, device)
    bridge = bridge_from_config(case, context, config)
    transfer = transfer_operator_parameters(source_bridge, bridge)
    if not transfer["passed"]:
        raise RuntimeError(f"Could not transfer operator to {case.name}")
    bridge.eval()

    by_snr: dict[str, Any] = {}
    for snr_index, value in enumerate(config["evaluation"]["ebno_db"]):
        snr = float(value)
        proposed_logits: list[torch.Tensor] = []
        off_logits: list[torch.Tensor] = []
        bit_targets: list[torch.Tensor] = []
        normalized_errors: list[torch.Tensor] = []
        coverage_values: list[float] = []

        for rep in range(int(calibration["repetitions"])):
            seed = (
                int(config["seed"])
                + 1_000_000
                + 100_000 * case_index
                + 1_000 * snr_index
                + rep
            )
            set_all_seeds(seed)
            batch = context.sample(
                batch_size=int(calibration["batch_size"]), ebno_db=snr
            )
            with torch.no_grad():
                outputs = bridge.forward_variants(
                    batch,
                    ["proposed", "uncertainty_off_fixed_graph"],
                    random_seed=seed + 500_000,
                )
            proposed = outputs["proposed"]
            off = outputs["uncertainty_off_fixed_graph"]
            proposed_logits.append(proposed["bit_logits"].detach().cpu())
            off_logits.append(off["bit_logits"].detach().cpu())
            bit_targets.append(batch.coded_bits.detach().cpu())

            posterior = proposed["posterior"]
            mean = posterior.mean[..., batch.data_idx]
            truth = batch.h[..., batch.data_idx]
            var = posterior.var_diag[None, :, None, batch.data_idx].to(device)
            normalized_errors.append(
                (torch.abs(mean - truth) ** 2 / var.clamp_min(1e-8))
                .detach()
                .float()
                .cpu()
                .reshape(-1)
            )
            threshold = -math.log(0.05) * var
            coverage_values.append(
                float((torch.abs(mean - truth) ** 2 <= threshold).float().mean().item())
            )

        proposed_tensor = torch.cat([x.reshape(-1) for x in proposed_logits])
        off_tensor = torch.cat([x.reshape(-1) for x in off_logits])
        bits_tensor = torch.cat([x.reshape(-1) for x in bit_targets])
        normalized = torch.cat(normalized_errors)
        variance_scale = float(normalized.mean().item())
        variance_scale = float(
            min(
                max(variance_scale, float(calibration["variance_scale_min"])),
                float(calibration["variance_scale_max"]),
            )
        )
        proposed_temperature = fit_temperature(
            proposed_tensor,
            bits_tensor,
            minimum=float(calibration["temperature_min"]),
            maximum=float(calibration["temperature_max"]),
            points=int(calibration["temperature_points"]),
            max_samples=int(calibration["temperature_max_samples"]),
        )
        off_temperature = fit_temperature(
            off_tensor,
            bits_tensor,
            minimum=float(calibration["temperature_min"]),
            maximum=float(calibration["temperature_max"]),
            points=int(calibration["temperature_points"]),
            max_samples=int(calibration["temperature_max_samples"]),
        )
        by_snr[str(snr)] = {
            "variance_scale": variance_scale,
            "normalized_error_mean": float(normalized.mean().item()),
            "normalized_error_median": float(normalized.median().item()),
            "coverage95": float(np.mean(coverage_values)),
            "proposed": proposed_temperature,
            "uncertainty_off_fixed_graph": off_temperature,
            "calibration_samples": int(bits_tensor.numel()),
        }

    del bridge, context
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {"case": case.__dict__, "by_snr": by_snr}


def load_or_fit_calibration(
    config: dict[str, Any],
    config_path: Path,
    preliminary: dict[str, Any],
    source_bridge: NRBayesRouteBridge,
    device: torch.device,
) -> dict[str, Any]:
    path = ROOT / "outputs/reports/gate1_nr_failure_mode_calibration.json"
    contract = calibration_contract(config, config_path, preliminary)
    cases: dict[str, Any] = {}
    if path.is_file():
        value = json.loads(path.read_text(encoding="utf-8"))
        if value.get("contract", {}).get("signature") != contract["signature"]:
            raise RuntimeError("Calibration resume contract mismatch")
        if value.get("complete") is True:
            return value
        cases = dict(value.get("cases", {}))

    for case_index, raw in enumerate(config["evaluation"]["cases"]):
        case = NRCase.from_mapping(raw)
        if case.name in cases:
            continue
        cases[case.name] = fit_case_calibration(
            case, case_index, config, source_bridge, device
        )
        partial = {
            "version": DIAGNOSTIC_VERSION,
            "complete": False,
            "contract": contract,
            "cases": cases,
        }
        save_json(partial, path)

    result = {
        "version": DIAGNOSTIC_VERSION,
        "complete": True,
        "contract": contract,
        "cases": cases,
    }
    save_json(result, path)
    return result


def scale_posterior(posterior: PosteriorOutput, scale: float) -> PosteriorOutput:
    value = float(scale)
    return replace(
        posterior,
        var_diag=posterior.var_diag * value,
        local_cov=posterior.local_cov * value,
        latent_cov=posterior.latent_cov * value,
    )


def detector_output_with_fixed_graph(
    bridge: NRBayesRouteBridge,
    batch: Any,
    posterior: PosteriorOutput,
    graph_mask: torch.Tensor,
    *,
    use_uncertainty: bool,
) -> dict[str, Any]:
    bit_logits, symbol_logits, x_mean, x_var, observed_mask = bridge.detector(
        batch.y,
        posterior.mean,
        posterior.local_cov,
        batch.data_idx,
        batch.noise_var,
        kappa=mask_as_kappa(graph_mask),
        edge_mass=1.0,
        use_uncertainty=use_uncertainty,
    )
    if not torch.equal(observed_mask, graph_mask):
        raise RuntimeError("Fixed graph changed inside diagnostic detector")
    return {
        "bit_logits": bit_logits,
        "symbol_logits": symbol_logits,
        "x_mean": x_mean,
        "x_var": x_var,
        "posterior": posterior,
        "graph_mask": observed_mask,
        "reference_graph_mask": graph_mask,
        "edge_density": edge_density(observed_mask),
    }


def true_channel_output(
    detector: BayesRouteDetector,
    batch: Any,
    edge_mass: float,
) -> dict[str, Any]:
    streams = int(batch.h.shape[1])
    resource_elements = int(batch.h.shape[-1])
    zero_cov = torch.zeros(
        (streams, streams, resource_elements),
        dtype=torch.complex64,
        device=batch.h.device,
    )
    kappa = coupling_matrix(
        batch.h.detach(), zero_cov, batch.data_idx, batch.noise_var.detach()
    )
    bit_logits, symbol_logits, x_mean, x_var, graph_mask = detector(
        batch.y,
        batch.h,
        zero_cov,
        batch.data_idx,
        batch.noise_var,
        kappa=kappa,
        edge_mass=float(edge_mass),
        use_uncertainty=False,
    )
    return {
        "bit_logits": bit_logits,
        "symbol_logits": symbol_logits,
        "x_mean": x_mean,
        "x_var": x_var,
        "graph_mask": graph_mask,
        "edge_density": edge_density(graph_mask),
    }


def temperature_output(output: dict[str, Any], temperature: float) -> dict[str, Any]:
    result = dict(output)
    result["bit_logits"] = output["bit_logits"] / float(temperature)
    return result


def crc_disagreement(decoded: dict[str, Any], information_bits: torch.Tensor) -> float:
    bit_error = decoded["b_hat"] != information_bits
    block_error = bit_error.reshape(bit_error.shape[0], bit_error.shape[1], -1).any(-1)
    crc_failure = ~decoded["crc"].bool()
    if crc_failure.shape != block_error.shape:
        crc_failure = crc_failure.reshape(block_error.shape)
    return float((block_error != crc_failure).float().mean().item())


def custom_row(
    *,
    case: NRCase,
    variant: str,
    snr: float,
    rep: int,
    seed: int,
    output: dict[str, Any],
    batch: Any,
    decoded: dict[str, Any],
    contract_signature: str,
    temperature: float | None = None,
    variance_scale: float | None = None,
    detector_iterations: int | None = None,
    posterior_available: bool = True,
    trainable_parameters: int = 17,
) -> dict[str, Any]:
    coded = coded_bit_metrics(output["bit_logits"], batch.coded_bits)
    if posterior_available:
        channel = channel_metrics(output, batch)
    else:
        channel = {
            "channel_nmse": 0.0 if variant.startswith("true_channel") else float("nan"),
            "channel_marginal_nll": float("nan"),
            "channel_coverage95": float("nan"),
        }
    return {
        "case": case.name,
        "scenario": case.scenario,
        "dmrs_config_type": case.dmrs_config_type,
        "num_users": case.num_users,
        "num_layers_per_user": case.num_layers_per_user,
        "num_streams": case.num_streams,
        "variant": variant,
        "ebno_db": float(snr),
        "rep": int(rep),
        "eval_seed": int(seed),
        "information_ber": decoded["information_ber"],
        "tbler": decoded["tbler"],
        "crc_failure_rate": decoded["crc_failure_rate"],
        "crc_block_disagreement_rate": crc_disagreement(
            decoded, batch.information_bits
        ),
        **coded,
        **channel,
        "edge_density": float(output["edge_density"].item()),
        "temperature": float(temperature) if temperature is not None else float("nan"),
        "variance_scale": float(variance_scale) if variance_scale is not None else float("nan"),
        "detector_iterations": (
            int(detector_iterations) if detector_iterations is not None else 4
        ),
        "trainable_parameters": int(trainable_parameters),
        "diagnostic_contract_signature": contract_signature,
    }


def standard_row(
    *,
    case: NRCase,
    variant: str,
    snr: float,
    rep: int,
    seed: int,
    metrics: dict[str, Any],
    contract_signature: str,
) -> dict[str, Any]:
    return {
        "case": case.name,
        "scenario": case.scenario,
        "dmrs_config_type": case.dmrs_config_type,
        "num_users": case.num_users,
        "num_layers_per_user": case.num_layers_per_user,
        "num_streams": case.num_streams,
        "variant": variant,
        "ebno_db": float(snr),
        "rep": int(rep),
        "eval_seed": int(seed),
        "information_ber": metrics["information_ber"],
        "tbler": metrics["tbler"],
        "crc_failure_rate": metrics["crc_failure_rate"],
        "crc_block_disagreement_rate": float("nan"),
        "coded_ber": float("nan"),
        "coded_bit_nll": float("nan"),
        "coded_brier": float("nan"),
        "channel_nmse": float("nan"),
        "channel_marginal_nll": float("nan"),
        "channel_coverage95": float("nan"),
        "edge_density": float("nan"),
        "temperature": float("nan"),
        "variance_scale": float("nan"),
        "detector_iterations": float("nan"),
        "trainable_parameters": 0,
        "diagnostic_contract_signature": contract_signature,
    }


def append_rows_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    new = pd.DataFrame(rows)
    if path.is_file():
        old = pd.read_csv(path)
        if list(old.columns) != list(new.columns):
            raise RuntimeError("Diagnostic CSV column contract mismatch")
        combined = pd.concat([old, new], ignore_index=True)
    else:
        combined = new
    temporary = path.with_suffix(path.suffix + ".tmp")
    combined.to_csv(temporary, index=False)
    temporary.replace(path)


def diagnostic_contract(
    config: dict[str, Any],
    config_path: Path,
    preliminary: dict[str, Any],
    calibration: dict[str, Any],
) -> dict[str, Any]:
    payload = {
        "version": DIAGNOSTIC_VERSION,
        "config_sha256": sha256_file(config_path),
        "checkpoint_sha256": preliminary["checkpoint_sha256"],
        "calibration_signature": calibration["contract"]["signature"],
        "source_sha256": source_hashes(),
        "evaluation": config["evaluation"],
        "bridge": config["bridge"],
    }
    payload["signature"] = package_contract_signature(payload)
    return payload


def decode_custom_variants(
    *,
    context: Any,
    batch: Any,
    outputs: dict[str, dict[str, Any]],
    bp_iterations: int,
    device: torch.device,
) -> dict[str, dict[str, Any]]:
    from sionna.phy.nr import LayerDemapper, TBDecoder

    decoder = TBDecoder(
        context.transmitter._tb_encoder,
        num_bp_iter=int(bp_iterations),
        device=str(device),
    )
    demapper = LayerDemapper(
        context.transmitter._layer_mapper,
        num_bits_per_symbol=int(context.grid.bits_per_symbol),
        device=str(device),
    )
    decoded: dict[str, dict[str, Any]] = {}
    for name, output in outputs.items():
        with torch.no_grad():
            decoded[name] = decode_bridge(
                context.transmitter,
                output,
                batch.information_bits,
                num_bp_iter=int(bp_iterations),
                device=device,
                decoder=decoder,
                layer_demapper=demapper,
            )
    return decoded


def evaluate(
    config: dict[str, Any],
    config_path: Path,
    preliminary: dict[str, Any],
    calibration: dict[str, Any],
    source_bridge: NRBayesRouteBridge,
    device: torch.device,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    evaluation = config["evaluation"]
    contract = diagnostic_contract(config, config_path, preliminary, calibration)
    eval_dir = ROOT / "outputs/eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    raw_path = eval_dir / "gate1_nr_failure_mode_diagnostic.csv"
    contract_path = eval_dir / "gate1_nr_failure_mode_diagnostic_contract.json"

    if raw_path.is_file():
        if not contract_path.is_file():
            raise RuntimeError("Diagnostic CSV exists without contract")
        existing = json.loads(contract_path.read_text(encoding="utf-8"))
        if existing.get("signature") != contract["signature"]:
            raise RuntimeError("Diagnostic resume contract mismatch")
    else:
        save_json(contract, contract_path)

    done: set[tuple[str, str, float, int]] = set()
    if raw_path.is_file():
        old = pd.read_csv(raw_path)
        keys = ["case", "variant", "ebno_db", "rep"]
        if old[keys].duplicated().any():
            raise RuntimeError("Diagnostic CSV contains duplicate keys")
        for _, row in old.iterrows():
            done.add(
                (
                    str(row["case"]),
                    str(row["variant"]),
                    float(row["ebno_db"]),
                    int(row["rep"]),
                )
            )

    variants = [str(x) for x in evaluation["variants"]]
    for case_index, raw_case in enumerate(evaluation["cases"]):
        case = NRCase.from_mapping(raw_case)
        context = build_nr_context(case, device)
        bridge4 = bridge_from_config(case, context, config, detector_iterations=4)
        if not transfer_operator_parameters(source_bridge, bridge4)["passed"]:
            raise RuntimeError(f"Could not transfer trained operator to {case.name}")
        bridge4.eval()

        iteration_bridges: dict[int, NRBayesRouteBridge] = {}
        for iterations in (1, 2, 3):
            item = bridge_from_config(
                case, context, config, detector_iterations=iterations
            )
            if not transfer_operator_parameters(source_bridge, item)["passed"]:
                raise RuntimeError(
                    f"Could not transfer operator to {case.name}/iter{iterations}"
                )
            item.eval()
            iteration_bridges[iterations] = item

        untrained = bridge_from_config(case, context, config, detector_iterations=4)
        untrained.eval()
        true_detectors = {
            1: BayesRouteDetector(
                bits_per_symbol=int(context.grid.bits_per_symbol),
                n_iter=1,
                use_uncertainty=False,
            ).to(device),
            4: BayesRouteDetector(
                bits_per_symbol=int(context.grid.bits_per_symbol),
                n_iter=4,
                use_uncertainty=False,
            ).to(device),
        }
        ls_receiver = standard_receiver(context, perfect_csi=False, return_crc=True)
        perfect_receiver = standard_receiver(
            context, perfect_csi=True, return_crc=True
        )

        for snr_index, snr_value in enumerate(evaluation["ebno_db"]):
            snr = float(snr_value)
            cal = calibration["cases"][case.name]["by_snr"][str(snr)]
            for rep in range(int(evaluation["repetitions"])):
                missing = [
                    name
                    for name in variants
                    if (case.name, name, snr, rep) not in done
                ]
                if not missing:
                    continue
                seed = (
                    int(config["seed"])
                    + 10_000_000
                    + 100_000 * case_index
                    + 1_000 * snr_index
                    + rep
                )
                set_all_seeds(seed)
                batch = context.sample(
                    batch_size=int(evaluation["batch_size"]), ebno_db=snr
                )

                outputs: dict[str, dict[str, Any]] = {}
                existing_needed = [x for x in missing if x in EXISTING_VARIANTS]
                # Temperature and variance controls require the corresponding bases.
                if any(
                    x
                    in {
                        "proposed_temperature_calibrated",
                        "variance_scaled_proposed",
                    }
                    for x in missing
                ) and "proposed" not in existing_needed:
                    existing_needed.append("proposed")
                if (
                    "uncertainty_off_temperature_calibrated" in missing
                    and "uncertainty_off_fixed_graph" not in existing_needed
                ):
                    existing_needed.append("uncertainty_off_fixed_graph")
                if existing_needed:
                    with torch.no_grad():
                        outputs.update(
                            bridge4.forward_variants(
                                batch,
                                existing_needed,
                                random_seed=seed + 700_000,
                            )
                        )

                if "untrained_proposed" in missing:
                    with torch.no_grad():
                        outputs["untrained_proposed"] = untrained(batch)
                for iterations in (1, 2, 3):
                    name = f"proposed_iter{iterations}"
                    if name in missing:
                        with torch.no_grad():
                            outputs[name] = iteration_bridges[iterations](batch)
                if "true_channel_iter1" in missing:
                    with torch.no_grad():
                        outputs["true_channel_iter1"] = true_channel_output(
                            true_detectors[1],
                            batch,
                            float(config["bridge"]["edge_mass"]),
                        )
                if "true_channel_iter4" in missing:
                    with torch.no_grad():
                        outputs["true_channel_iter4"] = true_channel_output(
                            true_detectors[4],
                            batch,
                            float(config["bridge"]["edge_mass"]),
                        )
                if "proposed_temperature_calibrated" in missing:
                    outputs["proposed_temperature_calibrated"] = temperature_output(
                        outputs["proposed"],
                        float(cal["proposed"]["temperature"]),
                    )
                if "uncertainty_off_temperature_calibrated" in missing:
                    outputs[
                        "uncertainty_off_temperature_calibrated"
                    ] = temperature_output(
                        outputs["uncertainty_off_fixed_graph"],
                        float(
                            cal["uncertainty_off_fixed_graph"]["temperature"]
                        ),
                    )
                if "variance_scaled_proposed" in missing:
                    base = outputs["proposed"]
                    scaled = scale_posterior(
                        base["posterior"], float(cal["variance_scale"])
                    )
                    with torch.no_grad():
                        outputs["variance_scaled_proposed"] = (
                            detector_output_with_fixed_graph(
                                bridge4,
                                batch,
                                scaled,
                                base["reference_graph_mask"],
                                use_uncertainty=True,
                            )
                        )

                custom_missing = [
                    name
                    for name in missing
                    if name not in {"ls_lmmse", "perfect_csi_lmmse"}
                ]
                decoded = decode_custom_variants(
                    context=context,
                    batch=batch,
                    outputs={name: outputs[name] for name in custom_missing},
                    bp_iterations=int(evaluation["bp_iterations"]),
                    device=device,
                )

                rows: list[dict[str, Any]] = []
                for name in missing:
                    if name == "ls_lmmse":
                        with torch.no_grad():
                            metrics = run_standard_receiver(
                                ls_receiver,
                                batch,
                                batch.information_bits,
                                perfect_csi=False,
                            )
                        rows.append(
                            standard_row(
                                case=case,
                                variant=name,
                                snr=snr,
                                rep=rep,
                                seed=seed,
                                metrics=metrics,
                                contract_signature=contract["signature"],
                            )
                        )
                        continue
                    if name == "perfect_csi_lmmse":
                        with torch.no_grad():
                            metrics = run_standard_receiver(
                                perfect_receiver,
                                batch,
                                batch.information_bits,
                                perfect_csi=True,
                            )
                        rows.append(
                            standard_row(
                                case=case,
                                variant=name,
                                snr=snr,
                                rep=rep,
                                seed=seed,
                                metrics=metrics,
                                contract_signature=contract["signature"],
                            )
                        )
                        continue

                    temperature = None
                    variance_scale = None
                    iterations = 4
                    posterior_available = not name.startswith("true_channel")
                    if name == "proposed_temperature_calibrated":
                        temperature = float(cal["proposed"]["temperature"])
                    elif name == "uncertainty_off_temperature_calibrated":
                        temperature = float(
                            cal["uncertainty_off_fixed_graph"]["temperature"]
                        )
                    elif name == "variance_scaled_proposed":
                        variance_scale = float(cal["variance_scale"])
                    elif name.startswith("proposed_iter"):
                        iterations = int(name.rsplit("iter", 1)[1])
                    elif name.startswith("true_channel_iter"):
                        iterations = int(name.rsplit("iter", 1)[1])

                    rows.append(
                        custom_row(
                            case=case,
                            variant=name,
                            snr=snr,
                            rep=rep,
                            seed=seed,
                            output=outputs[name],
                            batch=batch,
                            decoded=decoded[name],
                            contract_signature=contract["signature"],
                            temperature=temperature,
                            variance_scale=variance_scale,
                            detector_iterations=iterations,
                            posterior_available=posterior_available,
                            trainable_parameters=(
                                0 if name.startswith("true_channel") else 17
                            ),
                        )
                    )

                append_rows_atomic(raw_path, rows)
                for row in rows:
                    done.add((case.name, str(row["variant"]), snr, rep))
                print(
                    json.dumps(
                        {
                            "case": case.name,
                            "ebno_db": snr,
                            "rep": rep,
                            "rows_committed": len(rows),
                            "completed_keys": len(done),
                        }
                    ),
                    flush=True,
                )

        del (
            bridge4,
            iteration_bridges,
            untrained,
            true_detectors,
            ls_receiver,
            perfect_receiver,
            context,
        )
        if device.type == "cuda":
            torch.cuda.empty_cache()

    df = pd.read_csv(raw_path)
    keys = ["case", "variant", "ebno_db", "rep"]
    unique = int(len(df.drop_duplicates(keys)))
    expected = expected_rows(config)
    complete = bool(len(df) == expected and unique == expected)
    if not complete:
        raise RuntimeError(
            f"Diagnostic incomplete: rows={len(df)}, unique={unique}, expected={expected}"
        )
    return df, {
        "contract": contract,
        "raw_csv": str(raw_path.relative_to(ROOT)),
        "rows": int(len(df)),
        "unique_rows": unique,
        "expected_rows": expected,
        "complete": complete,
    }


def paired_summary(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    keys = ["case", "ebno_db", "rep", "eval_seed"]
    reference = df[df["variant"] == "proposed"]
    rows: list[dict[str, Any]] = []
    for comparator in sorted(set(df["variant"]) - {"proposed"}):
        other = df[df["variant"] == comparator]
        merged = reference.merge(other, on=keys, suffixes=("_proposed", "_comparator"))
        for _, item in merged.iterrows():
            row = {
                "comparator": comparator,
                "case": item["case"],
                "ebno_db": float(item["ebno_db"]),
                "rep": int(item["rep"]),
                "eval_seed": int(item["eval_seed"]),
            }
            for metric in ("tbler", "information_ber", "coded_bit_nll"):
                a = item.get(f"{metric}_proposed")
                b = item.get(f"{metric}_comparator")
                row[f"{metric}_delta_proposed_minus_comparator"] = (
                    float(a - b) if pd.notna(a) and pd.notna(b) else float("nan")
                )
            rows.append(row)
    paired = pd.DataFrame(rows)

    summary_rows: list[dict[str, Any]] = []
    group_specs: list[tuple[str, Iterable[str]]] = [
        ("pooled", ["comparator"]),
        ("per_case", ["comparator", "case"]),
    ]
    for scope, columns in group_specs:
        for group_key, sub in paired.groupby(list(columns)):
            if not isinstance(group_key, tuple):
                group_key = (group_key,)
            row: dict[str, Any] = {"scope": scope, "pairs": int(len(sub))}
            for name, value in zip(columns, group_key):
                row[name] = value
            for metric in (
                "tbler_delta_proposed_minus_comparator",
                "information_ber_delta_proposed_minus_comparator",
                "coded_bit_nll_delta_proposed_minus_comparator",
            ):
                values = pd.to_numeric(sub[metric], errors="coerce").dropna()
                if values.empty:
                    continue
                mean = float(values.mean())
                std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
                half = 1.96 * std / math.sqrt(len(values))
                row[f"{metric}_mean"] = mean
                row[f"{metric}_ci95_low"] = mean - half
                row[f"{metric}_ci95_high"] = mean + half
            summary_rows.append(row)
    return paired, pd.DataFrame(summary_rows)


def aggregate_table(df: pd.DataFrame) -> pd.DataFrame:
    numeric = [
        "information_ber",
        "tbler",
        "crc_failure_rate",
        "crc_block_disagreement_rate",
        "coded_ber",
        "coded_bit_nll",
        "coded_brier",
        "channel_nmse",
        "channel_marginal_nll",
        "channel_coverage95",
        "edge_density",
        "temperature",
        "variance_scale",
    ]
    result = (
        df.groupby(["case", "scenario", "variant", "ebno_db"], as_index=False)[
            numeric
        ]
        .agg(["mean", "std", "count"])
    )
    result.columns = [
        "_".join(str(x) for x in item if str(x))
        for item in result.columns.to_flat_index()
    ]
    return result


def summary_value(
    summary: pd.DataFrame,
    comparator: str,
    metric: str,
    *,
    scope: str = "pooled",
    case: str | None = None,
    field: str = "mean",
) -> float:
    sub = summary[
        (summary["scope"] == scope) & (summary["comparator"] == comparator)
    ]
    if case is not None:
        sub = sub[sub["case"] == case]
    column = f"{metric}_{field}"
    if sub.empty or column not in sub.columns:
        return float("nan")
    return float(sub.iloc[0][column])


def mean_metric(
    df: pd.DataFrame,
    *,
    variant: str,
    metric: str,
    cases: list[str] | None = None,
    snrs: list[float] | None = None,
) -> float:
    sub = df[df["variant"] == variant]
    if cases is not None:
        sub = sub[sub["case"].isin(cases)]
    if snrs is not None:
        sub = sub[sub["ebno_db"].isin(snrs)]
    values = pd.to_numeric(sub[metric], errors="coerce").dropna()
    return float(values.mean()) if not values.empty else float("nan")


def classify(
    df: pd.DataFrame,
    paired_summary_table: pd.DataFrame,
    calibration: dict[str, Any],
) -> dict[str, Any]:
    multiuser_cases = sorted(
        df.loc[df["num_streams"] >= 4, "case"].drop_duplicates().tolist()
    )
    high_snrs = [6.0, 10.0]

    uncertainty_ci_high = summary_value(
        paired_summary_table,
        "uncertainty_off_fixed_graph",
        "tbler_delta_proposed_minus_comparator",
        field="ci95_high",
    )
    random_ci_high = summary_value(
        paired_summary_table,
        "random_graph_fixed_cardinality",
        "tbler_delta_proposed_minus_comparator",
        field="ci95_high",
    )
    graph_off_ci_high = summary_value(
        paired_summary_table,
        "graph_off",
        "tbler_delta_proposed_minus_comparator",
        field="ci95_high",
    )
    diagonal_ci_high = summary_value(
        paired_summary_table,
        "diagonal_posterior_fixed_graph",
        "tbler_delta_proposed_minus_comparator",
        field="ci95_high",
    )
    mean_graph_ci_high = summary_value(
        paired_summary_table,
        "mean_only_graph_fixed_cardinality",
        "tbler_delta_proposed_minus_comparator",
        field="ci95_high",
    )

    proposed_high = mean_metric(
        df,
        variant="proposed",
        metric="tbler",
        cases=multiuser_cases,
        snrs=high_snrs,
    )
    ls_high = mean_metric(
        df,
        variant="ls_lmmse",
        metric="tbler",
        cases=multiuser_cases,
        snrs=high_snrs,
    )
    true4_high = mean_metric(
        df,
        variant="true_channel_iter4",
        metric="tbler",
        cases=multiuser_cases,
        snrs=high_snrs,
    )
    true1_high = mean_metric(
        df,
        variant="true_channel_iter1",
        metric="tbler",
        cases=multiuser_cases,
        snrs=high_snrs,
    )
    perfect_high = mean_metric(
        df,
        variant="perfect_csi_lmmse",
        metric="tbler",
        cases=multiuser_cases,
        snrs=high_snrs,
    )
    iter1_high = mean_metric(
        df,
        variant="proposed_iter1",
        metric="tbler",
        cases=multiuser_cases,
        snrs=high_snrs,
    )
    iter3_high = mean_metric(
        df,
        variant="proposed_iter3",
        metric="tbler",
        cases=multiuser_cases,
        snrs=high_snrs,
    )
    temp_high = mean_metric(
        df,
        variant="proposed_temperature_calibrated",
        metric="tbler",
        cases=multiuser_cases,
        snrs=high_snrs,
    )
    variance_high = mean_metric(
        df,
        variant="variance_scaled_proposed",
        metric="tbler",
        cases=multiuser_cases,
        snrs=high_snrs,
    )
    untrained_nll = mean_metric(
        df, variant="untrained_proposed", metric="coded_bit_nll"
    )
    trained_nll = mean_metric(df, variant="proposed", metric="coded_bit_nll")

    calibration_scales: list[float] = []
    calibration_coverages: list[float] = []
    temperatures: list[float] = []
    for case_value in calibration["cases"].values():
        for record in case_value["by_snr"].values():
            calibration_scales.append(float(record["variance_scale"]))
            calibration_coverages.append(float(record["coverage95"]))
            temperatures.append(float(record["proposed"]["temperature"]))

    crc_values = pd.to_numeric(
        df.loc[~df["variant"].isin(["ls_lmmse", "perfect_csi_lmmse"]),
               "crc_block_disagreement_rate"],
        errors="coerce",
    ).dropna()
    max_crc_disagreement = float(crc_values.max()) if not crc_values.empty else 0.0
    mean_crc_disagreement = float(crc_values.mean()) if not crc_values.empty else 0.0

    software_checks = {
        "complete_rows": True,
        "all_core_metrics_finite": bool(
            np.isfinite(
                df[["information_ber", "tbler", "crc_failure_rate"]].to_numpy()
            ).all()
        ),
        "all_variants_present": bool(len(set(df["variant"])) == 18),
        "paired_seeds_complete": bool(
            df.groupby(["case", "ebno_db", "rep"])["variant"].nunique().min()
            == 18
        ),
        "mean_crc_block_disagreement_below_0p5pct": mean_crc_disagreement <= 0.005,
    }
    mechanism_checks = {
        "uncertainty_gain_reproduced": bool(
            math.isfinite(uncertainty_ci_high) and uncertainty_ci_high < 0.0
        ),
        "coupling_beats_random_reproduced": bool(
            math.isfinite(random_ci_high) and random_ci_high < 0.0
        ),
        "graph_beats_graph_off": bool(
            math.isfinite(graph_off_ci_high) and graph_off_ci_high < 0.0
        ),
        "full_cross_layer_covariance_needed": bool(
            math.isfinite(diagonal_ci_high) and diagonal_ci_high < 0.0
        ),
        "covariance_terms_in_graph_needed": bool(
            math.isfinite(mean_graph_ci_high) and mean_graph_ci_high < 0.0
        ),
        "training_improves_coded_nll": bool(
            math.isfinite(trained_nll)
            and math.isfinite(untrained_nll)
            and trained_nll < untrained_nll
        ),
    }
    failure_diagnoses = {
        "standard_ls_lmmse_outperforms_proposed_at_high_snr": bool(
            math.isfinite(proposed_high)
            and math.isfinite(ls_high)
            and proposed_high > ls_high + 0.02
        ),
        "true_channel_detector_far_from_perfect_lmmse": bool(
            math.isfinite(true4_high)
            and math.isfinite(perfect_high)
            and true4_high > perfect_high + 0.05
        ),
        "four_iterations_worse_than_one_with_true_channel": bool(
            math.isfinite(true4_high)
            and math.isfinite(true1_high)
            and true4_high > true1_high + 0.02
        ),
        "four_iterations_worse_than_one_with_posterior": bool(
            math.isfinite(proposed_high)
            and math.isfinite(iter1_high)
            and proposed_high > iter1_high + 0.02
        ),
        "temperature_scaling_materially_reduces_high_snr_tbler": bool(
            math.isfinite(proposed_high)
            and math.isfinite(temp_high)
            and temp_high < proposed_high - 0.02
        ),
        "variance_rescaling_materially_reduces_high_snr_tbler": bool(
            math.isfinite(proposed_high)
            and math.isfinite(variance_high)
            and variance_high < proposed_high - 0.02
        ),
        "posterior_variance_scale_mismatch": bool(
            any(scale < 0.8 or scale > 1.25 for scale in calibration_scales)
        ),
        "posterior_coverage_mismatch": bool(
            any(value < 0.92 or value > 0.98 for value in calibration_coverages)
        ),
        "llr_temperature_far_from_one": bool(
            any(value < 0.8 or value > 1.25 for value in temperatures)
        ),
    }

    if not all(software_checks.values()):
        classification = "GATE1_FAILURE_DIAGNOSTIC_BLOCKED"
        next_action = "REPAIR_DIAGNOSTIC_PIPELINE"
    elif (
        failure_diagnoses["true_channel_detector_far_from_perfect_lmmse"]
        or failure_diagnoses["four_iterations_worse_than_one_with_true_channel"]
    ):
        classification = "GATE1_DETECTOR_REPAIR_REQUIRED"
        next_action = "REPLACE_SOFT_PIC_WITH_DAMPED_EXTRINSIC_EP_OR_LMMSE_EP"
    elif (
        failure_diagnoses["posterior_variance_scale_mismatch"]
        or failure_diagnoses["posterior_coverage_mismatch"]
    ):
        classification = "GATE1_POSTERIOR_CALIBRATION_OR_OPERATOR_REPAIR_REQUIRED"
        next_action = "CALIBRATE_OR_CONDITION_STOCHASTIC_OPERATOR"
    elif failure_diagnoses[
        "standard_ls_lmmse_outperforms_proposed_at_high_snr"
    ]:
        classification = "GATE1_OPERATOR_OR_DETECTOR_REPAIR_REQUIRED"
        next_action = "SEPARATE_CHANNEL_AND_DETECTOR_GAPS_WITH_TRUE_CHANNEL_CONTROL"
    else:
        classification = "GATE1_READY_FOR_PUBLICATION_SCALE_DESIGN"
        next_action = "DESIGN_PUBLICATION_SCALE_CAMPAIGN"

    return {
        "classification": classification,
        "next_action": next_action,
        "software_checks": software_checks,
        "mechanism_checks": mechanism_checks,
        "failure_diagnoses": failure_diagnoses,
        "key_metrics": {
            "multiuser_high_snr_proposed_tbler": proposed_high,
            "multiuser_high_snr_ls_lmmse_tbler": ls_high,
            "multiuser_high_snr_true_channel_iter1_tbler": true1_high,
            "multiuser_high_snr_true_channel_iter4_tbler": true4_high,
            "multiuser_high_snr_perfect_csi_lmmse_tbler": perfect_high,
            "multiuser_high_snr_proposed_iter1_tbler": iter1_high,
            "multiuser_high_snr_proposed_iter3_tbler": iter3_high,
            "multiuser_high_snr_temperature_calibrated_tbler": temp_high,
            "multiuser_high_snr_variance_scaled_tbler": variance_high,
            "trained_coded_bit_nll": trained_nll,
            "untrained_coded_bit_nll": untrained_nll,
            "max_crc_block_disagreement_rate": max_crc_disagreement,
            "mean_crc_block_disagreement_rate": mean_crc_disagreement,
            "calibration_variance_scale_min": min(calibration_scales),
            "calibration_variance_scale_max": max(calibration_scales),
            "calibration_coverage95_min": min(calibration_coverages),
            "calibration_coverage95_max": max(calibration_coverages),
            "calibration_temperature_min": min(temperatures),
            "calibration_temperature_max": max(temperatures),
        },
    }


def make_plots(df: pd.DataFrame) -> list[str]:
    output_dir = ROOT / "outputs/plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    selected = [
        "proposed",
        "proposed_iter1",
        "proposed_temperature_calibrated",
        "variance_scaled_proposed",
        "true_channel_iter1",
        "true_channel_iter4",
        "ls_lmmse",
        "perfect_csi_lmmse",
    ]
    for case in sorted(df["case"].unique()):
        sub_case = df[df["case"] == case]
        fig, ax = plt.subplots(figsize=(7.4, 4.9))
        for variant in selected:
            sub = sub_case[sub_case["variant"] == variant]
            if sub.empty:
                continue
            curve = sub.groupby("ebno_db", as_index=False)["tbler"].mean()
            ax.semilogy(
                curve["ebno_db"],
                curve["tbler"].clip(lower=1e-4),
                marker="o",
                label=variant,
            )
        ax.set_xlabel("Eb/N0 [dB]")
        ax.set_ylabel("Decoded TBLER")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=6.7)
        ax.set_title(case)
        fig.tight_layout()
        path = output_dir / f"gate1_failure_{case}_tbler.png"
        fig.savefig(path, dpi=220)
        plt.close(fig)
        paths.append(str(path.relative_to(ROOT)))

    fig, ax = plt.subplots(figsize=(7.4, 4.9))
    for variant in (
        "proposed",
        "uncertainty_off_fixed_graph",
        "proposed_temperature_calibrated",
        "variance_scaled_proposed",
    ):
        sub = df[df["variant"] == variant]
        curve = sub.groupby("ebno_db", as_index=False)["coded_bit_nll"].mean()
        ax.plot(curve["ebno_db"], curve["coded_bit_nll"], marker="o", label=variant)
    ax.set_xlabel("Eb/N0 [dB]")
    ax.set_ylabel("Coded-bit NLL")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    path = output_dir / "gate1_failure_calibration_nll.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    paths.append(str(path.relative_to(ROOT)))
    return paths


def preflight(config: dict[str, Any], config_path: Path) -> dict[str, Any]:
    if str(config.get("diagnostic_revision")) != DIAGNOSTIC_VERSION:
        raise RuntimeError("Diagnostic revision mismatch")
    if str(config.get("gate1_revision")) != GATE1_NR_VERSION:
        raise RuntimeError("Gate-1 revision mismatch")
    preliminary = verify_preliminary_evidence(config)
    rows = expected_rows(config)
    if rows != 2304:
        raise RuntimeError(f"Expected 2304 rows, computed {rows}")
    if len(config["evaluation"]["variants"]) != 18:
        raise RuntimeError("Expected 18 diagnostic variants")
    if len(config["evaluation"]["cases"]) != 4:
        raise RuntimeError("Expected four diagnostic cases")
    report = {
        "passed": True,
        "diagnostic_revision": DIAGNOSTIC_VERSION,
        "gate1_revision": GATE1_NR_VERSION,
        "checkpoint_sha256": preliminary["checkpoint_sha256"],
        "expected_rows": rows,
        "source_contract_files": len(source_hashes()),
        "config_sha256": sha256_file(config_path),
        "publication_nr_ready": False,
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="configs/gate1_nr_failure_mode_diagnostic.yaml"
    )
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()

    config_path = ROOT / args.config
    config = load_yaml(config_path)
    preflight_report = preflight(config, config_path)
    if args.preflight_only:
        print(json.dumps(preflight_report, indent=2, sort_keys=True))
        return

    import sionna
    import sionna.phy

    device = torch.device(str(config["device"]))
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda:0")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("Gate-1 diagnostic requires a CUDA compute node")
    if str(getattr(sionna, "__version__", "")) != "2.0.1":
        raise SystemExit(f"Expected Sionna 2.0.1, found {sionna.__version__}")
    sionna.phy.config.device = str(device)
    set_all_seeds(int(config["seed"]))

    preliminary = verify_preliminary_evidence(config)
    source_bridge, source_context, source_case = load_source_bridge(
        config, preliminary, device
    )
    del source_context
    if device.type == "cuda":
        torch.cuda.empty_cache()
    calibration = load_or_fit_calibration(
        config, config_path, preliminary, source_bridge, device
    )
    df, evaluation = evaluate(
        config,
        config_path,
        preliminary,
        calibration,
        source_bridge,
        device,
    )

    paired, paired_summary_table = paired_summary(df)
    aggregate = aggregate_table(df)
    eval_dir = ROOT / "outputs/eval"
    report_dir = ROOT / "outputs/reports"
    gate_dir = ROOT / "outputs/gates"
    report_dir.mkdir(parents=True, exist_ok=True)
    gate_dir.mkdir(parents=True, exist_ok=True)
    paired_path = eval_dir / "gate1_nr_failure_mode_paired.csv"
    paired_summary_path = report_dir / "gate1_nr_failure_mode_paired_summary.csv"
    aggregate_path = eval_dir / "gate1_nr_failure_mode_aggregate.csv"
    paired.to_csv(paired_path, index=False)
    paired_summary_table.to_csv(paired_summary_path, index=False)
    aggregate.to_csv(aggregate_path, index=False)

    decision = classify(df, paired_summary_table, calibration)
    plots = make_plots(df)
    result = {
        **decision,
        "version": DIAGNOSTIC_VERSION,
        "complete": True,
        "publication_nr_ready": False,
        "pgca_agmp_baseline_included": False,
        "preflight": preflight_report,
        "preliminary": {
            key: value for key, value in preliminary.items() if key != "summary"
        },
        "source_case": source_case.__dict__,
        "calibration_report": "outputs/reports/gate1_nr_failure_mode_calibration.json",
        "evaluation": evaluation,
        "raw_csv": evaluation["raw_csv"],
        "aggregate_csv": str(aggregate_path.relative_to(ROOT)),
        "paired_csv": str(paired_path.relative_to(ROOT)),
        "paired_summary_csv": str(paired_summary_path.relative_to(ROOT)),
        "plots": plots,
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
    save_json(result, report_dir / "gate1_nr_failure_mode_diagnostic.json")
    save_json(result, gate_dir / "GATE1_NR_FAILURE_MODE_DIAGNOSTIC.json")

    lines: list[str] = []
    lines.extend(
        f"{name}: {'PASS' if value else 'FAIL'}"
        for name, value in result["software_checks"].items()
    )
    lines.extend(
        f"{name}: {'PASS' if value else 'FAIL'}"
        for name, value in result["mechanism_checks"].items()
    )
    lines.extend(
        f"{name}: {'YES' if value else 'NO'}"
        for name, value in result["failure_diagnoses"].items()
    )
    lines.extend(
        [
            f"CLASSIFICATION: {result['classification']}",
            f"NEXT_ACTION: {result['next_action']}",
            "PGCA_AGMP_BASELINE_INCLUDED: NO",
            "PUBLICATION_NR_READY: NO",
        ]
    )
    text = "\n".join(lines) + "\n"
    (gate_dir / "GATE1_NR_FAILURE_MODE_DIAGNOSTIC.txt").write_text(
        text, encoding="utf-8"
    )
    print(text, end="")

    del source_bridge
    if result["classification"] == "GATE1_FAILURE_DIAGNOSTIC_BLOCKED":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
