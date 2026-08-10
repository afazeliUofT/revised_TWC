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
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bayesroute.lmmse_ep import (  # noqa: E402
    DETECTOR_VERSION,
    DampedExtrinsicLMMSEDetector,
    full_directed_graph,
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
    channel_metrics,
    coded_bit_metrics,
    decode_bridge,
    fixed_cardinality_mask,
    package_contract_signature,
    run_standard_receiver,
    standard_receiver,
    transfer_operator_parameters,
)

SCREEN_VERSION = "gate1_nr_detector_repair_screen_v1"
REPAIR_REVISION = "gate1_nr_detector_repair_v1"
SMOKE_CLASSIFICATION = "GATE1_NR_DETECTOR_REPAIR_SMOKE_PASS"
CHECKPOINT_SHA256 = "ca3243386e3d0511236a3c2c68f0396df9d05b7dc7f4118a8748d150613d3576"
SOURCE_CONTRACT_FILES = (
    "src/bayesroute/lmmse_ep.py",
    "scripts/gate1_nr_detector_repair_screen.py",
    "configs/gate1_nr_detector_repair_screen.yaml",
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


def damping_token(value: float) -> str:
    return str(float(value)).replace(".", "p")


def candidate_name(iterations: int, damping: float) -> str:
    return f"delmmse_sparse_i{int(iterations)}_d{damping_token(damping)}"


def candidate_variants(config: dict[str, Any]) -> list[str]:
    return [
        candidate_name(iterations, damping)
        for iterations in config["screen"]["iterations"]
        for damping in config["screen"]["damping"]
    ]


def all_variants(config: dict[str, Any]) -> list[str]:
    return candidate_variants(config) + [
        "old_pic_posterior",
        "old_pic_true_channel",
        "delmmse_posterior_full_graph_i1",
        "delmmse_posterior_full_graph_i4_d0p5",
        "delmmse_posterior_graph_off_i4_d0p5",
        "delmmse_posterior_random_i4_d0p5",
        "delmmse_posterior_uncertainty_off_i4_d0p5",
        "delmmse_true_full_i1",
        "delmmse_true_sparse_i4_d0p5",
        "delmmse_true_full_i4_d0p5",
        "ls_lmmse",
        "perfect_csi_lmmse",
    ]


def expected_rows(config: dict[str, Any]) -> int:
    evaluation = config["evaluation"]
    return (
        len(evaluation["cases"])
        * len(evaluation["ebno_db"])
        * int(evaluation["repetitions"])
        * len(all_variants(config))
    )


def verify_preconditions(config: dict[str, Any]) -> dict[str, Any]:
    smoke_path = ROOT / "outputs/gates/GATE1_NR_DETECTOR_REPAIR_SMOKE.json"
    failure_path = ROOT / "outputs/gates/GATE1_NR_FAILURE_MODE_DIAGNOSTIC.json"
    checkpoint_path = ROOT / str(config["checkpoint_path"])
    revision_path = ROOT / "GATE1_NR_DETECTOR_REPAIR_REVISION.json"
    for path in (smoke_path, failure_path, checkpoint_path, revision_path):
        if not path.is_file():
            raise RuntimeError(f"Missing required file: {path}")
    smoke = json.loads(smoke_path.read_text(encoding="utf-8"))
    failure = json.loads(failure_path.read_text(encoding="utf-8"))
    revision = json.loads(revision_path.read_text(encoding="utf-8"))
    checkpoint_sha = sha256_file(checkpoint_path)
    checks = {
        "smoke_pass": (
            smoke.get("classification") == SMOKE_CLASSIFICATION
            and smoke.get("overall_pass") is True
            and smoke.get("screen_ready") is True
        ),
        "failure_diagnostic": (
            failure.get("classification") == "GATE1_DETECTOR_REPAIR_REQUIRED"
        ),
        "checkpoint": checkpoint_sha == CHECKPOINT_SHA256,
        "repair_revision": revision.get("revision") == REPAIR_REVISION,
        "gate1_revision": revision.get("gate1_revision") == GATE1_NR_VERSION,
        "detector_version": DETECTOR_VERSION == "damped_extrinsic_lmmse_v1",
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "checkpoint_sha256": checkpoint_sha,
    }


def bridge_from_config(case: NRCase, context: Any, config: dict[str, Any]) -> NRBayesRouteBridge:
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


def load_source_bridge(config: dict[str, Any], device: torch.device) -> tuple[NRBayesRouteBridge, Any]:
    summary = json.loads(
        (ROOT / "outputs/reports/gate1_nr_preliminary_train_summary.json").read_text(
            encoding="utf-8"
        )
    )
    source_case = NRCase.from_mapping(summary["source_case"])
    source_context = build_nr_context(source_case, device)
    source = bridge_from_config(source_case, source_context, config)
    state = torch.load(
        ROOT / str(config["checkpoint_path"]), map_location=device, weights_only=False
    )
    source.load_state_dict(state["model"], strict=True)
    source.eval()
    return source, source_context


def attach_posterior(
    output: dict[str, Any], posterior: Any, graph: torch.Tensor
) -> dict[str, Any]:
    result = dict(output)
    result["posterior"] = posterior
    result["reference_graph_mask"] = graph
    result["edge_density"] = edge_density(graph)
    return result


def random_fixed_cardinality_graph(
    reference: torch.Tensor, seed: int
) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    scores = torch.rand(reference.shape, generator=generator, dtype=torch.float32)
    return fixed_cardinality_mask(scores.to(reference.device), reference)


def true_channel_graphs(
    batch: Any, edge_mass: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
    sparse = coupling_selection_mask(kappa, float(edge_mass))
    full = full_directed_graph(
        int(batch.h.shape[0]), int(batch.data_idx.numel()), streams,
        device=batch.h.device,
    )
    return zero_cov, sparse, full


def decode_outputs(
    context: Any,
    batch: Any,
    outputs: dict[str, dict[str, Any]],
    *,
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
    result: dict[str, dict[str, Any]] = {}
    for name, output in outputs.items():
        result[name] = decode_bridge(
            context.transmitter,
            output,
            batch.information_bits,
            num_bp_iter=int(bp_iterations),
            device=device,
            decoder=decoder,
            layer_demapper=demapper,
        )
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
    posterior_available: bool,
) -> dict[str, Any]:
    coded = coded_bit_metrics(output["bit_logits"], batch.coded_bits)
    channel = (
        channel_metrics(output, batch)
        if posterior_available
        else {
            "channel_nmse": 0.0 if "true" in variant else float("nan"),
            "channel_marginal_nll": float("nan"),
            "channel_coverage95": float("nan"),
        }
    )
    return {
        "case": case.name,
        "scenario": case.scenario,
        "num_streams": case.num_streams,
        "variant": variant,
        "ebno_db": float(snr),
        "rep": int(rep),
        "eval_seed": int(seed),
        "information_ber": decoded["information_ber"],
        "tbler": decoded["tbler"],
        "crc_failure_rate": decoded["crc_failure_rate"],
        "crc_block_disagreement_rate": crc_disagreement(decoded, batch.information_bits),
        **coded,
        **channel,
        "edge_density": float(output["edge_density"].item()),
        "detector_version": str(output.get("detector_version", "old_soft_pic")),
        "detector_iterations": int(output.get("detector_iterations", 4)),
        "damping": float(output.get("damping", float("nan"))),
        "covariance_mode": str(output.get("covariance_mode", "legacy")),
        "contract_signature": contract_signature,
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
        "detector_version": variant,
        "detector_iterations": float("nan"),
        "damping": float("nan"),
        "covariance_mode": "standard",
        "contract_signature": contract_signature,
    }


def append_rows_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    new = pd.DataFrame(rows)
    if path.is_file():
        old = pd.read_csv(path)
        if list(old.columns) != list(new.columns):
            raise RuntimeError("Detector-repair CSV column contract mismatch")
        combined = pd.concat([old, new], ignore_index=True)
    else:
        combined = new
    temporary = path.with_suffix(path.suffix + ".tmp")
    combined.to_csv(temporary, index=False)
    temporary.replace(path)


def contract(config: dict[str, Any], config_path: Path, preflight: dict[str, Any]) -> dict[str, Any]:
    payload = {
        "version": SCREEN_VERSION,
        "repair_revision": REPAIR_REVISION,
        "checkpoint_sha256": preflight["checkpoint_sha256"],
        "config_sha256": sha256_file(config_path),
        "source_sha256": source_hashes(),
        "variants": all_variants(config),
        "evaluation": config["evaluation"],
        "screen": config["screen"],
        "bridge": config["bridge"],
    }
    payload["signature"] = package_contract_signature(payload)
    return payload


def evaluate(
    config: dict[str, Any],
    config_path: Path,
    preflight: dict[str, Any],
    source: NRBayesRouteBridge,
    device: torch.device,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    evaluation = config["evaluation"]
    variants = all_variants(config)
    experiment_contract = contract(config, config_path, preflight)
    eval_dir = ROOT / "outputs/eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    raw_path = eval_dir / "gate1_nr_detector_repair_screen.csv"
    contract_path = eval_dir / "gate1_nr_detector_repair_screen_contract.json"
    if raw_path.is_file():
        if not contract_path.is_file():
            raise RuntimeError("Detector-repair CSV exists without contract")
        old_contract = json.loads(contract_path.read_text(encoding="utf-8"))
        if old_contract.get("signature") != experiment_contract["signature"]:
            raise RuntimeError("Detector-repair resume contract mismatch")
    else:
        save_json(experiment_contract, contract_path)

    done: set[tuple[str, str, float, int]] = set()
    if raw_path.is_file():
        old = pd.read_csv(raw_path)
        keys = ["case", "variant", "ebno_db", "rep"]
        if old[keys].duplicated().any():
            raise RuntimeError("Detector-repair CSV contains duplicate keys")
        for _, row in old.iterrows():
            done.add((str(row["case"]), str(row["variant"]), float(row["ebno_db"]), int(row["rep"])))

    candidates = candidate_variants(config)
    for case_index, raw_case in enumerate(evaluation["cases"]):
        case = NRCase.from_mapping(raw_case)
        context = build_nr_context(case, device)
        bridge = bridge_from_config(case, context, config)
        if not transfer_operator_parameters(source, bridge)["passed"]:
            raise RuntimeError(f"Could not transfer operator to {case.name}")
        bridge.eval()

        candidate_detectors: dict[str, DampedExtrinsicLMMSEDetector] = {}
        for iterations in config["screen"]["iterations"]:
            for damping in config["screen"]["damping"]:
                name = candidate_name(int(iterations), float(damping))
                candidate_detectors[name] = DampedExtrinsicLMMSEDetector(
                    int(context.grid.bits_per_symbol),
                    n_iter=int(iterations),
                    damping=float(damping),
                    covariance_mode="diagonal",
                ).to(device)
        default_detector = DampedExtrinsicLMMSEDetector(
            int(context.grid.bits_per_symbol),
            n_iter=4,
            damping=0.5,
            covariance_mode="diagonal",
        ).to(device)
        one_step_detector = DampedExtrinsicLMMSEDetector(
            int(context.grid.bits_per_symbol),
            n_iter=1,
            damping=1.0,
            covariance_mode="diagonal",
        ).to(device)
        old_true_detector = BayesRouteDetector(
            int(context.grid.bits_per_symbol), n_iter=4, use_uncertainty=False
        ).to(device)
        ls_receiver = standard_receiver(context, perfect_csi=False, return_crc=True)
        perfect_receiver = standard_receiver(context, perfect_csi=True, return_crc=True)

        for snr_index, value in enumerate(evaluation["ebno_db"]):
            snr = float(value)
            for rep in range(int(evaluation["repetitions"])):
                missing = [
                    name for name in variants
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
                with torch.inference_mode():
                    old_posterior = bridge.forward_variant(
                        batch, "proposed", random_seed=seed + 800_000
                    )
                    posterior = old_posterior["posterior"]
                    reference = old_posterior["reference_graph_mask"]
                    full_graph = full_directed_graph(
                        int(batch.y.shape[0]), int(batch.data_idx.numel()),
                        case.num_streams, device=device
                    )
                    off_graph = torch.zeros_like(reference)
                    random_graph = random_fixed_cardinality_graph(
                        reference, seed + 900_000
                    )
                    zero_cov, true_sparse, true_full = true_channel_graphs(
                        batch, float(config["bridge"]["edge_mass"])
                    )

                    outputs: dict[str, dict[str, Any]] = {}
                    if "old_pic_posterior" in missing:
                        outputs["old_pic_posterior"] = old_posterior
                    if "old_pic_true_channel" in missing:
                        kappa_true = coupling_matrix(
                            batch.h.detach(), zero_cov, batch.data_idx,
                            batch.noise_var.detach()
                        )
                        bit, sym, mean, var, mask = old_true_detector(
                            batch.y, batch.h, zero_cov, batch.data_idx,
                            batch.noise_var, kappa=kappa_true,
                            edge_mass=float(config["bridge"]["edge_mass"]),
                            use_uncertainty=False,
                        )
                        outputs["old_pic_true_channel"] = {
                            "bit_logits": bit,
                            "symbol_logits": sym,
                            "x_mean": mean,
                            "x_var": var,
                            "graph_mask": mask,
                            "edge_density": edge_density(mask),
                        }

                    for name in candidates:
                        if name not in missing:
                            continue
                        result = candidate_detectors[name](
                            batch.y, posterior.mean, posterior.local_cov,
                            batch.data_idx, batch.noise_var, reference
                        )
                        outputs[name] = attach_posterior(
                            result, posterior, reference
                        )

                    one_step_specs = {
                        "delmmse_posterior_full_graph_i1": (
                            batch.y, posterior.mean, posterior.local_cov,
                            full_graph, "diagonal", True
                        ),
                        "delmmse_true_full_i1": (
                            batch.y, batch.h, zero_cov, true_full, "none", False
                        ),
                    }
                    for name, (obs, mean, cov, graph, mode, has_posterior) in one_step_specs.items():
                        if name not in missing:
                            continue
                        result = one_step_detector(
                            obs, mean, cov, batch.data_idx, batch.noise_var,
                            graph, covariance_mode=mode
                        )
                        outputs[name] = (
                            attach_posterior(result, posterior, graph)
                            if has_posterior else result
                        )

                    default_specs = {
                        "delmmse_posterior_full_graph_i4_d0p5": (
                            posterior.mean, posterior.local_cov, full_graph,
                            "diagonal", True
                        ),
                        "delmmse_posterior_graph_off_i4_d0p5": (
                            posterior.mean, posterior.local_cov, off_graph,
                            "diagonal", True
                        ),
                        "delmmse_posterior_random_i4_d0p5": (
                            posterior.mean, posterior.local_cov, random_graph,
                            "diagonal", True
                        ),
                        "delmmse_posterior_uncertainty_off_i4_d0p5": (
                            posterior.mean, posterior.local_cov, reference,
                            "none", True
                        ),
                        "delmmse_true_sparse_i4_d0p5": (
                            batch.h, zero_cov, true_sparse, "none", False
                        ),
                        "delmmse_true_full_i4_d0p5": (
                            batch.h, zero_cov, true_full, "none", False
                        ),
                    }
                    for name, (mean, cov, graph, mode, has_posterior) in default_specs.items():
                        if name not in missing:
                            continue
                        result = default_detector(
                            batch.y, mean, cov, batch.data_idx,
                            batch.noise_var, graph, covariance_mode=mode
                        )
                        outputs[name] = (
                            attach_posterior(result, posterior, graph)
                            if has_posterior else result
                        )

                custom_missing = [
                    name for name in missing
                    if name not in {"ls_lmmse", "perfect_csi_lmmse"}
                ]
                with torch.inference_mode():
                    decoded = decode_outputs(
                        context, batch,
                        {name: outputs[name] for name in custom_missing},
                        bp_iterations=int(evaluation["bp_iterations"]),
                        device=device,
                    )
                rows: list[dict[str, Any]] = []
                for name in missing:
                    if name == "ls_lmmse":
                        with torch.inference_mode():
                            metrics = run_standard_receiver(
                                ls_receiver, batch, batch.information_bits,
                                perfect_csi=False
                            )
                        rows.append(standard_row(
                            case=case, variant=name, snr=snr, rep=rep,
                            seed=seed, metrics=metrics,
                            contract_signature=experiment_contract["signature"]
                        ))
                    elif name == "perfect_csi_lmmse":
                        with torch.inference_mode():
                            metrics = run_standard_receiver(
                                perfect_receiver, batch, batch.information_bits,
                                perfect_csi=True
                            )
                        rows.append(standard_row(
                            case=case, variant=name, snr=snr, rep=rep,
                            seed=seed, metrics=metrics,
                            contract_signature=experiment_contract["signature"]
                        ))
                    else:
                        posterior_available = name not in {
                            "old_pic_true_channel",
                            "delmmse_true_full_i1",
                            "delmmse_true_sparse_i4_d0p5",
                            "delmmse_true_full_i4_d0p5",
                        }
                        rows.append(custom_row(
                            case=case, variant=name, snr=snr, rep=rep,
                            seed=seed, output=outputs[name], batch=batch,
                            decoded=decoded[name],
                            contract_signature=experiment_contract["signature"],
                            posterior_available=posterior_available,
                        ))
                append_rows_atomic(raw_path, rows)
                for row in rows:
                    done.add((case.name, str(row["variant"]), snr, rep))
                print(json.dumps({
                    "case": case.name,
                    "ebno_db": snr,
                    "rep": rep,
                    "rows_committed": len(rows),
                    "completed_keys": len(done),
                }), flush=True)

        del (
            bridge, context, candidate_detectors, default_detector,
            one_step_detector, old_true_detector, ls_receiver, perfect_receiver
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
            f"Detector-repair screen incomplete: rows={len(df)}, unique={unique}, expected={expected}"
        )
    return df, {
        "complete": complete,
        "rows": int(len(df)),
        "unique_rows": unique,
        "expected_rows": expected,
        "raw_csv": str(raw_path.relative_to(ROOT)),
        "contract": experiment_contract,
    }


def mean_metric(
    df: pd.DataFrame,
    *,
    variant: str,
    metric: str,
    reps: list[int],
    multiuser_only: bool = True,
    high_snr_only: bool = True,
) -> float:
    sub = df[(df["variant"] == variant) & (df["rep"].isin(reps))]
    if multiuser_only:
        sub = sub[sub["num_streams"] >= 4]
    if high_snr_only:
        sub = sub[sub["ebno_db"].isin([6.0, 10.0])]
    values = pd.to_numeric(sub[metric], errors="coerce").dropna()
    return float(values.mean()) if not values.empty else float("nan")


def paired_delta(
    df: pd.DataFrame,
    reference: str,
    comparator: str,
    metric: str,
    reps: list[int],
) -> dict[str, float]:
    keys = ["case", "ebno_db", "rep", "eval_seed"]
    a = df[(df["variant"] == reference) & (df["rep"].isin(reps))]
    b = df[(df["variant"] == comparator) & (df["rep"].isin(reps))]
    merged = a.merge(b, on=keys, suffixes=("_a", "_b"))
    values = pd.to_numeric(merged[f"{metric}_a"], errors="coerce") - pd.to_numeric(
        merged[f"{metric}_b"], errors="coerce"
    )
    values = values.dropna()
    mean = float(values.mean()) if len(values) else float("nan")
    std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    half = 1.96 * std / math.sqrt(max(len(values), 1))
    return {
        "pairs": int(len(values)),
        "mean": mean,
        "ci95_low": mean - half,
        "ci95_high": mean + half,
    }


def select_and_classify(
    df: pd.DataFrame,
    config: dict[str, Any],
) -> dict[str, Any]:
    selection_reps = [int(x) for x in config["screen"]["selection_reps"]]
    holdout_reps = [int(x) for x in config["screen"]["holdout_reps"]]
    candidates = candidate_variants(config)
    ranking: list[dict[str, Any]] = []
    for name in candidates:
        ranking.append({
            "variant": name,
            "selection_tbler": mean_metric(
                df, variant=name, metric="tbler", reps=selection_reps
            ),
            "selection_information_ber": mean_metric(
                df, variant=name, metric="information_ber", reps=selection_reps
            ),
            "selection_coded_nll": mean_metric(
                df, variant=name, metric="coded_bit_nll", reps=selection_reps
            ),
        })
    ranking.sort(key=lambda item: (
        item["selection_tbler"],
        item["selection_information_ber"],
        item["selection_coded_nll"],
        item["variant"],
    ))
    selected = ranking[0]["variant"]

    selected_high = mean_metric(
        df, variant=selected, metric="tbler", reps=holdout_reps
    )
    old_high = mean_metric(
        df, variant="old_pic_posterior", metric="tbler", reps=holdout_reps
    )
    ls_high = mean_metric(
        df, variant="ls_lmmse", metric="tbler", reps=holdout_reps
    )
    perfect_high = mean_metric(
        df, variant="perfect_csi_lmmse", metric="tbler", reps=holdout_reps
    )
    true_one = mean_metric(
        df, variant="delmmse_true_full_i1", metric="tbler",
        reps=holdout_reps
    )
    true_new = mean_metric(
        df, variant="delmmse_true_full_i4_d0p5", metric="tbler",
        reps=holdout_reps
    )
    true_old = mean_metric(
        df, variant="old_pic_true_channel", metric="tbler", reps=holdout_reps
    )

    selected_6 = mean_metric(
        df[df["ebno_db"] == 6.0], variant=selected, metric="tbler",
        reps=holdout_reps, high_snr_only=False
    )
    selected_10 = mean_metric(
        df[df["ebno_db"] == 10.0], variant=selected, metric="tbler",
        reps=holdout_reps, high_snr_only=False
    )

    software_checks = {
        "complete_rows": True,
        "all_metrics_finite": bool(
            np.isfinite(df[["information_ber", "tbler", "crc_failure_rate"]].to_numpy()).all()
        ),
        "all_variants_present": len(set(df["variant"])) == len(all_variants(config)),
        "paired_seed_batches": bool(
            df.groupby(["case", "ebno_db", "rep"])["variant"].nunique().min()
            == len(all_variants(config))
        ),
        "selection_holdout_disjoint": not bool(set(selection_reps) & set(holdout_reps)),
    }
    scientific_checks = {
        "true_channel_one_step_matches_lmmse": bool(
            true_one <= perfect_high + 0.03
        ),
        "true_channel_iterative_repair_beats_old_pic": bool(
            true_new <= true_old - 0.10
        ),
        "true_channel_iterations_not_harmful": bool(
            true_new <= true_one + 0.02
        ),
        "posterior_detector_improves_old_pic": bool(
            selected_high <= old_high - 0.05
        ),
        "posterior_detector_within_0p10_of_ls": bool(
            selected_high <= ls_high + 0.10
        ),
        "no_high_snr_reversal": bool(selected_10 <= selected_6 + 0.05),
    }

    if not all(software_checks.values()):
        classification = "GATE1_DETECTOR_REPAIR_SCREEN_BLOCKED"
        next_action = "REPAIR_SCREEN_PIPELINE"
    elif not scientific_checks["true_channel_one_step_matches_lmmse"]:
        classification = "GATE1_LMMSE_REPAIR_ALGORITHM_NOT_YET_VALID"
        next_action = "AUDIT_SPATIAL_LMMSE_AND_LLR_EQUATIONS"
    elif not scientific_checks["true_channel_iterations_not_harmful"]:
        classification = "GATE1_LMMSE_DAMPING_OR_EXTRINSIC_UPDATE_REPAIR_REQUIRED"
        next_action = "REVISE_DAMPING_AND_CAVITY_MOMENT_UPDATE"
    elif not scientific_checks["true_channel_iterative_repair_beats_old_pic"]:
        classification = "GATE1_LMMSE_REPAIR_INSUFFICIENT"
        next_action = "TEST_EXPECTATION_PROPAGATION_PRECISION_SUBTRACTION"
    elif not scientific_checks["posterior_detector_improves_old_pic"]:
        classification = "GATE1_DETECTOR_REPAIRED_OPERATOR_NEXT"
        next_action = "CONDITION_OR_RETRAIN_POSTERIOR_OPERATOR_FOR_NR_CHANNEL_FAMILIES"
    elif not scientific_checks["posterior_detector_within_0p10_of_ls"]:
        classification = "GATE1_DETECTOR_REPAIR_PARTIAL"
        next_action = "JOINTLY_TUNE_POSTERIOR_OPERATOR_AND_LMMSE_MESSAGE_DETECTOR"
    else:
        classification = "GATE1_DETECTOR_REPAIR_SUPPORTED"
        next_action = "RUN_CLEAN_SELECTED_DETECTOR_ABLATIONS_ON_NEW_SEEDS"

    comparisons = {}
    for comparator in (
        "old_pic_posterior",
        "ls_lmmse",
        "perfect_csi_lmmse",
    ):
        comparisons[comparator] = {
            metric: paired_delta(
                df, selected, comparator, metric, holdout_reps
            )
            for metric in ("tbler", "information_ber")
        }

    return {
        "classification": classification,
        "next_action": next_action,
        "selected_variant": selected,
        "selection_ranking": ranking,
        "selection_reps": selection_reps,
        "holdout_reps": holdout_reps,
        "software_checks": software_checks,
        "scientific_checks": scientific_checks,
        "holdout_high_snr_metrics": {
            "selected_tbler": selected_high,
            "old_pic_posterior_tbler": old_high,
            "ls_lmmse_tbler": ls_high,
            "perfect_csi_lmmse_tbler": perfect_high,
            "new_true_channel_one_step_tbler": true_one,
            "new_true_channel_iterative_tbler": true_new,
            "old_true_channel_pic_tbler": true_old,
            "selected_tbler_6db": selected_6,
            "selected_tbler_10db": selected_10,
        },
        "holdout_paired_comparisons": comparisons,
    }


def aggregate_and_plot(df: pd.DataFrame, selected: str) -> tuple[str, list[str]]:
    eval_dir = ROOT / "outputs/eval"
    report_dir = ROOT / "outputs/reports"
    plot_dir = ROOT / "outputs/plots"
    report_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)
    numeric = [
        "information_ber", "tbler", "crc_failure_rate", "coded_ber",
        "coded_bit_nll", "coded_brier", "channel_nmse",
        "channel_coverage95", "edge_density",
    ]
    aggregate = (
        df.groupby(["case", "scenario", "variant", "ebno_db"], as_index=False)[numeric]
        .agg(["mean", "std", "count"])
    )
    aggregate.columns = [
        "_".join(str(x) for x in item if str(x))
        for item in aggregate.columns.to_flat_index()
    ]
    aggregate_path = eval_dir / "gate1_nr_detector_repair_aggregate.csv"
    aggregate.to_csv(aggregate_path, index=False)

    paths: list[str] = []
    selected_variants = [
        selected,
        "old_pic_posterior",
        "delmmse_true_full_i1",
        "delmmse_true_full_i4_d0p5",
        "old_pic_true_channel",
        "ls_lmmse",
        "perfect_csi_lmmse",
    ]
    for case in sorted(df["case"].unique()):
        fig, ax = plt.subplots(figsize=(7.4, 4.9))
        for variant in selected_variants:
            sub = df[(df["case"] == case) & (df["variant"] == variant)]
            if sub.empty:
                continue
            curve = sub.groupby("ebno_db", as_index=False)["tbler"].mean()
            ax.semilogy(
                curve["ebno_db"], curve["tbler"].clip(lower=1e-4),
                marker="o", label=variant
            )
        ax.set_xlabel("Eb/N0 [dB]")
        ax.set_ylabel("Decoded TBLER")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=6.5)
        ax.set_title(case)
        fig.tight_layout()
        path = plot_dir / f"gate1_detector_repair_{case}_tbler.png"
        fig.savefig(path, dpi=220)
        plt.close(fig)
        paths.append(str(path.relative_to(ROOT)))
    return str(aggregate_path.relative_to(ROOT)), paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="configs/gate1_nr_detector_repair_screen.yaml"
    )
    args = parser.parse_args()
    config_path = ROOT / args.config
    config = load_yaml(config_path)
    preflight = verify_preconditions(config)
    if not preflight["passed"]:
        raise SystemExit(f"BLOCKED: detector-repair screen preflight failed: {preflight}")

    import sionna
    import sionna.phy

    if str(getattr(sionna, "__version__", "")) != "2.0.1":
        raise SystemExit("Gate-1 detector repair requires Sionna 2.0.1")
    device = torch.device(str(config["device"]))
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda:0")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("Detector-repair screen requires a CUDA compute node")
    sionna.phy.config.device = str(device)

    source, source_context = load_source_bridge(config, device)
    df, evaluation = evaluate(
        config, config_path, preflight, source, device
    )
    decision = select_and_classify(df, config)
    aggregate_csv, plots = aggregate_and_plot(df, decision["selected_variant"])

    result = {
        "version": SCREEN_VERSION,
        "repair_revision": REPAIR_REVISION,
        "detector_version": DETECTOR_VERSION,
        "complete": evaluation["complete"],
        "publication_nr_ready": False,
        "preflight": preflight,
        "evaluation": evaluation,
        "aggregate_csv": aggregate_csv,
        "plots": plots,
        **decision,
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
    report_dir = ROOT / "outputs/reports"
    gate_dir = ROOT / "outputs/gates"
    report_dir.mkdir(parents=True, exist_ok=True)
    gate_dir.mkdir(parents=True, exist_ok=True)
    save_json(result, report_dir / "gate1_nr_detector_repair_screen.json")
    save_json(result, gate_dir / "GATE1_NR_DETECTOR_REPAIR_SCREEN.json")
    lines = [
        *[f"{name}: {'PASS' if value else 'FAIL'}" for name, value in decision["software_checks"].items()],
        *[f"{name}: {'PASS' if value else 'FAIL'}" for name, value in decision["scientific_checks"].items()],
        f"SELECTED_VARIANT: {decision['selected_variant']}",
        f"CLASSIFICATION: {decision['classification']}",
        f"NEXT_ACTION: {decision['next_action']}",
        "PUBLICATION_NR_READY: NO",
    ]
    (gate_dir / "GATE1_NR_DETECTOR_REPAIR_SCREEN.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print("\n".join(lines))
    if not all(decision["software_checks"].values()):
        raise SystemExit(2)

    del source, source_context
    if device.type == "cuda":
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
