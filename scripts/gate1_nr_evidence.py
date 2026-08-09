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
import torch.nn.functional as F
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bayesroute.config import capture_rng_state, restore_rng_state
from bayesroute.nr_gate1 import (
    GATE1_NR_VERSION,
    NRBayesRouteBridge,
    NRCase,
    build_nr_context,
    channel_metrics,
    coded_bit_metrics,
    decode_bridge,
    package_contract_signature,
    run_standard_receiver,
    standard_receiver,
    transfer_operator_parameters,
)

TRAINING_VERSION = "gate1_nr_train_v1"
EVALUATION_VERSION = "gate1_nr_eval_v1"
CUSTOM_VARIANTS = {
    "proposed",
    "uncertainty_off_fixed_graph",
    "diagonal_posterior_fixed_graph",
    "mean_only_graph_fixed_cardinality",
    "random_graph_fixed_cardinality",
    "full_graph",
    "graph_off",
}


def load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected mapping in {path}")
    return value


def save_json(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    temp.replace(path)


def atomic_torch_save(value: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temp)
    temp.replace(path)


def set_all_seeds(seed: int) -> None:
    import sionna.phy

    random.seed(int(seed))
    np.random.seed(int(seed) % (2**32 - 1))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    sionna.phy.config.seed = int(seed)


def smoke_gate(root: Path, expected: str) -> dict[str, Any]:
    path = root / "outputs/gates/GATE1_NR_SMOKE.json"
    if not path.is_file():
        return {"passed": False, "reason": "missing", "path": str(path)}
    value = json.loads(path.read_text(encoding="utf-8"))
    return {
        "passed": bool(
            value.get("classification") == expected
            and value.get("overall_pass") is True
            and value.get("evidence_ready") is True
        ),
        "classification": value.get("classification"),
        "path": str(path),
    }


def bridge_from_config(case: NRCase, context: Any, config: dict[str, Any]) -> NRBayesRouteBridge:
    bridge = config["bridge"]
    return NRBayesRouteBridge(
        context.grid,
        num_streams=case.num_streams,
        rank=int(bridge["rank"]),
        bank_rank=int(bridge["bank_rank"]),
        detector_iterations=int(bridge["detector_iterations"]),
        edge_mass=float(bridge["edge_mass"]),
        length_f=float(bridge["length_f"]),
        length_t=float(bridge["length_t"]),
        operator_seed=int(bridge["operator_seed"]),
    ).to(context.device)


def differentiable_channel_nll(output: dict[str, Any], batch: Any) -> torch.Tensor:
    posterior = output["posterior"]
    mean = posterior.mean[..., batch.data_idx]
    truth = batch.h[..., batch.data_idx]
    var = posterior.var_diag[None, :, None, batch.data_idx].to(mean.device).clamp_min(1e-8)
    return torch.mean(torch.abs(mean - truth) ** 2 / var + torch.log(var)).real


def fixed_validation_nll(
    bridge: NRBayesRouteBridge,
    context: Any,
    *,
    seed: int,
    batch_size: int,
    ebno_db: float,
) -> float:
    state = capture_rng_state()
    was_training = bridge.training
    try:
        set_all_seeds(seed)
        bridge.eval()
        with torch.no_grad():
            batch = context.sample(batch_size=batch_size, ebno_db=ebno_db)
            output = bridge(batch)
            return float(
                F.binary_cross_entropy_with_logits(
                    output["bit_logits"], batch.coded_bits
                ).item()
            )
    finally:
        restore_rng_state(state)
        bridge.train(was_training)


def training_contract(config: dict[str, Any], config_path: Path) -> dict[str, Any]:
    payload = {
        "version": TRAINING_VERSION,
        "gate1_revision": config["gate1_revision"],
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "seed": int(config["seed"]),
        "bridge": config["bridge"],
        "training": config["training"],
    }
    payload["signature"] = package_contract_signature(payload)
    return payload


def train_or_resume(
    config: dict[str, Any],
    config_path: Path,
    output_root: Path,
    device: torch.device,
) -> tuple[NRBayesRouteBridge, dict[str, Any], Any]:
    training = config["training"]
    case = NRCase.from_mapping(training["source_case"])
    context = build_nr_context(case, device)
    bridge = bridge_from_config(case, context, config)
    optimizer = torch.optim.AdamW(
        bridge.parameters(),
        lr=float(training["lr"]),
        weight_decay=float(training["weight_decay"]),
    )
    contract = training_contract(config, config_path)

    checkpoint_dir = output_root / "checkpoints/preliminary"
    report_dir = Path("outputs/reports")
    log_dir = Path("outputs/logs")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    last_path = checkpoint_dir / "last.pt"
    best_path = checkpoint_dir / "best.pt"
    metrics_path = log_dir / "gate1_nr_preliminary_train.csv"

    start_step = 0
    best_metric = float("inf")
    if last_path.is_file():
        state = torch.load(last_path, map_location=device, weights_only=False)
        if state.get("contract") != contract:
            raise RuntimeError("Gate-1 training checkpoint contract mismatch")
        bridge.load_state_dict(state["model"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        start_step = int(state["step"]) + 1
        best_metric = float(state["best_metric"])
        restore_rng_state(state.get("rng_state"))
        print(f"Resuming Gate-1 training from step {start_step}", flush=True)
        if metrics_path.is_file():
            old = pd.read_csv(metrics_path)
            old = old[old["step"] < start_step]
            old.to_csv(metrics_path, index=False)

    fields = [
        "step",
        "ebno_db",
        "loss",
        "coded_bit_nll",
        "channel_nll",
        "coded_ber",
        "channel_nmse",
        "channel_coverage95",
        "edge_density",
        "grad_norm",
        "validation_coded_bit_nll",
        "contract_signature",
    ]
    if not metrics_path.is_file():
        with metrics_path.open("w", newline="", encoding="utf-8") as stream:
            csv.DictWriter(stream, fieldnames=fields).writeheader()

    steps = int(training["steps"])
    for step in range(start_step, steps):
        # Per-step deterministic seeds make resumption independent of hidden
        # channel-model RNG state.
        step_seed = int(config["seed"]) + 100_000 + step
        set_all_seeds(step_seed)
        fraction = ((step * 2654435761) % 1_000_003) / 1_000_003.0
        ebno_db = float(training["ebno_db_min"]) + fraction * (
            float(training["ebno_db_max"]) - float(training["ebno_db_min"])
        )
        bridge.train()
        batch = context.sample(batch_size=int(training["batch_size"]), ebno_db=ebno_db)
        output = bridge(batch)
        bit_nll = F.binary_cross_entropy_with_logits(
            output["bit_logits"], batch.coded_bits
        )
        channel_nll = differentiable_channel_nll(output, batch)
        loss = bit_nll + float(training["channel_loss_weight"]) * channel_nll
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite Gate-1 training loss at step {step}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(
            bridge.parameters(), float(training["grad_clip"])
        )
        if not torch.isfinite(grad):
            raise RuntimeError(f"Non-finite Gate-1 gradient at step {step}")
        optimizer.step()

        validation_due = (
            step % int(training["validation_every"]) == 0 or step == steps - 1
        )
        validation = float("nan")
        if validation_due:
            validation = fixed_validation_nll(
                bridge,
                context,
                seed=int(training["validation_seed"]),
                batch_size=int(training["validation_batch_size"]),
                ebno_db=float(training["validation_ebno_db"]),
            )
            if validation < best_metric:
                best_metric = validation
                atomic_torch_save(
                    {
                        "model": bridge.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "step": step,
                        "best_metric": best_metric,
                        "contract": contract,
                        "rng_state": capture_rng_state(),
                    },
                    best_path,
                )

        log_due = step % int(training["save_every"]) == 0 or step == steps - 1
        if log_due:
            with torch.no_grad():
                coded = coded_bit_metrics(output["bit_logits"], batch.coded_bits)
                channel = channel_metrics(output, batch)
            row = {
                "step": step,
                "ebno_db": ebno_db,
                "loss": float(loss.item()),
                "coded_bit_nll": coded["coded_bit_nll"],
                "channel_nll": float(channel_nll.item()),
                "coded_ber": coded["coded_ber"],
                "channel_nmse": channel["channel_nmse"],
                "channel_coverage95": channel["channel_coverage95"],
                "edge_density": float(output["edge_density"].item()),
                "grad_norm": float(grad.item()),
                "validation_coded_bit_nll": validation,
                "contract_signature": contract["signature"],
            }
            with metrics_path.open("a", newline="", encoding="utf-8") as stream:
                csv.DictWriter(stream, fieldnames=fields).writerow(row)
            print(json.dumps(row), flush=True)

        if log_due:
            atomic_torch_save(
                {
                    "model": bridge.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "step": step,
                    "best_metric": best_metric,
                    "contract": contract,
                    "rng_state": capture_rng_state(),
                },
                last_path,
            )

    if not best_path.is_file() or not last_path.is_file():
        raise RuntimeError("Gate-1 training completed without best/last checkpoints")
    best_state = torch.load(best_path, map_location=device, weights_only=False)
    if best_state.get("contract") != contract:
        raise RuntimeError("Best Gate-1 checkpoint contract mismatch")
    bridge.load_state_dict(best_state["model"], strict=True)
    bridge.eval()

    summary = {
        "complete": True,
        "version": TRAINING_VERSION,
        "steps": steps,
        "best_metric": best_metric,
        "best_checkpoint": str(best_path),
        "last_checkpoint": str(last_path),
        "best_checkpoint_sha256": hashlib.sha256(best_path.read_bytes()).hexdigest(),
        "trainable_parameters": int(sum(p.numel() for p in bridge.parameters() if p.requires_grad)),
        "source_case": case.__dict__,
        "contract": contract,
    }
    save_json(summary, report_dir / "gate1_nr_preliminary_train_summary.json")
    return bridge, summary, context


def evaluation_contract(
    config: dict[str, Any],
    config_path: Path,
    train_summary: dict[str, Any],
) -> dict[str, Any]:
    payload = {
        "version": EVALUATION_VERSION,
        "gate1_revision": config["gate1_revision"],
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "training_checkpoint_sha256": train_summary["best_checkpoint_sha256"],
        "evaluation": config["evaluation"],
        "bridge": config["bridge"],
    }
    payload["signature"] = package_contract_signature(payload)
    return payload


def append_row(path: Path, row: dict[str, Any]) -> None:
    pd.DataFrame([row]).to_csv(
        path, mode="a", header=not path.is_file(), index=False
    )


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
        "dmrs_length": case.dmrs_length,
        "dmrs_ports": json.dumps(list(case.dmrs_ports)),
        "num_users": case.num_users,
        "num_layers_per_user": case.num_layers_per_user,
        "num_streams": case.num_streams,
        "num_rx_ant": case.num_rx_ant,
        "variant": variant,
        "ebno_db": float(snr),
        "rep": int(rep),
        "eval_seed": int(seed),
        "information_ber": metrics["information_ber"],
        "tbler": metrics["tbler"],
        "crc_failure_rate": metrics["crc_failure_rate"],
        "coded_ber": float("nan"),
        "coded_bit_nll": float("nan"),
        "coded_brier": float("nan"),
        "channel_nmse": float("nan"),
        "channel_marginal_nll": float("nan"),
        "channel_coverage95": float("nan"),
        "edge_density": float("nan"),
        "graph_count_match": True,
        "trainable_parameters": 0,
        "evaluation_contract_signature": contract_signature,
    }


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
    graph_count_match: bool,
    trainable_parameters: int,
    contract_signature: str,
) -> dict[str, Any]:
    coded = coded_bit_metrics(output["bit_logits"], batch.coded_bits)
    channel = channel_metrics(output, batch)
    return {
        "case": case.name,
        "scenario": case.scenario,
        "dmrs_config_type": case.dmrs_config_type,
        "dmrs_length": case.dmrs_length,
        "dmrs_ports": json.dumps(list(case.dmrs_ports)),
        "num_users": case.num_users,
        "num_layers_per_user": case.num_layers_per_user,
        "num_streams": case.num_streams,
        "num_rx_ant": case.num_rx_ant,
        "variant": variant,
        "ebno_db": float(snr),
        "rep": int(rep),
        "eval_seed": int(seed),
        "information_ber": decoded["information_ber"],
        "tbler": decoded["tbler"],
        "crc_failure_rate": decoded["crc_failure_rate"],
        **coded,
        **channel,
        "edge_density": float(output["edge_density"].item()),
        "graph_count_match": bool(graph_count_match),
        "trainable_parameters": int(trainable_parameters),
        "evaluation_contract_signature": contract_signature,
    }


def evaluate_or_resume(
    config: dict[str, Any],
    config_path: Path,
    trained_bridge: NRBayesRouteBridge,
    train_summary: dict[str, Any],
    device: torch.device,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    from sionna.phy.nr import LayerDemapper, TBDecoder

    evaluation = config["evaluation"]
    contract = evaluation_contract(config, config_path, train_summary)
    eval_dir = Path("outputs/eval")
    report_dir = Path("outputs/reports")
    eval_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    raw_path = eval_dir / "gate1_nr_preliminary_eval.csv"
    contract_path = eval_dir / "gate1_nr_preliminary_contract.json"

    if raw_path.is_file():
        if not contract_path.is_file():
            raise RuntimeError("Gate-1 evaluation CSV exists without contract")
        old_contract = json.loads(contract_path.read_text(encoding="utf-8"))
        if old_contract.get("signature") != contract["signature"]:
            raise RuntimeError("Gate-1 evaluation resume contract mismatch")
    else:
        save_json(contract, contract_path)

    done: set[tuple[str, str, float, int]] = set()
    if raw_path.is_file():
        old = pd.read_csv(raw_path)
        keys = ["case", "variant", "ebno_db", "rep"]
        if old[keys].duplicated().any():
            raise RuntimeError("Gate-1 evaluation CSV contains duplicate keys")
        for _, row in old.iterrows():
            done.add((str(row["case"]), str(row["variant"]), float(row["ebno_db"]), int(row["rep"])))

    base_variants = [str(x) for x in evaluation["variants"]]
    cases = [NRCase.from_mapping(item) for item in evaluation["cases"]]
    expected_rows = 0
    for raw_case, case in zip(evaluation["cases"], cases):
        expected_rows += len(base_variants) * len(evaluation["ebno_db"]) * int(evaluation["repetitions"])
        if bool(raw_case.get("run_kbest", False)):
            expected_rows += len(evaluation["ebno_db"]) * int(evaluation["repetitions"])

    for case_index, (raw_case, case) in enumerate(zip(evaluation["cases"], cases)):
        context = build_nr_context(case, device)
        bridge = bridge_from_config(case, context, config)
        transfer = transfer_operator_parameters(trained_bridge, bridge)
        if not transfer["passed"]:
            raise RuntimeError(f"Failed learned-operator transfer for {case.name}")
        bridge.eval()
        trainable = int(sum(p.numel() for p in bridge.parameters() if p.requires_grad))
        tb_decoder = TBDecoder(
            context.transmitter._tb_encoder,
            num_bp_iter=int(evaluation["bp_iterations"]),
            device=str(device),
        )
        layer_demapper = LayerDemapper(
            context.transmitter._layer_mapper,
            num_bits_per_symbol=int(context.grid.bits_per_symbol),
            device=str(device),
        )
        ls = standard_receiver(context, perfect_csi=False, return_crc=True)
        perfect = standard_receiver(context, perfect_csi=True, return_crc=True)
        kbest = None
        if bool(raw_case.get("run_kbest", False)):
            kbest = standard_receiver(
                context,
                perfect_csi=True,
                kbest_k=int(raw_case.get("kbest_k", 16)),
                return_crc=True,
            )

        case_variants = list(base_variants)
        if kbest is not None:
            case_variants.append("perfect_csi_kbest")

        for snr_index, snr_value in enumerate(evaluation["ebno_db"]):
            snr = float(snr_value)
            for rep in range(int(evaluation["repetitions"])):
                missing = [
                    variant for variant in case_variants
                    if (case.name, variant, snr, rep) not in done
                ]
                if not missing:
                    continue
                eval_seed = int(config["seed"]) + 10_000_000 + 100_000 * case_index + 1_000 * snr_index + rep
                set_all_seeds(eval_seed)
                batch = context.sample(
                    batch_size=int(evaluation["batch_size"]), ebno_db=snr
                )

                custom_missing = [name for name in missing if name in CUSTOM_VARIANTS]
                custom_outputs: dict[str, dict[str, Any]] = {}
                if custom_missing:
                    custom_outputs = bridge.forward_variants(
                        batch, custom_missing, random_seed=eval_seed + 800_000
                    )
                    # Proposed reference is needed to verify cardinality even if it
                    # was already completed in an earlier interrupted invocation.
                    if "proposed" not in custom_outputs:
                        custom_outputs["proposed"] = bridge.forward_variant(batch, "proposed")
                    reference_counts = custom_outputs["proposed"]["graph_mask"].sum(dim=-1)
                else:
                    reference_counts = None

                for variant in missing:
                    if variant in CUSTOM_VARIANTS:
                        output = custom_outputs[variant]
                        counts_match = True
                        if variant in {
                            "uncertainty_off_fixed_graph",
                            "diagonal_posterior_fixed_graph",
                            "mean_only_graph_fixed_cardinality",
                            "random_graph_fixed_cardinality",
                        }:
                            counts_match = bool(
                                torch.equal(
                                    output["graph_mask"].sum(dim=-1), reference_counts
                                )
                            )
                        with torch.no_grad():
                            decoded = decode_bridge(
                                context.transmitter,
                                output,
                                batch.information_bits,
                                num_bp_iter=int(evaluation["bp_iterations"]),
                                device=device,
                                decoder=tb_decoder,
                                layer_demapper=layer_demapper,
                            )
                        row = custom_row(
                            case=case,
                            variant=variant,
                            snr=snr,
                            rep=rep,
                            seed=eval_seed,
                            output=output,
                            batch=batch,
                            decoded=decoded,
                            graph_count_match=counts_match,
                            trainable_parameters=trainable,
                            contract_signature=contract["signature"],
                        )
                    elif variant == "ls_lmmse":
                        with torch.no_grad():
                            metrics = run_standard_receiver(
                                ls, batch, batch.information_bits, perfect_csi=False
                            )
                        row = standard_row(
                            case=case,
                            variant=variant,
                            snr=snr,
                            rep=rep,
                            seed=eval_seed,
                            metrics=metrics,
                            contract_signature=contract["signature"],
                        )
                    elif variant == "perfect_csi_lmmse":
                        with torch.no_grad():
                            metrics = run_standard_receiver(
                                perfect, batch, batch.information_bits, perfect_csi=True
                            )
                        row = standard_row(
                            case=case,
                            variant=variant,
                            snr=snr,
                            rep=rep,
                            seed=eval_seed,
                            metrics=metrics,
                            contract_signature=contract["signature"],
                        )
                    elif variant == "perfect_csi_kbest":
                        if kbest is None:
                            raise RuntimeError("KBest variant requested without detector")
                        with torch.no_grad():
                            metrics = run_standard_receiver(
                                kbest, batch, batch.information_bits, perfect_csi=True
                            )
                        row = standard_row(
                            case=case,
                            variant=variant,
                            snr=snr,
                            rep=rep,
                            seed=eval_seed,
                            metrics=metrics,
                            contract_signature=contract["signature"],
                        )
                    else:
                        raise RuntimeError(f"Unknown Gate-1 evidence variant {variant}")
                    append_row(raw_path, row)
                    done.add((case.name, variant, snr, rep))
                    print(json.dumps({
                        "case": case.name,
                        "variant": variant,
                        "ebno_db": snr,
                        "rep": rep,
                        "tbler": row["tbler"],
                    }), flush=True)

        del bridge, context, ls, perfect, kbest, tb_decoder, layer_demapper
        if device.type == "cuda":
            torch.cuda.empty_cache()

    df = pd.read_csv(raw_path)
    keys = ["case", "variant", "ebno_db", "rep"]
    unique_rows = int(len(df.drop_duplicates(keys)))
    complete = bool(len(df) == expected_rows and unique_rows == expected_rows)
    if not complete:
        raise RuntimeError(
            f"Gate-1 evidence incomplete: rows={len(df)}, unique={unique_rows}, expected={expected_rows}"
        )

    numeric = [
        "information_ber", "tbler", "crc_failure_rate", "coded_ber",
        "coded_bit_nll", "coded_brier", "channel_nmse",
        "channel_marginal_nll", "channel_coverage95", "edge_density",
        "graph_count_match", "trainable_parameters",
    ]
    aggregate = (
        df.groupby(["case", "scenario", "variant", "ebno_db"], as_index=False)[numeric]
        .agg(["mean", "std", "count"])
    )
    aggregate.columns = [
        "_".join([str(x) for x in item if str(x)]) for item in aggregate.columns.to_flat_index()
    ]
    aggregate_path = eval_dir / "gate1_nr_preliminary_aggregate.csv"
    aggregate.to_csv(aggregate_path, index=False)

    paired_rows: list[dict[str, Any]] = []
    reference = df[df["variant"] == "proposed"]
    pair_keys = ["case", "ebno_db", "rep", "eval_seed"]
    for comparator in sorted(set(df["variant"]) - {"proposed"}):
        other = df[df["variant"] == comparator]
        merged = reference.merge(other, on=pair_keys, suffixes=("_proposed", "_comparator"))
        for _, row in merged.iterrows():
            paired_rows.append(
                {
                    "comparator": comparator,
                    "case": row["case"],
                    "ebno_db": float(row["ebno_db"]),
                    "rep": int(row["rep"]),
                    "eval_seed": int(row["eval_seed"]),
                    "tbler_delta_proposed_minus_comparator": float(
                        row["tbler_proposed"] - row["tbler_comparator"]
                    ),
                    "information_ber_delta_proposed_minus_comparator": float(
                        row["information_ber_proposed"] - row["information_ber_comparator"]
                    ),
                    "coded_bit_nll_delta_proposed_minus_comparator": float(
                        row["coded_bit_nll_proposed"] - row["coded_bit_nll_comparator"]
                    ) if pd.notna(row["coded_bit_nll_comparator"]) else float("nan"),
                }
            )
    paired = pd.DataFrame(paired_rows)
    paired_path = eval_dir / "gate1_nr_preliminary_paired.csv"
    paired.to_csv(paired_path, index=False)

    summary_rows: list[dict[str, Any]] = []
    if not paired.empty:
        for comparator, sub in paired.groupby("comparator"):
            row: dict[str, Any] = {"comparator": comparator, "pairs": int(len(sub))}
            for metric in [
                "tbler_delta_proposed_minus_comparator",
                "information_ber_delta_proposed_minus_comparator",
                "coded_bit_nll_delta_proposed_minus_comparator",
            ]:
                values = pd.to_numeric(sub[metric], errors="coerce").dropna()
                if len(values):
                    mean = float(values.mean())
                    std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
                    half = 1.96 * std / math.sqrt(len(values))
                    row[f"{metric}_mean"] = mean
                    row[f"{metric}_ci95_low"] = mean - half
                    row[f"{metric}_ci95_high"] = mean + half
            summary_rows.append(row)
    paired_summary = pd.DataFrame(summary_rows)
    paired_summary_path = report_dir / "gate1_nr_preliminary_paired_summary.csv"
    paired_summary.to_csv(paired_summary_path, index=False)

    return df, {
        "contract": contract,
        "raw_csv": str(raw_path),
        "aggregate_csv": str(aggregate_path),
        "paired_csv": str(paired_path),
        "paired_summary_csv": str(paired_summary_path),
        "rows": int(len(df)),
        "unique_rows": unique_rows,
        "expected_rows": expected_rows,
        "complete": complete,
    }


def make_plots(df: pd.DataFrame) -> list[str]:
    plot_dir = Path("outputs/plots")
    plot_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    selected_variants = [
        "proposed",
        "uncertainty_off_fixed_graph",
        "mean_only_graph_fixed_cardinality",
        "random_graph_fixed_cardinality",
        "ls_lmmse",
        "perfect_csi_lmmse",
    ]
    for case_name in sorted(df["case"].unique()):
        sub_case = df[df["case"] == case_name]
        fig, ax = plt.subplots(figsize=(7.2, 4.8))
        for variant in selected_variants:
            sub = sub_case[sub_case["variant"] == variant]
            if sub.empty:
                continue
            curve = sub.groupby("ebno_db", as_index=False)["tbler"].mean()
            ax.semilogy(curve["ebno_db"], curve["tbler"].clip(lower=1e-4), marker="o", label=variant)
        ax.set_xlabel("Eb/N0 [dB]")
        ax.set_ylabel("Decoded TBLER")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=7)
        ax.set_title(case_name)
        fig.tight_layout()
        path = plot_dir / f"gate1_nr_{case_name}_tbler.png"
        fig.savefig(path, dpi=220)
        plt.close(fig)
        paths.append(str(path))

    custom = df[df["variant"].isin(selected_variants[:4])]
    if not custom.empty:
        fig, ax = plt.subplots(figsize=(7.2, 4.8))
        for variant in selected_variants[:4]:
            sub = custom[custom["variant"] == variant]
            if sub.empty:
                continue
            curve = sub.groupby("ebno_db", as_index=False)["coded_bit_nll"].mean()
            ax.plot(curve["ebno_db"], curve["coded_bit_nll"], marker="o", label=variant)
        ax.set_xlabel("Eb/N0 [dB]")
        ax.set_ylabel("Coded-bit NLL")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)
        fig.tight_layout()
        path = plot_dir / "gate1_nr_coded_bit_nll.png"
        fig.savefig(path, dpi=220)
        plt.close(fig)
        paths.append(str(path))
    return paths


def _paired_mean(path: str, comparator: str, metric: str) -> float:
    table = pd.read_csv(path)
    row = table[table["comparator"] == comparator]
    if row.empty or metric not in row.columns:
        return float("nan")
    return float(row.iloc[0][metric])


def classify(
    df: pd.DataFrame,
    evaluation_meta: dict[str, Any],
    train_summary: dict[str, Any],
) -> dict[str, Any]:
    paired_path = evaluation_meta["paired_summary_csv"]
    uncertainty_nll = _paired_mean(
        paired_path,
        "uncertainty_off_fixed_graph",
        "coded_bit_nll_delta_proposed_minus_comparator_mean",
    )
    uncertainty_tbler = _paired_mean(
        paired_path,
        "uncertainty_off_fixed_graph",
        "tbler_delta_proposed_minus_comparator_mean",
    )
    random_tbler = _paired_mean(
        paired_path,
        "random_graph_fixed_cardinality",
        "tbler_delta_proposed_minus_comparator_mean",
    )
    full_tbler = _paired_mean(
        paired_path,
        "full_graph",
        "tbler_delta_proposed_minus_comparator_mean",
    )

    finite_core = bool(
        np.isfinite(df[["information_ber", "tbler", "crc_failure_rate"]].to_numpy()).all()
    )
    graph_rows = df[df["variant"].isin([
        "uncertainty_off_fixed_graph",
        "diagonal_posterior_fixed_graph",
        "mean_only_graph_fixed_cardinality",
        "random_graph_fixed_cardinality",
    ])]
    cardinality = bool(graph_rows["graph_count_match"].astype(bool).all())

    perfect = df[df["variant"] == "perfect_csi_lmmse"]
    improving_cases = 0
    tested_cases = 0
    for _, sub in perfect.groupby("case"):
        curve = sub.groupby("ebno_db")["tbler"].mean().sort_index()
        if len(curve) >= 2:
            tested_cases += 1
            improving_cases += int(float(curve.iloc[-1]) <= float(curve.iloc[0]))
    perfect_improves = bool(tested_cases > 0 and improving_cases == tested_cases)
    kbest_present = "perfect_csi_kbest" in set(df["variant"])

    checks = {
        "complete_rows": bool(evaluation_meta["complete"]),
        "finite_core_metrics": finite_core,
        "fixed_graph_cardinality_exact": cardinality,
        "training_complete": bool(train_summary["complete"]),
        "perfect_csi_lmmse_improves_with_snr": perfect_improves,
        "kbest_reference_executed": kbest_present,
    }
    mechanism_signals = {
        "uncertainty_coded_nll_gain": bool(math.isfinite(uncertainty_nll) and uncertainty_nll < 0.0),
        "uncertainty_tbler_gain": bool(math.isfinite(uncertainty_tbler) and uncertainty_tbler < 0.0),
        "coupling_beats_random_tbler": bool(math.isfinite(random_tbler) and random_tbler < 0.0),
        "sparse_not_worse_than_full_by_0p01": bool(math.isfinite(full_tbler) and full_tbler <= 0.01),
    }
    software_pass = all(checks.values())
    signal_count = sum(mechanism_signals.values())
    if software_pass and signal_count >= 3:
        classification = "GATE1_NR_PRELIMINARY_PROMISING"
    elif software_pass:
        classification = "GATE1_NR_PRELIMINARY_MIXED"
    else:
        classification = "GATE1_NR_PRELIMINARY_BLOCKED"
    return {
        "classification": classification,
        "checks": checks,
        "mechanism_signals": mechanism_signals,
        "key_deltas_proposed_minus_comparator": {
            "coded_bit_nll_vs_uncertainty_off_fixed_graph": uncertainty_nll,
            "tbler_vs_uncertainty_off_fixed_graph": uncertainty_tbler,
            "tbler_vs_random_graph_fixed_cardinality": random_tbler,
            "tbler_vs_full_graph": full_tbler,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/gate1_nr_evidence.yaml")
    args = parser.parse_args()
    config_path = Path(args.config)
    config = load_yaml(config_path)
    gate = smoke_gate(ROOT, str(config["required_smoke_classification"]))
    if not gate["passed"]:
        raise SystemExit(f"BLOCKED: Gate-1 NR smoke has not passed: {gate}")
    if str(config["gate1_revision"]) != GATE1_NR_VERSION:
        raise SystemExit("Gate-1 evidence/revision mismatch")

    import sionna
    import sionna.phy

    device = torch.device(str(config["device"]))
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda:0")
    sionna.phy.config.device = str(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("Gate-1 evidence requires a CUDA compute node")
    if str(getattr(sionna, "__version__", "")) != "2.0.1":
        raise SystemExit(f"Gate-1 requires Sionna 2.0.1, found {getattr(sionna, '__version__', None)}")

    output_root = Path("outputs/gate1_nr")
    output_root.mkdir(parents=True, exist_ok=True)
    trained, train_summary, _ = train_or_resume(
        config, config_path, output_root, device
    )
    df, evaluation_meta = evaluate_or_resume(
        config, config_path, trained, train_summary, device
    )
    plots = make_plots(df)
    result = classify(df, evaluation_meta, train_summary)
    result.update(
        {
            "gate1_revision": GATE1_NR_VERSION,
            "complete": bool(evaluation_meta["complete"]),
            "publication_nr_ready": False,
            "scope": "PRELIMINARY_STANDARD_COMPLIANT_GATE_NOT_PUBLICATION_CAMPAIGN",
            "gate0_pgca_agmp_baseline_included": False,
            "smoke_gate": gate,
            "training": train_summary,
            "evaluation": evaluation_meta,
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
    )
    report_dir = Path("outputs/reports")
    gate_dir = Path("outputs/gates")
    report_dir.mkdir(parents=True, exist_ok=True)
    gate_dir.mkdir(parents=True, exist_ok=True)
    save_json(result, report_dir / "gate1_nr_preliminary_summary.json")
    save_json(result, gate_dir / "GATE1_NR_PRELIMINARY_EVIDENCE.json")
    lines = [
        *[f"{name}: {'PASS' if value else 'FAIL'}" for name, value in result["checks"].items()],
        *[f"{name}: {'PASS' if value else 'FAIL'}" for name, value in result["mechanism_signals"].items()],
        f"CLASSIFICATION: {result['classification']}",
        "PGCA_AGMP_BASELINE_INCLUDED: NO",
        "PUBLICATION_NR_READY: NO",
    ]
    (gate_dir / "GATE1_NR_PRELIMINARY_EVIDENCE.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print("\n".join(lines))
    if result["classification"] == "GATE1_NR_PRELIMINARY_BLOCKED":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
