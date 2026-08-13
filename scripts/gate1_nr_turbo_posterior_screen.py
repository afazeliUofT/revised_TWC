#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import time
from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]

from bayesroute.nr_gate1 import (
    decode_bridge,
    normalize_device,
    run_standard_receiver,
    standard_receiver,
)
from bayesroute.turbo_posterior import posterior_batch_metrics
from gate1_nr_joint_operator_common import coded_metrics, make_repaired_detector, repaired_forward
from gate1_nr_posterior_factorial_common import ls_repaired_forward, save_json, sha256_file
from gate1_nr_turbo_posterior_common import (
    TURBO_GATE_VERSION,
    TurboSetting,
    build_loaded_bridge,
    experiment_signature,
    extension_preconditions,
    initial_detector_output,
    make_case_context,
    pilot_state_and_reference,
    set_all_seeds,
    source_hashes,
    turbo_forward,
)

SELECTION_RAW = ROOT / "outputs/eval/gate1_nr_turbo_posterior_selection.csv"
SELECTION_CONTRACT = ROOT / "outputs/eval/gate1_nr_turbo_posterior_selection_contract.json"
HOLDOUT_RAW = ROOT / "outputs/eval/gate1_nr_turbo_posterior_holdout.csv"
HOLDOUT_CONTRACT = ROOT / "outputs/eval/gate1_nr_turbo_posterior_holdout_contract.json"
AGGREGATE_PATH = ROOT / "outputs/eval/gate1_nr_turbo_posterior_aggregate.csv"
REPORT_PATH = ROOT / "outputs/reports/gate1_nr_turbo_posterior.json"
GATE_JSON = ROOT / "outputs/gates/GATE1_NR_TURBO_POSTERIOR.json"
GATE_TXT = ROOT / "outputs/gates/GATE1_NR_TURBO_POSTERIOR.txt"
EXPECTED_SELECTION_ROWS = 240
EXPECTED_HOLDOUT_ROWS = 252
EXPECTED_ROWS = 492
SOURCE_FILES = [
    "configs/gate1_nr_turbo_posterior_screen.yaml",
    "scripts/gate1_nr_turbo_posterior_common.py",
    "scripts/gate1_nr_turbo_posterior_screen.py",
    "src/bayesroute/turbo_posterior.py",
    "src/bayesroute/multiscale_posterior.py",
    "src/bayesroute/lmmse_ep.py",
    "src/bayesroute/nr_gate1.py",
    "src/bayesroute/qam.py",
]


def load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected YAML mapping: {path}")
    return value


def append_rows_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    new = pd.DataFrame(rows)
    if path.is_file():
        old = pd.read_csv(path)
        if list(old.columns) != list(new.columns):
            raise RuntimeError(f"CSV column contract mismatch: {path}")
        combined = pd.concat([old, new], ignore_index=True)
    else:
        combined = new
    temp = path.with_suffix(path.suffix + ".tmp")
    combined.to_csv(temp, index=False)
    temp.replace(path)


def crc_disagreement(decoded: dict[str, Any], bits: torch.Tensor) -> float:
    block_error = (decoded["b_hat"] != bits).reshape(
        bits.shape[0], bits.shape[1], -1
    ).any(-1)
    crc_failure = ~decoded["crc"].bool()
    if crc_failure.shape != block_error.shape:
        crc_failure = crc_failure.reshape(block_error.shape)
    return float((block_error != crc_failure).float().mean().item())


def custom_row(
    *,
    case: Any,
    group: str,
    variant: str,
    snr: float,
    rep: int,
    seed: int,
    output: dict[str, Any],
    batch: Any,
    decoded: dict[str, Any],
    signature: str,
) -> dict[str, Any]:
    posterior_metrics = output.get("posterior_metrics")
    if not isinstance(posterior_metrics, dict):
        posterior_metrics = posterior_batch_metrics(
            output["posterior"], batch.h, batch.data_idx
        )
    diagnostics = output.get("turbo_diagnostics", {})
    return {
        "case": case.name,
        "group": group,
        "num_prb": int(case.num_prb),
        "variant": variant,
        "ebno_db": float(snr),
        "rep": int(rep),
        "eval_seed": int(seed),
        "information_ber": float(decoded["information_ber"]),
        "tbler": float(decoded["tbler"]),
        "crc_failure_rate": float(decoded["crc_failure_rate"]),
        "crc_block_disagreement_rate": crc_disagreement(
            decoded, batch.information_bits
        ),
        **coded_metrics(output, batch),
        **posterior_metrics,
        "edge_density": float(output["edge_density"].item()),
        "selected_observations": int(diagnostics.get("selected_observations", 0)),
        "information_damping": float(diagnostics.get("information_damping", 0.0)),
        "latent_trace_reduction_fraction": float(
            diagnostics.get("latent_trace_reduction_fraction", 0.0)
        ),
        "contract_signature": signature,
    }


def standard_row(
    *, case: Any, group: str, variant: str, snr: float, rep: int, seed: int,
    metrics: dict[str, Any], signature: str,
) -> dict[str, Any]:
    return {
        "case": case.name,
        "group": group,
        "num_prb": int(case.num_prb),
        "variant": variant,
        "ebno_db": float(snr),
        "rep": int(rep),
        "eval_seed": int(seed),
        "information_ber": float(metrics["information_ber"]),
        "tbler": float(metrics["tbler"]),
        "crc_failure_rate": float(metrics["crc_failure_rate"]),
        "crc_block_disagreement_rate": float("nan"),
        "coded_ber": float("nan"),
        "coded_bit_nll": float("nan"),
        "coded_brier": float("nan"),
        "channel_nmse": float("nan"),
        "normalized_error_mean": float("nan"),
        "coverage95": float("nan"),
        "edge_density": float("nan"),
        "selected_observations": 0,
        "information_damping": 0.0,
        "latent_trace_reduction_fraction": 0.0,
        "contract_signature": signature,
    }


def decode_outputs(
    context: Any,
    batch: Any,
    outputs: dict[str, dict[str, Any]],
    *, bp_iterations: int, device: torch.device,
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
    return {
        name: decode_bridge(
            context.transmitter,
            output,
            batch.information_bits,
            num_bp_iter=int(bp_iterations),
            device=device,
            decoder=decoder,
            layer_demapper=demapper,
        )
        for name, output in outputs.items()
    }


def completed_batches(path: Path, variants: Sequence[str]) -> set[tuple[str, float, int]]:
    if not path.is_file():
        return set()
    frame = pd.read_csv(path)
    keys = ["case", "variant", "ebno_db", "rep"]
    if frame[keys].duplicated().any():
        raise RuntimeError(f"Duplicate evaluation rows: {path}")
    counts = frame.groupby(["case", "ebno_db", "rep"])["variant"].nunique()
    partial = counts[counts != len(variants)]
    if len(partial):
        raise RuntimeError(f"Partial paired batch in {path}: {partial.index.tolist()}")
    return {(str(a), float(b), int(c)) for a, b, c in counts.index}


def ensure_contract(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    payload = dict(payload)
    payload["signature"] = experiment_signature(payload)
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing.get("signature") != payload["signature"]:
            raise RuntimeError(f"Evaluation contract mismatch: {path}")
    else:
        save_json(payload, path)
    return payload


def selection_variants(candidates: Sequence[TurboSetting]) -> list[str]:
    return [item.name for item in candidates] + [
        "pilot_only_extended",
        "oracle_symbol_turbo",
        "ls_lmmse",
        "perfect_csi_lmmse",
    ]


def run_selection(
    config: dict[str, Any], config_path: Path, device: torch.device,
    pre: dict[str, Any], *, deadline_epoch: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    candidates = [TurboSetting.from_mapping(item) for item in config["candidates"]]
    oracle_setting = TurboSetting.from_mapping(config["oracle_setting"])
    variants = selection_variants(candidates)
    section = config["selection"]
    payload = {
        "version": TURBO_GATE_VERSION,
        "phase": "selection_4prb_8prb_only",
        "config_sha256": sha256_file(config_path),
        "source_sha256": source_hashes(SOURCE_FILES),
        "checkpoint_sha256": pre["checkpoint_sha256"],
        "cases": section["cases"],
        "ebno_db": section["ebno_db"],
        "repetitions": section["repetitions"],
        "candidates": [item.as_dict() for item in candidates],
        "variants": variants,
        "holdout_prb_used": False,
    }
    contract = ensure_contract(SELECTION_CONTRACT, payload)
    done = completed_batches(SELECTION_RAW, variants)
    for case_index, raw_case in enumerate(section["cases"]):
        case, context = make_case_context(raw_case, device)
        bridge = build_loaded_bridge(
            case, context, operator_seed=int(config["operator_seed"])
        )
        detector = make_repaired_detector(int(context.grid.bits_per_symbol)).to(device)
        ls_receiver = standard_receiver(context, perfect_csi=False, return_crc=True)
        perfect_receiver = standard_receiver(context, perfect_csi=True, return_crc=True)
        for snr_index, raw_snr in enumerate(section["ebno_db"]):
            snr = float(raw_snr)
            for rep in range(int(section["repetitions"])):
                key = (case.name, snr, rep)
                if key in done:
                    continue
                seed = int(config["seed"]) + case_index * 100_000 + snr_index * 1_000 + rep
                set_all_seeds(seed)
                batch = context.sample(int(section["batch_size"]), snr)
                with torch.inference_mode():
                    state, graph, _ = pilot_state_and_reference(bridge, batch)
                    initial = initial_detector_output(bridge, detector, batch, state, graph)
                    outputs: dict[str, dict[str, Any]] = {
                        "pilot_only_extended": initial,
                    }
                    for setting in candidates:
                        outputs[setting.name] = turbo_forward(
                            bridge, detector, batch, setting,
                            state=state, reference_graph=graph,
                            initial_output=initial,
                        )
                    outputs["oracle_symbol_turbo"] = turbo_forward(
                        bridge, detector, batch, oracle_setting,
                        state=state, reference_graph=graph,
                        initial_output=initial, oracle_symbols=True,
                    )
                    decoded = decode_outputs(
                        context, batch, outputs,
                        bp_iterations=int(config["bp_iterations"]), device=device,
                    )
                    ls_metrics = run_standard_receiver(
                        ls_receiver, batch, batch.information_bits, perfect_csi=False
                    )
                    perfect_metrics = run_standard_receiver(
                        perfect_receiver, batch, batch.information_bits, perfect_csi=True
                    )
                group = str(raw_case["group"])
                rows = [
                    custom_row(
                        case=case, group=group, variant=name, snr=snr, rep=rep,
                        seed=seed, output=output, batch=batch,
                        decoded=decoded[name], signature=contract["signature"],
                    )
                    for name, output in outputs.items()
                ]
                rows.extend([
                    standard_row(
                        case=case, group=group, variant="ls_lmmse", snr=snr,
                        rep=rep, seed=seed, metrics=ls_metrics,
                        signature=contract["signature"],
                    ),
                    standard_row(
                        case=case, group=group, variant="perfect_csi_lmmse", snr=snr,
                        rep=rep, seed=seed, metrics=perfect_metrics,
                        signature=contract["signature"],
                    ),
                ])
                if {row["variant"] for row in rows} != set(variants):
                    raise RuntimeError("Selection paired variant set mismatch")
                append_rows_atomic(SELECTION_RAW, rows)
                done.add(key)
                print(json.dumps({
                    "phase": "selection", "case": case.name, "ebno_db": snr,
                    "rep": rep, "rows_committed": len(rows),
                    "completed_rows": len(done) * len(variants),
                    "expected_rows": EXPECTED_SELECTION_ROWS,
                }), flush=True)
                if time.time() >= deadline_epoch:
                    frame = pd.read_csv(SELECTION_RAW)
                    return frame, {"complete": False, "rows": len(frame), "expected_rows": EXPECTED_SELECTION_ROWS, "contract": contract}
        del bridge, detector, ls_receiver, perfect_receiver
        if device.type == "cuda":
            torch.cuda.empty_cache()
    frame = pd.read_csv(SELECTION_RAW)
    unique = len(frame.drop_duplicates(["case", "variant", "ebno_db", "rep"]))
    if len(frame) != EXPECTED_SELECTION_ROWS or unique != EXPECTED_SELECTION_ROWS:
        raise RuntimeError(f"Selection incomplete: rows={len(frame)}, unique={unique}")
    return frame, {"complete": True, "rows": len(frame), "unique_rows": unique, "expected_rows": EXPECTED_SELECTION_ROWS, "contract": contract}


def choose_winner(frame: pd.DataFrame, candidates: Sequence[TurboSetting]) -> tuple[str, list[dict[str, Any]]]:
    ranking: list[dict[str, Any]] = []
    for setting in candidates:
        sub = frame[frame["variant"] == setting.name]
        condition = sub.groupby(["case", "ebno_db"])["tbler"].mean()
        objective = (
            float(sub["tbler"].mean())
            + 0.25 * float(condition.max())
            + 0.05 * float(sub["coded_bit_nll"].mean())
        )
        ranking.append({
            "variant": setting.name,
            "selection_objective": objective,
            "selection_tbler": float(sub["tbler"].mean()),
            "selection_worst_condition_tbler": float(condition.max()),
            "selection_coded_bit_nll": float(sub["coded_bit_nll"].mean()),
        })
    ranking.sort(key=lambda item: (item["selection_objective"], item["variant"]))
    return str(ranking[0]["variant"]), ranking


def run_holdout(
    config: dict[str, Any], config_path: Path, device: torch.device,
    pre: dict[str, Any], winner: str, selection_contract: dict[str, Any],
    *, deadline_epoch: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    settings = {TurboSetting.from_mapping(item).name: TurboSetting.from_mapping(item) for item in config["candidates"]}
    winner_setting = settings[winner]
    oracle_setting = TurboSetting.from_mapping(config["oracle_setting"])
    variants = [
        "winner_turbo", "pilot_only_extended", "winner_uncertainty_off",
        "oracle_symbol_turbo", "ls_estimate_repaired", "ls_lmmse",
        "perfect_csi_lmmse",
    ]
    section = config["holdout"]
    payload = {
        "version": TURBO_GATE_VERSION,
        "phase": "fresh_12prb_holdout",
        "config_sha256": sha256_file(config_path),
        "source_sha256": source_hashes(SOURCE_FILES),
        "checkpoint_sha256": pre["checkpoint_sha256"],
        "winner": winner,
        "winner_setting": winner_setting.as_dict(),
        "selection_contract_signature": selection_contract["signature"],
        "selection_csv_sha256": sha256_file(SELECTION_RAW),
        "cases": section["cases"],
        "ebno_db": section["ebno_db"],
        "repetitions": section["repetitions"],
        "variants": variants,
        "holdout_used_for_selection": False,
    }
    contract = ensure_contract(HOLDOUT_CONTRACT, payload)
    done = completed_batches(HOLDOUT_RAW, variants)
    for case_index, raw_case in enumerate(section["cases"]):
        case, context = make_case_context(raw_case, device)
        bridge = build_loaded_bridge(case, context, operator_seed=int(config["operator_seed"]))
        detector = make_repaired_detector(int(context.grid.bits_per_symbol)).to(device)
        ls_receiver = standard_receiver(context, perfect_csi=False, return_crc=True)
        perfect_receiver = standard_receiver(context, perfect_csi=True, return_crc=True)
        ls_detector = make_repaired_detector(int(context.grid.bits_per_symbol)).to(device)
        for snr_index, raw_snr in enumerate(section["ebno_db"]):
            snr = float(raw_snr)
            for rep in range(int(section["repetitions"])):
                key = (case.name, snr, rep)
                if key in done:
                    continue
                seed = int(config["seed"]) + 10_000_000 + case_index * 100_000 + snr_index * 1_000 + rep
                set_all_seeds(seed)
                batch = context.sample(int(section["batch_size"]), snr)
                with torch.inference_mode():
                    state, graph, _ = pilot_state_and_reference(bridge, batch)
                    initial = initial_detector_output(bridge, detector, batch, state, graph)
                    winner_output = turbo_forward(
                        bridge, detector, batch, winner_setting,
                        state=state, reference_graph=graph, initial_output=initial,
                    )
                    uncertainty_off = repaired_forward(
                        bridge, detector, batch,
                        posterior=winner_output["posterior"],
                        reference_graph=graph,
                        covariance_mode="none",
                    )
                    uncertainty_off["posterior_metrics"] = winner_output["posterior_metrics"]
                    uncertainty_off["turbo_diagnostics"] = winner_output["turbo_diagnostics"]
                    oracle = turbo_forward(
                        bridge, detector, batch, oracle_setting,
                        state=state, reference_graph=graph, initial_output=initial,
                        oracle_symbols=True,
                    )
                    ls_repaired = ls_repaired_forward(
                        ls_receiver, context, ls_detector, batch
                    )
                    outputs = {
                        "winner_turbo": winner_output,
                        "pilot_only_extended": initial,
                        "winner_uncertainty_off": uncertainty_off,
                        "oracle_symbol_turbo": oracle,
                        "ls_estimate_repaired": ls_repaired,
                    }
                    decoded = decode_outputs(
                        context, batch, outputs,
                        bp_iterations=int(config["bp_iterations"]), device=device,
                    )
                    ls_metrics = run_standard_receiver(
                        ls_receiver, batch, batch.information_bits, perfect_csi=False
                    )
                    perfect_metrics = run_standard_receiver(
                        perfect_receiver, batch, batch.information_bits, perfect_csi=True
                    )
                group = str(raw_case["group"])
                rows = [
                    custom_row(
                        case=case, group=group, variant=name, snr=snr, rep=rep,
                        seed=seed, output=output, batch=batch,
                        decoded=decoded[name], signature=contract["signature"],
                    )
                    for name, output in outputs.items()
                ]
                rows.extend([
                    standard_row(case=case, group=group, variant="ls_lmmse", snr=snr, rep=rep, seed=seed, metrics=ls_metrics, signature=contract["signature"]),
                    standard_row(case=case, group=group, variant="perfect_csi_lmmse", snr=snr, rep=rep, seed=seed, metrics=perfect_metrics, signature=contract["signature"]),
                ])
                if {row["variant"] for row in rows} != set(variants):
                    raise RuntimeError("Holdout paired variant set mismatch")
                append_rows_atomic(HOLDOUT_RAW, rows)
                done.add(key)
                print(json.dumps({
                    "phase": "holdout", "case": case.name, "ebno_db": snr,
                    "rep": rep, "rows_committed": len(rows),
                    "completed_rows": len(done) * len(variants),
                    "expected_rows": EXPECTED_HOLDOUT_ROWS,
                }), flush=True)
                if time.time() >= deadline_epoch:
                    frame = pd.read_csv(HOLDOUT_RAW)
                    return frame, {"complete": False, "rows": len(frame), "expected_rows": EXPECTED_HOLDOUT_ROWS, "contract": contract}
        del bridge, detector, ls_receiver, perfect_receiver, ls_detector
        if device.type == "cuda":
            torch.cuda.empty_cache()
    frame = pd.read_csv(HOLDOUT_RAW)
    unique = len(frame.drop_duplicates(["case", "variant", "ebno_db", "rep"]))
    if len(frame) != EXPECTED_HOLDOUT_ROWS or unique != EXPECTED_HOLDOUT_ROWS:
        raise RuntimeError(f"Holdout incomplete: rows={len(frame)}, unique={unique}")
    return frame, {"complete": True, "rows": len(frame), "unique_rows": unique, "expected_rows": EXPECTED_HOLDOUT_ROWS, "contract": contract}


def paired_delta(frame: pd.DataFrame, reference: str, comparator: str, metric: str) -> dict[str, float]:
    keys = ["case", "ebno_db", "rep", "eval_seed"]
    a = frame[frame["variant"] == reference]
    b = frame[frame["variant"] == comparator]
    merged = a.merge(b, on=keys, suffixes=("_a", "_b"))
    values = (
        pd.to_numeric(merged[f"{metric}_a"], errors="coerce")
        - pd.to_numeric(merged[f"{metric}_b"], errors="coerce")
    ).dropna()
    mean = float(values.mean())
    std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    if len(values) > 1:
        try:
            from scipy.stats import t as student_t
            critical = float(student_t.ppf(0.975, len(values) - 1))
        except Exception:
            critical = 1.96
    else:
        critical = 0.0
    half = critical * std / math.sqrt(max(len(values), 1))
    return {"pairs": int(len(values)), "mean": mean, "ci95_low": mean-half, "ci95_high": mean+half}


def mean_metric(frame: pd.DataFrame, variant: str, metric: str, *, snr: float | None = None) -> float:
    sub = frame[frame["variant"] == variant]
    if snr is not None:
        sub = sub[sub["ebno_db"] == float(snr)]
    values = pd.to_numeric(sub[metric], errors="coerce").dropna()
    return float(values.mean()) if len(values) else float("nan")


def make_plots(selection: pd.DataFrame, holdout: pd.DataFrame, ranking: list[dict[str, Any]]) -> list[str]:
    out = ROOT / "outputs/plots"
    out.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    plt.figure(figsize=(8, 4.8))
    names = [item["variant"] for item in ranking]
    values = [item["selection_objective"] for item in ranking]
    plt.bar(np.arange(len(names)), values)
    plt.xticks(np.arange(len(names)), names, rotation=35, ha="right", fontsize=7)
    plt.ylabel("Selection objective")
    plt.tight_layout()
    path = out / "gate1_turbo_posterior_selection.png"
    plt.savefig(path, dpi=180); plt.close(); paths.append(str(path.relative_to(ROOT)))
    plt.figure(figsize=(7, 4.7))
    for variant in ["winner_turbo", "pilot_only_extended", "ls_lmmse", "perfect_csi_lmmse"]:
        sub = holdout[holdout["variant"] == variant]
        grouped = sub.groupby("ebno_db")["tbler"].mean().sort_index()
        plt.plot(grouped.index, grouped.values, marker="o", label=variant)
    plt.xlabel("$E_b/N_0$ (dB)"); plt.ylabel("TBLER"); plt.ylim(-0.01, 0.25)
    plt.grid(True, alpha=0.3); plt.legend(fontsize=8); plt.tight_layout()
    path = out / "gate1_turbo_posterior_12prb_tbler.png"
    plt.savefig(path, dpi=180); plt.close(); paths.append(str(path.relative_to(ROOT)))
    return paths


def classify(holdout: pd.DataFrame) -> dict[str, Any]:
    comparisons = {
        "winner_minus_pilot_tbler": paired_delta(holdout, "winner_turbo", "pilot_only_extended", "tbler"),
        "winner_minus_uncertainty_off_tbler": paired_delta(holdout, "winner_turbo", "winner_uncertainty_off", "tbler"),
        "winner_minus_ls_tbler": paired_delta(holdout, "winner_turbo", "ls_lmmse", "tbler"),
        "winner_minus_ls_repaired_tbler": paired_delta(holdout, "winner_turbo", "ls_estimate_repaired", "tbler"),
        "oracle_minus_pilot_tbler": paired_delta(holdout, "oracle_symbol_turbo", "pilot_only_extended", "tbler"),
        "ls_repaired_minus_ls_tbler": paired_delta(holdout, "ls_estimate_repaired", "ls_lmmse", "tbler"),
    }
    winner_tbler = mean_metric(holdout, "winner_turbo", "tbler")
    ls_tbler = mean_metric(holdout, "ls_lmmse", "tbler")
    coverage = mean_metric(holdout, "winner_turbo", "coverage95")
    normalized = mean_metric(holdout, "winner_turbo", "normalized_error_mean")
    tbler10 = mean_metric(holdout, "winner_turbo", "tbler", snr=10.0)
    tbler14 = mean_metric(holdout, "winner_turbo", "tbler", snr=14.0)
    checks = {
        "winner_improves_pilot_only": comparisons["winner_minus_pilot_tbler"]["ci95_high"] < 0.0,
        "uncertainty_helps_after_turbo": comparisons["winner_minus_uncertainty_off_tbler"]["ci95_high"] < 0.0,
        "winner_beats_ls": comparisons["winner_minus_ls_tbler"]["ci95_high"] < 0.0,
        "winner_within_0p01_of_ls": comparisons["winner_minus_ls_tbler"]["mean"] <= 0.01,
        "ls_factorized_matches_standard": abs(comparisons["ls_repaired_minus_ls_tbler"]["mean"]) <= 0.01,
        "oracle_soft_data_has_value": comparisons["oracle_minus_pilot_tbler"]["ci95_high"] < 0.0,
        "no_high_snr_reversal": tbler14 <= tbler10 + 0.01,
        "coverage_reasonable": 0.90 <= coverage <= 0.99,
        "normalized_error_reasonable": 0.70 <= normalized <= 1.40,
    }
    if not checks["ls_factorized_matches_standard"]:
        classification = "GATE1_TURBO_DETECTOR_ESTIMATOR_INTERFACE_REPAIR_REQUIRED"
        next_action = "REPAIR_LS_FACTORIZATION_BEFORE_FURTHER_ARCHITECTURE_CHANGES"
    elif checks["winner_beats_ls"] and checks["no_high_snr_reversal"]:
        classification = "GATE1_TURBO_POSTERIOR_BEATS_LS"
        next_action = "FREEZE_ARCHITECTURE_AND_RUN_PUBLICATION_SCALE_CAMPAIGN"
    elif checks["winner_within_0p01_of_ls"] and checks["winner_improves_pilot_only"]:
        classification = "GATE1_TURBO_POSTERIOR_NEAR_LS"
        next_action = "ADD_DECODER_EXTRINSIC_FEEDBACK_AND_RUN_FINAL_GATE"
    elif checks["winner_improves_pilot_only"]:
        classification = "GATE1_SOFT_DATA_POSTERIOR_REFINEMENT_SUPPORTED"
        next_action = "ADD_DECODER_EXTRINSIC_FEEDBACK_OR_LOCALIZED_POSTERIOR"
    elif checks["oracle_soft_data_has_value"]:
        classification = "GATE1_DECODER_RELIABILITY_FEEDBACK_REQUIRED"
        next_action = "REPLACE_DETECTOR_MOMENTS_WITH_DECODER_EXTRINSIC_SOFT_SYMBOLS"
    else:
        classification = "GATE1_POSTERIOR_BASIS_LOCALIZATION_REQUIRED"
        next_action = "REPLACE_GLOBAL_RFF_WITH_LOCALIZED_DELAY_DOPPLER_POSTERIOR"
    return {
        "classification": classification,
        "next_action": next_action,
        "scientific_checks": checks,
        "paired_comparisons": comparisons,
        "metrics": {
            "winner_12prb_tbler": winner_tbler,
            "pilot_only_12prb_tbler": mean_metric(holdout, "pilot_only_extended", "tbler"),
            "ls_12prb_tbler": ls_tbler,
            "ls_repaired_12prb_tbler": mean_metric(holdout, "ls_estimate_repaired", "tbler"),
            "perfect_12prb_tbler": mean_metric(holdout, "perfect_csi_lmmse", "tbler"),
            "oracle_turbo_12prb_tbler": mean_metric(holdout, "oracle_symbol_turbo", "tbler"),
            "winner_12prb_10db_tbler": tbler10,
            "winner_12prb_14db_tbler": tbler14,
            "winner_12prb_coverage95": coverage,
            "winner_12prb_normalized_error": normalized,
            "winner_12prb_channel_nmse": mean_metric(holdout, "winner_turbo", "channel_nmse"),
        },
    }


def aggregate(selection: pd.DataFrame, holdout: pd.DataFrame) -> pd.DataFrame:
    frame = pd.concat([selection.assign(phase="selection"), holdout.assign(phase="holdout")], ignore_index=True)
    metrics = ["information_ber", "tbler", "crc_failure_rate", "coded_ber", "coded_bit_nll", "channel_nmse", "normalized_error_mean", "coverage95", "edge_density", "selected_observations", "latent_trace_reduction_fraction"]
    return frame.groupby(["phase", "case", "group", "num_prb", "variant", "ebno_db"], dropna=False)[metrics].agg(["mean", "std", "count"]).reset_index()


def write_incomplete(selection: dict[str, Any] | None, holdout: dict[str, Any] | None, winner: str | None) -> None:
    report = {
        "version": TURBO_GATE_VERSION,
        "complete": False,
        "selection": selection,
        "holdout": holdout,
        "winner": winner,
        "classification": "GATE1_NR_TURBO_POSTERIOR_INCOMPLETE",
        "next_action": "RESUBMIT_SAME_COMMAND",
        "publication_nr_ready": False,
    }
    save_json(report, REPORT_PATH); save_json(report, GATE_JSON)
    GATE_TXT.parent.mkdir(parents=True, exist_ok=True)
    GATE_TXT.write_text("CLASSIFICATION: GATE1_NR_TURBO_POSTERIOR_INCOMPLETE\nNEXT_ACTION: RESUBMIT_SAME_COMMAND\nPUBLICATION_NR_READY: NO\n", encoding="utf-8")
    print("GATE1_NR_TURBO_POSTERIOR_INCOMPLETE: RESUBMIT")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/gate1_nr_turbo_posterior_screen.yaml")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--deadline-minutes", type=float, default=40.0)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    config_path = ROOT / args.config
    config = load_yaml(config_path)
    pre = extension_preconditions()
    candidates = [TurboSetting.from_mapping(item) for item in config["candidates"]]
    selection_expected = len(config["selection"]["cases"]) * len(config["selection"]["ebno_db"]) * int(config["selection"]["repetitions"]) * (len(candidates) + 4)
    holdout_expected = len(config["holdout"]["cases"]) * len(config["holdout"]["ebno_db"]) * int(config["holdout"]["repetitions"]) * 7
    if selection_expected != EXPECTED_SELECTION_ROWS or holdout_expected != EXPECTED_HOLDOUT_ROWS:
        raise RuntimeError((selection_expected, holdout_expected))
    if args.preflight_only:
        smoke = json.loads((ROOT / "outputs/gates/GATE1_NR_TURBO_POSTERIOR_SMOKE.json").read_text(encoding="utf-8"))
        if smoke.get("classification") != "GATE1_NR_TURBO_POSTERIOR_SMOKE_PASS" or smoke.get("overall_pass") is not True:
            raise RuntimeError("Turbo posterior smoke has not passed")
        print("GATE1_NR_TURBO_POSTERIOR_SCREEN_PREFLIGHT_PASS")
        print("CANDIDATES", len(candidates))
        print("SELECTION_ROWS", EXPECTED_SELECTION_ROWS)
        print("HOLDOUT_ROWS", EXPECTED_HOLDOUT_ROWS)
        print("EXPECTED_ROWS", EXPECTED_ROWS)
        print("HOLDOUT_USED_FOR_SELECTION NO")
        print("TRAINING_REQUIRED NO")
        return
    device = normalize_device(args.device)
    deadline_epoch = time.time() + 60.0 * float(args.deadline_minutes)
    selection_frame, selection_report = run_selection(config, config_path, device, pre, deadline_epoch=deadline_epoch)
    if not selection_report["complete"]:
        write_incomplete(selection_report, None, None); return
    winner, ranking = choose_winner(selection_frame, candidates)
    if time.time() >= deadline_epoch:
        write_incomplete(selection_report, None, winner); return
    holdout_frame, holdout_report = run_holdout(config, config_path, device, pre, winner, selection_report["contract"], deadline_epoch=deadline_epoch)
    if not holdout_report["complete"]:
        write_incomplete(selection_report, holdout_report, winner); return
    combined_unique = len(pd.concat([selection_frame, holdout_frame]).drop_duplicates(["case", "variant", "ebno_db", "rep"]))
    if len(selection_frame) + len(holdout_frame) != EXPECTED_ROWS or combined_unique != EXPECTED_ROWS:
        raise RuntimeError("Combined turbo evidence row count mismatch")
    aggregate_frame = aggregate(selection_frame, holdout_frame)
    AGGREGATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    aggregate_frame.to_csv(AGGREGATE_PATH, index=False)
    scientific = classify(holdout_frame)
    plots = make_plots(selection_frame, holdout_frame, ranking)
    report = {
        "version": TURBO_GATE_VERSION,
        "complete": True,
        "classification": scientific["classification"],
        "next_action": scientific["next_action"],
        "publication_nr_ready": False,
        "preconditions": pre,
        "winner": winner,
        "winner_setting": next(item.as_dict() for item in candidates if item.name == winner),
        "selection_ranking": ranking,
        "selection": selection_report,
        "holdout": holdout_report,
        "evaluation": {"rows": EXPECTED_ROWS, "unique_rows": combined_unique, "expected_rows": EXPECTED_ROWS, "complete": True},
        "aggregate_csv": str(AGGREGATE_PATH.relative_to(ROOT)),
        "plots": plots,
        "environment": {"device": str(device), "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None, "torch": torch.__version__, "python": platform.python_version(), "slurm_job_id": os.environ.get("SLURM_JOB_ID")},
        **scientific,
    }
    save_json(report, REPORT_PATH); save_json(report, GATE_JSON)
    lines = [
        "complete_rows: PASS", "selection_holdout_disjoint: PASS",
        "holdout_used_for_selection: NO", "training_required: NO",
        *(f"{name}: {'PASS' if value else 'FAIL'}" for name, value in report["scientific_checks"].items()),
        f"WINNER: {winner}", f"CLASSIFICATION: {report['classification']}",
        f"NEXT_ACTION: {report['next_action']}", "PUBLICATION_NR_READY: NO",
    ]
    GATE_TXT.parent.mkdir(parents=True, exist_ok=True)
    GATE_TXT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
