#!/usr/bin/env python3
from __future__ import annotations

"""No-retraining audit of allocation-normalized versus fixed physical grid coordinates."""

import argparse
from dataclasses import replace
import hashlib
import json
import math
import platform
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml

from bayesroute.nr_gate1 import (
    NRBayesRouteBridge,
    NRCase,
    build_nr_context,
    decode_bridge,
    normalize_device,
    run_standard_receiver,
    standard_receiver,
)
from gate1_nr_joint_operator_common import (
    coded_metrics,
    load_operator_state,
    make_repaired_detector,
    package_signature,
    posterior_metrics,
    repaired_forward,
    save_json,
    set_all_seeds,
    sha256_file,
)

ROOT = Path(__file__).resolve().parents[1]
AUDIT_VERSION = "gate1_nr_grid_scale_audit_v1"
EXPECTED_ROWS = 360
FROZEN_CHECKPOINT_SHA256 = (
    "4f71c7a0a925005d676687e90c5a241668cfcfed21503e2874c3528721c66980"
)
CLEAN_CLASSIFICATION = "GATE1_CLEAN_GENERALIZATION_SUPPORTED"
REANALYSIS_CLASSIFICATION = "CLEAN_GENERALIZATION_SUPPORTED_WITH_GRID_SCALE_EXCEPTION"
RAW_PATH = ROOT / "outputs/eval/gate1_nr_grid_scale_audit.csv"
CONTRACT_PATH = ROOT / "outputs/eval/gate1_nr_grid_scale_audit_contract.json"
REPORT_PATH = ROOT / "outputs/reports/gate1_nr_grid_scale_audit.json"
GATE_JSON_PATH = ROOT / "outputs/gates/GATE1_NR_GRID_SCALE_AUDIT.json"
GATE_TXT_PATH = ROOT / "outputs/gates/GATE1_NR_GRID_SCALE_AUDIT.txt"
AGGREGATE_PATH = ROOT / "outputs/eval/gate1_nr_grid_scale_audit_aggregate.csv"

SOURCE_FILES = (
    "configs/gate1_nr_grid_scale_audit.yaml",
    "scripts/gate1_nr_grid_scale_audit.py",
    "scripts/gate1_nr_joint_operator_common.py",
    "src/bayesroute/lmmse_ep.py",
    "src/bayesroute/models.py",
    "src/bayesroute/nr_gate1.py",
    "src/bayesroute/qam.py",
)


def load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def source_hashes() -> dict[str, str]:
    return {name: sha256_file(ROOT / name) for name in SOURCE_FILES}


def expected_rows(config: dict[str, Any]) -> int:
    evaluation = config["evaluation"]
    return (
        len(evaluation["cases"])
        * len(evaluation["ebno_db"])
        * int(evaluation["repetitions"])
        * len(evaluation["variants"])
    )


def verify_preconditions(config: dict[str, Any]) -> dict[str, Any]:
    clean_report_path = (
        ROOT / "outputs/reports/gate1_nr_clean_generalization.json"
    )
    clean_gate_path = ROOT / "outputs/gates/GATE1_NR_CLEAN_GENERALIZATION.txt"
    reanalysis_path = (
        ROOT / "outputs/reports/gate1_nr_clean_generalization_reanalysis.json"
    )
    checkpoint = ROOT / str(config["frozen_candidate"]["checkpoint_path"])
    for path in (clean_report_path, clean_gate_path, reanalysis_path, checkpoint):
        if not path.is_file():
            raise RuntimeError(f"Missing grid-audit precondition: {path}")
    clean = json.loads(clean_report_path.read_text(encoding="utf-8"))
    reanalysis = json.loads(reanalysis_path.read_text(encoding="utf-8"))
    checkpoint_sha = sha256_file(checkpoint)
    checks = {
        "clean_complete": clean.get("complete") is True,
        "clean_classification": clean.get("classification") == CLEAN_CLASSIFICATION,
        "clean_rows": clean.get("evaluation", {}).get("rows") == 4800,
        "clean_source_contract": clean.get("software_checks", {}).get(
            "source_contract"
        )
        is True,
        "reanalysis_complete": reanalysis.get("complete") is True,
        "reanalysis_classification": (
            reanalysis.get("classification") == REANALYSIS_CLASSIFICATION
        ),
        "grid_exception_recorded": reanalysis.get("scientific_checks", {}).get(
            "grid_scale_exception_detected"
        )
        is True,
        "checkpoint": checkpoint_sha == FROZEN_CHECKPOINT_SHA256,
        "config_checkpoint": (
            config["frozen_candidate"]["checkpoint_sha256"]
            == FROZEN_CHECKPOINT_SHA256
        ),
        "revision": config.get("grid_scale_revision") == AUDIT_VERSION,
        "expected_rows": expected_rows(config) == EXPECTED_ROWS,
    }
    if not all(checks.values()):
        raise RuntimeError(f"Grid-audit precondition failure: {checks}")
    return {
        "passed": True,
        "checks": checks,
        "checkpoint_sha256": checkpoint_sha,
        "clean_classification": clean.get("classification"),
        "reanalysis_classification": reanalysis.get("classification"),
    }


def reference_physical_coords(
    grid: Any,
    *,
    subcarrier_spacing_khz: float,
    reference_num_subcarriers: int,
    reference_scs_khz: float,
) -> torch.Tensor:
    """Coordinates with fixed physical spacing across allocation widths.

    One frequency coordinate unit equals the bandwidth of a 4-PRB, 30-kHz
    reference allocation. The 4-PRB/30-kHz case is therefore exactly equivalent
    to the existing normalization, while 8/12-PRB grids span a wider interval.
    """
    device = grid.coords.device
    num_symbols = int(grid.num_ofdm_symbols)
    num_subcarriers = int(grid.num_effective_subcarriers)
    time_axis = torch.arange(num_symbols, dtype=torch.float32, device=device)
    freq_axis = torch.arange(num_subcarriers, dtype=torch.float32, device=device)
    tt, ff = torch.meshgrid(time_axis, freq_axis, indexing="ij")
    coords = torch.stack([ff.reshape(-1), tt.reshape(-1)], dim=-1)
    reference_span = float(reference_num_subcarriers - 1)
    if reference_span <= 0:
        raise ValueError("reference_num_subcarriers must exceed one")
    frequency_scale = float(subcarrier_spacing_khz) / float(reference_scs_khz)
    coords[:, 0] = (
        (coords[:, 0] - coords[:, 0].mean())
        * frequency_scale
        / reference_span
    )
    if num_symbols > 1:
        coords[:, 1] = (coords[:, 1] - coords[:, 1].mean()) / float(
            num_symbols - 1
        )
    return coords


def build_bridge(case: NRCase, grid: Any, config: dict[str, Any], device: torch.device) -> NRBayesRouteBridge:
    spec = config["frozen_candidate"]
    detector = config["selected_detector"]
    return NRBayesRouteBridge(
        grid,
        num_streams=case.num_streams,
        rank=int(spec["rank"]),
        bank_rank=int(spec["bank_rank"]),
        detector_iterations=int(detector["iterations"]),
        edge_mass=float(detector["edge_mass"]),
        length_f=float(spec["length_f"]),
        length_t=float(spec["length_t"]),
        operator_seed=58001,
    ).to(device)


def load_frozen_operator(bridge: NRBayesRouteBridge, checkpoint_path: Path, device: torch.device) -> None:
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    operators = state.get("operators")
    if not isinstance(operators, list) or len(operators) != 1:
        raise RuntimeError("Frozen global checkpoint must contain one operator state")
    load_operator_state(bridge, operators[0])


def experiment_contract(
    config: dict[str, Any], config_path: Path, preconditions: dict[str, Any]
) -> dict[str, Any]:
    payload = {
        "version": AUDIT_VERSION,
        "config_sha256": sha256_file(config_path),
        "source_sha256": source_hashes(),
        "frozen_checkpoint_sha256": preconditions["checkpoint_sha256"],
        "frozen_candidate": config["frozen_candidate"],
        "selected_detector": config["selected_detector"],
        "coordinate_systems": config["coordinate_systems"],
        "evaluation": config["evaluation"],
    }
    payload["signature"] = package_signature(payload)
    return payload


def append_rows_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    new = pd.DataFrame(rows)
    if path.is_file():
        old = pd.read_csv(path)
        if list(old.columns) != list(new.columns):
            raise RuntimeError("Grid-audit CSV column contract mismatch")
        combined = pd.concat([old, new], ignore_index=True)
    else:
        combined = new
    tmp = path.with_suffix(path.suffix + ".tmp")
    combined.to_csv(tmp, index=False)
    tmp.replace(path)


def decode_pair(
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
    group: str,
    variant: str,
    coordinate_mode: str,
    snr: float,
    rep: int,
    seed: int,
    output: dict[str, Any],
    batch: Any,
    decoded: dict[str, Any],
    contract_signature: str,
    coords_max_abs_diff: float,
    logits_max_abs_diff: float,
    posterior_mean_max_abs_diff: float,
) -> dict[str, Any]:
    return {
        "case": case.name,
        "group": group,
        "scenario": case.scenario,
        "num_users": int(case.num_users),
        "num_streams": int(case.num_streams),
        "num_rx_ant": int(case.num_rx_ant),
        "num_prb": int(case.num_prb),
        "subcarrier_spacing_khz": int(case.subcarrier_spacing_khz),
        "variant": variant,
        "coordinate_mode": coordinate_mode,
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
        **posterior_metrics(output, batch),
        "edge_density": float(output["edge_density"].item()),
        "coords_max_abs_diff": float(coords_max_abs_diff),
        "paired_logits_max_abs_diff": float(logits_max_abs_diff),
        "paired_posterior_mean_max_abs_diff": float(posterior_mean_max_abs_diff),
        "contract_signature": contract_signature,
    }


def standard_row(
    *,
    case: NRCase,
    group: str,
    variant: str,
    snr: float,
    rep: int,
    seed: int,
    metrics: dict[str, Any],
    contract_signature: str,
) -> dict[str, Any]:
    return {
        "case": case.name,
        "group": group,
        "scenario": case.scenario,
        "num_users": int(case.num_users),
        "num_streams": int(case.num_streams),
        "num_rx_ant": int(case.num_rx_ant),
        "num_prb": int(case.num_prb),
        "subcarrier_spacing_khz": int(case.subcarrier_spacing_khz),
        "variant": variant,
        "coordinate_mode": "standard",
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
        "coords_max_abs_diff": float("nan"),
        "paired_logits_max_abs_diff": float("nan"),
        "paired_posterior_mean_max_abs_diff": float("nan"),
        "contract_signature": contract_signature,
    }


def evaluate(
    config: dict[str, Any],
    config_path: Path,
    preconditions: dict[str, Any],
    device: torch.device,
    *,
    deadline_epoch: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    evaluation = config["evaluation"]
    variants = [str(x) for x in evaluation["variants"]]
    contract = experiment_contract(config, config_path, preconditions)
    RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
    if RAW_PATH.is_file():
        if not CONTRACT_PATH.is_file():
            raise RuntimeError("Grid-audit CSV exists without contract")
        existing = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
        if existing.get("signature") != contract["signature"]:
            raise RuntimeError("Grid-audit resume contract mismatch")
    else:
        save_json(contract, CONTRACT_PATH)

    done_batches: set[tuple[str, float, int]] = set()
    if RAW_PATH.is_file():
        existing = pd.read_csv(RAW_PATH)
        keys = ["case", "variant", "ebno_db", "rep"]
        if existing[keys].duplicated().any():
            raise RuntimeError("Grid-audit contains duplicate keys")
        batch_counts = existing.groupby(["case", "ebno_db", "rep"])[
            "variant"
        ].nunique()
        partial = batch_counts[batch_counts != len(variants)]
        if len(partial):
            raise RuntimeError(
                "Grid-audit contains a partial non-atomic paired batch: "
                + partial.to_string()
            )
        done_batches = set(
            (str(case), float(snr), int(rep))
            for case, snr, rep in batch_counts.index
        )

    checkpoint_path = ROOT / str(config["frozen_candidate"]["checkpoint_path"])
    ref_cfg = config["coordinate_systems"]["reference_physical"]
    raw_cases = list(evaluation["cases"])
    group_by_name = {str(item["name"]): str(item["group"]) for item in raw_cases}
    for case_index, raw_case in enumerate(raw_cases):
        case = NRCase.from_mapping(raw_case)
        context = build_nr_context(case, device)
        physical_coords = reference_physical_coords(
            context.grid,
            subcarrier_spacing_khz=float(case.subcarrier_spacing_khz),
            reference_num_subcarriers=int(ref_cfg["reference_num_subcarriers"]),
            reference_scs_khz=float(ref_cfg["reference_scs_khz"]),
        )
        physical_grid = replace(context.grid, coords=physical_coords)
        coord_diff = float(
            torch.max(torch.abs(context.grid.coords - physical_coords)).item()
        )
        allocation_bridge = build_bridge(case, context.grid, config, device)
        physical_bridge = build_bridge(case, physical_grid, config, device)
        load_frozen_operator(allocation_bridge, checkpoint_path, device)
        load_frozen_operator(physical_bridge, checkpoint_path, device)
        allocation_bridge.eval()
        physical_bridge.eval()
        allocation_detector = make_repaired_detector(
            int(context.grid.bits_per_symbol)
        ).to(device)
        physical_detector = make_repaired_detector(
            int(context.grid.bits_per_symbol)
        ).to(device)
        ls_receiver = standard_receiver(context, perfect_csi=False, return_crc=True)
        perfect_receiver = standard_receiver(context, perfect_csi=True, return_crc=True)

        for snr_index, raw_snr in enumerate(evaluation["ebno_db"]):
            snr = float(raw_snr)
            for rep_index in range(int(evaluation["repetitions"])):
                batch_key = (case.name, snr, rep_index)
                if batch_key in done_batches:
                    continue
                seed = (
                    int(config["seed"])
                    + 10_000_000
                    + 100_000 * case_index
                    + 1_000 * snr_index
                    + rep_index
                )
                set_all_seeds(seed)
                batch = context.sample(
                    batch_size=int(evaluation["batch_size"]), ebno_db=snr
                )
                with torch.inference_mode():
                    allocation_output = repaired_forward(
                        allocation_bridge, allocation_detector, batch
                    )
                    physical_output = repaired_forward(
                        physical_bridge, physical_detector, batch
                    )
                    logits_diff = float(
                        torch.max(
                            torch.abs(
                                allocation_output["bit_logits"]
                                - physical_output["bit_logits"]
                            )
                        ).item()
                    )
                    posterior_diff = float(
                        torch.max(
                            torch.abs(
                                allocation_output["posterior"].mean
                                - physical_output["posterior"].mean
                            )
                        ).item()
                    )
                    decoded = decode_pair(
                        context,
                        batch,
                        {
                            "frozen_allocation_normalized": allocation_output,
                            "frozen_reference_physical": physical_output,
                        },
                        bp_iterations=int(evaluation["bp_iterations"]),
                        device=device,
                    )
                    ls_metrics = run_standard_receiver(
                        ls_receiver,
                        batch,
                        batch.information_bits,
                        perfect_csi=False,
                    )
                    perfect_metrics = run_standard_receiver(
                        perfect_receiver,
                        batch,
                        batch.information_bits,
                        perfect_csi=True,
                    )

                group = group_by_name[case.name]
                rows = [
                    custom_row(
                        case=case,
                        group=group,
                        variant="frozen_allocation_normalized",
                        coordinate_mode="allocation_normalized",
                        snr=snr,
                        rep=rep_index,
                        seed=seed,
                        output=allocation_output,
                        batch=batch,
                        decoded=decoded["frozen_allocation_normalized"],
                        contract_signature=contract["signature"],
                        coords_max_abs_diff=coord_diff,
                        logits_max_abs_diff=logits_diff,
                        posterior_mean_max_abs_diff=posterior_diff,
                    ),
                    custom_row(
                        case=case,
                        group=group,
                        variant="frozen_reference_physical",
                        coordinate_mode="reference_physical",
                        snr=snr,
                        rep=rep_index,
                        seed=seed,
                        output=physical_output,
                        batch=batch,
                        decoded=decoded["frozen_reference_physical"],
                        contract_signature=contract["signature"],
                        coords_max_abs_diff=coord_diff,
                        logits_max_abs_diff=logits_diff,
                        posterior_mean_max_abs_diff=posterior_diff,
                    ),
                    standard_row(
                        case=case,
                        group=group,
                        variant="ls_lmmse",
                        snr=snr,
                        rep=rep_index,
                        seed=seed,
                        metrics=ls_metrics,
                        contract_signature=contract["signature"],
                    ),
                    standard_row(
                        case=case,
                        group=group,
                        variant="perfect_csi_lmmse",
                        snr=snr,
                        rep=rep_index,
                        seed=seed,
                        metrics=perfect_metrics,
                        contract_signature=contract["signature"],
                    ),
                ]
                append_rows_atomic(RAW_PATH, rows)
                done_batches.add(batch_key)
                print(
                    json.dumps(
                        {
                            "case": case.name,
                            "num_prb": case.num_prb,
                            "ebno_db": snr,
                            "rep": rep_index,
                            "rows_committed": len(rows),
                            "completed_rows": len(done_batches) * len(variants),
                            "expected_rows": EXPECTED_ROWS,
                        }
                    ),
                    flush=True,
                )
                if time.time() >= deadline_epoch:
                    frame = pd.read_csv(RAW_PATH)
                    return frame, {
                        "complete": False,
                        "rows": int(len(frame)),
                        "unique_rows": int(
                            len(
                                frame.drop_duplicates(
                                    ["case", "variant", "ebno_db", "rep"]
                                )
                            )
                        ),
                        "expected_rows": EXPECTED_ROWS,
                        "stop_reason": "internal_deadline",
                        "contract": contract,
                    }

        del (
            context,
            allocation_bridge,
            physical_bridge,
            allocation_detector,
            physical_detector,
            ls_receiver,
            perfect_receiver,
        )
        if device.type == "cuda":
            torch.cuda.empty_cache()

    frame = pd.read_csv(RAW_PATH)
    keys = ["case", "variant", "ebno_db", "rep"]
    unique = int(len(frame.drop_duplicates(keys)))
    if len(frame) != EXPECTED_ROWS or unique != EXPECTED_ROWS:
        raise RuntimeError(
            f"Grid-audit incomplete: rows={len(frame)}, unique={unique}, "
            f"expected={EXPECTED_ROWS}"
        )
    return frame, {
        "complete": True,
        "rows": int(len(frame)),
        "unique_rows": unique,
        "expected_rows": EXPECTED_ROWS,
        "raw_csv": str(RAW_PATH.relative_to(ROOT)),
        "contract": contract,
    }


def paired_delta(
    frame: pd.DataFrame,
    reference: str,
    comparator: str,
    metric: str,
    *,
    prb: int | None = None,
    wide_only: bool = False,
) -> dict[str, float]:
    sub = frame.copy()
    if prb is not None:
        sub = sub[sub["num_prb"] == int(prb)]
    if wide_only:
        sub = sub[sub["num_prb"] >= 8]
    keys = ["case", "ebno_db", "rep", "eval_seed"]
    a = sub[sub["variant"] == reference]
    b = sub[sub["variant"] == comparator]
    merged = a.merge(b, on=keys, suffixes=("_a", "_b"))
    values = (
        pd.to_numeric(merged[f"{metric}_a"], errors="coerce")
        - pd.to_numeric(merged[f"{metric}_b"], errors="coerce")
    ).dropna()
    mean = float(values.mean()) if len(values) else float("nan")
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
    return {
        "pairs": int(len(values)),
        "mean": mean,
        "ci95_low": mean - half,
        "ci95_high": mean + half,
    }


def mean_metric(
    frame: pd.DataFrame,
    *,
    variant: str,
    metric: str,
    prb: int,
    snr: float | None = None,
) -> float:
    sub = frame[(frame["variant"] == variant) & (frame["num_prb"] == int(prb))]
    if snr is not None:
        sub = sub[sub["ebno_db"] == float(snr)]
    values = pd.to_numeric(sub[metric], errors="coerce").dropna()
    return float(values.mean()) if len(values) else float("nan")


def classify(frame: pd.DataFrame) -> dict[str, Any]:
    comparisons = {
        "wide_reference_minus_allocation_tbler": paired_delta(
            frame,
            "frozen_reference_physical",
            "frozen_allocation_normalized",
            "tbler",
            wide_only=True,
        ),
        "wide_reference_minus_allocation_nll": paired_delta(
            frame,
            "frozen_reference_physical",
            "frozen_allocation_normalized",
            "coded_bit_nll",
            wide_only=True,
        ),
        "wide_reference_minus_allocation_nmse": paired_delta(
            frame,
            "frozen_reference_physical",
            "frozen_allocation_normalized",
            "channel_nmse",
            wide_only=True,
        ),
        "8prb_reference_minus_allocation_tbler": paired_delta(
            frame,
            "frozen_reference_physical",
            "frozen_allocation_normalized",
            "tbler",
            prb=8,
        ),
        "12prb_reference_minus_allocation_tbler": paired_delta(
            frame,
            "frozen_reference_physical",
            "frozen_allocation_normalized",
            "tbler",
            prb=12,
        ),
        "wide_reference_minus_ls_tbler": paired_delta(
            frame,
            "frozen_reference_physical",
            "ls_lmmse",
            "tbler",
            wide_only=True,
        ),
    }
    custom = frame[
        frame["variant"].isin(
            ["frozen_allocation_normalized", "frozen_reference_physical"]
        )
    ]
    ref4 = custom[custom["num_prb"] == 4]
    max_coord_diff_4 = float(
        pd.to_numeric(ref4["coords_max_abs_diff"], errors="coerce").max()
    )
    max_logits_diff_4 = float(
        pd.to_numeric(ref4["paired_logits_max_abs_diff"], errors="coerce").max()
    )
    max_posterior_diff_4 = float(
        pd.to_numeric(
            ref4["paired_posterior_mean_max_abs_diff"], errors="coerce"
        ).max()
    )

    per_prb: dict[str, Any] = {}
    for prb in (4, 8, 12):
        allocation_coverage = mean_metric(
            frame,
            variant="frozen_allocation_normalized",
            metric="coverage95",
            prb=prb,
        )
        physical_coverage = mean_metric(
            frame,
            variant="frozen_reference_physical",
            metric="coverage95",
            prb=prb,
        )
        allocation_normalized = mean_metric(
            frame,
            variant="frozen_allocation_normalized",
            metric="normalized_error_mean",
            prb=prb,
        )
        physical_normalized = mean_metric(
            frame,
            variant="frozen_reference_physical",
            metric="normalized_error_mean",
            prb=prb,
        )
        per_prb[str(prb)] = {
            "allocation_tbler": mean_metric(
                frame,
                variant="frozen_allocation_normalized",
                metric="tbler",
                prb=prb,
            ),
            "physical_tbler": mean_metric(
                frame,
                variant="frozen_reference_physical",
                metric="tbler",
                prb=prb,
            ),
            "ls_tbler": mean_metric(
                frame, variant="ls_lmmse", metric="tbler", prb=prb
            ),
            "perfect_tbler": mean_metric(
                frame,
                variant="perfect_csi_lmmse",
                metric="tbler",
                prb=prb,
            ),
            "allocation_coverage95": allocation_coverage,
            "physical_coverage95": physical_coverage,
            "allocation_normalized_error": allocation_normalized,
            "physical_normalized_error": physical_normalized,
            "allocation_10db_tbler": mean_metric(
                frame,
                variant="frozen_allocation_normalized",
                metric="tbler",
                prb=prb,
                snr=10.0,
            ),
            "allocation_14db_tbler": mean_metric(
                frame,
                variant="frozen_allocation_normalized",
                metric="tbler",
                prb=prb,
                snr=14.0,
            ),
            "physical_10db_tbler": mean_metric(
                frame,
                variant="frozen_reference_physical",
                metric="tbler",
                prb=prb,
                snr=10.0,
            ),
            "physical_14db_tbler": mean_metric(
                frame,
                variant="frozen_reference_physical",
                metric="tbler",
                prb=prb,
                snr=14.0,
            ),
            "allocation_calibration_distance": abs(allocation_coverage - 0.95)
            + abs(math.log(max(allocation_normalized, 1e-8))),
            "physical_calibration_distance": abs(physical_coverage - 0.95)
            + abs(math.log(max(physical_normalized, 1e-8))),
        }

    checks = {
        "complete_rows": len(frame) == EXPECTED_ROWS,
        "all_core_metrics_finite": bool(
            np.isfinite(
                frame[
                    frame["variant"].isin(
                        [
                            "frozen_allocation_normalized",
                            "frozen_reference_physical",
                        ]
                    )
                ][
                    [
                        "tbler",
                        "coded_bit_nll",
                        "channel_nmse",
                        "normalized_error_mean",
                        "coverage95",
                    ]
                ].to_numpy(dtype=float)
            ).all()
        ),
        "four_prb_coordinate_equivalence": bool(max_coord_diff_4 <= 1e-7),
        "four_prb_output_equivalence": bool(
            max_logits_diff_4 <= 2e-6 and max_posterior_diff_4 <= 2e-6
        ),
        "wide_grid_tbler_improves": bool(
            comparisons["wide_reference_minus_allocation_tbler"]["ci95_high"]
            < 0.0
        ),
        "wide_grid_nll_improves": bool(
            comparisons["wide_reference_minus_allocation_nll"]["ci95_high"]
            < 0.0
        ),
        "wide_grid_nmse_improves": bool(
            comparisons["wide_reference_minus_allocation_nmse"]["ci95_high"]
            < 0.0
        ),
        "wide_grid_calibration_improves": bool(
            np.mean(
                [
                    per_prb["8"]["physical_calibration_distance"],
                    per_prb["12"]["physical_calibration_distance"],
                ]
            )
            < np.mean(
                [
                    per_prb["8"]["allocation_calibration_distance"],
                    per_prb["12"]["allocation_calibration_distance"],
                ]
            )
        ),
        "physical_no_strict_reversal": bool(
            all(
                per_prb[str(prb)]["physical_14db_tbler"]
                <= per_prb[str(prb)]["physical_10db_tbler"] + 0.01
                for prb in (8, 12)
            )
        ),
    }
    strong_count = sum(
        int(checks[name])
        for name in (
            "wide_grid_tbler_improves",
            "wide_grid_nll_improves",
            "wide_grid_nmse_improves",
            "wide_grid_calibration_improves",
        )
    )
    if (
        checks["four_prb_coordinate_equivalence"]
        and checks["four_prb_output_equivalence"]
        and checks["wide_grid_tbler_improves"]
        and strong_count >= 3
    ):
        classification = "GRID_SCALE_COORDINATE_BUG_CONFIRMED"
        next_action = (
            "RETRAIN_GLOBAL_OPERATOR_WITH_FIXED_PHYSICAL_COORDINATES_"
            "AND_HOLD_OUT_12PRB"
        )
    elif (
        checks["four_prb_coordinate_equivalence"]
        and checks["four_prb_output_equivalence"]
        and strong_count >= 2
    ):
        classification = "GRID_SCALE_COORDINATE_BUG_PARTIALLY_CONFIRMED"
        next_action = "RUN_COORDINATE_PLUS_RANK_DIAGNOSTIC"
    else:
        classification = "GRID_SCALE_COORDINATE_HYPOTHESIS_NOT_SUPPORTED"
        next_action = "EXPAND_OR_LOCALIZE_POSTERIOR_KERNEL_BASIS"
    return {
        "classification": classification,
        "next_action": next_action,
        "scientific_checks": checks,
        "equivalence_maxima": {
            "coords_4prb": max_coord_diff_4,
            "bit_logits_4prb": max_logits_diff_4,
            "posterior_mean_4prb": max_posterior_diff_4,
        },
        "paired_comparisons": comparisons,
        "per_prb_metrics": per_prb,
    }


def make_outputs(
    frame: pd.DataFrame,
    evaluation_report: dict[str, Any],
    preconditions: dict[str, Any],
    classification: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    numeric = [
        "information_ber",
        "tbler",
        "coded_ber",
        "coded_bit_nll",
        "coded_brier",
        "channel_nmse",
        "normalized_error_mean",
        "coverage95",
        "edge_density",
    ]
    aggregate = (
        frame.groupby(
            ["case", "group", "num_prb", "variant", "ebno_db"],
            dropna=False,
        )[numeric]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    aggregate.columns = [
        "_".join(str(x) for x in item if str(x)) if isinstance(item, tuple) else str(item)
        for item in aggregate.columns
    ]
    AGGREGATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    aggregate.to_csv(AGGREGATE_PATH, index=False)

    plot_dir = ROOT / "outputs/plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    variants = [
        "frozen_allocation_normalized",
        "frozen_reference_physical",
        "ls_lmmse",
        "perfect_csi_lmmse",
    ]
    for prb in (4, 8, 12):
        subset = frame[frame["num_prb"] == prb]
        plt.figure(figsize=(7.0, 4.5))
        for variant in variants:
            curve = (
                subset[subset["variant"] == variant]
                .groupby("ebno_db")["tbler"]
                .mean()
                .sort_index()
            )
            plt.plot(curve.index, curve.values, marker="o", label=variant)
        plt.xlabel("Eb/N0 (dB)")
        plt.ylabel("Decoded TBLER")
        plt.title(f"Grid-scale audit: {prb} PRB")
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(plot_dir / f"gate1_grid_scale_{prb}prb_tbler.png", dpi=180)
        plt.close()

    plt.figure(figsize=(7.0, 4.5))
    for variant in (
        "frozen_allocation_normalized",
        "frozen_reference_physical",
    ):
        curve = (
            frame[frame["variant"] == variant]
            .groupby("num_prb")["coverage95"]
            .mean()
            .sort_index()
        )
        plt.plot(curve.index, curve.values, marker="o", label=variant)
    plt.axhline(0.95, linestyle="--", linewidth=1.0)
    plt.xlabel("Allocated PRBs")
    plt.ylabel("Posterior 95% coverage")
    plt.title("Grid-scale posterior calibration")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(plot_dir / "gate1_grid_scale_coverage.png", dpi=180)
    plt.close()

    report = {
        "version": AUDIT_VERSION,
        "complete": True,
        "classification": classification["classification"],
        "next_action": classification["next_action"],
        "publication_nr_ready": False,
        "environment": {
            "device": str(device),
            "gpu_name": (
                torch.cuda.get_device_name(0) if device.type == "cuda" else None
            ),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
        },
        "preconditions": preconditions,
        "evaluation": evaluation_report,
        "aggregate_csv": str(AGGREGATE_PATH.relative_to(ROOT)),
        **classification,
        "plots": [
            str(path.relative_to(ROOT))
            for path in sorted(plot_dir.glob("gate1_grid_scale_*.png"))
        ],
    }
    save_json(report, REPORT_PATH)
    save_json(report, GATE_JSON_PATH)
    lines = [
        f"complete_rows: {'PASS' if classification['scientific_checks']['complete_rows'] else 'FAIL'}",
        f"all_core_metrics_finite: {'PASS' if classification['scientific_checks']['all_core_metrics_finite'] else 'FAIL'}",
        f"four_prb_coordinate_equivalence: {'PASS' if classification['scientific_checks']['four_prb_coordinate_equivalence'] else 'FAIL'}",
        f"four_prb_output_equivalence: {'PASS' if classification['scientific_checks']['four_prb_output_equivalence'] else 'FAIL'}",
        f"wide_grid_tbler_improves: {'PASS' if classification['scientific_checks']['wide_grid_tbler_improves'] else 'FAIL'}",
        f"wide_grid_nll_improves: {'PASS' if classification['scientific_checks']['wide_grid_nll_improves'] else 'FAIL'}",
        f"wide_grid_nmse_improves: {'PASS' if classification['scientific_checks']['wide_grid_nmse_improves'] else 'FAIL'}",
        f"wide_grid_calibration_improves: {'PASS' if classification['scientific_checks']['wide_grid_calibration_improves'] else 'FAIL'}",
        f"physical_no_strict_reversal: {'PASS' if classification['scientific_checks']['physical_no_strict_reversal'] else 'FAIL'}",
        f"CLASSIFICATION: {classification['classification']}",
        f"NEXT_ACTION: {classification['next_action']}",
        "PUBLICATION_NR_READY: NO",
    ]
    GATE_TXT_PATH.parent.mkdir(parents=True, exist_ok=True)
    GATE_TXT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return report


def preflight(
    config: dict[str, Any], preconditions: dict[str, Any], device: torch.device
) -> dict[str, Any]:
    cases = [NRCase.from_mapping(item) for item in config["evaluation"]["cases"]]
    for case in cases:
        case.validate()
    rows = expected_rows(config)
    checks = {
        "preconditions": preconditions.get("passed") is True,
        "version": config.get("grid_scale_revision") == AUDIT_VERSION,
        "three_cases": len(cases) == 3,
        "prb_grid": sorted(case.num_prb for case in cases) == [4, 8, 12],
        "expected_rows": rows == EXPECTED_ROWS,
        "variants": set(config["evaluation"]["variants"])
        == {
            "frozen_allocation_normalized",
            "frozen_reference_physical",
            "ls_lmmse",
            "perfect_csi_lmmse",
        },
        "source_contract": len(source_hashes()) == len(SOURCE_FILES),
    }
    # Definitive Sionna/PUSCH construction check, without channel sampling.
    coordinate_report: list[dict[str, Any]] = []
    ref_cfg = config["coordinate_systems"]["reference_physical"]
    for case in cases:
        context = build_nr_context(case, device)
        physical = reference_physical_coords(
            context.grid,
            subcarrier_spacing_khz=float(case.subcarrier_spacing_khz),
            reference_num_subcarriers=int(ref_cfg["reference_num_subcarriers"]),
            reference_scs_khz=float(ref_cfg["reference_scs_khz"]),
        )
        coordinate_report.append(
            {
                "case": case.name,
                "num_prb": case.num_prb,
                "num_subcarriers": context.grid.num_effective_subcarriers,
                "max_abs_difference": float(
                    torch.max(torch.abs(context.grid.coords - physical)).item()
                ),
                "finite": bool(torch.isfinite(physical).all().item()),
            }
        )
    checks["all_pusch_configs_valid"] = all(
        item["finite"] for item in coordinate_report
    )
    checks["four_prb_reference_equivalence"] = bool(
        coordinate_report[0]["num_prb"] == 4
        and coordinate_report[0]["max_abs_difference"] <= 1e-7
    )
    report = {
        "version": AUDIT_VERSION,
        "passed": all(checks.values()),
        "checks": checks,
        "expected_rows": rows,
        "coordinate_report": coordinate_report,
        "source_sha256": source_hashes(),
    }
    if not report["passed"]:
        raise RuntimeError(f"Grid-scale preflight failed: {report}")
    print("GATE1_NR_GRID_SCALE_AUDIT_PREFLIGHT_PASS")
    print("EXPECTED_ROWS", rows)
    print("PRB_GRID", [case.num_prb for case in cases])
    print("FOUR_PRB_REFERENCE_EQUIVALENCE YES")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="configs/gate1_nr_grid_scale_audit.yaml"
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    config_path = ROOT / args.config
    config = load_yaml(config_path)
    preconditions = verify_preconditions(config)
    requested_device = args.device or str(config.get("device", "cuda"))
    device = normalize_device(requested_device)
    preflight(config, preconditions, device)
    if args.preflight_only:
        return
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Grid-scale audit requires a CUDA compute node")
    deadline = time.time() + float(config["internal_deadline_seconds"])
    frame, evaluation_report = evaluate(
        config,
        config_path,
        preconditions,
        device,
        deadline_epoch=deadline,
    )
    if not evaluation_report.get("complete", False):
        partial = {
            "version": AUDIT_VERSION,
            "complete": False,
            "classification": "GATE1_NR_GRID_SCALE_AUDIT_INCOMPLETE",
            "next_action": "RESUBMIT_SAME_RESUME_SAFE_COMMAND",
            "evaluation": evaluation_report,
            "publication_nr_ready": False,
        }
        save_json(partial, REPORT_PATH)
        save_json(partial, GATE_JSON_PATH)
        GATE_TXT_PATH.parent.mkdir(parents=True, exist_ok=True)
        GATE_TXT_PATH.write_text(
            "GATE1_NR_GRID_SCALE_AUDIT_INCOMPLETE: RESUBMIT\n"
            f"rows: {evaluation_report['rows']}\n"
            f"expected_rows: {EXPECTED_ROWS}\n"
            "PUBLICATION_NR_READY: NO\n",
            encoding="utf-8",
        )
        print("GATE1_NR_GRID_SCALE_AUDIT_INCOMPLETE: RESUBMIT")
        return
    classification = classify(frame)
    make_outputs(
        frame,
        evaluation_report,
        preconditions,
        classification,
        device,
    )


if __name__ == "__main__":
    main()
