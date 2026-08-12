#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from bayesroute.nr_gate1 import (  # noqa: E402
    NRBayesRouteBridge,
    NRCase,
    build_nr_context,
    build_pusch_configs,
    decode_bridge,
    run_standard_receiver,
    standard_receiver,
)
from gate1_nr_joint_operator_common import (  # noqa: E402
    CandidateSpec,
    JOINT_OPERATOR_VERSION,
    SELECTED_COVARIANCE_MODE,
    SELECTED_DETECTOR_DAMPING,
    SELECTED_DETECTOR_ITERATIONS,
    SELECTED_EDGE_MASS,
    coded_metrics,
    copy_old_operator_if_compatible,
    edge_density,
    load_operator_state,
    make_repaired_detector,
    package_signature,
    posterior_metrics,
    random_fixed_cardinality_graph,
    repaired_forward,
    save_json,
    set_all_seeds,
    sha256_file,
)
from bayesroute.lmmse_ep import full_directed_graph  # noqa: E402

CLEAN_VERSION = "gate1_nr_clean_generalization_v1"
REQUIRED_CAPACITY_VERSION = "gate1_nr_joint_operator_capacity_v1_1"
REQUIRED_CAPACITY_CLASSIFICATION = "GATE1_JOINT_OPERATOR_SUPPORTED"
REQUIRED_BEST_GLOBAL = "global_r24_cold_lf1_lt0p5"
EXPECTED_ROWS = 4800
SOURCE_CONTRACT_FILES = (
    "scripts/gate1_nr_clean_generalization.py",
    "configs/gate1_nr_clean_generalization.yaml",
    "scripts/gate1_nr_joint_operator_common.py",
    "src/bayesroute/lmmse_ep.py",
    "src/bayesroute/nr_gate1.py",
    "src/bayesroute/models.py",
    "src/bayesroute/qam.py",
)


def load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected mapping in {path}")
    return value


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


def verify_preconditions(config: dict[str, Any]) -> dict[str, Any]:
    capacity_path = ROOT / "outputs/reports/gate1_nr_joint_operator_capacity.json"
    candidate_path = ROOT / (
        "outputs/reports/gate1_nr_joint_operator_"
        f"{config['frozen_candidate']['name']}.json"
    )
    checkpoint_path = ROOT / str(config["frozen_candidate"]["checkpoint_path"])
    old_checkpoint_path = ROOT / str(config["old_checkpoint"]["path"])
    revision_path = ROOT / "GATE1_NR_JOINT_OPERATOR_REVISION.json"
    for path in (
        capacity_path,
        candidate_path,
        checkpoint_path,
        old_checkpoint_path,
        revision_path,
    ):
        if not path.is_file():
            raise RuntimeError(f"Missing clean-gate precondition: {path}")

    capacity = json.loads(capacity_path.read_text(encoding="utf-8"))
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    revision = json.loads(revision_path.read_text(encoding="utf-8"))
    checkpoint_sha = sha256_file(checkpoint_path)
    old_sha = sha256_file(old_checkpoint_path)
    checks = {
        "capacity_complete": capacity.get("complete") is True,
        "capacity_version": capacity.get("version") == REQUIRED_CAPACITY_VERSION,
        "capacity_classification": (
            capacity.get("classification") == REQUIRED_CAPACITY_CLASSIFICATION
        ),
        "capacity_rows": (
            capacity.get("evaluation", {}).get("rows") == 704
            and capacity.get("evaluation", {}).get("unique_rows") == 704
        ),
        "capacity_best_global": (
            capacity.get("best_global_candidate") == REQUIRED_BEST_GLOBAL
        ),
        "capacity_equal_exposure": (
            capacity.get("software_checks", {}).get("equal_case_exposure") is True
        ),
        "candidate_complete": candidate.get("complete") is True,
        "candidate_name": (
            candidate.get("candidate", {}).get("name") == REQUIRED_BEST_GLOBAL
        ),
        "candidate_checkpoint_record": (
            candidate.get("best_checkpoint_sha256")
            == str(config["frozen_candidate"]["checkpoint_sha256"])
        ),
        "candidate_checkpoint_file": (
            checkpoint_sha == str(config["frozen_candidate"]["checkpoint_sha256"])
        ),
        "old_checkpoint": old_sha == str(config["old_checkpoint"]["sha256"]),
        "joint_revision": revision.get("revision") == JOINT_OPERATOR_VERSION,
        "capacity_workflow": (
            revision.get("capacity_workflow_revision")
            == REQUIRED_CAPACITY_VERSION
        ),
        "selected_detector": bool(
            int(config["selected_detector"]["iterations"])
            == SELECTED_DETECTOR_ITERATIONS
            and abs(
                float(config["selected_detector"]["damping"])
                - SELECTED_DETECTOR_DAMPING
            )
            < 1e-12
            and abs(
                float(config["selected_detector"]["edge_mass"])
                - SELECTED_EDGE_MASS
            )
            < 1e-12
            and str(config["selected_detector"]["covariance_mode"])
            == SELECTED_COVARIANCE_MODE
        ),
        "expected_rows": expected_rows(config) == EXPECTED_ROWS,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "capacity_classification": capacity.get("classification"),
        "capacity_commit_scope": "frozen global checkpoint; no test-set tuning",
        "checkpoint_sha256": checkpoint_sha,
        "old_checkpoint_sha256": old_sha,
    }


def build_bridge(
    case: NRCase,
    context: Any,
    spec: CandidateSpec,
    *,
    operator_seed: int = 58001,
) -> NRBayesRouteBridge:
    return NRBayesRouteBridge(
        context.grid,
        num_streams=case.num_streams,
        rank=int(spec.rank),
        bank_rank=int(spec.bank_rank),
        detector_iterations=SELECTED_DETECTOR_ITERATIONS,
        edge_mass=SELECTED_EDGE_MASS,
        length_f=float(spec.length_f),
        length_t=float(spec.length_t),
        operator_seed=int(operator_seed),
    ).to(context.device)


def load_frozen_operator(
    bridge: NRBayesRouteBridge,
    checkpoint_path: Path,
    device: torch.device,
) -> None:
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    operators = state.get("operators")
    if not isinstance(operators, list) or len(operators) != 1:
        raise RuntimeError("Frozen global checkpoint must contain one operator state")
    load_operator_state(bridge, operators[0])


def experiment_contract(
    config: dict[str, Any],
    config_path: Path,
    preconditions: dict[str, Any],
) -> dict[str, Any]:
    payload = {
        "version": CLEAN_VERSION,
        "config_sha256": sha256_file(config_path),
        "source_sha256": source_hashes(),
        "frozen_checkpoint_sha256": preconditions["checkpoint_sha256"],
        "old_checkpoint_sha256": preconditions["old_checkpoint_sha256"],
        "frozen_candidate": config["frozen_candidate"],
        "selected_detector": config["selected_detector"],
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
            raise RuntimeError("Clean evaluation CSV column contract mismatch")
        combined = pd.concat([old, new], ignore_index=True)
    else:
        combined = new
    temporary = path.with_suffix(path.suffix + ".tmp")
    combined.to_csv(temporary, index=False)
    temporary.replace(path)


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
    snr: float,
    rep: int,
    seed: int,
    output: dict[str, Any],
    batch: Any,
    decoded: dict[str, Any],
    contract_signature: str,
    graph_count_match: bool,
) -> dict[str, Any]:
    return {
        "case": case.name,
        "group": group,
        "scenario": case.scenario,
        "num_users": int(case.num_users),
        "num_layers_per_user": int(case.num_layers_per_user),
        "num_streams": int(case.num_streams),
        "num_rx_ant": int(case.num_rx_ant),
        "num_prb": int(case.num_prb),
        "mcs_index": int(case.mcs_index),
        "dmrs_config_type": int(case.dmrs_config_type),
        "dmrs_length": int(case.dmrs_length),
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
        **posterior_metrics(output, batch),
        "edge_density": float(output["edge_density"].item()),
        "graph_count_match": bool(graph_count_match),
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
        "num_layers_per_user": int(case.num_layers_per_user),
        "num_streams": int(case.num_streams),
        "num_rx_ant": int(case.num_rx_ant),
        "num_prb": int(case.num_prb),
        "mcs_index": int(case.mcs_index),
        "dmrs_config_type": int(case.dmrs_config_type),
        "dmrs_length": int(case.dmrs_length),
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
        "graph_count_match": True,
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
    eval_dir = ROOT / "outputs/eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    raw_path = eval_dir / "gate1_nr_clean_generalization.csv"
    contract_path = eval_dir / "gate1_nr_clean_generalization_contract.json"
    if raw_path.is_file():
        if not contract_path.is_file():
            raise RuntimeError("Clean CSV exists without its contract")
        existing = json.loads(contract_path.read_text(encoding="utf-8"))
        if existing.get("signature") != contract["signature"]:
            raise RuntimeError("Clean evaluation resume contract mismatch")
    else:
        save_json(contract, contract_path)

    done: set[tuple[str, str, float, int]] = set()
    if raw_path.is_file():
        old = pd.read_csv(raw_path)
        keys = ["case", "variant", "ebno_db", "rep"]
        if old[keys].duplicated().any():
            raise RuntimeError("Clean evaluation contains duplicate keys")
        for _, row in old.iterrows():
            done.add(
                (
                    str(row["case"]),
                    str(row["variant"]),
                    float(row["ebno_db"]),
                    int(row["rep"]),
                )
            )

    frozen_spec = CandidateSpec.from_mapping(config["frozen_candidate"])
    frozen_checkpoint = ROOT / str(config["frozen_candidate"]["checkpoint_path"])
    old_checkpoint = torch.load(
        ROOT / str(config["old_checkpoint"]["path"]),
        map_location=device,
        weights_only=False,
    )
    if "model" not in old_checkpoint:
        raise RuntimeError("Old preliminary checkpoint has no model state")

    raw_cases = list(evaluation["cases"])
    group_by_name = {str(item["name"]): str(item["group"]) for item in raw_cases}
    for case_index, raw_case in enumerate(raw_cases):
        case = NRCase.from_mapping(raw_case)
        context = build_nr_context(case, device)
        frozen_bridge = build_bridge(case, context, frozen_spec)
        load_frozen_operator(frozen_bridge, frozen_checkpoint, device)
        frozen_bridge.eval()
        frozen_detector = make_repaired_detector(int(context.grid.bits_per_symbol)).to(device)

        old_spec = CandidateSpec(
            name="old_checkpoint",
            mode="global",
            rank=16,
            bank_rank=24,
            length_f=2.0,
            length_t=1.0,
            initialization="old_checkpoint",
            learning_rate=0.0,
            channel_loss_weight=0.0,
            calibration_loss_weight=0.0,
            steps=1,
        )
        old_bridge = build_bridge(case, context, old_spec)
        if not copy_old_operator_if_compatible(old_bridge, old_checkpoint["model"]):
            raise RuntimeError(f"Old checkpoint cannot initialize {case.name}")
        old_bridge.eval()
        old_detector = make_repaired_detector(int(context.grid.bits_per_symbol)).to(device)
        ls_receiver = standard_receiver(context, perfect_csi=False, return_crc=True)
        perfect_receiver = standard_receiver(context, perfect_csi=True, return_crc=True)

        for snr_index, raw_snr in enumerate(evaluation["ebno_db"]):
            snr = float(raw_snr)
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
                    + 20_000_000
                    + 100_000 * case_index
                    + 1_000 * snr_index
                    + rep
                )
                set_all_seeds(seed)
                batch = context.sample(
                    batch_size=int(evaluation["batch_size"]),
                    ebno_db=snr,
                )
                outputs: dict[str, dict[str, Any]] = {}
                graph_match: dict[str, bool] = {}
                with torch.inference_mode():
                    need_frozen = any(
                        name in missing
                        for name in (
                            "frozen_global",
                            "uncertainty_off_fixed_graph",
                            "random_graph_fixed_cardinality",
                            "full_graph",
                            "graph_off",
                        )
                    )
                    base: dict[str, Any] | None = None
                    if need_frozen:
                        base = repaired_forward(
                            frozen_bridge,
                            frozen_detector,
                            batch,
                        )
                        posterior = base["posterior"]
                        reference = base["reference_graph_mask"]
                        if "frozen_global" in missing:
                            outputs["frozen_global"] = base
                            graph_match["frozen_global"] = True
                        if "uncertainty_off_fixed_graph" in missing:
                            outputs["uncertainty_off_fixed_graph"] = repaired_forward(
                                frozen_bridge,
                                frozen_detector,
                                batch,
                                covariance_mode="none",
                                posterior=posterior,
                                reference_graph=reference,
                            )
                            graph_match["uncertainty_off_fixed_graph"] = True
                        if "random_graph_fixed_cardinality" in missing:
                            random_graph = random_fixed_cardinality_graph(
                                reference, seed + 900_000
                            )
                            output = repaired_forward(
                                frozen_bridge,
                                frozen_detector,
                                batch,
                                posterior=posterior,
                                reference_graph=reference,
                                graph_mode="random",
                                random_seed=seed + 900_000,
                            )
                            outputs["random_graph_fixed_cardinality"] = output
                            graph_match["random_graph_fixed_cardinality"] = bool(
                                torch.equal(
                                    random_graph.sum(dim=(-2, -1)),
                                    reference.sum(dim=(-2, -1)),
                                )
                                and torch.equal(
                                    output["graph_mask"].sum(dim=(-2, -1)),
                                    reference.sum(dim=(-2, -1)),
                                )
                            )
                        if "full_graph" in missing:
                            output = repaired_forward(
                                frozen_bridge,
                                frozen_detector,
                                batch,
                                posterior=posterior,
                                reference_graph=reference,
                                graph_mode="full",
                            )
                            outputs["full_graph"] = output
                            graph_match["full_graph"] = True
                        if "graph_off" in missing:
                            output = repaired_forward(
                                frozen_bridge,
                                frozen_detector,
                                batch,
                                posterior=posterior,
                                reference_graph=reference,
                                graph_mode="off",
                            )
                            outputs["graph_off"] = output
                            graph_match["graph_off"] = True
                    if "old_checkpoint_repaired_detector" in missing:
                        output = repaired_forward(old_bridge, old_detector, batch)
                        outputs["old_checkpoint_repaired_detector"] = output
                        graph_match["old_checkpoint_repaired_detector"] = True

                custom_missing = [
                    name
                    for name in missing
                    if name not in {"ls_lmmse", "perfect_csi_lmmse"}
                ]
                with torch.inference_mode():
                    decoded = decode_outputs(
                        context,
                        batch,
                        {name: outputs[name] for name in custom_missing},
                        bp_iterations=int(evaluation["bp_iterations"]),
                        device=device,
                    )
                rows: list[dict[str, Any]] = []
                group = group_by_name[case.name]
                for name in missing:
                    if name == "ls_lmmse":
                        with torch.inference_mode():
                            metrics = run_standard_receiver(
                                ls_receiver,
                                batch,
                                batch.information_bits,
                                perfect_csi=False,
                            )
                        rows.append(
                            standard_row(
                                case=case,
                                group=group,
                                variant=name,
                                snr=snr,
                                rep=rep,
                                seed=seed,
                                metrics=metrics,
                                contract_signature=contract["signature"],
                            )
                        )
                    elif name == "perfect_csi_lmmse":
                        with torch.inference_mode():
                            metrics = run_standard_receiver(
                                perfect_receiver,
                                batch,
                                batch.information_bits,
                                perfect_csi=True,
                            )
                        rows.append(
                            standard_row(
                                case=case,
                                group=group,
                                variant=name,
                                snr=snr,
                                rep=rep,
                                seed=seed,
                                metrics=metrics,
                                contract_signature=contract["signature"],
                            )
                        )
                    else:
                        rows.append(
                            custom_row(
                                case=case,
                                group=group,
                                variant=name,
                                snr=snr,
                                rep=rep,
                                seed=seed,
                                output=outputs[name],
                                batch=batch,
                                decoded=decoded[name],
                                contract_signature=contract["signature"],
                                graph_count_match=graph_match.get(name, True),
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
                            "expected_rows": EXPECTED_ROWS,
                        }
                    ),
                    flush=True,
                )
                if time.time() >= deadline_epoch:
                    frame = pd.read_csv(raw_path)
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
                        "raw_csv": str(raw_path.relative_to(ROOT)),
                        "contract": contract,
                        "stop_reason": "internal_deadline",
                    }

        del (
            context,
            frozen_bridge,
            frozen_detector,
            old_bridge,
            old_detector,
            ls_receiver,
            perfect_receiver,
        )
        if device.type == "cuda":
            torch.cuda.empty_cache()

    frame = pd.read_csv(raw_path)
    keys = ["case", "variant", "ebno_db", "rep"]
    unique = int(len(frame.drop_duplicates(keys)))
    complete = bool(len(frame) == EXPECTED_ROWS and unique == EXPECTED_ROWS)
    if not complete:
        raise RuntimeError(
            f"Clean evaluation incomplete: rows={len(frame)}, "
            f"unique={unique}, expected={EXPECTED_ROWS}"
        )
    return frame, {
        "complete": True,
        "rows": int(len(frame)),
        "unique_rows": unique,
        "expected_rows": EXPECTED_ROWS,
        "raw_csv": str(raw_path.relative_to(ROOT)),
        "contract": contract,
    }


def subset_for_comparison(
    frame: pd.DataFrame,
    *,
    high_snr: bool = True,
    multi_stream: bool = True,
    holdout_only: bool = False,
    loaded_only: bool = False,
) -> pd.DataFrame:
    sub = frame
    if high_snr:
        sub = sub[sub["ebno_db"].isin([6.0, 10.0, 14.0])]
    if multi_stream:
        sub = sub[sub["num_streams"] >= 4]
    if holdout_only:
        sub = sub[sub["group"] != "seen_new_seed"]
    if loaded_only:
        sub = sub[sub["num_streams"] >= 4]
    return sub


def mean_metric(
    frame: pd.DataFrame,
    *,
    variant: str,
    metric: str,
    high_snr: bool = True,
    multi_stream: bool = True,
    holdout_only: bool = False,
    snr: float | None = None,
) -> float:
    sub = subset_for_comparison(
        frame,
        high_snr=high_snr,
        multi_stream=multi_stream,
        holdout_only=holdout_only,
    )
    sub = sub[sub["variant"] == variant]
    if snr is not None:
        sub = sub[sub["ebno_db"] == float(snr)]
    values = pd.to_numeric(sub[metric], errors="coerce").dropna()
    return float(values.mean()) if len(values) else float("nan")


def paired_delta(
    frame: pd.DataFrame,
    reference: str,
    comparator: str,
    metric: str,
    *,
    high_snr: bool = True,
    multi_stream: bool = True,
    holdout_only: bool = False,
) -> dict[str, float]:
    sub = subset_for_comparison(
        frame,
        high_snr=high_snr,
        multi_stream=multi_stream,
        holdout_only=holdout_only,
    )
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


def classify(frame: pd.DataFrame) -> dict[str, Any]:
    comparisons = {
        "old_checkpoint_tbler": paired_delta(
            frame,
            "frozen_global",
            "old_checkpoint_repaired_detector",
            "tbler",
            holdout_only=True,
        ),
        "uncertainty_tbler": paired_delta(
            frame,
            "frozen_global",
            "uncertainty_off_fixed_graph",
            "tbler",
            holdout_only=True,
        ),
        "uncertainty_nll": paired_delta(
            frame,
            "frozen_global",
            "uncertainty_off_fixed_graph",
            "coded_bit_nll",
            holdout_only=True,
        ),
        "random_graph_tbler": paired_delta(
            frame,
            "frozen_global",
            "random_graph_fixed_cardinality",
            "tbler",
            holdout_only=True,
        ),
        "full_graph_tbler": paired_delta(
            frame,
            "frozen_global",
            "full_graph",
            "tbler",
            holdout_only=True,
        ),
        "graph_off_tbler": paired_delta(
            frame,
            "frozen_global",
            "graph_off",
            "tbler",
            holdout_only=True,
        ),
        "ls_lmmse_tbler": paired_delta(
            frame,
            "frozen_global",
            "ls_lmmse",
            "tbler",
            holdout_only=True,
        ),
        "perfect_csi_lmmse_tbler": paired_delta(
            frame,
            "frozen_global",
            "perfect_csi_lmmse",
            "tbler",
            holdout_only=True,
        ),
    }
    values = {
        "frozen_holdout_high_snr_tbler": mean_metric(
            frame,
            variant="frozen_global",
            metric="tbler",
            holdout_only=True,
        ),
        "old_holdout_high_snr_tbler": mean_metric(
            frame,
            variant="old_checkpoint_repaired_detector",
            metric="tbler",
            holdout_only=True,
        ),
        "ls_holdout_high_snr_tbler": mean_metric(
            frame,
            variant="ls_lmmse",
            metric="tbler",
            holdout_only=True,
        ),
        "perfect_holdout_high_snr_tbler": mean_metric(
            frame,
            variant="perfect_csi_lmmse",
            metric="tbler",
            holdout_only=True,
        ),
        "frozen_holdout_6db_tbler": mean_metric(
            frame,
            variant="frozen_global",
            metric="tbler",
            high_snr=False,
            holdout_only=True,
            snr=6.0,
        ),
        "frozen_holdout_10db_tbler": mean_metric(
            frame,
            variant="frozen_global",
            metric="tbler",
            high_snr=False,
            holdout_only=True,
            snr=10.0,
        ),
        "frozen_holdout_14db_tbler": mean_metric(
            frame,
            variant="frozen_global",
            metric="tbler",
            high_snr=False,
            holdout_only=True,
            snr=14.0,
        ),
    }
    posterior_rows = frame[
        (frame["variant"] == "frozen_global")
        & (frame["group"] != "seen_new_seed")
    ]
    normalized = pd.to_numeric(
        posterior_rows["normalized_error_mean"], errors="coerce"
    ).dropna()
    coverage = pd.to_numeric(posterior_rows["coverage95"], errors="coerce").dropna()
    values["holdout_normalized_error_median"] = float(normalized.median())
    values["holdout_coverage95_mean"] = float(coverage.mean())

    checks = {
        "frozen_improves_old_checkpoint": bool(
            comparisons["old_checkpoint_tbler"]["ci95_high"] < 0.0
        ),
        "uncertainty_improves_tbler": bool(
            comparisons["uncertainty_tbler"]["ci95_high"] < 0.0
        ),
        "uncertainty_improves_nll": bool(
            comparisons["uncertainty_nll"]["ci95_high"] < 0.0
        ),
        "coupling_beats_random_loaded": bool(
            comparisons["random_graph_tbler"]["ci95_high"] < 0.0
        ),
        "graph_beats_graph_off": bool(
            comparisons["graph_off_tbler"]["ci95_high"] < 0.0
        ),
        "sparse_not_worse_than_full_by_0p01": bool(
            comparisons["full_graph_tbler"]["ci95_high"] <= 0.01
        ),
        "within_0p05_of_ls_on_holdout": bool(
            comparisons["ls_lmmse_tbler"]["mean"] <= 0.05
        ),
        "no_high_snr_reversal": bool(
            values["frozen_holdout_14db_tbler"]
            <= values["frozen_holdout_10db_tbler"] + 0.03
        ),
        "posterior_not_grossly_miscalibrated": bool(
            0.4 <= values["holdout_normalized_error_median"] <= 2.5
            and 0.80 <= values["holdout_coverage95_mean"] <= 1.0
        ),
    }
    if (
        checks["within_0p05_of_ls_on_holdout"]
        and checks["frozen_improves_old_checkpoint"]
        and checks["uncertainty_improves_tbler"]
        and checks["coupling_beats_random_loaded"]
        and checks["no_high_snr_reversal"]
    ):
        classification = "GATE1_CLEAN_GENERALIZATION_SUPPORTED"
        next_action = "DESIGN_PUBLICATION_SCALE_CAMPAIGN_AND_COMPLEXITY_AUDIT"
    elif (
        checks["within_0p05_of_ls_on_holdout"]
        and checks["frozen_improves_old_checkpoint"]
        and checks["uncertainty_improves_tbler"]
        and checks["no_high_snr_reversal"]
    ):
        classification = "GATE1_UNCERTAINTY_SUPPORTED_ROUTING_UNRESOLVED"
        next_action = "NARROW_ROUTING_CLAIM_OR_RUN_HIGH_LOAD_ROUTING_STRESS"
    elif checks["frozen_improves_old_checkpoint"] and checks["uncertainty_improves_tbler"]:
        classification = "GATE1_CLEAN_GENERALIZATION_PARTIAL"
        next_action = "IMPROVE_POSTERIOR_OPERATOR_BEFORE_PUBLICATION_SCALE"
    else:
        classification = "GATE1_CLEAN_GENERALIZATION_INSUFFICIENT"
        next_action = "REASSESS_FROZEN_OPERATOR_AND_TRAINING_OBJECTIVE"
    return {
        "classification": classification,
        "next_action": next_action,
        "scientific_checks": checks,
        "high_snr_metrics": values,
        "paired_comparisons_frozen_minus_comparator": comparisons,
    }


def make_outputs(
    frame: pd.DataFrame,
    decision: dict[str, Any],
) -> tuple[str, list[str]]:
    eval_dir = ROOT / "outputs/eval"
    plot_dir = ROOT / "outputs/plots"
    eval_dir.mkdir(parents=True, exist_ok=True)
    numeric = [
        "information_ber",
        "tbler",
        "crc_failure_rate",
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
            ["case", "group", "scenario", "variant", "ebno_db"],
            as_index=False,
        )[numeric]
        .agg(["mean", "std", "count"])
    )
    aggregate.columns = [
        "_".join(str(x) for x in item if str(x))
        for item in aggregate.columns.to_flat_index()
    ]
    aggregate_path = eval_dir / "gate1_nr_clean_generalization_aggregate.csv"
    aggregate.to_csv(aggregate_path, index=False)

    plot_dir.mkdir(parents=True, exist_ok=True)
    selected = [
        "frozen_global",
        "old_checkpoint_repaired_detector",
        "uncertainty_off_fixed_graph",
        "random_graph_fixed_cardinality",
        "full_graph",
        "ls_lmmse",
        "perfect_csi_lmmse",
    ]
    plots: list[str] = []
    for case in sorted(frame["case"].unique()):
        fig, ax = plt.subplots(figsize=(7.6, 5.0))
        case_frame = frame[frame["case"] == case]
        for variant in selected:
            sub = case_frame[case_frame["variant"] == variant]
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
        ax.legend(fontsize=6.2)
        ax.set_title(case)
        fig.tight_layout()
        path = plot_dir / f"gate1_clean_{case}_tbler.png"
        fig.savefig(path, dpi=220)
        plt.close(fig)
        plots.append(str(path.relative_to(ROOT)))

    posterior = frame[frame["variant"] == "frozen_global"]
    calibration = (
        posterior.groupby(["case", "ebno_db"], as_index=False)[
            ["normalized_error_mean", "coverage95"]
        ]
        .mean()
    )
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    for case in sorted(calibration["case"].unique()):
        sub = calibration[calibration["case"] == case]
        ax.plot(
            sub["ebno_db"],
            sub["normalized_error_mean"],
            marker="o",
            label=case,
        )
    ax.axhline(1.0, linestyle="--", linewidth=1.0)
    ax.set_xlabel("Eb/N0 [dB]")
    ax.set_ylabel("Mean normalized channel error")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=5.8, ncol=2)
    fig.tight_layout()
    path = plot_dir / "gate1_clean_posterior_calibration.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    plots.append(str(path.relative_to(ROOT)))

    return str(aggregate_path.relative_to(ROOT)), plots


def preflight(config: dict[str, Any], preconditions: dict[str, Any]) -> dict[str, Any]:
    case_reports: list[dict[str, Any]] = []
    for raw in config["evaluation"]["cases"]:
        case = NRCase.from_mapping(raw)
        case.validate()
        pusch = build_pusch_configs(case)
        case_reports.append(
            {
                "name": case.name,
                "group": raw["group"],
                "num_streams": case.num_streams,
                "mcs_index": int(case.mcs_index),
                "pusch_config_count": len(pusch),
                "dmrs_ports": list(case.dmrs_ports),
                "passed": True,
            }
        )
    variants = [str(x) for x in config["evaluation"]["variants"]]
    checks = {
        "preconditions": bool(preconditions["passed"]),
        "expected_rows": expected_rows(config) == EXPECTED_ROWS,
        "unique_cases": len(case_reports)
        == len({item["name"] for item in case_reports}),
        "unique_variants": len(variants) == len(set(variants)),
        "all_pusch_configs_valid": all(item["passed"] for item in case_reports),
        "source_contract": len(source_hashes()) == len(SOURCE_CONTRACT_FILES),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "case_reports": case_reports,
        "expected_rows": EXPECTED_ROWS,
        "variants": variants,
        "source_sha256": source_hashes(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/gate1_nr_clean_generalization.yaml",
    )
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    config_path = ROOT / args.config
    config = load_yaml(config_path)
    preconditions = verify_preconditions(config)
    if not preconditions["passed"]:
        raise RuntimeError(f"Clean-gate preconditions failed: {preconditions}")
    preflight_report = preflight(config, preconditions)
    if not preflight_report["passed"]:
        raise RuntimeError(f"Clean-gate preflight failed: {preflight_report}")
    if args.preflight_only:
        print(json.dumps(preflight_report, indent=2, sort_keys=True))
        print("GATE1_NR_CLEAN_GENERALIZATION_PREFLIGHT_PASS")
        return

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    deadline_epoch = time.time() + float(config["internal_deadline_seconds"])
    frame, evaluation = evaluate(
        config,
        config_path,
        preconditions,
        device,
        deadline_epoch=deadline_epoch,
    )
    report_path = ROOT / "outputs/reports/gate1_nr_clean_generalization.json"
    if evaluation.get("complete") is not True:
        report = {
            "version": CLEAN_VERSION,
            "complete": False,
            "preconditions": preconditions,
            "preflight": preflight_report,
            "evaluation": evaluation,
            "publication_nr_ready": False,
        }
        save_json(report, report_path)
        print("GATE1_NR_CLEAN_GENERALIZATION_INCOMPLETE: RESUBMIT", flush=True)
        return

    decision = classify(frame)
    aggregate_csv, plots = make_outputs(frame, decision)
    software_checks = {
        "complete_rows": bool(evaluation["complete"]),
        "all_core_metrics_finite": bool(
            np.isfinite(
                frame[["information_ber", "tbler", "crc_failure_rate"]].to_numpy()
            ).all()
        ),
        "all_variants_present": bool(
            frame["variant"].nunique() == len(config["evaluation"]["variants"])
        ),
        "paired_seed_batches": bool(
            frame.groupby(["case", "ebno_db", "rep"])["variant"].nunique().min()
            == len(config["evaluation"]["variants"])
        ),
        "fixed_graph_cardinality_exact": bool(
            frame[frame["variant"] == "random_graph_fixed_cardinality"][
                "graph_count_match"
            ].all()
        ),
        "crc_consistency": bool(
            pd.to_numeric(
                frame["crc_block_disagreement_rate"], errors="coerce"
            ).dropna().mean()
            <= 0.005
        ),
        "source_contract": len(source_hashes()) == len(SOURCE_CONTRACT_FILES),
    }
    complete = all(software_checks.values())
    if not complete:
        decision["classification"] = "GATE1_CLEAN_GENERALIZATION_BLOCKED"
        decision["next_action"] = "REPAIR_CLEAN_GENERALIZATION_PIPELINE"

    report = {
        "version": CLEAN_VERSION,
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
        "complete": complete,
        "preconditions": preconditions,
        "preflight": preflight_report,
        "software_checks": software_checks,
        "evaluation": evaluation,
        **decision,
        "aggregate_csv": aggregate_csv,
        "plots": plots,
        "frozen_model_scope": (
            "single frozen checkpoint selected before these evaluation seeds and "
            "expanded cases; no retraining or holdout-driven tuning"
        ),
        "publication_nr_ready": False,
    }
    save_json(report, report_path)
    gate_dir = ROOT / "outputs/gates"
    gate_dir.mkdir(parents=True, exist_ok=True)
    save_json(report, gate_dir / "GATE1_NR_CLEAN_GENERALIZATION.json")
    lines = [
        *(f"{key}: {'PASS' if value else 'FAIL'}" for key, value in software_checks.items()),
        *(
            f"{key}: {'PASS' if value else 'FAIL'}"
            for key, value in decision["scientific_checks"].items()
        ),
        f"CLASSIFICATION: {decision['classification']}",
        f"NEXT_ACTION: {decision['next_action']}",
        "PUBLICATION_NR_READY: NO",
    ]
    (gate_dir / "GATE1_NR_CLEAN_GENERALIZATION.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print("\n".join(lines), flush=True)
    if not complete:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
