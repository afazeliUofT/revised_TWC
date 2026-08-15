#!/usr/bin/env python3
from __future__ import annotations

"""Frozen-checkpoint confirmation of the implementable localized receiver.

This gate performs no training, no retuning, and no model selection. It loads the
checkpoint selected before the fresh 12-PRB confirmation cases were constructed,
then compares it with LS+LMMSE using paired channel realizations. The primary
analysis treats each independently seeded simulation batch as a cluster. A
transport-block-level exact McNemar test is reported as secondary evidence.
"""

import argparse
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

ROOT = Path(__file__).resolve().parents[1]

from bayesroute.ls_anchored_localized_posterior import load_shared_localized_state
from bayesroute.nr_gate1 import NRCase, normalize_device, standard_receiver
from gate1_nr_implementable_localized_common import (
    build_stack_item,
    custom_row,
    decode_outputs,
    mean_only_forward,
    observable_forward,
    package_signature,
    save_json,
    selected_basis_spec,
    set_all_seeds,
    sha256_file,
    standard_row,
    uncertainty_off_forward,
)
from gate1_nr_joint_operator_common import make_repaired_detector
from gate1_nr_posterior_factorial_common import ls_repaired_forward

VERSION = "gate1_nr_implementable_localized_confirmation_v1"
MODEL_VERSION = "ls_anchored_localized_residual_v1"
REQUIRED_SOURCE_CLASSIFICATION = "GATE1_IMPLEMENTABLE_LOCALIZED_POSSIBLY_BEATS_LS"
REQUIRED_SOURCE_NEXT_ACTION = "RUN_ONE_LARGER_FIXED_CONFIRMATION_WITHOUT_RETUNING"
REQUIRED_SOURCE_ROWS = 2016
FROZEN_CHECKPOINT_SHA256 = (
    "cfb1afd665d6df89e6590b611badd68a77b9ada43de07daeee7b8d8a27dd70aa"
)
EXPECTED_ROWS = 2304

OUTPUT_ROOT = ROOT / "outputs/gate1_nr_implementable_localized_confirmation"
RAW_PATH = ROOT / "outputs/eval/gate1_nr_implementable_localized_confirmation.csv"
CONTRACT_PATH = ROOT / "outputs/eval/gate1_nr_implementable_localized_confirmation_contract.json"
AGGREGATE_PATH = ROOT / "outputs/eval/gate1_nr_implementable_localized_confirmation_aggregate.csv"
PAIRED_PATH = ROOT / "outputs/reports/gate1_nr_implementable_localized_confirmation_paired.csv"
REPORT_PATH = ROOT / "outputs/reports/gate1_nr_implementable_localized_confirmation.json"
GATE_JSON = ROOT / "outputs/gates/GATE1_NR_IMPLEMENTABLE_LOCALIZED_CONFIRMATION.json"
GATE_TXT = ROOT / "outputs/gates/GATE1_NR_IMPLEMENTABLE_LOCALIZED_CONFIRMATION.txt"
SOURCE_REPORT_PATH = ROOT / "outputs/reports/gate1_nr_implementable_localized.json"

SOURCE_FILES = (
    "configs/gate1_nr_implementable_localized_confirmation.yaml",
    "scripts/gate1_nr_implementable_localized_confirmation.py",
    "scripts/gate1_nr_implementable_localized_common.py",
    "scripts/gate1_nr_joint_operator_common.py",
    "scripts/gate1_nr_posterior_factorial_common.py",
    "src/bayesroute/ls_anchored_localized_posterior.py",
    "src/bayesroute/localized_delay_doppler.py",
    "src/bayesroute/lmmse_ep.py",
    "src/bayesroute/models.py",
    "src/bayesroute/nr_gate1.py",
    "src/bayesroute/qam.py",
)


def load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected YAML mapping: {path}")
    return value


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected JSON object: {path}")
    return value


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


def preconditions(config: dict[str, Any]) -> dict[str, Any]:
    if not SOURCE_REPORT_PATH.is_file():
        raise RuntimeError(f"Missing source result: {SOURCE_REPORT_PATH}")
    source = load_json(SOURCE_REPORT_PATH)
    checkpoint = ROOT / str(config["checkpoint"]["path"])
    checkpoint_hash = sha256_file(checkpoint) if checkpoint.is_file() else "missing"
    evaluation = config["evaluation"]
    checks = {
        "source_complete": source.get("complete") is True,
        "source_classification": (
            source.get("classification") == REQUIRED_SOURCE_CLASSIFICATION
        ),
        "source_next_action": (
            source.get("next_action") == REQUIRED_SOURCE_NEXT_ACTION
        ),
        "source_rows": (
            source.get("evaluation", {}).get("rows") == REQUIRED_SOURCE_ROWS
            and source.get("evaluation", {}).get("unique_rows")
            == REQUIRED_SOURCE_ROWS
        ),
        "source_training_converged": (
            source.get("training", {}).get("training_converged") is True
        ),
        "source_no_truth": (
            source.get("training", {}).get("inference_uses_true_channel") is False
        ),
        "checkpoint_present": checkpoint.is_file(),
        "checkpoint_hash": checkpoint_hash == FROZEN_CHECKPOINT_SHA256,
        "checkpoint_hash_config": (
            str(config["checkpoint"]["sha256"]) == FROZEN_CHECKPOINT_SHA256
        ),
        "no_training_section": "training" not in config,
        "two_fresh_cases": len(evaluation["cases"]) == 2,
        "all_12prb": all(int(item["num_prb"]) == 12 for item in evaluation["cases"]),
        "fresh_cell_ids": len({int(item["n_cell_id"]) for item in evaluation["cases"]}) == 2,
        "expected_rows": expected_rows(config) == EXPECTED_ROWS,
        "fixed_variants": evaluation["variants"] == [
            "trained_localized",
            "trained_uncertainty_off",
            "trained_mean_only",
            "ls_estimate_repaired",
            "ls_lmmse",
            "perfect_csi_lmmse",
        ],
        "large_block_count": (
            len(evaluation["cases"])
            * int(evaluation["repetitions"])
            * int(evaluation["batch_size"])
            * int(evaluation["cases"][0]["num_users"])
            >= 32768
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(f"Confirmation preconditions failed: {checks}")
    return {
        "passed": True,
        "checks": checks,
        "source_report": str(SOURCE_REPORT_PATH.relative_to(ROOT)),
        "source_classification": source["classification"],
        "source_next_action": source["next_action"],
        "source_metrics": source.get("metrics", {}),
        "checkpoint": str(checkpoint.relative_to(ROOT)),
        "checkpoint_sha256": checkpoint_hash,
        "no_retraining": True,
        "no_retuning": True,
    }


def contract(config: dict[str, Any], config_path: Path, pre: dict[str, Any]) -> dict[str, Any]:
    payload = {
        "version": VERSION,
        "config_sha256": sha256_file(config_path),
        "source_sha256": source_hashes(),
        "source_result": {
            "path": pre["source_report"],
            "classification": pre["source_classification"],
            "next_action": pre["source_next_action"],
        },
        "frozen_checkpoint_sha256": pre["checkpoint_sha256"],
        "evaluation": config["evaluation"],
        "decision": config["decision"],
        "policy": {
            "training_required": False,
            "retuning_allowed": False,
            "checkpoint_frozen_before_confirmation_cases": True,
            "inference_uses_true_channel": False,
        },
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
            raise RuntimeError("Confirmation CSV column contract mismatch")
        combined = pd.concat([old, new], ignore_index=True)
    else:
        combined = new
    temporary = path.with_suffix(path.suffix + ".tmp")
    combined.to_csv(temporary, index=False)
    temporary.replace(path)


def block_error_vector(decoded: dict[str, Any], bits: torch.Tensor) -> torch.Tensor:
    return (decoded["b_hat"] != bits).reshape(bits.shape[0], bits.shape[1], -1).any(-1)


def standard_decode(
    receiver: Any,
    batch: Any,
    bits: torch.Tensor,
    *,
    perfect_csi: bool,
) -> dict[str, Any]:
    result = (
        receiver(batch.raw_y, batch.noise_var, batch.raw_h)
        if perfect_csi
        else receiver(batch.raw_y, batch.noise_var)
    )
    if isinstance(result, tuple):
        b_hat, crc = result
    else:
        b_hat = result
        crc = torch.ones(
            b_hat.shape[:-1] + (1,), dtype=torch.bool, device=b_hat.device
        )
    bit_error = (b_hat != bits).float()
    block_error = bit_error.reshape(bit_error.shape[0], bit_error.shape[1], -1).any(-1)
    return {
        "information_ber": float(bit_error.mean().item()),
        "tbler": float(block_error.float().mean().item()),
        "crc_failure_rate": float((~crc.bool()).float().mean().item()),
        "b_hat": b_hat,
        "crc": crc,
        "block_error": block_error,
    }


def add_pair_fields(
    row: dict[str, Any],
    *,
    trained_error: torch.Tensor,
    ls_error: torch.Tensor,
    uncertainty_error: torch.Tensor,
    mean_only_error: torch.Tensor,
) -> dict[str, Any]:
    if not (
        trained_error.shape == ls_error.shape == uncertainty_error.shape == mean_only_error.shape
    ):
        raise RuntimeError("Paired block-error shapes disagree")
    trained_only = trained_error & ~ls_error
    ls_only = ~trained_error & ls_error
    both_error = trained_error & ls_error
    both_correct = ~trained_error & ~ls_error
    total = int(trained_error.numel())
    row.update(
        {
            "transport_blocks": total,
            "trained_block_errors": int(trained_error.sum().item()),
            "ls_block_errors": int(ls_error.sum().item()),
            "uncertainty_off_block_errors": int(uncertainty_error.sum().item()),
            "mean_only_block_errors": int(mean_only_error.sum().item()),
            "trained_only_errors": int(trained_only.sum().item()),
            "ls_only_errors": int(ls_only.sum().item()),
            "both_errors": int(both_error.sum().item()),
            "both_correct": int(both_correct.sum().item()),
            "trained_minus_ls_batch_tbler": float(
                trained_error.float().mean().item() - ls_error.float().mean().item()
            ),
        }
    )
    return row


def evaluate(
    config: dict[str, Any],
    config_path: Path,
    pre: dict[str, Any],
    device: torch.device,
    *,
    deadline_epoch: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    evaluation = config["evaluation"]
    variants = [str(value) for value in evaluation["variants"]]
    experiment = contract(config, config_path, pre)
    if RAW_PATH.is_file():
        if not CONTRACT_PATH.is_file():
            raise RuntimeError("Confirmation CSV exists without contract")
        prior = load_json(CONTRACT_PATH)
        if prior.get("signature") != experiment["signature"]:
            raise RuntimeError("Confirmation evaluation contract mismatch")
    else:
        save_json(experiment, CONTRACT_PATH)

    done: set[tuple[str, float, int]] = set()
    if RAW_PATH.is_file():
        frame = pd.read_csv(RAW_PATH)
        keys = ["case", "variant", "ebno_db", "rep"]
        if frame[keys].duplicated().any():
            raise RuntimeError("Confirmation CSV contains duplicate keys")
        counts = frame.groupby(["case", "ebno_db", "rep"])["variant"].nunique()
        if len(counts[counts != len(variants)]):
            raise RuntimeError("Confirmation CSV contains a partial paired batch")
        done = {(str(a), float(b), int(c)) for a, b, c in counts.index}

    spec = selected_basis_spec()
    checkpoint_path = ROOT / str(config["checkpoint"]["path"])
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if sha256_file(checkpoint_path) != FROZEN_CHECKPOINT_SHA256:
        raise RuntimeError("Frozen confirmation checkpoint hash changed")
    if "operator" not in checkpoint:
        raise RuntimeError("Frozen checkpoint does not contain operator state")

    groups = {str(item["name"]): str(item["group"]) for item in evaluation["cases"]}
    for case_index, raw_case in enumerate(evaluation["cases"]):
        item = build_stack_item(
            raw_case,
            device,
            spec,
            num_knots=int(config["model"]["num_knots"]),
        )
        load_shared_localized_state(item.operator, checkpoint["operator"])
        item.operator.eval()
        ls_repaired_detector = make_repaired_detector(
            int(item.context.grid.bits_per_symbol)
        ).to(device)
        perfect_receiver = standard_receiver(
            item.context, perfect_csi=True, return_crc=True
        )

        for snr_index, raw_snr in enumerate(evaluation["ebno_db"]):
            snr = float(raw_snr)
            for rep in range(int(evaluation["repetitions"])):
                key = (item.case.name, snr, rep)
                if key in done:
                    continue
                seed = (
                    int(config["seed"])
                    + 10_000_000
                    + case_index * 1_000_000
                    + snr_index * 10_000
                    + rep
                )
                set_all_seeds(seed)
                batch = item.context.sample(int(evaluation["batch_size"]), snr)
                with torch.inference_mode():
                    trained = observable_forward(item, batch)
                    uncertainty_off = uncertainty_off_forward(item, batch, trained)
                    mean_only = mean_only_forward(item, batch, trained)
                    ls_repaired = ls_repaired_forward(
                        item.ls_receiver,
                        item.context,
                        ls_repaired_detector,
                        batch,
                    )
                    custom_outputs = {
                        "trained_localized": trained,
                        "trained_uncertainty_off": uncertainty_off,
                        "trained_mean_only": mean_only,
                        "ls_estimate_repaired": ls_repaired,
                    }
                    decoded = decode_outputs(
                        item.context,
                        batch,
                        custom_outputs,
                        bp_iterations=int(evaluation["bp_iterations"]),
                        device=device,
                    )
                    ls_decoded = standard_decode(
                        item.ls_receiver,
                        batch,
                        batch.information_bits,
                        perfect_csi=False,
                    )
                    perfect_decoded = standard_decode(
                        perfect_receiver,
                        batch,
                        batch.information_bits,
                        perfect_csi=True,
                    )

                trained_error = block_error_vector(
                    decoded["trained_localized"], batch.information_bits
                )
                uncertainty_error = block_error_vector(
                    decoded["trained_uncertainty_off"], batch.information_bits
                )
                mean_only_error = block_error_vector(
                    decoded["trained_mean_only"], batch.information_bits
                )
                ls_error = ls_decoded["block_error"]
                group = groups[item.case.name]
                rows = [
                    custom_row(
                        case=item.case,
                        group=group,
                        variant=name,
                        snr=snr,
                        rep=rep,
                        seed=seed,
                        output=output,
                        batch=batch,
                        decoded=decoded[name],
                        signature=experiment["signature"],
                    )
                    for name, output in custom_outputs.items()
                ]
                rows.extend(
                    [
                        standard_row(
                            case=item.case,
                            group=group,
                            variant="ls_lmmse",
                            snr=snr,
                            rep=rep,
                            seed=seed,
                            metrics=ls_decoded,
                            signature=experiment["signature"],
                        ),
                        standard_row(
                            case=item.case,
                            group=group,
                            variant="perfect_csi_lmmse",
                            snr=snr,
                            rep=rep,
                            seed=seed,
                            metrics=perfect_decoded,
                            signature=experiment["signature"],
                        ),
                    ]
                )
                for row in rows:
                    add_pair_fields(
                        row,
                        trained_error=trained_error,
                        ls_error=ls_error,
                        uncertainty_error=uncertainty_error,
                        mean_only_error=mean_only_error,
                    )
                if {row["variant"] for row in rows} != set(variants):
                    raise RuntimeError("Confirmation variant-set mismatch")
                append_rows_atomic(RAW_PATH, rows)
                done.add(key)
                print(
                    json.dumps(
                        {
                            "case": item.case.name,
                            "ebno_db": snr,
                            "rep": rep,
                            "rows_committed": len(rows),
                            "completed_rows": len(done) * len(variants),
                            "expected_rows": expected_rows(config),
                            "trained_block_errors": int(trained_error.sum().item()),
                            "ls_block_errors": int(ls_error.sum().item()),
                        }
                    ),
                    flush=True,
                )
                if time.time() >= deadline_epoch:
                    frame = pd.read_csv(RAW_PATH)
                    return frame, {
                        "complete": False,
                        "rows": len(frame),
                        "expected_rows": expected_rows(config),
                        "contract": experiment,
                    }
        del item, ls_repaired_detector, perfect_receiver
        if device.type == "cuda":
            torch.cuda.empty_cache()

    frame = pd.read_csv(RAW_PATH)
    unique = len(frame.drop_duplicates(["case", "variant", "ebno_db", "rep"]))
    expected = expected_rows(config)
    if len(frame) != expected or unique != expected:
        raise RuntimeError(
            f"Confirmation incomplete: rows={len(frame)}, unique={unique}, expected={expected}"
        )
    return frame, {
        "complete": True,
        "rows": len(frame),
        "unique_rows": unique,
        "expected_rows": expected,
        "contract": experiment,
    }


def paired_delta(
    frame: pd.DataFrame,
    reference: str,
    comparator: str,
    metric: str,
    *,
    snr: float | None = None,
) -> dict[str, float]:
    keys = ["case", "ebno_db", "rep", "eval_seed"]
    work = frame
    if snr is not None:
        work = work[work["ebno_db"] == float(snr)]
    first = work[work["variant"] == reference]
    second = work[work["variant"] == comparator]
    merged = first.merge(second, on=keys, suffixes=("_a", "_b"))
    values = (
        pd.to_numeric(merged[f"{metric}_a"], errors="coerce")
        - pd.to_numeric(merged[f"{metric}_b"], errors="coerce")
    ).dropna()
    if not len(values):
        raise RuntimeError(f"No paired values for {reference} versus {comparator}")
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
    return {
        "pairs": int(len(values)),
        "mean": mean,
        "std": std,
        "ci95_low": mean - half,
        "ci95_high": mean + half,
    }


def stratified_bootstrap(
    frame: pd.DataFrame,
    *,
    repetitions: int,
    seed: int,
) -> dict[str, float]:
    trained = frame[frame["variant"] == "trained_localized"].copy()
    if not len(trained):
        raise RuntimeError("No trained rows for bootstrap")
    groups = [
        pd.to_numeric(group["trained_minus_ls_batch_tbler"], errors="coerce")
        .dropna()
        .to_numpy(dtype=float)
        for _, group in trained.groupby(["case", "ebno_db"], sort=True)
    ]
    if not groups or any(len(group) == 0 for group in groups):
        raise RuntimeError("Bootstrap strata are incomplete")
    rng = np.random.default_rng(int(seed))
    draws = np.empty(int(repetitions), dtype=float)
    total = float(sum(len(group) for group in groups))
    for index in range(int(repetitions)):
        accumulator = 0.0
        for values in groups:
            sample = rng.choice(values, size=len(values), replace=True)
            accumulator += float(sample.sum())
        draws[index] = accumulator / total
    return {
        "repetitions": int(repetitions),
        "seed": int(seed),
        "mean": float(np.mean(draws)),
        "ci95_low": float(np.quantile(draws, 0.025)),
        "ci95_high": float(np.quantile(draws, 0.975)),
    }


def exact_mcnemar(frame: pd.DataFrame) -> dict[str, Any]:
    trained = frame[frame["variant"] == "trained_localized"]
    trained_only = int(pd.to_numeric(trained["trained_only_errors"]).sum())
    ls_only = int(pd.to_numeric(trained["ls_only_errors"]).sum())
    discordant = trained_only + ls_only
    if discordant:
        try:
            from scipy.stats import binomtest

            p_one_sided = float(
                binomtest(ls_only, discordant, p=0.5, alternative="greater").pvalue
            )
        except Exception:
            probability = 0.0
            for value in range(ls_only, discordant + 1):
                probability += math.comb(discordant, value) * (0.5 ** discordant)
            p_one_sided = float(min(probability, 1.0))
    else:
        p_one_sided = 1.0
    return {
        "trained_only_errors": trained_only,
        "ls_only_errors": ls_only,
        "discordant_blocks": discordant,
        "one_sided_p_trained_better": p_one_sided,
        "secondary_only": True,
    }


def mean_metric(
    frame: pd.DataFrame,
    variant: str,
    metric: str,
    *,
    snr: float | None = None,
) -> float:
    subset = frame[frame["variant"] == variant]
    if snr is not None:
        subset = subset[subset["ebno_db"] == float(snr)]
    values = pd.to_numeric(subset[metric], errors="coerce").dropna()
    return float(values.mean()) if len(values) else float("nan")


def classify(
    frame: pd.DataFrame,
    config: dict[str, Any],
    evaluation: dict[str, Any],
    pre: dict[str, Any],
) -> dict[str, Any]:
    expected = expected_rows(config)
    variants = set(config["evaluation"]["variants"])
    complete_rows = len(frame) == expected
    unique_rows = len(
        frame.drop_duplicates(["case", "variant", "ebno_db", "rep"])
    ) == expected
    all_variants = set(frame["variant"].unique()) == variants
    paired_counts = frame.groupby(["case", "ebno_db", "rep"])["variant"].nunique()
    paired_batches = bool(len(paired_counts) and (paired_counts == len(variants)).all())
    trained_rows = frame[frame["variant"] == "trained_localized"]
    finite = bool(
        np.isfinite(pd.to_numeric(trained_rows["tbler"], errors="coerce")).all()
    )
    crc_ok = float(
        pd.to_numeric(trained_rows["crc_block_disagreement_rate"], errors="coerce").mean()
    ) <= float(config["decision"]["crc_disagreement_limit"])
    inference_no_truth = bool(
        (frame[frame["variant"].str.startswith("trained_")]["inference_uses_true_channel"] == False).all()  # noqa: E712
    )
    checkpoint_unchanged = pre["checkpoint_sha256"] == FROZEN_CHECKPOINT_SHA256

    overall = paired_delta(frame, "trained_localized", "ls_lmmse", "tbler")
    uncertainty = paired_delta(
        frame, "trained_localized", "trained_uncertainty_off", "tbler"
    )
    mean_only = paired_delta(
        frame, "trained_localized", "trained_mean_only", "tbler"
    )
    ls_control = paired_delta(
        frame, "ls_estimate_repaired", "ls_lmmse", "tbler"
    )
    per_snr = {
        str(float(snr)): paired_delta(
            frame, "trained_localized", "ls_lmmse", "tbler", snr=float(snr)
        )
        for snr in config["evaluation"]["ebno_db"]
    }
    bootstrap = stratified_bootstrap(
        frame,
        repetitions=int(config["decision"]["bootstrap_repetitions"]),
        seed=int(config["decision"]["bootstrap_seed"]),
    )
    mcnemar = exact_mcnemar(frame)

    ten = mean_metric(frame, "trained_localized", "tbler", snr=10.0)
    fourteen = mean_metric(frame, "trained_localized", "tbler", snr=14.0)
    no_reversal = fourteen <= ten + float(config["decision"]["max_14db_reversal"])
    no_snr_harm = all(
        value["mean"] <= float(config["decision"]["max_per_snr_harm"])
        for value in per_snr.values()
    )
    ls_match = abs(ls_control["mean"]) <= float(
        config["decision"]["ls_factorization_tolerance"]
    )

    software_checks = {
        "complete_rows": complete_rows,
        "unique_rows": unique_rows,
        "all_variants_present": all_variants,
        "paired_seed_batches": paired_batches,
        "all_core_metrics_finite": finite,
        "crc_consistency": crc_ok,
        "inference_uses_no_true_channel": inference_no_truth,
        "frozen_checkpoint_unchanged": checkpoint_unchanged,
        "no_retraining_or_retuning": pre["no_retraining"] and pre["no_retuning"],
        "ls_factorized_matches_standard": ls_match,
    }
    statistically_better = overall["ci95_high"] < 0.0 and bootstrap["ci95_high"] < 0.0
    point_better = overall["mean"] < 0.0
    parity = (
        abs(overall["mean"]) <= float(config["decision"]["final_parity_margin"])
        and overall["ci95_low"] <= 0.0 <= overall["ci95_high"]
    )
    scientific_checks = {
        "trained_point_estimate_beats_ls": point_better,
        "trained_statistically_beats_ls_student_t": overall["ci95_high"] < 0.0,
        "trained_statistically_beats_ls_bootstrap": bootstrap["ci95_high"] < 0.0,
        "no_material_per_snr_harm": no_snr_harm,
        "no_high_snr_reversal": no_reversal,
        "uncertainty_improves_tbler": uncertainty["ci95_high"] < 0.0,
        "localized_mean_improves_ls_variance_only": mean_only["ci95_high"] < 0.0,
        "secondary_mcnemar_supports_trained": mcnemar["one_sided_p_trained_better"] < 0.05,
    }

    if not all(software_checks.values()):
        classification = "GATE1_IMPLEMENTABLE_LOCALIZED_CONFIRMATION_INTERFACE_REPAIR_REQUIRED"
        next_action = "REPAIR_ONLY_FAILED_SOFTWARE_OR_BASELINE_CONTROL_AND_RERUN_IDENTICAL_CONFIRMATION"
    elif statistically_better and no_snr_harm and no_reversal:
        classification = "GATE1_IMPLEMENTABLE_LOCALIZED_CONFIRMED_BEATS_LS"
        next_action = "PROCEED_TO_PUBLICATION_SCALE_GENERALIZATION_COMPLEXITY_AND_LATENCY"
    elif parity and no_snr_harm:
        classification = "GATE1_IMPLEMENTABLE_LOCALIZED_FINAL_PARITY_WITH_LS"
        next_action = "STOP_LS_BEATING_CLAIM_AND_RETAIN_ONLY_IF_COMPLEXITY_OR_INTERPRETABILITY_CASE_IS_STRONG"
    else:
        classification = "GATE1_ABANDON_BAYESROUTE_FOR_LS_BEATING"
        next_action = "STOP_ARCHITECTURE_SEARCH_AND_DO_NOT_RETRAIN_RETUNE_OR_REDESIGN"

    trained_blocks = int(trained_rows["transport_blocks"].sum())
    metrics = {
        "trained_tbler": mean_metric(frame, "trained_localized", "tbler"),
        "ls_tbler": mean_metric(frame, "ls_lmmse", "tbler"),
        "perfect_tbler": mean_metric(frame, "perfect_csi_lmmse", "tbler"),
        "uncertainty_off_tbler": mean_metric(
            frame, "trained_uncertainty_off", "tbler"
        ),
        "mean_only_tbler": mean_metric(frame, "trained_mean_only", "tbler"),
        "trained_10db_tbler": ten,
        "trained_14db_tbler": fourteen,
        "transport_blocks_per_receiver": trained_blocks,
        "transport_blocks_per_snr_per_receiver": int(trained_blocks / 3),
        "trained_block_errors": int(trained_rows["trained_block_errors"].sum()),
        "ls_block_errors": int(trained_rows["ls_block_errors"].sum()),
        "trained_channel_nmse": mean_metric(
            frame, "trained_localized", "channel_nmse"
        ),
        "trained_coverage95": mean_metric(
            frame, "trained_localized", "coverage95"
        ),
    }
    return {
        "classification": classification,
        "next_action": next_action,
        "software_checks": software_checks,
        "scientific_checks": scientific_checks,
        "metrics": metrics,
        "per_snr_trained_minus_ls": per_snr,
        "paired_comparisons": {
            "trained_minus_ls_tbler": overall,
            "trained_minus_uncertainty_off_tbler": uncertainty,
            "trained_minus_mean_only_tbler": mean_only,
            "ls_repaired_minus_ls_tbler": ls_control,
        },
        "stratified_cluster_bootstrap": bootstrap,
        "mcnemar_secondary": mcnemar,
        "final_confirmation": True,
    }


def aggregate(frame: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "information_ber",
        "tbler",
        "crc_failure_rate",
        "coded_ber",
        "coded_bit_nll",
        "channel_nmse",
        "normalized_error_mean",
        "coverage95",
        "residual_gate",
        "correction_power",
        "effective_rank",
        "delta_gain_mean_abs",
        "edge_density",
        "transport_blocks",
        "trained_block_errors",
        "ls_block_errors",
    ]
    return (
        frame.groupby(["case", "variant", "ebno_db"], dropna=False)[metrics]
        .agg(["mean", "std", "count"])
        .reset_index()
    )


def make_plots(frame: pd.DataFrame, result: dict[str, Any]) -> list[str]:
    out = ROOT / "outputs/plots"
    out.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []

    plt.figure(figsize=(7.2, 4.8))
    for variant in (
        "trained_localized",
        "trained_uncertainty_off",
        "ls_lmmse",
        "perfect_csi_lmmse",
    ):
        subset = frame[frame["variant"] == variant]
        grouped = subset.groupby("ebno_db")["tbler"].mean().sort_index()
        plt.plot(grouped.index, grouped.values, marker="o", label=variant)
    plt.xlabel("$E_b/N_0$ (dB)")
    plt.ylabel("TBLER")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    path = out / "gate1_implementable_localized_confirmation_tbler.png"
    plt.savefig(path, dpi=180)
    plt.close()
    paths.append(str(path.relative_to(ROOT)))

    snrs = sorted(float(value) for value in frame["ebno_db"].unique())
    means = [result["per_snr_trained_minus_ls"][str(value)]["mean"] for value in snrs]
    lows = [result["per_snr_trained_minus_ls"][str(value)]["ci95_low"] for value in snrs]
    highs = [result["per_snr_trained_minus_ls"][str(value)]["ci95_high"] for value in snrs]
    lower_error = [mean - low for mean, low in zip(means, lows)]
    upper_error = [high - mean for mean, high in zip(means, highs)]
    plt.figure(figsize=(6.8, 4.5))
    plt.errorbar(snrs, means, yerr=[lower_error, upper_error], marker="o", capsize=4)
    plt.axhline(0.0, linewidth=1.0)
    plt.xlabel("$E_b/N_0$ (dB)")
    plt.ylabel("Proposed $-$ LS+LMMSE TBLER")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    path = out / "gate1_implementable_localized_confirmation_paired_delta.png"
    plt.savefig(path, dpi=180)
    plt.close()
    paths.append(str(path.relative_to(ROOT)))
    return paths


def write_incomplete(pre: dict[str, Any], evaluation: dict[str, Any]) -> None:
    report = {
        "version": VERSION,
        "complete": False,
        "preconditions": pre,
        "evaluation": evaluation,
        "classification": "GATE1_NR_IMPLEMENTABLE_LOCALIZED_CONFIRMATION_INCOMPLETE",
        "next_action": "RESUBMIT_SAME_COMMAND",
        "publication_nr_ready": False,
    }
    save_json(report, REPORT_PATH)
    save_json(report, GATE_JSON)
    GATE_TXT.parent.mkdir(parents=True, exist_ok=True)
    GATE_TXT.write_text(
        "CLASSIFICATION: GATE1_NR_IMPLEMENTABLE_LOCALIZED_CONFIRMATION_INCOMPLETE\n"
        "NEXT_ACTION: RESUBMIT_SAME_COMMAND\n"
        "PUBLICATION_NR_READY: NO\n",
        encoding="utf-8",
    )
    print("GATE1_NR_IMPLEMENTABLE_LOCALIZED_CONFIRMATION_INCOMPLETE: RESUBMIT")


def write_final(report: dict[str, Any]) -> None:
    save_json(report, REPORT_PATH)
    save_json(report, GATE_JSON)
    lines = [
        f"{name}: {'PASS' if value else 'FAIL'}"
        for name, value in report["software_checks"].items()
    ]
    lines.extend(
        f"{name}: {'PASS' if value else 'FAIL'}"
        for name, value in report["scientific_checks"].items()
    )
    lines.extend(
        [
            f"CLASSIFICATION: {report['classification']}",
            f"NEXT_ACTION: {report['next_action']}",
            "TRAINING_REQUIRED: NO",
            "RETUNING_ALLOWED: NO",
            "INFERENCE_USES_TRUE_CHANNEL: NO",
            "PUBLICATION_NR_READY: NO",
        ]
    )
    GATE_TXT.parent.mkdir(parents=True, exist_ok=True)
    GATE_TXT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--deadline-minutes", type=float, default=40.0)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    config_path = Path(args.config)
    config = load_yaml(config_path)
    pre = preconditions(config)
    if args.preflight_only:
        blocks_per_snr = (
            len(config["evaluation"]["cases"])
            * int(config["evaluation"]["repetitions"])
            * int(config["evaluation"]["batch_size"])
            * int(config["evaluation"]["cases"][0]["num_users"])
        )
        print("GATE1_NR_IMPLEMENTABLE_LOCALIZED_CONFIRMATION_PREFLIGHT_PASS")
        print("SOURCE_CLASSIFICATION", pre["source_classification"])
        print("FROZEN_CHECKPOINT", pre["checkpoint_sha256"])
        print("CASES", len(config["evaluation"]["cases"]))
        print("EXPECTED_ROWS", expected_rows(config))
        print("TRANSPORT_BLOCKS_PER_SNR_PER_RECEIVER", blocks_per_snr)
        print("TRAINING_REQUIRED NO")
        print("RETUNING_ALLOWED NO")
        print("INFERENCE_USES_TRUE_CHANNEL NO")
        return

    device = normalize_device(args.device)
    deadline = time.time() + 60.0 * float(args.deadline_minutes)
    frame, evaluation = evaluate(
        config,
        config_path,
        pre,
        device,
        deadline_epoch=deadline,
    )
    if not evaluation.get("complete"):
        write_incomplete(pre, evaluation)
        return

    result = classify(frame, config, evaluation, pre)
    aggregate_frame = aggregate(frame)
    AGGREGATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    aggregate_frame.to_csv(AGGREGATE_PATH, index=False)
    paired = frame[frame["variant"] == "trained_localized"][
        [
            "case",
            "ebno_db",
            "rep",
            "eval_seed",
            "transport_blocks",
            "trained_block_errors",
            "ls_block_errors",
            "trained_only_errors",
            "ls_only_errors",
            "both_errors",
            "both_correct",
            "trained_minus_ls_batch_tbler",
        ]
    ].copy()
    PAIRED_PATH.parent.mkdir(parents=True, exist_ok=True)
    paired.to_csv(PAIRED_PATH, index=False)
    plots = make_plots(frame, result)
    report = {
        "version": VERSION,
        "model_version": MODEL_VERSION,
        "complete": True,
        "preconditions": pre,
        "evaluation": evaluation,
        **result,
        "raw_csv": str(RAW_PATH.relative_to(ROOT)),
        "aggregate_csv": str(AGGREGATE_PATH.relative_to(ROOT)),
        "paired_csv": str(PAIRED_PATH.relative_to(ROOT)),
        "plots": plots,
        "environment": {
            "device": str(device),
            "gpu_name": (
                torch.cuda.get_device_name(device)
                if device.type == "cuda"
                else "cpu"
            ),
            "torch": torch.__version__,
            "python": platform.python_version(),
        },
        "publication_nr_ready": False,
    }
    write_final(report)


if __name__ == "__main__":
    main()
