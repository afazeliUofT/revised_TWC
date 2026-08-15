#!/usr/bin/env python3
from __future__ import annotations

"""Final implementable LS-anchored localized receiver go/no-go.

Exactly one model is trained on 4-/8-PRB UMi cases. A fresh 12-PRB case is
constructed only after training and checkpoint selection are complete. Failure
to beat LS+LMMSE ends the BayesRoute search for that objective.
"""

import argparse
import csv
import json
import math
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

from bayesroute.config import capture_rng_state, restore_rng_state
from bayesroute.ls_anchored_localized_posterior import (
    IMPLEMENTABLE_LOCALIZED_VERSION,
    bind_shared_localized_parameters,
    load_shared_localized_state,
    shared_localized_state,
    unique_localized_parameters,
)
from bayesroute.nr_gate1 import (
    NRCase,
    build_nr_context,
    normalize_device,
    run_standard_receiver,
    standard_receiver,
)
from gate1_nr_implementable_localized_common import (
    GATE_VERSION,
    atomic_torch_save,
    build_shared_stack,
    build_stack_item,
    channel_mse,
    custom_row,
    decode_outputs,
    differentiable_training_loss,
    localized_ceiling_preconditions,
    mean_only_forward,
    observable_forward,
    observable_metrics,
    package_signature,
    save_json,
    selected_basis_spec,
    set_all_seeds,
    sha256_file,
    source_hashes,
    standard_row,
    uncertainty_off_forward,
)
from gate1_nr_joint_operator_common import make_repaired_detector, repaired_forward
from gate1_nr_posterior_factorial_common import ls_repaired_forward
from gate1_nr_turbo_posterior_common import build_loaded_bridge


OUTPUT_ROOT = ROOT / "outputs/gate1_nr_implementable_localized"
TRAIN_REPORT = ROOT / "outputs/reports/gate1_nr_implementable_localized_train.json"
FINAL_REPORT = ROOT / "outputs/reports/gate1_nr_implementable_localized.json"
RAW_PATH = ROOT / "outputs/eval/gate1_nr_implementable_localized.csv"
CONTRACT_PATH = ROOT / "outputs/eval/gate1_nr_implementable_localized_contract.json"
AGGREGATE_PATH = ROOT / "outputs/eval/gate1_nr_implementable_localized_aggregate.csv"
LOG_PATH = ROOT / "outputs/logs/gate1_nr_implementable_localized_train.csv"
GATE_JSON = ROOT / "outputs/gates/GATE1_NR_IMPLEMENTABLE_LOCALIZED.json"
GATE_TXT = ROOT / "outputs/gates/GATE1_NR_IMPLEMENTABLE_LOCALIZED.txt"
SMOKE_REPORT = ROOT / "outputs/gates/GATE1_NR_IMPLEMENTABLE_LOCALIZED_SMOKE.json"


def load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected YAML mapping: {path}")
    return value


def expected_rows(config: dict[str, Any]) -> int:
    evaluation = config["evaluation"]
    return (
        len(evaluation["cases"])
        * len(evaluation["ebno_db"])
        * int(evaluation["repetitions"])
        * len(evaluation["variants"])
    )


def preconditions(config: dict[str, Any]) -> dict[str, Any]:
    ceiling = localized_ceiling_preconditions()
    if not SMOKE_REPORT.is_file():
        raise RuntimeError(f"Missing implementable-localized smoke report: {SMOKE_REPORT}")
    smoke = json.loads(SMOKE_REPORT.read_text(encoding="utf-8"))
    training_prb = [int(item["num_prb"]) for item in config["training_cases"]]
    validation_prb = [int(item["num_prb"]) for item in config["validation_cases"]]
    holdout = [
        item for item in config["evaluation"]["cases"]
        if int(item["num_prb"]) == 12
    ]
    checks = {
        "ceiling": ceiling["passed"],
        "smoke_complete": smoke.get("complete") is True,
        "smoke_pass": smoke.get("classification") == "GATE1_NR_IMPLEMENTABLE_LOCALIZED_SMOKE_PASS",
        "smoke_ready": smoke.get("screen_ready") is True,
        "model_version": smoke.get("model_version") == IMPLEMENTABLE_LOCALIZED_VERSION,
        "revision": config.get("revision") == GATE_VERSION,
        "training_only_4_8_prb": training_prb == [4, 8],
        "validation_only_4_8_prb": validation_prb == [4, 8],
        "one_fresh_12prb_holdout": len(holdout) == 1
        and holdout[0].get("group") == "fresh_untouched_12prb_holdout",
        "holdout_not_training": all(
            int(item["num_prb"]) != 12
            for item in config["training_cases"] + config["validation_cases"]
        ),
        "expected_variants": config["evaluation"]["variants"] == [
            "trained_localized",
            "trained_uncertainty_off",
            "trained_mean_only",
            "old_global_repaired",
            "ls_estimate_repaired",
            "ls_lmmse",
            "perfect_csi_lmmse",
        ],
        "expected_rows_positive": expected_rows(config) > 0,
    }
    if not all(checks.values()):
        raise RuntimeError(f"Implementable-localized preconditions failed: {checks}")
    return {
        "passed": True,
        "checks": checks,
        "ceiling": ceiling,
        "smoke_report": str(SMOKE_REPORT.relative_to(ROOT)),
        "smoke_classification": smoke["classification"],
    }


def training_contract(
    config: dict[str, Any], config_path: Path, pre: dict[str, Any]
) -> dict[str, Any]:
    payload = {
        "version": GATE_VERSION,
        "config_sha256": sha256_file(config_path),
        "source_sha256": source_hashes(),
        "basis": config["basis"],
        "model": config["model"],
        "training": config["training"],
        "training_cases": config["training_cases"],
        "validation_cases": config["validation_cases"],
        "holdout_policy": {
            "training_prb": [4, 8],
            "validation_prb": [4, 8],
            "holdout_prb": 12,
            "holdout_used_for_training": False,
            "holdout_used_for_validation": False,
            "holdout_constructed_after_training_complete": True,
        },
        "ceiling_classification": pre["ceiling"]["classification"],
        "ceiling_winner": pre["ceiling"]["winner"],
        "inference_uses_true_channel": False,
    }
    payload["signature"] = package_signature(payload)
    return payload


def validation_score(
    items: Sequence[Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    training = config["training"]
    records: list[dict[str, Any]] = []
    with torch.no_grad():
        for case_index, item in enumerate(items):
            for snr_index, raw_snr in enumerate(training["validation_ebno_db"]):
                snr = float(raw_snr)
                seed = (
                    int(training["validation_seed"])
                    + case_index * 100_000
                    + snr_index * 1_000
                )
                set_all_seeds(seed)
                batch = item.context.sample(
                    int(training["validation_batch_size"]), snr
                )
                output = observable_forward(item, batch)
                metrics = observable_metrics(output, batch)
                fused_mse = float(channel_mse(output["posterior"], batch).item())
                ls_mse = float(channel_mse(output["ls_posterior"], batch).item())
                records.append(
                    {
                        "case": item.case.name,
                        "num_prb": int(item.case.num_prb),
                        "ebno_db": snr,
                        **metrics,
                        "fused_channel_mse": fused_mse,
                        "ls_channel_mse": ls_mse,
                        "mse_ratio_to_ls": fused_mse / max(ls_mse, 1e-12),
                    }
                )
    nll = np.asarray([item["coded_bit_nll"] for item in records], dtype=float)
    nmse = np.asarray([item["channel_nmse"] for item in records], dtype=float)
    normalized = np.asarray(
        [item["normalized_error_mean"] for item in records], dtype=float
    )
    ratios = np.asarray([item["mse_ratio_to_ls"] for item in records], dtype=float)
    calibration = np.abs(np.log(np.clip(normalized, 1e-6, None)))
    score = (
        float(nll.mean())
        + 0.25 * float(nll.max())
        + 0.10 * float(nmse.mean())
        + 0.05 * float(calibration.mean())
        + 0.15 * float(np.maximum(ratios - 1.0, 0.0).mean())
    )
    return {
        "score": score,
        "mean_coded_bit_nll": float(nll.mean()),
        "worst_coded_bit_nll": float(nll.max()),
        "mean_channel_nmse": float(nmse.mean()),
        "mean_mse_ratio_to_ls": float(ratios.mean()),
        "worst_mse_ratio_to_ls": float(ratios.max()),
        "mean_calibration_abs_log": float(calibration.mean()),
        "records": records,
    }


def cosine_lr(step: int, total: int, start: float, end: float) -> float:
    if total <= 1:
        return float(end)
    progress = min(max(float(step) / float(total - 1), 0.0), 1.0)
    return float(
        end + 0.5 * (start - end) * (1.0 + math.cos(math.pi * progress))
    )


def train(
    config: dict[str, Any],
    config_path: Path,
    pre: dict[str, Any],
    device: torch.device,
    *,
    deadline_epoch: float,
) -> dict[str, Any]:
    if TRAIN_REPORT.is_file():
        prior = json.loads(TRAIN_REPORT.read_text(encoding="utf-8"))
        if prior.get("complete") is True:
            return prior

    spec = selected_basis_spec(pre["ceiling"])
    training = config["training"]
    train_items = build_shared_stack(
        config["training_cases"],
        device,
        spec,
        num_knots=int(config["model"]["num_knots"]),
    )
    validation_items = build_shared_stack(
        config["validation_cases"],
        device,
        spec,
        num_knots=int(config["model"]["num_knots"]),
    )
    bind_shared_localized_parameters(
        [item.operator for item in train_items + validation_items]
    )
    parameters = unique_localized_parameters(
        [item.operator for item in train_items + validation_items]
    )
    optimizer = torch.optim.AdamW(
        parameters,
        lr=float(training["learning_rate_start"]),
        weight_decay=float(training["weight_decay"]),
    )
    contract = training_contract(config, config_path, pre)
    checkpoint_dir = OUTPUT_ROOT / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_path = checkpoint_dir / "best.pt"
    last_path = checkpoint_dir / "last.pt"
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    max_steps = int(training["max_steps"])
    min_steps = int(training["min_steps"])
    validation_every = int(training["validation_every"])
    patience_limit = int(training["early_stopping_patience"])
    start_step = 0
    best_score = float("inf")
    best_step = -1
    no_improve = 0
    validation_history: list[dict[str, Any]] = []
    baseline_validation: dict[str, Any] | None = None

    if last_path.is_file():
        state = torch.load(last_path, map_location=device, weights_only=False)
        if state.get("contract") != contract:
            raise RuntimeError("Implementable-localized checkpoint contract mismatch")
        load_shared_localized_state(train_items[0].operator, state["operator"])
        optimizer.load_state_dict(state["optimizer"])
        start_step = int(state["step"]) + 1
        best_score = float(state["best_score"])
        best_step = int(state["best_step"])
        no_improve = int(state["no_improve"])
        validation_history = list(state.get("validation_history", []))
        baseline_validation = state.get("baseline_validation")
        restore_rng_state(state.get("rng_state"))
        print(f"Resuming implementable-localized training at step {start_step}", flush=True)
    else:
        set_all_seeds(int(config["seed"]))
        baseline_validation = validation_score(validation_items, config)
        best_score = float(baseline_validation["score"])
        validation_history = [
            {"step": -1, "score": best_score, "source": "safe_ls_anchored_initialization"}
        ]
        initial = {
            "version": GATE_VERSION,
            "operator": shared_localized_state(train_items[0].operator),
            "optimizer": optimizer.state_dict(),
            "step": -1,
            "best_score": best_score,
            "best_step": -1,
            "no_improve": 0,
            "validation": baseline_validation,
            "baseline_validation": baseline_validation,
            "validation_history": validation_history,
            "contract": contract,
            "rng_state": capture_rng_state(),
        }
        atomic_torch_save(initial, best_path)
        atomic_torch_save(initial, last_path)

    fields = [
        "step",
        "case",
        "num_prb",
        "ebno_db",
        "learning_rate",
        "loss",
        "coded_bit_nll",
        "channel_nll",
        "calibration_penalty",
        "fused_channel_mse",
        "ls_channel_mse",
        "ls_gain_hinge",
        "gradient_norm",
        "validation_score",
        "best_score",
        "no_improve",
        "contract_signature",
    ]
    if not LOG_PATH.is_file():
        with LOG_PATH.open("w", newline="", encoding="utf-8") as stream:
            csv.DictWriter(stream, fieldnames=fields).writeheader()

    stopped_early = False
    last_validation = baseline_validation
    for step in range(start_step, max_steps):
        item = train_items[step % len(train_items)]
        learning_rate = cosine_lr(
            step,
            max_steps,
            float(training["learning_rate_start"]),
            float(training["learning_rate_end"]),
        )
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        snr = float(training["ebno_db_min"]) + (
            float(training["ebno_db_max"]) - float(training["ebno_db_min"])
        ) * float(torch.rand(1).item())
        batch = item.context.sample(int(training["batch_size"]), snr)
        optimizer.zero_grad(set_to_none=True)
        output = observable_forward(item, batch)
        loss, parts = differentiable_training_loss(
            output,
            batch,
            channel_loss_weight=float(training["channel_loss_weight"]),
            calibration_loss_weight=float(training["calibration_loss_weight"]),
            ls_gain_loss_weight=float(training["ls_gain_loss_weight"]),
            ls_gain_target_ratio=float(training["ls_gain_target_ratio"]),
        )
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite localized training loss at step {step}")
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(
            parameters, float(training["grad_clip"])
        )
        if not torch.isfinite(grad):
            raise RuntimeError(f"Non-finite localized gradient at step {step}")
        optimizer.step()

        validation_value = float("nan")
        validation_due = (
            step % validation_every == 0 or step == max_steps - 1
        )
        if validation_due:
            last_validation = validation_score(validation_items, config)
            validation_value = float(last_validation["score"])
            validation_history.append(
                {
                    "step": step,
                    "score": validation_value,
                    "mean_coded_bit_nll": last_validation["mean_coded_bit_nll"],
                    "mean_channel_nmse": last_validation["mean_channel_nmse"],
                    "mean_mse_ratio_to_ls": last_validation["mean_mse_ratio_to_ls"],
                }
            )
            if validation_value < best_score - float(training["minimum_improvement"]):
                best_score = validation_value
                best_step = step
                no_improve = 0
                atomic_torch_save(
                    {
                        "version": GATE_VERSION,
                        "operator": shared_localized_state(train_items[0].operator),
                        "optimizer": optimizer.state_dict(),
                        "step": step,
                        "best_score": best_score,
                        "best_step": best_step,
                        "no_improve": no_improve,
                        "validation": last_validation,
                        "baseline_validation": baseline_validation,
                        "validation_history": validation_history,
                        "contract": contract,
                        "rng_state": capture_rng_state(),
                    },
                    best_path,
                )
            else:
                no_improve += 1

        save_due = (
            step % int(training["save_every"]) == 0
            or validation_due
            or step == max_steps - 1
        )
        if save_due:
            row = {
                "step": step,
                "case": item.case.name,
                "num_prb": int(item.case.num_prb),
                "ebno_db": snr,
                "learning_rate": learning_rate,
                "loss": float(loss.detach().item()),
                "coded_bit_nll": float(parts["bit_nll"].detach().item()),
                "channel_nll": float(parts["channel_nll"].detach().item()),
                "calibration_penalty": float(parts["calibration_penalty"].detach().item()),
                "fused_channel_mse": float(parts["fused_channel_mse"].detach().item()),
                "ls_channel_mse": float(parts["ls_channel_mse"].detach().item()),
                "ls_gain_hinge": float(parts["ls_gain_hinge"].detach().item()),
                "gradient_norm": float(grad.detach().item()),
                "validation_score": validation_value,
                "best_score": best_score,
                "no_improve": no_improve,
                "contract_signature": contract["signature"],
            }
            with LOG_PATH.open("a", newline="", encoding="utf-8") as stream:
                csv.DictWriter(stream, fieldnames=fields).writerow(row)
            print(json.dumps(row), flush=True)
            atomic_torch_save(
                {
                    "version": GATE_VERSION,
                    "operator": shared_localized_state(train_items[0].operator),
                    "optimizer": optimizer.state_dict(),
                    "step": step,
                    "best_score": best_score,
                    "best_step": best_step,
                    "no_improve": no_improve,
                    "validation": last_validation,
                    "baseline_validation": baseline_validation,
                    "validation_history": validation_history,
                    "contract": contract,
                    "rng_state": capture_rng_state(),
                },
                last_path,
            )
            if time.time() >= deadline_epoch:
                return {
                    "complete": False,
                    "version": GATE_VERSION,
                    "step": step,
                    "max_steps": max_steps,
                    "best_score": best_score,
                    "contract": contract,
                    "stop_reason": "internal_deadline",
                }

        if (
            step + 1 >= min_steps
            and validation_due
            and no_improve >= patience_limit
        ):
            stopped_early = True
            break

    best = torch.load(best_path, map_location=device, weights_only=False)
    if best.get("contract") != contract:
        raise RuntimeError("Best localized checkpoint contract mismatch")
    history = list(best.get("validation_history", validation_history))
    scored = [item for item in history if int(item.get("step", -1)) >= 0]
    if len(scored) >= 3:
        tail_improvement = float(scored[-3]["score"] - scored[-1]["score"])
    else:
        tail_improvement = float("inf")
    training_converged = bool(
        stopped_early
        or best_step <= int(best["step"]) - 2 * validation_every
        or tail_improvement <= float(training["tail_improvement_tolerance"])
    )
    last_state = torch.load(last_path, map_location="cpu", weights_only=False)
    last_executed_step = int(last_state["step"])
    summary = {
        "complete": True,
        "version": GATE_VERSION,
        "model_version": IMPLEMENTABLE_LOCALIZED_VERSION,
        "steps_executed": last_executed_step + 1,
        "last_executed_step": last_executed_step,
        "max_steps": max_steps,
        "minimum_steps": min_steps,
        "stopped_early": stopped_early,
        "training_converged": training_converged,
        "best_step": int(best["best_step"]),
        "best_score": float(best["best_score"]),
        "baseline_score": float(baseline_validation["score"]),
        "validation_score_improvement": float(baseline_validation["score"] - best["best_score"]),
        "tail_validation_improvement": tail_improvement,
        "baseline_validation": baseline_validation,
        "best_validation": best["validation"],
        "validation_history": history,
        "best_checkpoint": str(best_path.relative_to(ROOT)),
        "best_checkpoint_sha256": sha256_file(best_path),
        "last_checkpoint": str(last_path.relative_to(ROOT)),
        "trainable_parameters": int(sum(parameter.numel() for parameter in parameters)),
        "parameter_report": train_items[0].operator.parameter_report(),
        "basis_reports": [item.basis_report for item in train_items],
        "contract": contract,
        "fresh_12prb_used_for_training_or_selection": False,
        "inference_uses_true_channel": False,
    }
    save_json(summary, TRAIN_REPORT)
    del train_items, validation_items, optimizer
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return summary


def append_rows_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    new = pd.DataFrame(rows)
    if path.is_file():
        old = pd.read_csv(path)
        if list(old.columns) != list(new.columns):
            raise RuntimeError("Implementable-localized CSV column contract mismatch")
        combined = pd.concat([old, new], ignore_index=True)
    else:
        combined = new
    temporary = path.with_suffix(path.suffix + ".tmp")
    combined.to_csv(temporary, index=False)
    temporary.replace(path)


def evaluation_contract(
    config: dict[str, Any], config_path: Path, train_report: dict[str, Any]
) -> dict[str, Any]:
    payload = {
        "version": GATE_VERSION,
        "config_sha256": sha256_file(config_path),
        "source_sha256": source_hashes(),
        "training_checkpoint_sha256": train_report["best_checkpoint_sha256"],
        "training_contract_signature": train_report["contract"]["signature"],
        "evaluation": config["evaluation"],
        "holdout_policy": {
            "holdout_used_for_training": False,
            "holdout_used_for_validation": False,
            "holdout_constructed_after_training_complete": True,
        },
        "inference_uses_true_channel": False,
    }
    payload["signature"] = package_signature(payload)
    return payload


def evaluate(
    config: dict[str, Any],
    config_path: Path,
    train_report: dict[str, Any],
    device: torch.device,
    *,
    deadline_epoch: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    evaluation = config["evaluation"]
    variants = [str(value) for value in evaluation["variants"]]
    contract = evaluation_contract(config, config_path, train_report)
    if RAW_PATH.is_file():
        if not CONTRACT_PATH.is_file():
            raise RuntimeError("Localized evaluation CSV exists without contract")
        prior = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
        if prior.get("signature") != contract["signature"]:
            raise RuntimeError("Localized evaluation contract mismatch")
    else:
        save_json(contract, CONTRACT_PATH)

    done: set[tuple[str, float, int]] = set()
    if RAW_PATH.is_file():
        frame = pd.read_csv(RAW_PATH)
        keys = ["case", "variant", "ebno_db", "rep"]
        if frame[keys].duplicated().any():
            raise RuntimeError("Localized evaluation contains duplicate keys")
        counts = frame.groupby(["case", "ebno_db", "rep"])["variant"].nunique()
        if len(counts[counts != len(variants)]):
            raise RuntimeError("Localized evaluation contains partial paired batches")
        done = {(str(a), float(b), int(c)) for a, b, c in counts.index}

    spec = selected_basis_spec()
    checkpoint_path = ROOT / train_report["best_checkpoint"]
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if checkpoint.get("contract") != train_report["contract"]:
        raise RuntimeError("Localized evaluation checkpoint contract mismatch")
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
        old_bridge = build_loaded_bridge(
            item.case, item.context, operator_seed=int(config["old_global_operator_seed"])
        )
        old_detector = make_repaired_detector(int(item.context.grid.bits_per_symbol)).to(device)
        ls_repaired_detector = make_repaired_detector(int(item.context.grid.bits_per_symbol)).to(device)
        perfect_receiver = standard_receiver(item.context, perfect_csi=True, return_crc=True)

        for snr_index, raw_snr in enumerate(evaluation["ebno_db"]):
            snr = float(raw_snr)
            for rep in range(int(evaluation["repetitions"])):
                key = (item.case.name, snr, rep)
                if key in done:
                    continue
                seed = (
                    int(config["seed"])
                    + 10_000_000
                    + case_index * 100_000
                    + snr_index * 1_000
                    + rep
                )
                set_all_seeds(seed)
                batch = item.context.sample(int(evaluation["batch_size"]), snr)
                with torch.inference_mode():
                    trained = observable_forward(item, batch)
                    uncertainty_off = uncertainty_off_forward(item, batch, trained)
                    mean_only = mean_only_forward(item, batch, trained)
                    old_global = repaired_forward(old_bridge, old_detector, batch)
                    old_global["inference_uses_true_channel"] = False
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
                        "old_global_repaired": old_global,
                        "ls_estimate_repaired": ls_repaired,
                    }
                    decoded = decode_outputs(
                        item.context,
                        batch,
                        custom_outputs,
                        bp_iterations=int(evaluation["bp_iterations"]),
                        device=device,
                    )
                    ls_metrics = run_standard_receiver(
                        item.ls_receiver,
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
                        signature=contract["signature"],
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
                            metrics=ls_metrics,
                            signature=contract["signature"],
                        ),
                        standard_row(
                            case=item.case,
                            group=group,
                            variant="perfect_csi_lmmse",
                            snr=snr,
                            rep=rep,
                            seed=seed,
                            metrics=perfect_metrics,
                            signature=contract["signature"],
                        ),
                    ]
                )
                if {row["variant"] for row in rows} != set(variants):
                    raise RuntimeError("Localized evaluation variant-set mismatch")
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
                        "contract": contract,
                    }
        del item, old_bridge, old_detector, ls_repaired_detector, perfect_receiver
        if device.type == "cuda":
            torch.cuda.empty_cache()

    frame = pd.read_csv(RAW_PATH)
    unique = len(frame.drop_duplicates(["case", "variant", "ebno_db", "rep"]))
    expected = expected_rows(config)
    if len(frame) != expected or unique != expected:
        raise RuntimeError(
            f"Localized evaluation incomplete: rows={len(frame)}, unique={unique}, expected={expected}"
        )
    return frame, {
        "complete": True,
        "rows": len(frame),
        "unique_rows": unique,
        "expected_rows": expected,
        "contract": contract,
    }


def paired_delta(
    frame: pd.DataFrame,
    reference: str,
    comparator: str,
    metric: str,
    *,
    prb: int | None = None,
) -> dict[str, float]:
    keys = ["case", "ebno_db", "rep", "eval_seed"]
    work = frame
    if prb is not None:
        work = work[work["num_prb"] == int(prb)]
    first = work[work["variant"] == reference]
    second = work[work["variant"] == comparator]
    merged = first.merge(second, on=keys, suffixes=("_a", "_b"))
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
    return {
        "pairs": int(len(values)),
        "mean": mean,
        "ci95_low": mean - half,
        "ci95_high": mean + half,
    }


def mean_metric(
    frame: pd.DataFrame,
    variant: str,
    metric: str,
    *,
    prb: int | None = None,
    snr: float | None = None,
) -> float:
    subset = frame[frame["variant"] == variant]
    if prb is not None:
        subset = subset[subset["num_prb"] == int(prb)]
    if snr is not None:
        subset = subset[subset["ebno_db"] == float(snr)]
    values = pd.to_numeric(subset[metric], errors="coerce").dropna()
    return float(values.mean()) if len(values) else float("nan")


def classify(
    frame: pd.DataFrame,
    config: dict[str, Any],
    train_report: dict[str, Any],
    evaluation: dict[str, Any],
) -> dict[str, Any]:
    variants = set(config["evaluation"]["variants"])
    complete_rows = len(frame) == expected_rows(config)
    unique_rows = len(
        frame.drop_duplicates(["case", "variant", "ebno_db", "rep"])
    ) == expected_rows(config)
    all_variants = set(frame["variant"].unique()) == variants
    paired_counts = frame.groupby(["case", "ebno_db", "rep"])["variant"].nunique()
    paired_seed_batches = bool(len(paired_counts) > 0 and (paired_counts == len(variants)).all())
    all_finite = bool(
        np.isfinite(
            pd.to_numeric(
                frame[frame["variant"] == "trained_localized"]["tbler"],
                errors="coerce",
            )
        ).all()
    )
    ls_factorization = paired_delta(
        frame, "ls_estimate_repaired", "ls_lmmse", "tbler", prb=12
    )
    trained_minus_ls = paired_delta(
        frame, "trained_localized", "ls_lmmse", "tbler", prb=12
    )
    trained_minus_old = paired_delta(
        frame, "trained_localized", "old_global_repaired", "tbler", prb=12
    )
    trained_minus_uncertainty_off = paired_delta(
        frame, "trained_localized", "trained_uncertainty_off", "tbler", prb=12
    )
    trained_minus_mean_only = paired_delta(
        frame, "trained_localized", "trained_mean_only", "tbler", prb=12
    )
    ten = mean_metric(frame, "trained_localized", "tbler", prb=12, snr=10.0)
    fourteen = mean_metric(frame, "trained_localized", "tbler", prb=12, snr=14.0)
    no_reversal = fourteen <= ten + float(config["decision"]["max_14db_reversal"])
    inference_no_truth = bool(
        (frame[frame["variant"].str.startswith("trained_")]["inference_uses_true_channel"] == False).all()  # noqa: E712
    )
    crc_ok = float(
        pd.to_numeric(
            frame[frame["variant"] == "trained_localized"]["crc_block_disagreement_rate"],
            errors="coerce",
        ).mean()
    ) <= 0.005
    ls_match = abs(ls_factorization["mean"]) <= 0.005
    software_checks = {
        "complete_rows": complete_rows,
        "unique_rows": unique_rows,
        "all_variants_present": all_variants,
        "all_core_metrics_finite": all_finite,
        "paired_seed_batches": paired_seed_batches,
        "crc_consistency": crc_ok,
        "ls_factorized_matches_standard": ls_match,
        "inference_uses_no_true_channel": inference_no_truth,
        "fresh_12prb_holdout": evaluation["contract"]["holdout_policy"]["holdout_constructed_after_training_complete"],
    }

    if not all(software_checks.values()):
        classification = "GATE1_IMPLEMENTABLE_LOCALIZED_INTERFACE_REPAIR_REQUIRED"
        next_action = "REPAIR_ONLY_FAILED_SOFTWARE_OR_BASELINE_CONTROL"
    elif trained_minus_ls["ci95_high"] < 0.0 and no_reversal:
        classification = "GATE1_IMPLEMENTABLE_LOCALIZED_BEATS_LS"
        next_action = "PROCEED_TO_PUBLICATION_SCALE_EVALUATION_AND_COMPLEXITY_AUDIT"
    elif trained_minus_ls["mean"] < 0.0 and no_reversal:
        classification = "GATE1_IMPLEMENTABLE_LOCALIZED_POSSIBLY_BEATS_LS"
        next_action = "RUN_ONE_LARGER_FIXED_CONFIRMATION_WITHOUT_RETUNING"
    else:
        classification = "GATE1_ABANDON_BAYESROUTE_FOR_LS_BEATING"
        next_action = "STOP_ARCHITECTURE_SEARCH_AND_DO_NOT_RETRAIN_OR_REDESIGN"

    batch_size = int(config["evaluation"]["batch_size"])
    repetitions = int(config["evaluation"]["repetitions"])
    users = int(
        next(
            item["num_users"]
            for item in config["evaluation"]["cases"]
            if int(item["num_prb"]) == 12
        )
    )
    blocks_per_snr = batch_size * repetitions * users
    metrics = {
        "trained_12prb_tbler": mean_metric(frame, "trained_localized", "tbler", prb=12),
        "ls_12prb_tbler": mean_metric(frame, "ls_lmmse", "tbler", prb=12),
        "perfect_12prb_tbler": mean_metric(frame, "perfect_csi_lmmse", "tbler", prb=12),
        "old_global_12prb_tbler": mean_metric(frame, "old_global_repaired", "tbler", prb=12),
        "trained_uncertainty_off_12prb_tbler": mean_metric(frame, "trained_uncertainty_off", "tbler", prb=12),
        "trained_mean_only_12prb_tbler": mean_metric(frame, "trained_mean_only", "tbler", prb=12),
        "trained_12prb_10db_tbler": ten,
        "trained_12prb_14db_tbler": fourteen,
        "trained_12prb_channel_nmse": mean_metric(frame, "trained_localized", "channel_nmse", prb=12),
        "trained_12prb_coverage95": mean_metric(frame, "trained_localized", "coverage95", prb=12),
        "trained_12prb_residual_gate": mean_metric(frame, "trained_localized", "residual_gate", prb=12),
        "transport_blocks_per_snr_per_receiver": blocks_per_snr,
        "estimated_trained_block_errors": int(round(mean_metric(frame, "trained_localized", "tbler", prb=12) * blocks_per_snr * len(config["evaluation"]["ebno_db"]))),
        "estimated_ls_block_errors": int(round(mean_metric(frame, "ls_lmmse", "tbler", prb=12) * blocks_per_snr * len(config["evaluation"]["ebno_db"]))),
    }
    return {
        "classification": classification,
        "next_action": next_action,
        "software_checks": software_checks,
        "scientific_checks": {
            "trained_mean_beats_ls": trained_minus_ls["mean"] < 0.0,
            "trained_statistically_beats_ls": trained_minus_ls["ci95_high"] < 0.0,
            "trained_beats_old_global": trained_minus_old["ci95_high"] < 0.0,
            "uncertainty_improves_tbler": trained_minus_uncertainty_off["ci95_high"] < 0.0,
            "calibrated_uncertainty_beats_ls_variance_only": trained_minus_mean_only["ci95_high"] < 0.0,
            "no_high_snr_reversal": no_reversal,
            "training_converged": bool(train_report["training_converged"]),
        },
        "metrics": metrics,
        "paired_comparisons": {
            "trained_minus_ls_12prb_tbler": trained_minus_ls,
            "trained_minus_old_global_12prb_tbler": trained_minus_old,
            "trained_minus_uncertainty_off_12prb_tbler": trained_minus_uncertainty_off,
            "trained_minus_mean_only_12prb_tbler": trained_minus_mean_only,
            "ls_repaired_minus_ls_12prb_tbler": ls_factorization,
        },
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
    ]
    return (
        frame.groupby(["case", "group", "num_prb", "variant", "ebno_db"], dropna=False)[metrics]
        .agg(["mean", "std", "count"])
        .reset_index()
    )


def make_plots(frame: pd.DataFrame, train_report: dict[str, Any]) -> list[str]:
    out = ROOT / "outputs/plots"
    out.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    selected = [
        "trained_localized",
        "old_global_repaired",
        "ls_lmmse",
        "perfect_csi_lmmse",
    ]
    for prb in sorted(int(value) for value in frame["num_prb"].unique()):
        plt.figure(figsize=(7.2, 4.8))
        for variant in selected:
            subset = frame[(frame["num_prb"] == prb) & (frame["variant"] == variant)]
            grouped = subset.groupby("ebno_db")["tbler"].mean().sort_index()
            plt.plot(grouped.index, grouped.values, marker="o", label=variant)
        plt.xlabel("$E_b/N_0$ (dB)")
        plt.ylabel("TBLER")
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=8)
        plt.tight_layout()
        path = out / f"gate1_implementable_localized_{prb}prb_tbler.png"
        plt.savefig(path, dpi=180)
        plt.close()
        paths.append(str(path.relative_to(ROOT)))

    history = train_report.get("validation_history", [])
    if history:
        x = [int(item["step"]) for item in history]
        y = [float(item["score"]) for item in history]
        plt.figure(figsize=(6.8, 4.5))
        plt.plot(x, y, marker="o")
        plt.xlabel("Training step")
        plt.ylabel("Validation score")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        path = out / "gate1_implementable_localized_validation.png"
        plt.savefig(path, dpi=180)
        plt.close()
        paths.append(str(path.relative_to(ROOT)))
    return paths


def write_incomplete(
    train_report: dict[str, Any], evaluation: dict[str, Any] | None = None
) -> None:
    report = {
        "version": GATE_VERSION,
        "complete": False,
        "training": train_report,
        "evaluation": evaluation,
        "classification": "GATE1_NR_IMPLEMENTABLE_LOCALIZED_INCOMPLETE",
        "next_action": "RESUBMIT_SAME_COMMAND",
        "publication_nr_ready": False,
    }
    save_json(report, FINAL_REPORT)
    save_json(report, GATE_JSON)
    GATE_TXT.parent.mkdir(parents=True, exist_ok=True)
    GATE_TXT.write_text(
        "CLASSIFICATION: GATE1_NR_IMPLEMENTABLE_LOCALIZED_INCOMPLETE\n"
        "NEXT_ACTION: RESUBMIT_SAME_COMMAND\n"
        "PUBLICATION_NR_READY: NO\n",
        encoding="utf-8",
    )
    print("GATE1_NR_IMPLEMENTABLE_LOCALIZED_INCOMPLETE: RESUBMIT", flush=True)


def write_final(report: dict[str, Any]) -> None:
    save_json(report, FINAL_REPORT)
    save_json(report, GATE_JSON)
    lines = [
        f"{name}: {'PASS' if value else 'FAIL'}"
        for name, value in report["software_checks"].items()
    ]
    lines.extend(
        [
            f"{name}: {'PASS' if value else 'FAIL'}"
            for name, value in report["scientific_checks"].items()
        ]
    )
    lines.extend(
        [
            f"CLASSIFICATION: {report['classification']}",
            f"NEXT_ACTION: {report['next_action']}",
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
    parser.add_argument("--deadline-minutes", type=float, default=85.0)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    config_path = Path(args.config)
    config = load_yaml(config_path)
    pre = preconditions(config)
    if args.preflight_only:
        print("GATE1_NR_IMPLEMENTABLE_LOCALIZED_FINAL_PREFLIGHT_PASS")
        print("MODEL_VERSION", IMPLEMENTABLE_LOCALIZED_VERSION)
        print("WINNER_BASIS", pre["ceiling"]["winner"]["name"])
        print("MAX_TRAINING_STEPS", config["training"]["max_steps"])
        print("MIN_TRAINING_STEPS", config["training"]["min_steps"])
        print("TRAINING_PRB", [item["num_prb"] for item in config["training_cases"]])
        print("HOLDOUT_PRB", 12)
        print("EXPECTED_ROWS", expected_rows(config))
        print("INFERENCE_USES_TRUE_CHANNEL NO")
        print("HARD_FINAL_STOP YES")
        return

    device = normalize_device(args.device)
    deadline = time.time() + 60.0 * float(args.deadline_minutes)
    train_report = train(
        config,
        config_path,
        pre,
        device,
        deadline_epoch=deadline,
    )
    if not train_report.get("complete"):
        write_incomplete(train_report)
        return
    frame, evaluation = evaluate(
        config,
        config_path,
        train_report,
        device,
        deadline_epoch=deadline,
    )
    if not evaluation.get("complete"):
        write_incomplete(train_report, evaluation)
        return
    classified = classify(frame, config, train_report, evaluation)
    aggregate_frame = aggregate(frame)
    AGGREGATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    aggregate_frame.to_csv(AGGREGATE_PATH, index=False)
    plots = make_plots(frame, train_report)
    report = {
        "version": GATE_VERSION,
        "model_version": IMPLEMENTABLE_LOCALIZED_VERSION,
        "complete": True,
        "preconditions": pre,
        "training": train_report,
        "evaluation": evaluation,
        "aggregate_csv": str(AGGREGATE_PATH.relative_to(ROOT)),
        "plots": plots,
        **classified,
        "environment": {
            "device": str(device),
            "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
            "torch": torch.__version__,
            "python": platform.python_version(),
        },
        "publication_nr_ready": False,
    }
    write_final(report)


if __name__ == "__main__":
    main()
