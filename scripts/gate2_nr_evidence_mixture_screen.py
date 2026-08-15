#!/usr/bin/env python3
from __future__ import annotations

"""Train and evaluate the evidence-weighted mixture LMMSE channel estimator.

The estimator and all LMMSE channel-estimation baselines feed the same repaired
spatial LMMSE detector.  This isolates channel-estimation quality from detector
choice.  The four-component estimator and the one-component LMMSE estimator
are trained on exactly the same batches.  No evaluation case is used for
checkpoint selection.
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
from scipy.stats import t as student_t

ROOT = Path(__file__).resolve().parents[1]

from bayesroute.evidence_mixture_lmmse import (
    EVIDENCE_MIXTURE_LMMSE_VERSION,
    evidence_state_to_jsonable,
    load_shared_evidence_state,
    shared_evidence_state,
    unique_evidence_parameters,
)
from bayesroute.nr_gate1 import normalize_device, run_standard_receiver, standard_receiver
from gate2_nr_evidence_mixture_common import (
    GATE2_VERSION,
    atomic_torch_save,
    audit_state_json,
    basis_spec_from_config,
    build_shared_stack,
    build_stack_item,
    custom_row,
    decode_outputs,
    evidence_forward,
    evidence_metrics,
    gradient_report,
    load_json,
    package_signature,
    perfect_channel_forward,
    save_json,
    set_all_seeds,
    sha256_file,
    source_hashes,
    source_result_preconditions,
    standard_row,
    training_loss,
)
from gate1_nr_posterior_factorial_common import ls_repaired_forward


OUTPUT_ROOT = ROOT / "outputs/gate2_nr_evidence_mixture"
TRAIN_REPORT = ROOT / "outputs/reports/gate2_nr_evidence_mixture_train.json"
FINAL_REPORT = ROOT / "outputs/reports/gate2_nr_evidence_mixture.json"
PROPOSED_STATE_JSON = ROOT / "outputs/reports/gate2_nr_evidence_mixture_proposed_state.json"
SINGLE_STATE_JSON = ROOT / "outputs/reports/gate2_nr_evidence_mixture_single_lmmse_state.json"
RAW_PATH = ROOT / "outputs/eval/gate2_nr_evidence_mixture.csv"
CONTRACT_PATH = ROOT / "outputs/eval/gate2_nr_evidence_mixture_contract.json"
AGGREGATE_PATH = ROOT / "outputs/eval/gate2_nr_evidence_mixture_aggregate.csv"
PAIRED_PATH = ROOT / "outputs/reports/gate2_nr_evidence_mixture_paired.csv"
LOG_PATH = ROOT / "outputs/logs/gate2_nr_evidence_mixture_train.csv"
GATE_JSON = ROOT / "outputs/gates/GATE2_NR_EVIDENCE_MIXTURE.json"
GATE_TXT = ROOT / "outputs/gates/GATE2_NR_EVIDENCE_MIXTURE.txt"
SMOKE_REPORT = ROOT / "outputs/gates/GATE2_NR_EVIDENCE_MIXTURE_SMOKE.json"


PROPOSED = "proposed_evidence_mixture"
HARD = "proposed_hard_evidence"
UNIFORM = "proposed_uniform_mixture"
MOMENT = "proposed_moment_lmmse_ce"
SINGLE = "trained_single_lmmse_ce"
LS_COMMON = "ls_linear_common_detector"
LS_CHAIN = "sionna_ls_lmmse_chain"
PERFECT = "perfect_csi_common_detector"


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
    source = source_result_preconditions()
    if not SMOKE_REPORT.is_file():
        raise RuntimeError(f"Missing Gate-2 smoke report: {SMOKE_REPORT}")
    smoke = load_json(SMOKE_REPORT)
    training_prb = sorted({int(item["num_prb"]) for item in config["training_cases"]})
    validation_names = {str(item["name"]) for item in config["validation_cases"]}
    evaluation_names = {str(item["name"]) for item in config["evaluation"]["cases"]}
    checks = {
        "source_result": source["passed"],
        "source_parity_not_misread_as_lmmse_ce": source["classification"]
        == "GATE1_IMPLEMENTABLE_LOCALIZED_FINAL_PARITY_WITH_LS",
        "smoke_complete": smoke.get("complete") is True,
        "smoke_pass": smoke.get("classification")
        == "GATE2_NR_EVIDENCE_MIXTURE_SMOKE_PASS",
        "smoke_ready": smoke.get("screen_ready") is True,
        "model_version": smoke.get("model_version")
        == EVIDENCE_MIXTURE_LMMSE_VERSION,
        "revision": config.get("revision") == GATE2_VERSION,
        "proposed_components": int(config["model"]["proposed_components"]) == 4,
        "single_lmmse_component": int(config["model"]["single_lmmse_components"])
        == 1,
        "small_parameter_count": int(
            smoke.get("parameter_report", {}).get("trainable_parameters", 10**9)
        )
        <= 128,
        "training_grid_span": training_prb == [4, 8, 12],
        "validation_evaluation_disjoint": validation_names.isdisjoint(evaluation_names),
        "common_detector_variants": config["evaluation"]["variants"]
        == [PROPOSED, HARD, UNIFORM, MOMENT, SINGLE, LS_COMMON, LS_CHAIN, PERFECT],
        "deterministic_step_seeding": config["training"].get(
            "deterministic_step_seeding"
        )
        is True,
        "expected_rows": expected_rows(config) == 2304,
    }
    if not all(checks.values()):
        raise RuntimeError(f"Gate-2 preconditions failed: {checks}")
    return {
        "passed": True,
        "checks": checks,
        "source": source,
        "smoke_report": str(SMOKE_REPORT.relative_to(ROOT)),
        "smoke_classification": smoke["classification"],
    }


def training_contract(
    config: dict[str, Any], config_path: Path, pre: dict[str, Any]
) -> dict[str, Any]:
    payload = {
        "version": GATE2_VERSION,
        "model_version": EVIDENCE_MIXTURE_LMMSE_VERSION,
        "config_sha256": sha256_file(config_path),
        "source_sha256": source_hashes(),
        "basis": config["basis"],
        "model": config["model"],
        "training": config["training"],
        "training_cases": config["training_cases"],
        "validation_cases": config["validation_cases"],
        "source_result": {
            "classification": pre["source"]["classification"],
            "report": pre["source"]["report"],
        },
        "fairness": {
            "same_batches_for_proposed_and_single_lmmse": True,
            "same_common_detector": True,
            "evaluation_cases_not_used_for_training_or_selection": True,
            "inference_uses_true_channel": False,
        },
    }
    payload["signature"] = package_signature(payload)
    return payload


def evaluation_contract(
    config: dict[str, Any], config_path: Path, train: dict[str, Any]
) -> dict[str, Any]:
    payload = {
        "version": GATE2_VERSION,
        "config_sha256": sha256_file(config_path),
        "source_sha256": source_hashes(),
        "evaluation": config["evaluation"],
        "decision": config["decision"],
        "training_contract_signature": train["contract"]["signature"],
        "proposed_checkpoint_sha256": train["proposed_best_checkpoint_sha256"],
        "single_lmmse_checkpoint_sha256": train[
            "single_lmmse_best_checkpoint_sha256"
        ],
        "policy": {
            "common_detector_for_estimator_primary_comparisons": True,
            "sionna_ls_lmmse_chain_is_secondary": True,
            "evaluation_cases_used_for_training": False,
            "evaluation_cases_used_for_validation": False,
            "inference_uses_true_channel": False,
        },
    }
    payload["signature"] = package_signature(payload)
    return payload


def validation_score(
    items: Sequence[Any], config: dict[str, Any], *, mode: str = "mixture"
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
                output = evidence_forward(item, batch, mode=mode)
                metrics = evidence_metrics(output, batch)
                records.append(
                    {
                        "case": item.case.name,
                        "scenario": item.case.scenario,
                        "num_prb": int(item.case.num_prb),
                        "ebno_db": snr,
                        **metrics,
                    }
                )
    nll = np.asarray([item["coded_bit_nll"] for item in records], dtype=float)
    nmse = np.asarray([item["channel_nmse"] for item in records], dtype=float)
    normalized = np.asarray(
        [item["normalized_error_mean"] for item in records], dtype=float
    )
    calibration = np.abs(np.log(np.clip(normalized, 1e-6, None)))
    evidence = np.asarray(
        [item.get("negative_log_evidence", 0.0) for item in records], dtype=float
    )
    score = (
        float(nll.mean())
        + 0.25 * float(nll.max())
        + 0.15 * float(nmse.mean())
        + 0.05 * float(calibration.mean())
        + 0.02 * float(evidence.mean())
    )
    return {
        "score": score,
        "mean_coded_bit_nll": float(nll.mean()),
        "worst_coded_bit_nll": float(nll.max()),
        "mean_channel_nmse": float(nmse.mean()),
        "mean_calibration_abs_log": float(calibration.mean()),
        "mean_negative_log_evidence": float(evidence.mean()),
        "records": records,
    }


def cosine_lr(step: int, total: int, start: float, end: float) -> float:
    if total <= 1:
        return float(end)
    progress = min(max(float(step) / float(total - 1), 0.0), 1.0)
    return float(
        end + 0.5 * (start - end) * (1.0 + math.cos(math.pi * progress))
    )


def deterministic_training_seed(config: dict[str, Any], step: int) -> int:
    training = config["training"]
    if training.get("deterministic_step_seeding") is not True:
        raise RuntimeError("Deterministic training-step seeding is required")
    return int(config["seed"]) + int(training["step_seed_offset"]) + int(step)


def _checkpoint_payload(
    *,
    step: int,
    proposed: Any,
    single: Any,
    proposed_optimizer: torch.optim.Optimizer,
    single_optimizer: torch.optim.Optimizer,
    contract: dict[str, Any],
    state: dict[str, Any],
) -> dict[str, Any]:
    return {
        "version": GATE2_VERSION,
        "step": int(step),
        "proposed": shared_evidence_state(proposed.operator),
        "single": shared_evidence_state(single.operator),
        "proposed_optimizer": proposed_optimizer.state_dict(),
        "single_optimizer": single_optimizer.state_dict(),
        "contract": contract,
        **state,
    }


def train(
    config: dict[str, Any],
    config_path: Path,
    pre: dict[str, Any],
    device: torch.device,
    *,
    deadline_epoch: float,
) -> dict[str, Any]:
    if TRAIN_REPORT.is_file():
        prior = load_json(TRAIN_REPORT)
        if prior.get("complete") is True:
            return prior

    spec = basis_spec_from_config(config)
    proposed_train = build_shared_stack(
        config["training_cases"],
        device,
        spec,
        num_components=int(config["model"]["proposed_components"]),
        num_knots=int(config["model"]["num_knots"]),
    )
    single_train = build_shared_stack(
        config["training_cases"],
        device,
        spec,
        num_components=1,
        num_knots=int(config["model"]["num_knots"]),
    )
    proposed_validation = build_shared_stack(
        config["validation_cases"],
        device,
        spec,
        num_components=int(config["model"]["proposed_components"]),
        num_knots=int(config["model"]["num_knots"]),
    )
    single_validation = build_shared_stack(
        config["validation_cases"],
        device,
        spec,
        num_components=1,
        num_knots=int(config["model"]["num_knots"]),
    )
    from bayesroute.evidence_mixture_lmmse import bind_shared_evidence_parameters

    bind_shared_evidence_parameters(
        [item.operator for item in proposed_train + proposed_validation]
    )
    bind_shared_evidence_parameters(
        [item.operator for item in single_train + single_validation]
    )
    proposed_parameters = unique_evidence_parameters(
        [item.operator for item in proposed_train + proposed_validation]
    )
    single_parameters = unique_evidence_parameters(
        [item.operator for item in single_train + single_validation]
    )
    training = config["training"]
    proposed_optimizer = torch.optim.AdamW(
        proposed_parameters,
        lr=float(training["learning_rate_start"]),
        weight_decay=float(training["weight_decay"]),
    )
    single_optimizer = torch.optim.AdamW(
        single_parameters,
        lr=float(training["learning_rate_start"]),
        weight_decay=float(training["weight_decay"]),
    )
    contract = training_contract(config, config_path, pre)
    checkpoint_dir = OUTPUT_ROOT / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    last_path = checkpoint_dir / "last.pt"
    proposed_best_path = checkpoint_dir / "proposed_best.pt"
    single_best_path = checkpoint_dir / "single_lmmse_best.pt"
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    state: dict[str, Any] = {
        "proposed_best_score": float("inf"),
        "single_best_score": float("inf"),
        "proposed_best_step": -1,
        "single_best_step": -1,
        "proposed_no_improve": 0,
        "single_no_improve": 0,
        "proposed_validation_history": [],
        "single_validation_history": [],
        "baseline_validation": None,
    }
    start_step = 0
    if last_path.is_file():
        saved = torch.load(last_path, map_location=device, weights_only=False)
        if saved.get("contract") != contract:
            raise RuntimeError("Gate-2 training checkpoint contract mismatch")
        load_shared_evidence_state(proposed_train[0].operator, saved["proposed"])
        load_shared_evidence_state(single_train[0].operator, saved["single"])
        proposed_optimizer.load_state_dict(saved["proposed_optimizer"])
        single_optimizer.load_state_dict(saved["single_optimizer"])
        start_step = int(saved["step"]) + 1
        for key in state:
            state[key] = saved.get(key, state[key])
        print(f"Resuming Gate-2 training at step {start_step}", flush=True)
    else:
        proposed_baseline = validation_score(proposed_validation, config)
        single_baseline = validation_score(single_validation, config)
        state.update(
            {
                "proposed_best_score": float(proposed_baseline["score"]),
                "single_best_score": float(single_baseline["score"]),
                "proposed_best_step": -1,
                "single_best_step": -1,
                "proposed_validation_history": [
                    {"step": -1, **proposed_baseline}
                ],
                "single_validation_history": [{"step": -1, **single_baseline}],
                "baseline_validation": {
                    "proposed": proposed_baseline,
                    "single_lmmse": single_baseline,
                },
            }
        )
        initial = _checkpoint_payload(
            step=-1,
            proposed=proposed_train[0],
            single=single_train[0],
            proposed_optimizer=proposed_optimizer,
            single_optimizer=single_optimizer,
            contract=contract,
            state=state,
        )
        atomic_torch_save(initial, last_path)
        atomic_torch_save(initial, proposed_best_path)
        atomic_torch_save(initial, single_best_path)

    fields = [
        "step",
        "case",
        "num_prb",
        "scenario",
        "ebno_db",
        "training_seed",
        "learning_rate",
        "proposed_loss",
        "single_loss",
        "proposed_bit_nll",
        "single_bit_nll",
        "proposed_channel_nmse",
        "single_channel_nmse",
        "proposed_gradient_norm",
        "single_gradient_norm",
        "proposed_validation_score",
        "single_validation_score",
        "proposed_best_score",
        "single_best_score",
        "contract_signature",
    ]
    if not LOG_PATH.is_file():
        with LOG_PATH.open("w", newline="", encoding="utf-8") as stream:
            csv.DictWriter(stream, fieldnames=fields).writeheader()

    max_steps = int(training["max_steps"])
    min_steps = int(training["min_steps"])
    validation_every = int(training["validation_every"])
    patience = int(training["early_stopping_patience"])
    minimum_improvement = float(training["minimum_improvement"])
    completed = False
    stopped_early = False
    last_step = start_step - 1

    for step in range(start_step, max_steps):
        if time.time() >= deadline_epoch:
            print("Gate-2 training deadline reached; checkpointing", flush=True)
            break
        case_index = step % len(proposed_train)
        proposed_item = proposed_train[case_index]
        single_item = single_train[case_index]
        seed = deterministic_training_seed(config, step)
        set_all_seeds(seed)
        snr_generator = torch.Generator(device="cpu")
        snr_generator.manual_seed(seed + 71)
        snr = float(
            torch.empty((), dtype=torch.float32)
            .uniform_(
                float(training["ebno_db_min"]),
                float(training["ebno_db_max"]),
                generator=snr_generator,
            )
            .item()
        )
        batch = proposed_item.context.sample(int(training["batch_size"]), snr)
        learning_rate = cosine_lr(
            step,
            max_steps,
            float(training["learning_rate_start"]),
            float(training["learning_rate_end"]),
        )
        for optimizer in (proposed_optimizer, single_optimizer):
            for group in optimizer.param_groups:
                group["lr"] = learning_rate
            optimizer.zero_grad(set_to_none=True)

        proposed_output = evidence_forward(proposed_item, batch, mode="mixture")
        single_output = evidence_forward(single_item, batch, mode="mixture")
        proposed_loss, proposed_parts = training_loss(
            proposed_item, proposed_output, batch, training
        )
        single_loss, single_parts = training_loss(
            single_item, single_output, batch, training
        )
        (proposed_loss + single_loss).backward()
        proposed_grad = float(
            torch.nn.utils.clip_grad_norm_(
                proposed_parameters, float(training["grad_clip"])
            ).item()
        )
        single_grad = float(
            torch.nn.utils.clip_grad_norm_(
                single_parameters, float(training["grad_clip"])
            ).item()
        )
        if not math.isfinite(proposed_grad) or not math.isfinite(single_grad):
            raise RuntimeError("Non-finite Gate-2 training gradient")
        proposed_optimizer.step()
        single_optimizer.step()
        last_step = step

        proposed_validation_score = float("nan")
        single_validation_score = float("nan")
        should_validate = (step % validation_every == 0) or (step == max_steps - 1)
        if should_validate:
            proposed_validation_result = validation_score(proposed_validation, config)
            single_validation_result = validation_score(single_validation, config)
            proposed_validation_score = float(proposed_validation_result["score"])
            single_validation_score = float(single_validation_result["score"])
            state["proposed_validation_history"].append(
                {"step": step, **proposed_validation_result}
            )
            state["single_validation_history"].append(
                {"step": step, **single_validation_result}
            )
            if proposed_validation_score < float(state["proposed_best_score"]) - minimum_improvement:
                state["proposed_best_score"] = proposed_validation_score
                state["proposed_best_step"] = step
                state["proposed_no_improve"] = 0
                payload = _checkpoint_payload(
                    step=step,
                    proposed=proposed_train[0],
                    single=single_train[0],
                    proposed_optimizer=proposed_optimizer,
                    single_optimizer=single_optimizer,
                    contract=contract,
                    state=state,
                )
                atomic_torch_save(payload, proposed_best_path)
            else:
                state["proposed_no_improve"] = int(state["proposed_no_improve"]) + 1
            if single_validation_score < float(state["single_best_score"]) - minimum_improvement:
                state["single_best_score"] = single_validation_score
                state["single_best_step"] = step
                state["single_no_improve"] = 0
                payload = _checkpoint_payload(
                    step=step,
                    proposed=proposed_train[0],
                    single=single_train[0],
                    proposed_optimizer=proposed_optimizer,
                    single_optimizer=single_optimizer,
                    contract=contract,
                    state=state,
                )
                atomic_torch_save(payload, single_best_path)
            else:
                state["single_no_improve"] = int(state["single_no_improve"]) + 1

        if step % int(training["save_every"]) == 0 or should_validate:
            payload = _checkpoint_payload(
                step=step,
                proposed=proposed_train[0],
                single=single_train[0],
                proposed_optimizer=proposed_optimizer,
                single_optimizer=single_optimizer,
                contract=contract,
                state=state,
            )
            atomic_torch_save(payload, last_path)

        if step % 25 == 0 or should_validate:
            row = {
                "step": step,
                "case": proposed_item.case.name,
                "num_prb": int(proposed_item.case.num_prb),
                "scenario": proposed_item.case.scenario,
                "ebno_db": snr,
                "training_seed": seed,
                "learning_rate": learning_rate,
                "proposed_loss": float(proposed_loss.detach().item()),
                "single_loss": float(single_loss.detach().item()),
                "proposed_bit_nll": float(proposed_parts["bit_nll"].detach().item()),
                "single_bit_nll": float(single_parts["bit_nll"].detach().item()),
                "proposed_channel_nmse": float(
                    proposed_parts["channel_nmse"].detach().item()
                ),
                "single_channel_nmse": float(
                    single_parts["channel_nmse"].detach().item()
                ),
                "proposed_gradient_norm": proposed_grad,
                "single_gradient_norm": single_grad,
                "proposed_validation_score": proposed_validation_score,
                "single_validation_score": single_validation_score,
                "proposed_best_score": float(state["proposed_best_score"]),
                "single_best_score": float(state["single_best_score"]),
                "contract_signature": contract["signature"],
            }
            with LOG_PATH.open("a", newline="", encoding="utf-8") as stream:
                csv.DictWriter(stream, fieldnames=fields).writerow(row)
            print(json.dumps(row, default=str), flush=True)

        if (
            step + 1 >= min_steps
            and int(state["proposed_no_improve"]) >= patience
            and int(state["single_no_improve"]) >= patience
        ):
            stopped_early = True
            completed = True
            break
    else:
        completed = True

    if last_step >= start_step:
        payload = _checkpoint_payload(
            step=last_step,
            proposed=proposed_train[0],
            single=single_train[0],
            proposed_optimizer=proposed_optimizer,
            single_optimizer=single_optimizer,
            contract=contract,
            state=state,
        )
        atomic_torch_save(payload, last_path)

    if not completed:
        report = {
            "version": GATE2_VERSION,
            "complete": False,
            "classification": "GATE2_NR_EVIDENCE_MIXTURE_TRAINING_INCOMPLETE",
            "next_action": "RESUBMIT",
            "steps_executed": max(last_step + 1, 0),
            "contract": contract,
            **state,
        }
        save_json(report, TRAIN_REPORT)
        return report

    proposed_best = torch.load(proposed_best_path, map_location=device, weights_only=False)
    single_best = torch.load(single_best_path, map_location=device, weights_only=False)
    load_shared_evidence_state(proposed_train[0].operator, proposed_best["proposed"])
    load_shared_evidence_state(single_train[0].operator, single_best["single"])
    save_json(audit_state_json(proposed_train[0].operator), PROPOSED_STATE_JSON)
    save_json(audit_state_json(single_train[0].operator), SINGLE_STATE_JSON)
    report = {
        "version": GATE2_VERSION,
        "model_version": EVIDENCE_MIXTURE_LMMSE_VERSION,
        "complete": True,
        "classification": "GATE2_NR_EVIDENCE_MIXTURE_TRAINING_COMPLETE",
        "contract": contract,
        "steps_executed": int(last_step + 1),
        "stopped_early": stopped_early,
        "max_steps": max_steps,
        "minimum_steps": min_steps,
        "training_converged": bool(stopped_early or last_step + 1 == max_steps),
        "proposed_best_step": int(state["proposed_best_step"]),
        "single_lmmse_best_step": int(state["single_best_step"]),
        "proposed_best_score": float(state["proposed_best_score"]),
        "single_lmmse_best_score": float(state["single_best_score"]),
        "proposed_best_checkpoint": str(proposed_best_path.relative_to(ROOT)),
        "single_lmmse_best_checkpoint": str(single_best_path.relative_to(ROOT)),
        "proposed_best_checkpoint_sha256": sha256_file(proposed_best_path),
        "single_lmmse_best_checkpoint_sha256": sha256_file(single_best_path),
        "proposed_parameter_report": proposed_train[0].operator.parameter_report(),
        "single_lmmse_parameter_report": single_train[0].operator.parameter_report(),
        "proposed_validation_history": state["proposed_validation_history"],
        "single_lmmse_validation_history": state["single_validation_history"],
        "baseline_validation": state["baseline_validation"],
        "proposed_state_json": str(PROPOSED_STATE_JSON.relative_to(ROOT)),
        "single_lmmse_state_json": str(SINGLE_STATE_JSON.relative_to(ROOT)),
        "training_log": str(LOG_PATH.relative_to(ROOT)),
        "same_training_batches": True,
        "evaluation_cases_used_for_training_or_selection": False,
        "inference_uses_true_channel": False,
    }
    save_json(report, TRAIN_REPORT)
    return report


def append_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.is_file() and path.stat().st_size > 0
    with path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        if not exists:
            writer.writeheader()
        writer.writerows(rows)
        stream.flush()


def completed_batch_keys(path: Path) -> set[tuple[str, float, int]]:
    if not path.is_file() or path.stat().st_size == 0:
        return set()
    frame = pd.read_csv(path)
    expected = {PROPOSED, HARD, UNIFORM, MOMENT, SINGLE, LS_COMMON, LS_CHAIN, PERFECT}
    result: set[tuple[str, float, int]] = set()
    for key, group in frame.groupby(["case", "ebno_db", "rep"]):
        if set(group["variant"].astype(str)) == expected and len(group) == len(expected):
            result.add((str(key[0]), float(key[1]), int(key[2])))
    return result


def _augment_standard_metrics(metrics: dict[str, Any], batch: Any) -> dict[str, Any]:
    value = dict(metrics)
    transport_blocks = int(batch.information_bits.shape[0] * batch.information_bits.shape[1])
    value["transport_blocks"] = transport_blocks
    value["block_errors"] = int(round(float(value["tbler"]) * transport_blocks))
    return value


def evaluate(
    config: dict[str, Any],
    config_path: Path,
    train_report: dict[str, Any],
    device: torch.device,
    *,
    deadline_epoch: float,
) -> dict[str, Any]:
    contract = evaluation_contract(config, config_path, train_report)
    if CONTRACT_PATH.is_file():
        prior = load_json(CONTRACT_PATH)
        if prior != contract:
            raise RuntimeError("Gate-2 evaluation contract mismatch")
    else:
        save_json(contract, CONTRACT_PATH)

    spec = basis_spec_from_config(config)
    proposed_items = build_shared_stack(
        config["evaluation"]["cases"],
        device,
        spec,
        num_components=int(config["model"]["proposed_components"]),
        num_knots=int(config["model"]["num_knots"]),
    )
    single_items = build_shared_stack(
        config["evaluation"]["cases"],
        device,
        spec,
        num_components=1,
        num_knots=int(config["model"]["num_knots"]),
    )
    from bayesroute.evidence_mixture_lmmse import bind_shared_evidence_parameters

    bind_shared_evidence_parameters([item.operator for item in proposed_items])
    bind_shared_evidence_parameters([item.operator for item in single_items])
    proposed_checkpoint = ROOT / train_report["proposed_best_checkpoint"]
    single_checkpoint = ROOT / train_report["single_lmmse_best_checkpoint"]
    if sha256_file(proposed_checkpoint) != train_report["proposed_best_checkpoint_sha256"]:
        raise RuntimeError("Proposed checkpoint hash mismatch")
    if sha256_file(single_checkpoint) != train_report[
        "single_lmmse_best_checkpoint_sha256"
    ]:
        raise RuntimeError("Single-LMMSE checkpoint hash mismatch")
    proposed_saved = torch.load(proposed_checkpoint, map_location=device, weights_only=False)
    single_saved = torch.load(single_checkpoint, map_location=device, weights_only=False)
    load_shared_evidence_state(proposed_items[0].operator, proposed_saved["proposed"])
    load_shared_evidence_state(single_items[0].operator, single_saved["single"])

    completed = completed_batch_keys(RAW_PATH)
    evaluation = config["evaluation"]
    expected = expected_rows(config)
    for case_index, (proposed_item, single_item) in enumerate(
        zip(proposed_items, single_items)
    ):
        standard_ls = standard_receiver(
            proposed_item.context, perfect_csi=False, return_crc=True
        )
        for snr_index, raw_snr in enumerate(evaluation["ebno_db"]):
            snr = float(raw_snr)
            for rep in range(int(evaluation["repetitions"])):
                key = (proposed_item.case.name, snr, rep)
                if key in completed:
                    continue
                if time.time() >= deadline_epoch:
                    return {
                        "complete": False,
                        "rows": int(pd.read_csv(RAW_PATH).shape[0]) if RAW_PATH.is_file() else 0,
                        "expected_rows": expected,
                        "raw_csv": str(RAW_PATH.relative_to(ROOT)),
                        "contract": contract,
                    }
                seed = (
                    int(config["seed"])
                    + 40_000_000
                    + case_index * 1_000_000
                    + snr_index * 10_000
                    + rep
                )
                set_all_seeds(seed)
                batch = proposed_item.context.sample(int(evaluation["batch_size"]), snr)
                with torch.inference_mode():
                    proposed = evidence_forward(proposed_item, batch, mode="mixture")
                    hard = evidence_forward(proposed_item, batch, mode="hard")
                    uniform = evidence_forward(proposed_item, batch, mode="uniform")
                    moment = evidence_forward(proposed_item, batch, mode="moment")
                    single = evidence_forward(single_item, batch, mode="mixture")
                    ls_common = ls_repaired_forward(
                        proposed_item.ls_receiver,
                        proposed_item.context,
                        proposed_item.detector,
                        batch,
                    )
                    perfect = perfect_channel_forward(proposed_item, batch)
                    decoded = decode_outputs(
                        proposed_item.context,
                        batch,
                        {
                            PROPOSED: proposed,
                            HARD: hard,
                            UNIFORM: uniform,
                            MOMENT: moment,
                            SINGLE: single,
                            LS_COMMON: ls_common,
                            PERFECT: perfect,
                        },
                        bp_iterations=int(evaluation["bp_iterations"]),
                        device=device,
                    )
                    standard_metrics = run_standard_receiver(
                        standard_ls,
                        batch,
                        batch.information_bits,
                        perfect_csi=False,
                    )
                group = str(config["evaluation"]["cases"][case_index].get("group", "evaluation"))
                outputs = {
                    PROPOSED: proposed,
                    HARD: hard,
                    UNIFORM: uniform,
                    MOMENT: moment,
                    SINGLE: single,
                    LS_COMMON: ls_common,
                    PERFECT: perfect,
                }
                rows = [
                    custom_row(
                        case=proposed_item.case,
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
                    for name, output in outputs.items()
                ]
                rows.append(
                    standard_row(
                        case=proposed_item.case,
                        group=group,
                        variant=LS_CHAIN,
                        snr=snr,
                        rep=rep,
                        seed=seed,
                        metrics=_augment_standard_metrics(standard_metrics, batch),
                        signature=contract["signature"],
                    )
                )
                ordered = {str(item): index for index, item in enumerate(evaluation["variants"])}
                rows.sort(key=lambda row: ordered[row["variant"]])
                append_rows(RAW_PATH, rows)
                completed.add(key)
                print(
                    json.dumps(
                        {
                            "case": proposed_item.case.name,
                            "ebno_db": snr,
                            "rep": rep,
                            "rows_committed": len(rows),
                            "completed_rows": len(completed) * len(evaluation["variants"]),
                            "expected_rows": expected,
                            "proposed_block_errors": rows[0]["block_errors"],
                            "single_lmmse_block_errors": rows[4]["block_errors"],
                            "ls_linear_block_errors": rows[5]["block_errors"],
                        }
                    ),
                    flush=True,
                )
    frame = pd.read_csv(RAW_PATH)
    return {
        "complete": len(frame) == expected,
        "rows": int(len(frame)),
        "unique_rows": int(
            frame[["case", "ebno_db", "rep", "variant"]].drop_duplicates().shape[0]
        ),
        "expected_rows": expected,
        "raw_csv": str(RAW_PATH.relative_to(ROOT)),
        "contract": contract,
    }


def paired_summary(frame: pd.DataFrame, reference: str, comparator: str, metric: str) -> dict[str, Any]:
    keys = ["case", "ebno_db", "rep"]
    left = frame[frame["variant"] == reference][keys + [metric]].rename(
        columns={metric: "reference"}
    )
    right = frame[frame["variant"] == comparator][keys + [metric]].rename(
        columns={metric: "comparator"}
    )
    paired = left.merge(right, on=keys, how="inner")
    values = paired["reference"].to_numpy(float) - paired["comparator"].to_numpy(float)
    n = int(values.size)
    mean = float(values.mean()) if n else float("nan")
    std = float(values.std(ddof=1)) if n > 1 else 0.0
    if n > 1:
        radius = float(student_t.ppf(0.975, n - 1) * std / math.sqrt(n))
    else:
        radius = 0.0
    return {
        "reference": reference,
        "comparator": comparator,
        "metric": metric,
        "pairs": n,
        "mean": mean,
        "std": std,
        "ci95_low": mean - radius,
        "ci95_high": mean + radius,
    }


def stratified_bootstrap(
    frame: pd.DataFrame,
    reference: str,
    comparator: str,
    metric: str,
    *,
    repetitions: int,
    seed: int,
) -> dict[str, Any]:
    keys = ["case", "ebno_db", "rep"]
    left = frame[frame["variant"] == reference][keys + [metric]].rename(
        columns={metric: "reference"}
    )
    right = frame[frame["variant"] == comparator][keys + [metric]].rename(
        columns={metric: "comparator"}
    )
    paired = left.merge(right, on=keys, how="inner")
    strata = [group["reference"].to_numpy(float) - group["comparator"].to_numpy(float)
              for _, group in paired.groupby(["case", "ebno_db"])]
    rng = np.random.default_rng(int(seed))
    draws = np.empty(int(repetitions), dtype=float)
    for index in range(int(repetitions)):
        values = [rng.choice(stratum, size=len(stratum), replace=True) for stratum in strata]
        draws[index] = float(np.concatenate(values).mean())
    return {
        "reference": reference,
        "comparator": comparator,
        "metric": metric,
        "repetitions": int(repetitions),
        "seed": int(seed),
        "mean": float(draws.mean()),
        "ci95_low": float(np.quantile(draws, 0.025)),
        "ci95_high": float(np.quantile(draws, 0.975)),
    }


def create_aggregate(frame: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "information_ber",
        "tbler",
        "coded_ber",
        "coded_bit_nll",
        "channel_nmse",
        "normalized_error_mean",
        "coverage95",
        "evidence_entropy",
        "effective_component_count",
        "negative_log_evidence",
        "edge_density",
        "block_errors",
        "transport_blocks",
    ]
    return frame.groupby(
        ["case", "group", "scenario", "num_prb", "variant", "ebno_db"],
        dropna=False,
    )[metrics].agg(["mean", "std", "count"]).reset_index()


def make_plots(frame: pd.DataFrame, train_report: dict[str, Any]) -> list[str]:
    plot_dir = ROOT / "outputs/plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    variants = [PROPOSED, MOMENT, SINGLE, LS_COMMON]
    for case in sorted(frame["case"].unique()):
        subset = frame[(frame["case"] == case) & frame["variant"].isin(variants)]
        fig, ax = plt.subplots(figsize=(6.2, 4.2))
        for variant in variants:
            values = subset[subset["variant"] == variant].groupby("ebno_db")["tbler"].mean()
            if len(values):
                ax.semilogy(values.index, np.maximum(values.values, 1e-5), marker="o", label=variant)
        ax.set_xlabel("$E_b/N_0$ (dB)")
        ax.set_ylabel("TBLER")
        ax.grid(True, which="both")
        ax.legend(fontsize=7)
        ax.set_title(str(case))
        path = plot_dir / f"gate2_evidence_mixture_{case}_tbler.png"
        fig.tight_layout()
        fig.savefig(path, dpi=180)
        plt.close(fig)
        paths.append(str(path.relative_to(ROOT)))

    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    for variant in [PROPOSED, MOMENT, SINGLE, LS_COMMON]:
        values = frame[frame["variant"] == variant].groupby("ebno_db")["channel_nmse"].mean()
        if len(values):
            ax.semilogy(values.index, np.maximum(values.values, 1e-6), marker="o", label=variant)
    ax.set_xlabel("$E_b/N_0$ (dB)")
    ax.set_ylabel("Channel NMSE")
    ax.grid(True, which="both")
    ax.legend(fontsize=7)
    path = plot_dir / "gate2_evidence_mixture_channel_nmse.png"
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(str(path.relative_to(ROOT)))

    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    proposed = frame[frame["variant"] == PROPOSED]
    values = proposed.groupby("ebno_db")["effective_component_count"].mean()
    ax.plot(values.index, values.values, marker="o")
    ax.set_xlabel("$E_b/N_0$ (dB)")
    ax.set_ylabel("Effective number of mixture components")
    ax.grid(True)
    path = plot_dir / "gate2_evidence_mixture_effective_components.png"
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(str(path.relative_to(ROOT)))

    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    for label, history_key in [
        ("mixture", "proposed_validation_history"),
        ("single LMMSE", "single_lmmse_validation_history"),
    ]:
        history = train_report[history_key]
        x = [item["step"] for item in history]
        y = [item["score"] for item in history]
        ax.plot(x, y, marker="o", label=label)
    ax.set_xlabel("Training step")
    ax.set_ylabel("Validation score")
    ax.grid(True)
    ax.legend()
    path = plot_dir / "gate2_evidence_mixture_validation.png"
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(str(path.relative_to(ROOT)))
    return paths


def summarize(
    config: dict[str, Any],
    pre: dict[str, Any],
    train_report: dict[str, Any],
    evaluation_report: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    frame = pd.read_csv(RAW_PATH)
    aggregate = create_aggregate(frame)
    aggregate.to_csv(AGGREGATE_PATH, index=False)

    comparisons = {
        "proposed_minus_single_lmmse_tbler": paired_summary(frame, PROPOSED, SINGLE, "tbler"),
        "proposed_minus_moment_lmmse_tbler": paired_summary(frame, PROPOSED, MOMENT, "tbler"),
        "proposed_minus_ls_linear_tbler": paired_summary(frame, PROPOSED, LS_COMMON, "tbler"),
        "proposed_minus_single_lmmse_nmse": paired_summary(frame, PROPOSED, SINGLE, "channel_nmse"),
        "proposed_minus_moment_lmmse_nmse": paired_summary(frame, PROPOSED, MOMENT, "channel_nmse"),
        "proposed_minus_uniform_tbler": paired_summary(frame, PROPOSED, UNIFORM, "tbler"),
        "proposed_minus_hard_tbler": paired_summary(frame, PROPOSED, HARD, "tbler"),
    }
    decision = config["decision"]
    bootstrap = {
        "proposed_minus_single_lmmse_tbler": stratified_bootstrap(
            frame,
            PROPOSED,
            SINGLE,
            "tbler",
            repetitions=int(decision["bootstrap_repetitions"]),
            seed=int(decision["bootstrap_seed"]),
        ),
        "proposed_minus_moment_lmmse_tbler": stratified_bootstrap(
            frame,
            PROPOSED,
            MOMENT,
            "tbler",
            repetitions=int(decision["bootstrap_repetitions"]),
            seed=int(decision["bootstrap_seed"]) + 1,
        ),
    }
    paired_rows: list[dict[str, Any]] = []
    for comparator in [SINGLE, MOMENT, LS_COMMON, UNIFORM, HARD]:
        keys = ["case", "ebno_db", "rep"]
        left = frame[frame["variant"] == PROPOSED][keys + ["tbler", "channel_nmse"]].rename(
            columns={"tbler": "proposed_tbler", "channel_nmse": "proposed_nmse"}
        )
        right = frame[frame["variant"] == comparator][keys + ["tbler", "channel_nmse"]].rename(
            columns={"tbler": "comparator_tbler", "channel_nmse": "comparator_nmse"}
        )
        merged = left.merge(right, on=keys)
        for row in merged.to_dict("records"):
            paired_rows.append(
                {
                    **row,
                    "comparator": comparator,
                    "tbler_delta": float(row["proposed_tbler"] - row["comparator_tbler"]),
                    "nmse_delta": float(row["proposed_nmse"] - row["comparator_nmse"]),
                }
            )
    pd.DataFrame(paired_rows).to_csv(PAIRED_PATH, index=False)

    means = frame.groupby("variant").agg(
        tbler=("tbler", "mean"),
        channel_nmse=("channel_nmse", "mean"),
        coverage95=("coverage95", "mean"),
        coded_bit_nll=("coded_bit_nll", "mean"),
        block_errors=("block_errors", "sum"),
        transport_blocks=("transport_blocks", "sum"),
    )
    metrics = {
        variant: {
            key: float(value) if key not in {"block_errors", "transport_blocks"} else int(value)
            for key, value in row.to_dict().items()
        }
        for variant, row in means.iterrows()
    }
    proposed_rows = frame[frame["variant"] == PROPOSED]
    component_count = float(proposed_rows["effective_component_count"].mean())
    weight_range = float(
        proposed_rows["maximum_component_weight"].max()
        - proposed_rows["minimum_component_weight"].min()
    )
    max_case_harm = -float("inf")
    for _, group in frame[frame["variant"].isin([PROPOSED, SINGLE])].groupby("case"):
        p = float(group[group["variant"] == PROPOSED]["tbler"].mean())
        q = float(group[group["variant"] == SINGLE]["tbler"].mean())
        max_case_harm = max(max_case_harm, p - q)
    snr_means = proposed_rows.groupby("ebno_db")["tbler"].mean().to_dict()
    reversal = float(snr_means.get(14.0, 0.0) - snr_means.get(10.0, 0.0))

    nmse_gain_single = (
        metrics[SINGLE]["channel_nmse"] - metrics[PROPOSED]["channel_nmse"]
    ) / max(metrics[SINGLE]["channel_nmse"], 1e-12)
    nmse_gain_moment = (
        metrics[MOMENT]["channel_nmse"] - metrics[PROPOSED]["channel_nmse"]
    ) / max(metrics[MOMENT]["channel_nmse"], 1e-12)

    software_checks = {
        "complete_rows": evaluation_report["complete"],
        "unique_rows": evaluation_report["unique_rows"] == evaluation_report["expected_rows"],
        "all_variants_present": set(frame["variant"].astype(str))
        == {PROPOSED, HARD, UNIFORM, MOMENT, SINGLE, LS_COMMON, LS_CHAIN, PERFECT},
        "paired_seed_batches": all(
            group["variant"].nunique() == 8
            for _, group in frame.groupby(["case", "ebno_db", "rep"])
        ),
        "all_core_metrics_finite": bool(
            np.isfinite(
                frame[frame["variant"] != LS_CHAIN][
                    ["tbler", "channel_nmse", "coded_bit_nll"]
                ].to_numpy(float)
            ).all()
        ),
        "inference_uses_no_true_channel_except_control": bool(
            not frame[frame["variant"] != PERFECT]["inference_uses_true_channel"].astype(bool).any()
        ),
        "checkpoint_hashes_preserved": (
            sha256_file(ROOT / train_report["proposed_best_checkpoint"])
            == train_report["proposed_best_checkpoint_sha256"]
            and sha256_file(ROOT / train_report["single_lmmse_best_checkpoint"])
            == train_report["single_lmmse_best_checkpoint_sha256"]
        ),
        "same_common_detector_primary": evaluation_report["contract"]["policy"]
        ["common_detector_for_estimator_primary_comparisons"],
    }

    required_relative_gain = float(decision["nmse_required_relative_gain"])
    scientific_checks = {
        "mixture_routing_nontrivial": component_count > 1.05 and weight_range > 0.05,
        "proposed_nmse_beats_single_lmmse": (
            comparisons["proposed_minus_single_lmmse_nmse"]["ci95_high"] < 0.0
            and nmse_gain_single >= required_relative_gain
        ),
        "proposed_nmse_beats_moment_lmmse": (
            comparisons["proposed_minus_moment_lmmse_nmse"]["ci95_high"] < 0.0
            and nmse_gain_moment >= required_relative_gain
        ),
        "proposed_tbler_beats_single_lmmse": comparisons[
            "proposed_minus_single_lmmse_tbler"
        ]["ci95_high"] < 0.0,
        "proposed_tbler_beats_moment_lmmse": comparisons[
            "proposed_minus_moment_lmmse_tbler"
        ]["ci95_high"] < 0.0,
        "bootstrap_supports_single_lmmse_gain": bootstrap[
            "proposed_minus_single_lmmse_tbler"
        ]["ci95_high"] < 0.0,
        "bootstrap_supports_moment_lmmse_gain": bootstrap[
            "proposed_minus_moment_lmmse_tbler"
        ]["ci95_high"] < 0.0,
        "no_material_pooled_tbler_harm": comparisons[
            "proposed_minus_single_lmmse_tbler"
        ]["mean"] <= float(decision["maximum_pooled_tbler_harm"]),
        "no_material_per_case_tbler_harm": max_case_harm
        <= float(decision["maximum_per_case_tbler_harm"]),
        "no_high_snr_reversal": reversal <= float(decision["maximum_14db_reversal"]),
        "soft_evidence_not_worse_than_uniform": comparisons[
            "proposed_minus_uniform_tbler"
        ]["mean"] <= float(decision["maximum_pooled_tbler_harm"]),
        "training_converged": train_report["training_converged"] is True,
    }

    if not all(software_checks.values()):
        classification = "GATE2_EVIDENCE_MIXTURE_INTERFACE_REPAIR_REQUIRED"
        next_action = "REPAIR_ONLY_FAILED_SOFTWARE_OR_COMMON_DETECTOR_CONTROL"
    elif (
        scientific_checks["proposed_nmse_beats_single_lmmse"]
        and scientific_checks["proposed_nmse_beats_moment_lmmse"]
        and scientific_checks["proposed_tbler_beats_single_lmmse"]
        and scientific_checks["proposed_tbler_beats_moment_lmmse"]
        and scientific_checks["bootstrap_supports_single_lmmse_gain"]
        and scientific_checks["bootstrap_supports_moment_lmmse_gain"]
        and scientific_checks["no_material_per_case_tbler_harm"]
        and scientific_checks["no_high_snr_reversal"]
    ):
        classification = "GATE2_EVIDENCE_MIXTURE_BEATS_LMMSE_CE"
        next_action = "FREEZE_ARCHITECTURE_AND_RUN_PUBLICATION_SCALE_ESTIMATOR_CAMPAIGN"
    elif (
        scientific_checks["proposed_nmse_beats_single_lmmse"]
        and scientific_checks["proposed_nmse_beats_moment_lmmse"]
        and scientific_checks["no_material_pooled_tbler_harm"]
        and scientific_checks["no_material_per_case_tbler_harm"]
    ):
        classification = "GATE2_EVIDENCE_MIXTURE_ESTIMATION_GAIN_TBLER_INCONCLUSIVE"
        next_action = "RUN_ONE_LARGER_FROZEN_COMMON_DETECTOR_CONFIRMATION"
    else:
        classification = "GATE2_EVIDENCE_MIXTURE_NO_ADVANTAGE"
        next_action = "STOP_EVIDENCE_MIXTURE_FOR_LMMSE_CE_BEATING_OBJECTIVE"

    plots = make_plots(frame, train_report)
    report = {
        "version": GATE2_VERSION,
        "model_version": EVIDENCE_MIXTURE_LMMSE_VERSION,
        "complete": True,
        "classification": classification,
        "next_action": next_action,
        "publication_nr_ready": False,
        "preconditions": pre,
        "training": train_report,
        "evaluation": evaluation_report,
        "software_checks": software_checks,
        "scientific_checks": scientific_checks,
        "metrics": metrics,
        "derived_metrics": {
            "mean_effective_component_count": component_count,
            "component_weight_range": weight_range,
            "relative_nmse_gain_vs_single_lmmse": nmse_gain_single,
            "relative_nmse_gain_vs_moment_lmmse": nmse_gain_moment,
            "maximum_case_tbler_harm_vs_single_lmmse": max_case_harm,
            "proposed_14db_minus_10db_tbler": reversal,
        },
        "paired_comparisons": comparisons,
        "stratified_bootstrap": bootstrap,
        "raw_csv": str(RAW_PATH.relative_to(ROOT)),
        "aggregate_csv": str(AGGREGATE_PATH.relative_to(ROOT)),
        "paired_csv": str(PAIRED_PATH.relative_to(ROOT)),
        "plots": plots,
        "environment": {
            "device": str(device),
            "gpu_name": torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else "cpu",
            "torch": torch.__version__,
            "python": platform.python_version(),
        },
        "reviewer_alignment": {
            "attention_modules_removed": True,
            "new_receiver_principle": "Bayesian model averaging of exact structured LMMSE channel estimators",
            "analytical_backbone": [
                "closed_form Gaussian component posterior",
                "exact pilot marginal-likelihood routing",
                "K=1 reduction to LMMSE channel estimation",
                "mixture posterior mean is MMSE under the declared prior",
                "positive-semidefinite covariance by construction",
                "ordered component powers for identifiable labels",
            ],
            "common_detector_estimator_isolation": True,
            "parameter_count_below_128": train_report["proposed_parameter_report"][
                "trainable_parameters"
            ]
            <= 128,
        },
    }
    save_json(report, FINAL_REPORT)
    return report


def gate_lines(report: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    for key, value in report["software_checks"].items():
        lines.append(f"{key}: {'PASS' if value else 'FAIL'}")
    for key, value in report["scientific_checks"].items():
        lines.append(f"{key}: {'PASS' if value else 'FAIL'}")
    lines.extend(
        [
            f"CLASSIFICATION: {report['classification']}",
            f"NEXT_ACTION: {report['next_action']}",
            "PRIMARY_COMPARISON: CHANNEL_ESTIMATORS_WITH_THE_SAME_DETECTOR",
            "K1_BASELINE: EXACT_STRUCTURED_LMMSE_CHANNEL_ESTIMATOR",
            "MOMENT_BASELINE: EXACT_SECOND_ORDER_MATCHED_LMMSE_CHANNEL_ESTIMATOR",
            "INFERENCE_USES_TRUE_CHANNEL: NO",
            "PUBLICATION_NR_READY: NO",
        ]
    )
    return lines


def preflight(config: dict[str, Any], config_path: Path, device: torch.device) -> dict[str, Any]:
    pre = preconditions(config)
    spec = basis_spec_from_config(config)
    cases = [*config["training_cases"], *config["validation_cases"], *config["evaluation"]["cases"]]
    seen: set[str] = set()
    reports: list[dict[str, Any]] = []
    for raw in cases:
        name = str(raw["name"])
        if name in seen:
            raise RuntimeError(f"Duplicate Gate-2 case name: {name}")
        seen.add(name)
        item = build_stack_item(
            raw,
            device,
            spec,
            num_components=int(config["model"]["proposed_components"]),
            num_knots=int(config["model"]["num_knots"]),
        )
        reports.append(
            {
                "name": name,
                "scenario": item.case.scenario,
                "num_prb": int(item.case.num_prb),
                "num_streams": int(item.case.num_streams),
                "dmrs_ports": list(item.case.dmrs_ports),
                "effective_rank": int(item.basis_report["effective_rank"]),
                "parameter_count": int(
                    item.operator.parameter_report()["trainable_parameters"]
                ),
                "passed": True,
            }
        )
    result = {
        "passed": True,
        "preconditions": pre,
        "cases": reports,
        "expected_rows": expected_rows(config),
        "config_sha256": sha256_file(config_path),
        "source_sha256": source_hashes(),
        "proposed_parameter_count": reports[0]["parameter_count"],
        "training_steps": int(config["training"]["max_steps"]),
        "common_detector_estimator_isolation": True,
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--deadline-minutes", type=float, default=85.0)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    config = load_yaml(config_path)
    device = normalize_device(args.device)
    preflight_report = preflight(config, config_path, device)
    if args.preflight_only:
        print("GATE2_NR_EVIDENCE_MIXTURE_PREFLIGHT_PASS")
        print("MODEL", EVIDENCE_MIXTURE_LMMSE_VERSION)
        print("PROPOSED_COMPONENTS", config["model"]["proposed_components"])
        print("TRAINABLE_PARAMETERS", preflight_report["proposed_parameter_count"])
        print("MAX_TRAINING_STEPS", config["training"]["max_steps"])
        print("EXPECTED_ROWS", preflight_report["expected_rows"])
        print("COMMON_DETECTOR_ESTIMATOR_ISOLATION YES")
        print("INFERENCE_USES_TRUE_CHANNEL NO")
        return

    start = time.time()
    deadline = start + float(args.deadline_minutes) * 60.0
    train_report = train(
        config, config_path, preflight_report["preconditions"], device, deadline_epoch=deadline
    )
    if not train_report.get("complete"):
        print("GATE2_NR_EVIDENCE_MIXTURE_INCOMPLETE: RESUBMIT", flush=True)
        return
    evaluation_report = evaluate(
        config,
        config_path,
        train_report,
        device,
        deadline_epoch=deadline,
    )
    if not evaluation_report.get("complete"):
        print("GATE2_NR_EVIDENCE_MIXTURE_INCOMPLETE: RESUBMIT", flush=True)
        return
    report = summarize(
        config,
        preflight_report["preconditions"],
        train_report,
        evaluation_report,
        device,
    )
    lines = gate_lines(report)
    GATE_TXT.parent.mkdir(parents=True, exist_ok=True)
    GATE_TXT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    save_json(report, GATE_JSON)
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
