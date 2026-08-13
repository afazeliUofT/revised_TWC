#!/usr/bin/env python3
from __future__ import annotations

"""One predeclared continuation of the selected posterior-factorial receiver.

The 12-PRB case is not used for training, validation, learning-rate selection,
or checkpoint selection. It is evaluated only after the 4/8-PRB extension has
finished. This gate prevents indefinite training of the same pilot-only model.
"""

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

from bayesroute.config import capture_rng_state, restore_rng_state
from bayesroute.nr_gate1 import (
    NRCase,
    build_nr_context,
    decode_bridge,
    normalize_device,
    run_standard_receiver,
    standard_receiver,
)
from gate1_nr_joint_operator_common import (
    coded_metrics,
    differentiable_loss,
    make_repaired_detector,
    posterior_metrics,
    repaired_forward,
)
from gate1_nr_posterior_factorial_common import (
    FactorialCandidate,
    atomic_torch_save,
    bind_candidate_parameters,
    build_candidate_bridge,
    candidate_parameter_tensors,
    extract_candidate_state,
    load_candidate_state,
    ls_repaired_forward,
    package_signature,
    save_json,
    set_all_seeds,
    sha256_file,
)

VERSION = "gate1_nr_posterior_extension_v1"
REQUIRED_FACTORIAL_CLASSIFICATION = (
    "GATE1_POSTERIOR_FACTORIAL_TRAINING_EXTENSION_REQUIRED"
)
REQUIRED_WINNER = "physical_context_multiscale_r64"
REQUIRED_ORIGINAL_CHECKPOINT_SHA256 = (
    "7f6545afdcfb581b36b2a5f9cb4f8814981717056d018d0cd4a18f4e4ce7d9e6"
)
EXPECTED_ORIGINAL_ROWS = 792
EXPECTED_ROWS = 648
OUTPUT_ROOT = ROOT / "outputs/gate1_nr_posterior_extension"
RAW_PATH = ROOT / "outputs/eval/gate1_nr_posterior_extension.csv"
CONTRACT_PATH = ROOT / "outputs/eval/gate1_nr_posterior_extension_contract.json"
AGGREGATE_PATH = ROOT / "outputs/eval/gate1_nr_posterior_extension_aggregate.csv"
REPORT_PATH = ROOT / "outputs/reports/gate1_nr_posterior_extension.json"
TRAIN_REPORT_PATH = ROOT / "outputs/reports/gate1_nr_posterior_extension_train.json"
GATE_JSON = ROOT / "outputs/gates/GATE1_NR_POSTERIOR_EXTENSION.json"
GATE_TXT = ROOT / "outputs/gates/GATE1_NR_POSTERIOR_EXTENSION.txt"
LOG_PATH = ROOT / "outputs/logs/gate1_nr_posterior_extension_train.csv"

SOURCE_FILES = (
    "configs/gate1_nr_posterior_extension.yaml",
    "scripts/gate1_nr_posterior_extension.py",
    "scripts/gate1_nr_posterior_factorial_common.py",
    "scripts/gate1_nr_posterior_factorial_screen.py",
    "src/bayesroute/multiscale_posterior.py",
    "src/bayesroute/models.py",
    "src/bayesroute/lmmse_ep.py",
    "src/bayesroute/nr_gate1.py",
    "src/bayesroute/qam.py",
)


def load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected a YAML mapping: {path}")
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


def find_original_summary(report: dict[str, Any]) -> dict[str, Any]:
    for item in report.get("candidate_summaries", []):
        if item.get("candidate", {}).get("name") == REQUIRED_WINNER:
            return item
    raise RuntimeError(f"Original winner summary not found: {REQUIRED_WINNER}")


def preconditions(config: dict[str, Any]) -> dict[str, Any]:
    factorial_path = ROOT / "outputs/reports/gate1_nr_posterior_factorial.json"
    gate_path = ROOT / "outputs/gates/GATE1_NR_POSTERIOR_FACTORIAL.txt"
    revision_path = ROOT / "GATE1_NR_POSTERIOR_FACTORIAL_REVISION.json"
    for path in (factorial_path, gate_path, revision_path):
        if not path.is_file():
            raise RuntimeError(f"Missing posterior-extension precondition: {path}")

    report = json.loads(factorial_path.read_text(encoding="utf-8"))
    revision = json.loads(revision_path.read_text(encoding="utf-8"))
    original = find_original_summary(report)
    checkpoint = ROOT / str(config["winner_checkpoint_path"])
    checkpoint_sha = sha256_file(checkpoint) if checkpoint.is_file() else "missing"
    training = config["training"]
    training_prb = [int(item["num_prb"]) for item in config["training_cases"]]
    eval_holdouts = [
        item for item in config["evaluation"]["cases"]
        if int(item["num_prb"]) == 12
    ]
    total_steps = int(training["total_extension_steps"])
    updates_per_grid = int(training["extension_updates_per_grid"])

    checks = {
        "factorial_complete": report.get("complete") is True,
        "factorial_rows": (
            report.get("evaluation", {}).get("rows") == EXPECTED_ORIGINAL_ROWS
            and report.get("evaluation", {}).get("unique_rows")
            == EXPECTED_ORIGINAL_ROWS
        ),
        "factorial_classification": (
            report.get("classification") == REQUIRED_FACTORIAL_CLASSIFICATION
        ),
        "factorial_winner": report.get("winner") == REQUIRED_WINNER,
        "factorial_next_action": (
            report.get("next_action")
            == "EXTEND_ONLY_THE_WINNING_CANDIDATE_WITH_FROZEN_HOLDOUT"
        ),
        "original_training_boundary": (
            original.get("best_step") == 1599
            and original.get("training_converged") is False
        ),
        "checkpoint_present": checkpoint.is_file(),
        "checkpoint_hash": checkpoint_sha == REQUIRED_ORIGINAL_CHECKPOINT_SHA256,
        "config_checkpoint_hash": (
            config.get("winner_checkpoint_sha256")
            == REQUIRED_ORIGINAL_CHECKPOINT_SHA256
        ),
        "winner_spec": config.get("winner_name") == REQUIRED_WINNER,
        "revision": config.get("revision") == VERSION,
        "factorial_revision": (
            revision.get("revision") == "gate1_nr_posterior_factorial_v1"
            and revision.get("ls_alignment_patch")
            == "gate1_nr_posterior_factorial_ls_alignment_v1"
        ),
        "equal_4prb_8prb_extension": (
            training_prb == [4, 8]
            and total_steps == 2 * updates_per_grid
            and updates_per_grid == 800
        ),
        "fresh_12prb_holdout": (
            len(eval_holdouts) == 1
            and str(eval_holdouts[0].get("group"))
            == "fresh_untouched_grid_holdout"
            and all(int(item["num_prb"]) in {4, 8} for item in config["training_cases"])
        ),
        "expected_rows": expected_rows(config) == EXPECTED_ROWS,
    }
    if not all(checks.values()):
        raise RuntimeError(f"Posterior-extension precondition failure: {checks}")
    return {
        "passed": True,
        "checks": checks,
        "factorial_report": str(factorial_path.relative_to(ROOT)),
        "factorial_classification": report["classification"],
        "original_summary": original,
        "original_checkpoint_sha256": checkpoint_sha,
    }


def candidate_spec(config: dict[str, Any]) -> FactorialCandidate:
    spec = FactorialCandidate.from_mapping(config["winner_spec"])
    if spec.name != REQUIRED_WINNER:
        raise RuntimeError("Posterior-extension winner specification mismatch")
    return spec


def build_bridges(
    spec: FactorialCandidate,
    cases: Sequence[NRCase],
    contexts: Sequence[Any],
    config: dict[str, Any],
) -> tuple[list[Any], list[Any]]:
    bridges = [
        build_candidate_bridge(
            case, context, spec, operator_seed=int(config["operator_seed"])
        )
        for case, context in zip(cases, contexts)
    ]
    bind_candidate_parameters(spec, bridges)
    detectors = [
        make_repaired_detector(int(context.grid.bits_per_symbol)).to(context.device)
        for context in contexts
    ]
    return bridges, detectors


def validation_score(
    bridges: Sequence[Any],
    detectors: Sequence[Any],
    contexts: Sequence[Any],
    cases: Sequence[NRCase],
    config: dict[str, Any],
) -> dict[str, Any]:
    training = config["training"]
    records: list[dict[str, Any]] = []
    with torch.no_grad():
        for case_index, (bridge, detector, context, case) in enumerate(
            zip(bridges, detectors, contexts, cases)
        ):
            for snr_index, raw_snr in enumerate(training["validation_ebno_db"]):
                snr = float(raw_snr)
                seed = (
                    int(training["validation_seed"])
                    + 100_000 * case_index
                    + 1_000 * snr_index
                )
                set_all_seeds(seed)
                batch = context.sample(int(training["validation_batch_size"]), snr)
                output = repaired_forward(bridge, detector, batch)
                records.append(
                    {
                        "case": case.name,
                        "num_prb": int(case.num_prb),
                        "ebno_db": snr,
                        **coded_metrics(output, batch),
                        **posterior_metrics(output, batch),
                    }
                )
    nll = np.asarray([float(item["coded_bit_nll"]) for item in records])
    nmse = np.asarray([float(item["channel_nmse"]) for item in records])
    normalized = np.asarray(
        [float(item["normalized_error_mean"]) for item in records]
    )
    calibration = np.abs(np.log(np.clip(normalized, 1e-6, None)))
    score = (
        float(np.mean(nll))
        + 0.25 * float(np.max(nll))
        + 0.10 * float(np.mean(nmse))
        + 0.05 * float(np.mean(calibration))
    )
    return {
        "score": score,
        "mean_coded_bit_nll": float(np.mean(nll)),
        "worst_coded_bit_nll": float(np.max(nll)),
        "mean_channel_nmse": float(np.mean(nmse)),
        "calibration_abs_log": float(np.mean(calibration)),
        "records": records,
    }


def extension_contract(
    config: dict[str, Any], config_path: Path, pre: dict[str, Any]
) -> dict[str, Any]:
    payload = {
        "version": VERSION,
        "config_sha256": sha256_file(config_path),
        "source_sha256": source_hashes(),
        "winner": REQUIRED_WINNER,
        "winner_spec": config["winner_spec"],
        "original_checkpoint_sha256": pre["original_checkpoint_sha256"],
        "training": config["training"],
        "training_cases": config["training_cases"],
        "holdout_policy": {
            "training_prb": [4, 8],
            "validation_prb": [4, 8],
            "selection_uses_12prb": False,
            "fresh_12prb_evaluated_only_after_training_complete": True,
        },
    }
    payload["signature"] = package_signature(payload)
    return payload


def cosine_lr(step: int, total: int, start: float, end: float) -> float:
    if total <= 1:
        return float(end)
    progress = min(max(float(step) / float(total - 1), 0.0), 1.0)
    return float(end + 0.5 * (start - end) * (1.0 + math.cos(math.pi * progress)))


def train_extension(
    config: dict[str, Any],
    config_path: Path,
    pre: dict[str, Any],
    cases: Sequence[NRCase],
    contexts: Sequence[Any],
    *,
    deadline_epoch: float,
) -> dict[str, Any]:
    spec = candidate_spec(config)
    training = config["training"]
    total_steps = int(training["total_extension_steps"])
    checkpoint_dir = OUTPUT_ROOT / "checkpoints"
    best_path = checkpoint_dir / "best.pt"
    last_path = checkpoint_dir / "last.pt"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    TRAIN_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    contract = extension_contract(config, config_path, pre)

    if TRAIN_REPORT_PATH.is_file():
        prior = json.loads(TRAIN_REPORT_PATH.read_text(encoding="utf-8"))
        if prior.get("contract", {}).get("signature") != contract["signature"]:
            raise RuntimeError("Posterior-extension training summary contract mismatch")
        if prior.get("complete") is True:
            return prior

    bridges, detectors = build_bridges(spec, cases, contexts, config)
    parameters = candidate_parameter_tensors(bridges)
    optimizer = torch.optim.AdamW(
        parameters,
        lr=float(training["learning_rate_start"]),
        weight_decay=float(training["weight_decay"]),
    )
    start_step = 0
    best_score = float("inf")
    baseline_validation: dict[str, Any] | None = None
    validation_history: list[dict[str, Any]] = []

    if last_path.is_file():
        state = torch.load(last_path, map_location=contexts[0].device, weights_only=False)
        if state.get("contract") != contract:
            raise RuntimeError("Posterior-extension checkpoint contract mismatch")
        load_candidate_state(spec, bridges[0], state["operator"])
        optimizer.load_state_dict(state["optimizer"])
        start_step = int(state["extension_step"]) + 1
        best_score = float(state["best_score"])
        baseline_validation = state["baseline_validation"]
        validation_history = list(state.get("validation_history", []))
        restore_rng_state(state.get("rng_state"))
        print(f"Resuming posterior extension at step {start_step}", flush=True)
    else:
        original_path = ROOT / str(config["winner_checkpoint_path"])
        original = torch.load(
            original_path, map_location=contexts[0].device, weights_only=False
        )
        if original.get("candidate", {}).get("name") != REQUIRED_WINNER:
            raise RuntimeError("Original winner checkpoint identity mismatch")
        load_candidate_state(spec, bridges[0], original["operator"])
        optimizer.load_state_dict(original["optimizer"])
        for group in optimizer.param_groups:
            group["lr"] = float(training["learning_rate_start"])
        baseline_validation = validation_score(
            bridges, detectors, contexts, cases, config
        )
        best_score = float(baseline_validation["score"])
        validation_history = [
            {
                "extension_step": -1,
                "global_step": 1599,
                "score": best_score,
                "source": "original_winner_before_extension",
            }
        ]
        initial_state = {
            "version": VERSION,
            "winner": REQUIRED_WINNER,
            "operator": extract_candidate_state(spec, bridges[0]),
            "optimizer": optimizer.state_dict(),
            "extension_step": -1,
            "global_step": 1599,
            "best_score": best_score,
            "baseline_validation": baseline_validation,
            "validation": baseline_validation,
            "validation_history": validation_history,
            "contract": contract,
            "rng_state": capture_rng_state(),
        }
        atomic_torch_save(initial_state, best_path)
        atomic_torch_save(initial_state, last_path)

    fields = [
        "extension_step",
        "global_step",
        "case",
        "num_prb",
        "ebno_db",
        "learning_rate",
        "loss",
        "coded_bit_nll",
        "channel_nll",
        "calibration_penalty",
        "gradient_norm",
        "validation_score",
        "contract_signature",
    ]
    if not LOG_PATH.is_file():
        with LOG_PATH.open("w", newline="", encoding="utf-8") as stream:
            csv.DictWriter(stream, fieldnames=fields).writeheader()
    elif start_step > 0:
        frame = pd.read_csv(LOG_PATH)
        frame = frame[frame["extension_step"] < start_step]
        frame.to_csv(LOG_PATH, index=False)

    last_validation: dict[str, Any] | None = None
    for extension_step in range(start_step, total_steps):
        case_index = extension_step % len(cases)
        case = cases[case_index]
        context = contexts[case_index]
        bridge = bridges[case_index]
        detector = detectors[case_index]
        seed = int(config["seed"]) + 2_000_000 + extension_step
        set_all_seeds(seed)
        fraction = (
            (extension_step * 2654435761 + case_index * 7919) % 1_000_003
        ) / 1_000_003.0
        snr = float(training["ebno_db_min"]) + fraction * (
            float(training["ebno_db_max"]) - float(training["ebno_db_min"])
        )
        learning_rate = cosine_lr(
            extension_step,
            total_steps,
            float(training["learning_rate_start"]),
            float(training["learning_rate_end"]),
        )
        for group in optimizer.param_groups:
            group["lr"] = learning_rate

        batch = context.sample(int(training["batch_size"]), snr)
        output = repaired_forward(bridge, detector, batch)
        loss, parts = differentiable_loss(
            output,
            batch,
            channel_loss_weight=float(spec.channel_loss_weight),
            calibration_loss_weight=float(spec.calibration_loss_weight),
        )
        if not torch.isfinite(loss):
            raise RuntimeError(
                f"Non-finite posterior-extension loss at step {extension_step}"
            )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(
            parameters, float(training["grad_clip"])
        )
        if not torch.isfinite(grad):
            raise RuntimeError(
                f"Non-finite posterior-extension gradient at step {extension_step}"
            )
        optimizer.step()

        validation_due = (
            extension_step % int(training["validation_every"]) == 0
            or extension_step == total_steps - 1
        )
        validation_value = float("nan")
        if validation_due:
            last_validation = validation_score(
                bridges, detectors, contexts, cases, config
            )
            validation_value = float(last_validation["score"])
            validation_history.append(
                {
                    "extension_step": int(extension_step),
                    "global_step": int(1600 + extension_step),
                    "score": validation_value,
                    "source": "extension_validation_4prb_8prb_only",
                }
            )
            if validation_value < best_score:
                best_score = validation_value
                atomic_torch_save(
                    {
                        "version": VERSION,
                        "winner": REQUIRED_WINNER,
                        "operator": extract_candidate_state(spec, bridges[0]),
                        "optimizer": optimizer.state_dict(),
                        "extension_step": extension_step,
                        "global_step": 1600 + extension_step,
                        "best_score": best_score,
                        "baseline_validation": baseline_validation,
                        "validation": last_validation,
                        "validation_history": validation_history,
                        "contract": contract,
                        "rng_state": capture_rng_state(),
                    },
                    best_path,
                )

        save_due = (
            extension_step % int(training["save_every"]) == 0
            or extension_step == total_steps - 1
        )
        if save_due:
            row = {
                "extension_step": extension_step,
                "global_step": 1600 + extension_step,
                "case": case.name,
                "num_prb": int(case.num_prb),
                "ebno_db": snr,
                "learning_rate": learning_rate,
                "loss": float(loss.detach().item()),
                "coded_bit_nll": float(parts["bit_nll"].detach().item()),
                "channel_nll": float(parts["channel_nll"].detach().item()),
                "calibration_penalty": float(
                    parts["calibration_penalty"].detach().item()
                ),
                "gradient_norm": float(grad.detach().item()),
                "validation_score": validation_value,
                "contract_signature": contract["signature"],
            }
            with LOG_PATH.open("a", newline="", encoding="utf-8") as stream:
                csv.DictWriter(stream, fieldnames=fields).writerow(row)
            print(json.dumps(row), flush=True)
            atomic_torch_save(
                {
                    "version": VERSION,
                    "winner": REQUIRED_WINNER,
                    "operator": extract_candidate_state(spec, bridges[0]),
                    "optimizer": optimizer.state_dict(),
                    "extension_step": extension_step,
                    "global_step": 1600 + extension_step,
                    "best_score": best_score,
                    "baseline_validation": baseline_validation,
                    "validation": last_validation,
                    "validation_history": validation_history,
                    "contract": contract,
                    "rng_state": capture_rng_state(),
                },
                last_path,
            )
            if time.time() >= deadline_epoch:
                return {
                    "complete": False,
                    "version": VERSION,
                    "extension_step": extension_step,
                    "total_extension_steps": total_steps,
                    "best_score": best_score,
                    "contract": contract,
                    "stop_reason": "internal_deadline",
                }

    best = torch.load(best_path, map_location=contexts[0].device, weights_only=False)
    if best.get("contract") != contract:
        raise RuntimeError("Posterior-extension best checkpoint contract mismatch")
    history = list(best.get("validation_history", validation_history))
    scored = [item for item in history if int(item["extension_step"]) >= 0]
    if len(scored) >= 3:
        tail_start = float(scored[-3]["score"])
        tail_end = float(scored[-1]["score"])
        tail_improvement = tail_start - tail_end
        tail_tolerance = max(0.001, 0.003 * abs(tail_start))
    else:
        tail_improvement = float("inf")
        tail_tolerance = 0.0
    best_extension_step = int(best["extension_step"])
    baseline_score = float(baseline_validation["score"])
    validation_gain = baseline_score - float(best["best_score"])
    training_converged = bool(
        best_extension_step <= total_steps - 2 * int(training["validation_every"])
        or tail_improvement <= tail_tolerance
    )
    summary = {
        "complete": True,
        "version": VERSION,
        "winner": REQUIRED_WINNER,
        "winner_spec": spec.as_dict(),
        "original_checkpoint": config["winner_checkpoint_path"],
        "original_checkpoint_sha256": pre["original_checkpoint_sha256"],
        "extension_steps": total_steps,
        "extension_updates_per_training_grid": total_steps // len(cases),
        "best_extension_step": best_extension_step,
        "best_global_step": int(best["global_step"]),
        "best_score": float(best["best_score"]),
        "baseline_score": baseline_score,
        "validation_score_improvement": validation_gain,
        "baseline_validation": baseline_validation,
        "best_validation": best["validation"],
        "validation_history": history,
        "tail_validation_improvement": float(tail_improvement),
        "tail_validation_tolerance": float(tail_tolerance),
        "training_converged": training_converged,
        "best_checkpoint": str(best_path.relative_to(ROOT)),
        "best_checkpoint_sha256": sha256_file(best_path),
        "last_checkpoint": str(last_path.relative_to(ROOT)),
        "contract": contract,
        "fresh_12prb_used_for_training_or_selection": False,
    }
    save_json(summary, TRAIN_REPORT_PATH)
    del bridges, detectors, optimizer
    if contexts[0].device.type == "cuda":
        torch.cuda.empty_cache()
    return summary


def load_operator(path: Path, expected_contract: dict[str, Any] | None, device: torch.device) -> dict[str, Any]:
    state = torch.load(path, map_location=device, weights_only=False)
    if expected_contract is not None and state.get("contract") != expected_contract:
        raise RuntimeError(f"Checkpoint contract mismatch: {path}")
    return state["operator"]


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


def append_rows_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    new = pd.DataFrame(rows)
    if path.is_file():
        old = pd.read_csv(path)
        if list(old.columns) != list(new.columns):
            raise RuntimeError("Posterior-extension CSV column contract mismatch")
        combined = pd.concat([old, new], ignore_index=True)
    else:
        combined = new
    temporary = path.with_suffix(path.suffix + ".tmp")
    combined.to_csv(temporary, index=False)
    temporary.replace(path)


def crc_disagreement(decoded: dict[str, Any], bits: torch.Tensor) -> float:
    block_error = (decoded["b_hat"] != bits).reshape(
        bits.shape[0], bits.shape[1], -1
    ).any(-1)
    crc_failure = ~decoded["crc"].bool()
    if crc_failure.shape != block_error.shape:
        crc_failure = crc_failure.reshape(block_error.shape)
    return float((block_error != crc_failure).float().mean().item())


def custom_row(
    *, case: NRCase, group: str, variant: str, snr: float, rep: int, seed: int,
    output: dict[str, Any], batch: Any, decoded: dict[str, Any], signature: str,
) -> dict[str, Any]:
    return {
        "case": case.name,
        "group": group,
        "scenario": case.scenario,
        "num_prb": int(case.num_prb),
        "num_streams": int(case.num_streams),
        "num_rx_ant": int(case.num_rx_ant),
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
        "contract_signature": signature,
    }


def standard_row(
    *, case: NRCase, group: str, variant: str, snr: float, rep: int, seed: int,
    metrics: dict[str, Any], signature: str,
) -> dict[str, Any]:
    return {
        "case": case.name,
        "group": group,
        "scenario": case.scenario,
        "num_prb": int(case.num_prb),
        "num_streams": int(case.num_streams),
        "num_rx_ant": int(case.num_rx_ant),
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
        "contract_signature": signature,
    }


def evaluation_contract(
    config: dict[str, Any], config_path: Path, pre: dict[str, Any], train: dict[str, Any]
) -> dict[str, Any]:
    payload = {
        "version": VERSION,
        "config_sha256": sha256_file(config_path),
        "source_sha256": source_hashes(),
        "original_checkpoint_sha256": pre["original_checkpoint_sha256"],
        "extended_checkpoint_sha256": train["best_checkpoint_sha256"],
        "evaluation": config["evaluation"],
        "fresh_12prb_holdout_policy": {
            "used_in_training": False,
            "used_in_validation": False,
            "evaluated_after_training_complete": True,
        },
    }
    payload["signature"] = package_signature(payload)
    return payload


def evaluate(
    config: dict[str, Any], config_path: Path, pre: dict[str, Any],
    train: dict[str, Any], device: torch.device, *, deadline_epoch: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    evaluation = config["evaluation"]
    variants = [str(x) for x in evaluation["variants"]]
    if len(variants) != 6 or expected_rows(config) != EXPECTED_ROWS:
        raise RuntimeError("Posterior-extension evaluation design mismatch")
    contract = evaluation_contract(config, config_path, pre, train)
    if RAW_PATH.is_file():
        if not CONTRACT_PATH.is_file():
            raise RuntimeError("Posterior-extension CSV exists without contract")
        existing = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
        if existing.get("signature") != contract["signature"]:
            raise RuntimeError("Posterior-extension evaluation contract mismatch")
    else:
        save_json(contract, CONTRACT_PATH)

    done: set[tuple[str, float, int]] = set()
    if RAW_PATH.is_file():
        frame = pd.read_csv(RAW_PATH)
        keys = ["case", "variant", "ebno_db", "rep"]
        if frame[keys].duplicated().any():
            raise RuntimeError("Posterior-extension CSV contains duplicate keys")
        counts = frame.groupby(["case", "ebno_db", "rep"])["variant"].nunique()
        if len(counts[counts != len(variants)]):
            raise RuntimeError("Posterior-extension CSV contains partial paired batch")
        done = {(str(a), float(b), int(c)) for a, b, c in counts.index}

    spec = candidate_spec(config)
    original_path = ROOT / str(config["winner_checkpoint_path"])
    extended_path = ROOT / str(train["best_checkpoint"])
    original_operator = load_operator(original_path, None, device)
    extended_operator = load_operator(extended_path, train["contract"], device)
    groups = {str(item["name"]): str(item["group"]) for item in evaluation["cases"]}

    for case_index, raw_case in enumerate(evaluation["cases"]):
        case = NRCase.from_mapping(raw_case)
        context = build_nr_context(case, device)
        original_bridge = build_candidate_bridge(
            case, context, spec, operator_seed=int(config["operator_seed"])
        )
        extended_bridge = build_candidate_bridge(
            case, context, spec, operator_seed=int(config["operator_seed"])
        )
        load_candidate_state(spec, original_bridge, original_operator)
        load_candidate_state(spec, extended_bridge, extended_operator)
        original_bridge.eval()
        extended_bridge.eval()
        original_detector = make_repaired_detector(
            int(context.grid.bits_per_symbol)
        ).to(device)
        extended_detector = make_repaired_detector(
            int(context.grid.bits_per_symbol)
        ).to(device)
        ls_receiver = standard_receiver(context, perfect_csi=False, return_crc=True)
        perfect_receiver = standard_receiver(context, perfect_csi=True, return_crc=True)
        ls_repaired_detector = make_repaired_detector(
            int(context.grid.bits_per_symbol)
        ).to(device)

        for snr_index, raw_snr in enumerate(evaluation["ebno_db"]):
            snr = float(raw_snr)
            for rep in range(int(evaluation["repetitions"])):
                key = (case.name, snr, rep)
                if key in done:
                    continue
                seed = (
                    int(config["seed"]) + 10_000_000 + case_index * 100_000
                    + snr_index * 1_000 + rep
                )
                set_all_seeds(seed)
                batch = context.sample(int(evaluation["batch_size"]), snr)
                with torch.inference_mode():
                    extended = repaired_forward(
                        extended_bridge, extended_detector, batch
                    )
                    original = repaired_forward(
                        original_bridge, original_detector, batch
                    )
                    posterior = extended["posterior"]
                    graph = extended["reference_graph_mask"]
                    uncertainty_raw = extended_detector(
                        batch.y,
                        posterior.mean,
                        posterior.local_cov,
                        batch.data_idx,
                        batch.noise_var,
                        graph,
                        covariance_mode="none",
                    )
                    uncertainty_off = dict(uncertainty_raw)
                    uncertainty_off["posterior"] = posterior
                    uncertainty_off["reference_graph_mask"] = graph
                    uncertainty_off["graph_mask"] = graph
                    uncertainty_off["edge_density"] = uncertainty_raw["edge_density"]
                    ls_repaired = ls_repaired_forward(
                        ls_receiver, context, ls_repaired_detector, batch
                    )
                    custom_outputs = {
                        "extended_winner": extended,
                        "original_winner": original,
                        "extended_uncertainty_off": uncertainty_off,
                        "ls_estimate_repaired": ls_repaired,
                    }
                    decoded = decode_outputs(
                        context,
                        batch,
                        custom_outputs,
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
                group = groups[case.name]
                rows = [
                    custom_row(
                        case=case,
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
                            case=case,
                            group=group,
                            variant="ls_lmmse",
                            snr=snr,
                            rep=rep,
                            seed=seed,
                            metrics=ls_metrics,
                            signature=contract["signature"],
                        ),
                        standard_row(
                            case=case,
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
                if {item["variant"] for item in rows} != set(variants):
                    raise RuntimeError("Posterior-extension paired variant mismatch")
                append_rows_atomic(RAW_PATH, rows)
                done.add(key)
                print(
                    json.dumps(
                        {
                            "case": case.name,
                            "num_prb": case.num_prb,
                            "ebno_db": snr,
                            "rep": rep,
                            "rows_committed": len(rows),
                            "completed_rows": len(done) * len(variants),
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
                            len(frame.drop_duplicates(["case", "variant", "ebno_db", "rep"]))
                        ),
                        "expected_rows": EXPECTED_ROWS,
                        "stop_reason": "internal_deadline",
                        "contract": contract,
                    }
        del (
            context,
            original_bridge,
            extended_bridge,
            original_detector,
            extended_detector,
            ls_receiver,
            perfect_receiver,
            ls_repaired_detector,
        )
        if device.type == "cuda":
            torch.cuda.empty_cache()

    frame = pd.read_csv(RAW_PATH)
    unique = int(len(frame.drop_duplicates(["case", "variant", "ebno_db", "rep"])))
    if len(frame) != EXPECTED_ROWS or unique != EXPECTED_ROWS:
        raise RuntimeError(
            f"Posterior-extension evaluation incomplete: rows={len(frame)}, unique={unique}"
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
    frame: pd.DataFrame, reference: str, comparator: str, metric: str, *, prb: int,
) -> dict[str, float]:
    sub = frame[frame["num_prb"] == int(prb)]
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
    frame: pd.DataFrame, variant: str, metric: str, *, prb: int, snr: float | None = None,
) -> float:
    sub = frame[(frame["variant"] == variant) & (frame["num_prb"] == int(prb))]
    if snr is not None:
        sub = sub[sub["ebno_db"] == float(snr)]
    values = pd.to_numeric(sub[metric], errors="coerce").dropna()
    return float(values.mean()) if len(values) else float("nan")


def aggregate(frame: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "information_ber", "tbler", "crc_failure_rate", "coded_ber",
        "coded_bit_nll", "coded_brier", "channel_nmse",
        "normalized_error_mean", "coverage95", "edge_density",
    ]
    return frame.groupby(
        ["case", "group", "num_prb", "variant", "ebno_db"], dropna=False
    )[metrics].agg(["mean", "std", "count"]).reset_index()


def make_plots(frame: pd.DataFrame, train: dict[str, Any]) -> list[str]:
    output_dir = ROOT / "outputs/plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    for prb in (4, 8, 12):
        plt.figure(figsize=(7.0, 4.7))
        for variant in (
            "extended_winner", "original_winner", "ls_lmmse", "perfect_csi_lmmse"
        ):
            sub = frame[(frame["num_prb"] == prb) & (frame["variant"] == variant)]
            grouped = sub.groupby("ebno_db")["tbler"].mean().sort_index()
            plt.plot(grouped.index, grouped.values, marker="o", label=variant)
        plt.xlabel("$E_b/N_0$ (dB)")
        plt.ylabel("TBLER")
        plt.ylim(-0.02, 1.02)
        plt.grid(True, which="both", alpha=0.3)
        plt.legend(fontsize=7)
        plt.tight_layout()
        path = output_dir / f"gate1_posterior_extension_{prb}prb_tbler.png"
        plt.savefig(path, dpi=180)
        plt.close()
        paths.append(str(path.relative_to(ROOT)))

    history = train.get("validation_history", [])
    if history:
        plt.figure(figsize=(7.0, 4.5))
        x = [int(item["global_step"]) for item in history]
        y = [float(item["score"]) for item in history]
        plt.plot(x, y, marker="o")
        plt.xlabel("Global training step")
        plt.ylabel("Fixed 4/8-PRB validation score")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        path = output_dir / "gate1_posterior_extension_validation.png"
        plt.savefig(path, dpi=180)
        plt.close()
        paths.append(str(path.relative_to(ROOT)))
    return paths


def classify(frame: pd.DataFrame, train: dict[str, Any]) -> dict[str, Any]:
    comparisons = {
        "extended_minus_original_12prb_tbler": paired_delta(
            frame, "extended_winner", "original_winner", "tbler", prb=12
        ),
        "extended_minus_ls_12prb_tbler": paired_delta(
            frame, "extended_winner", "ls_lmmse", "tbler", prb=12
        ),
        "extended_minus_ls_repaired_12prb_tbler": paired_delta(
            frame, "extended_winner", "ls_estimate_repaired", "tbler", prb=12
        ),
        "extended_minus_uncertainty_off_12prb_tbler": paired_delta(
            frame, "extended_winner", "extended_uncertainty_off", "tbler", prb=12
        ),
        "ls_repaired_minus_standard_ls_12prb_tbler": paired_delta(
            frame, "ls_estimate_repaired", "ls_lmmse", "tbler", prb=12
        ),
    }
    metrics = {
        "extended_4prb_tbler": mean_metric(frame, "extended_winner", "tbler", prb=4),
        "extended_8prb_tbler": mean_metric(frame, "extended_winner", "tbler", prb=8),
        "extended_12prb_tbler": mean_metric(frame, "extended_winner", "tbler", prb=12),
        "original_12prb_tbler": mean_metric(frame, "original_winner", "tbler", prb=12),
        "ls_12prb_tbler": mean_metric(frame, "ls_lmmse", "tbler", prb=12),
        "ls_repaired_12prb_tbler": mean_metric(frame, "ls_estimate_repaired", "tbler", prb=12),
        "perfect_12prb_tbler": mean_metric(frame, "perfect_csi_lmmse", "tbler", prb=12),
        "extended_12prb_10db_tbler": mean_metric(
            frame, "extended_winner", "tbler", prb=12, snr=10.0
        ),
        "extended_12prb_14db_tbler": mean_metric(
            frame, "extended_winner", "tbler", prb=12, snr=14.0
        ),
        "extended_12prb_coverage95": mean_metric(
            frame, "extended_winner", "coverage95", prb=12
        ),
        "extended_12prb_normalized_error": mean_metric(
            frame, "extended_winner", "normalized_error_mean", prb=12
        ),
        "extended_12prb_channel_nmse": mean_metric(
            frame, "extended_winner", "channel_nmse", prb=12
        ),
    }
    ls_gap = comparisons["extended_minus_ls_12prb_tbler"]
    ls_factorized_gap = comparisons[
        "ls_repaired_minus_standard_ls_12prb_tbler"
    ]
    checks = {
        "complete_rows": len(frame) == EXPECTED_ROWS,
        "all_metrics_finite": bool(
            np.isfinite(
                frame[frame["variant"].isin(
                    ["extended_winner", "original_winner", "extended_uncertainty_off"]
                )][
                    ["tbler", "coded_bit_nll", "channel_nmse", "normalized_error_mean", "coverage95"]
                ].to_numpy(dtype=float)
            ).all()
        ),
        "extension_validation_improves": (
            float(train["validation_score_improvement"]) > 0.0
        ),
        "extension_training_converged": bool(train["training_converged"]),
        "extension_improves_original_12prb": (
            comparisons["extended_minus_original_12prb_tbler"]["ci95_high"] < 0.0
        ),
        "uncertainty_helps_extended": (
            comparisons["extended_minus_uncertainty_off_12prb_tbler"]["ci95_high"] < 0.0
        ),
        "extended_beats_ls": ls_gap["ci95_high"] < 0.0,
        "extended_within_0p02_of_ls": ls_gap["mean"] <= 0.02,
        "ls_factorized_detector_close_to_standard": abs(ls_factorized_gap["mean"]) <= 0.02,
        "no_12prb_high_snr_reversal": (
            metrics["extended_12prb_14db_tbler"]
            <= metrics["extended_12prb_10db_tbler"] + 0.01
        ),
        "12prb_coverage_reasonable": (
            0.90 <= metrics["extended_12prb_coverage95"] <= 0.99
        ),
        "12prb_normalized_error_reasonable": (
            0.70 <= metrics["extended_12prb_normalized_error"] <= 1.40
        ),
    }
    if not checks["ls_factorized_detector_close_to_standard"]:
        classification = "GATE1_DETECTOR_ESTIMATOR_INTERFACE_REPAIR_REQUIRED"
        next_action = "REPAIR_LS_POSTERIOR_TO_DETECTOR_INTERFACE"
    elif (
        checks["extended_beats_ls"]
        and checks["no_12prb_high_snr_reversal"]
        and checks["12prb_coverage_reasonable"]
    ):
        classification = "GATE1_POSTERIOR_EXTENSION_BEATS_LS"
        next_action = "FREEZE_ARCHITECTURE_AND_RUN_PUBLICATION_SCALE_BASELINES"
    elif (
        checks["extended_within_0p02_of_ls"]
        and checks["no_12prb_high_snr_reversal"]
    ):
        classification = "GATE1_POSTERIOR_EXTENSION_NEAR_LS"
        next_action = "ADD_ONE_PRINCIPLED_TURBO_POSTERIOR_UPDATE"
    else:
        classification = "GATE1_TURBO_REFINEMENT_REQUIRED"
        next_action = "ADD_ONE_PRINCIPLED_DATA_AIDED_POSTERIOR_UPDATE"
    return {
        "classification": classification,
        "next_action": next_action,
        "scientific_checks": checks,
        "paired_comparisons": comparisons,
        "metrics": metrics,
    }


def write_incomplete(train: dict[str, Any] | None, evaluation: dict[str, Any] | None) -> None:
    report = {
        "version": VERSION,
        "complete": False,
        "training": train,
        "evaluation": evaluation,
        "classification": "GATE1_NR_POSTERIOR_EXTENSION_INCOMPLETE",
        "next_action": "RESUBMIT_SAME_COMMAND",
        "publication_nr_ready": False,
    }
    save_json(report, REPORT_PATH)
    save_json(report, GATE_JSON)
    GATE_TXT.parent.mkdir(parents=True, exist_ok=True)
    GATE_TXT.write_text(
        "CLASSIFICATION: GATE1_NR_POSTERIOR_EXTENSION_INCOMPLETE\n"
        "NEXT_ACTION: RESUBMIT_SAME_COMMAND\n"
        "PUBLICATION_NR_READY: NO\n",
        encoding="utf-8",
    )
    print("GATE1_NR_POSTERIOR_EXTENSION_INCOMPLETE: RESUBMIT")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="configs/gate1_nr_posterior_extension.yaml"
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--deadline-minutes", type=float, default=52.0)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()

    config_path = ROOT / args.config
    config = load_yaml(config_path)
    if expected_rows(config) != EXPECTED_ROWS:
        raise RuntimeError(
            f"Expected {EXPECTED_ROWS} rows, computed {expected_rows(config)}"
        )
    pre = preconditions(config)
    if args.preflight_only:
        print("GATE1_NR_POSTERIOR_EXTENSION_PREFLIGHT_PASS")
        print("WINNER", REQUIRED_WINNER)
        print("ORIGINAL_CHECKPOINT", pre["original_checkpoint_sha256"])
        print("EXTENSION_STEPS", config["training"]["total_extension_steps"])
        print("UPDATES_PER_TRAINING_GRID", config["training"]["extension_updates_per_grid"])
        print("TRAINING_GRIDS", [4, 8])
        print("FRESH_HOLDOUT_GRID", 12)
        print("EXPECTED_ROWS", EXPECTED_ROWS)
        print("HOLDOUT_USED_FOR_SELECTION NO")
        return

    device = normalize_device(args.device)
    deadline_epoch = time.time() + 60.0 * float(args.deadline_minutes)
    cases = [NRCase.from_mapping(item) for item in config["training_cases"]]
    contexts = [build_nr_context(case, device) for case in cases]
    train = train_extension(
        config,
        config_path,
        pre,
        cases,
        contexts,
        deadline_epoch=deadline_epoch,
    )
    if not train.get("complete", False):
        write_incomplete(train, None)
        return
    if time.time() >= deadline_epoch:
        write_incomplete(train, None)
        return

    frame, evaluation = evaluate(
        config,
        config_path,
        pre,
        train,
        device,
        deadline_epoch=deadline_epoch,
    )
    if not evaluation.get("complete", False):
        write_incomplete(train, evaluation)
        return

    aggregate_frame = aggregate(frame)
    AGGREGATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    aggregate_frame.to_csv(AGGREGATE_PATH, index=False)
    scientific = classify(frame, train)
    plots = make_plots(frame, train)
    report = {
        "version": VERSION,
        "complete": True,
        "classification": scientific["classification"],
        "next_action": scientific["next_action"],
        "publication_nr_ready": False,
        "timestamp_epoch": time.time(),
        "environment": {
            "device": str(device),
            "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "torch": torch.__version__,
            "python": platform.python_version(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        },
        "preconditions": pre,
        "training": train,
        "evaluation": evaluation,
        "aggregate_csv": str(AGGREGATE_PATH.relative_to(ROOT)),
        "plots": plots,
        **scientific,
    }
    save_json(report, REPORT_PATH)
    save_json(report, GATE_JSON)
    checks = report["scientific_checks"]
    lines = [
        "training_complete: PASS",
        "equal_4prb_8prb_extension_exposure: PASS",
        "fresh_12prb_holdout_frozen_until_training_complete: PASS",
        "complete_rows: PASS",
        *(f"{name}: {'PASS' if value else 'FAIL'}" for name, value in checks.items()),
        f"CLASSIFICATION: {report['classification']}",
        f"NEXT_ACTION: {report['next_action']}",
        "PUBLICATION_NR_READY: NO",
    ]
    GATE_TXT.parent.mkdir(parents=True, exist_ok=True)
    GATE_TXT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
