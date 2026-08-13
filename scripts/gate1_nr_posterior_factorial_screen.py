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
    posterior_graph,
    posterior_metrics,
    repaired_forward,
)
from gate1_nr_posterior_factorial_common import (
    POSTERIOR_FACTORIAL_VERSION,
    FactorialCandidate,
    atomic_torch_save,
    bind_candidate_parameters,
    build_candidate_bridge,
    candidate_parameter_tensors,
    extract_candidate_state,
    load_candidate_state,
    ls_repaired_forward,
    model_report,
    package_signature,
    save_json,
    set_all_seeds,
    sha256_file,
)

EXPECTED_ROWS = 792
REQUIRED_SMOKE_CLASSIFICATION = "GATE1_NR_POSTERIOR_FACTORIAL_SMOKE_PASS"
REQUIRED_GRID_CLASSIFICATION = "GRID_SCALE_COORDINATE_HYPOTHESIS_NOT_SUPPORTED"
REQUIRED_CHECKPOINT_SHA256 = (
    "4f71c7a0a925005d676687e90c5a241668cfcfed21503e2874c3528721c66980"
)
OUTPUT_ROOT = ROOT / "outputs/gate1_nr_posterior_factorial"
RAW_PATH = ROOT / "outputs/eval/gate1_nr_posterior_factorial.csv"
CONTRACT_PATH = ROOT / "outputs/eval/gate1_nr_posterior_factorial_contract.json"
AGGREGATE_PATH = ROOT / "outputs/eval/gate1_nr_posterior_factorial_aggregate.csv"
REPORT_PATH = ROOT / "outputs/reports/gate1_nr_posterior_factorial.json"
GATE_JSON = ROOT / "outputs/gates/GATE1_NR_POSTERIOR_FACTORIAL.json"
GATE_TXT = ROOT / "outputs/gates/GATE1_NR_POSTERIOR_FACTORIAL.txt"

SOURCE_FILES = (
    "configs/gate1_nr_posterior_factorial_screen.yaml",
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
    num_candidate_variants = len(config["candidates"])
    num_controls = 5
    return (
        len(evaluation["cases"])
        * len(evaluation["ebno_db"])
        * int(evaluation["repetitions"])
        * (num_candidate_variants + num_controls)
    )


def preconditions(config: dict[str, Any]) -> dict[str, Any]:
    smoke_path = ROOT / "outputs/gates/GATE1_NR_POSTERIOR_FACTORIAL_SMOKE.json"
    grid_path = ROOT / "outputs/reports/gate1_nr_grid_scale_audit.json"
    checkpoint = ROOT / str(config["frozen_checkpoint_path"])
    revision_path = ROOT / "GATE1_NR_POSTERIOR_FACTORIAL_REVISION.json"
    for path in (smoke_path, grid_path, checkpoint, revision_path):
        if not path.is_file():
            raise RuntimeError(f"Missing posterior-factorial precondition: {path}")
    smoke = json.loads(smoke_path.read_text(encoding="utf-8"))
    grid = json.loads(grid_path.read_text(encoding="utf-8"))
    revision = json.loads(revision_path.read_text(encoding="utf-8"))
    candidates = [FactorialCandidate.from_mapping(item) for item in config["candidates"]]
    num_training_grids = len(config["training_cases"])
    updates_per_grid = int(config["training"]["updates_per_training_grid"])
    expected_steps = num_training_grids * updates_per_grid
    schedule = {item.name: item.steps for item in candidates}
    checks = {
        "smoke": bool(
            smoke.get("classification") == REQUIRED_SMOKE_CLASSIFICATION
            and smoke.get("overall_pass") is True
            and smoke.get("screen_ready") is True
        ),
        "grid_audit": bool(
            grid.get("complete") is True
            and grid.get("classification") == REQUIRED_GRID_CLASSIFICATION
            and grid.get("evaluation", {}).get("rows") == 360
        ),
        "checkpoint": bool(
            sha256_file(checkpoint) == REQUIRED_CHECKPOINT_SHA256
            and config.get("frozen_checkpoint_sha256") == REQUIRED_CHECKPOINT_SHA256
        ),
        "revision": bool(
            config.get("revision") == POSTERIOR_FACTORIAL_VERSION
            and revision.get("revision") == POSTERIOR_FACTORIAL_VERSION
        ),
        "candidate_factorial": bool(
            len(candidates) == 6
            and {item.coordinate_mode for item in candidates}
            == {"allocation_normalized", "reference_physical"}
            and {item.model_type for item in candidates} == {"single", "multiscale"}
            and any(item.context_conditioned for item in candidates)
        ),
        "equal_grid_exposure": bool(
            updates_per_grid > 0
            and all(item.steps == expected_steps for item in candidates)
        ),
        "expected_rows": expected_rows(config) == EXPECTED_ROWS,
        "untouched_12prb_holdout": bool(
            all(int(item["num_prb"]) in {4, 8} for item in config["training_cases"])
            and any(
                int(item["num_prb"]) == 12
                and str(item.get("group")) == "untouched_grid_holdout"
                for item in config["evaluation"]["cases"]
            )
        ),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "schedule": schedule,
        "updates_per_grid": updates_per_grid,
        "expected_steps": expected_steps,
        "checkpoint_sha256": sha256_file(checkpoint),
    }


def candidate_contract(
    spec: FactorialCandidate,
    config: dict[str, Any],
    config_path: Path,
    pre: dict[str, Any],
) -> dict[str, Any]:
    payload = {
        "version": POSTERIOR_FACTORIAL_VERSION,
        "candidate": spec.as_dict(),
        "config_sha256": sha256_file(config_path),
        "source_sha256": source_hashes(),
        "selected_detector": config["selected_detector"],
        "training": config["training"],
        "training_cases": config["training_cases"],
        "grid_audit_classification": REQUIRED_GRID_CLASSIFICATION,
        "frozen_checkpoint_sha256": pre["checkpoint_sha256"],
    }
    payload["signature"] = package_signature(payload)
    return payload


def candidate_bridges(
    spec: FactorialCandidate,
    cases: Sequence[NRCase],
    contexts: Sequence[Any],
    *,
    operator_seed: int,
) -> tuple[list[Any], list[Any]]:
    bridges = [
        build_candidate_bridge(
            case, context, spec, operator_seed=int(operator_seed)
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
    *,
    seed_offset: int = 0,
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    training = config["training"]
    with torch.no_grad():
        for case_index, (bridge, detector, context, case) in enumerate(
            zip(bridges, detectors, contexts, cases)
        ):
            for snr_index, raw_snr in enumerate(training["validation_ebno_db"]):
                snr = float(raw_snr)
                seed = (
                    int(training["validation_seed"])
                    + int(seed_offset)
                    + 100_000 * case_index
                    + 1_000 * snr_index
                )
                set_all_seeds(seed)
                batch = context.sample(
                    int(training["validation_batch_size"]), snr
                )
                output = repaired_forward(bridge, detector, batch)
                coded = coded_metrics(output, batch)
                posterior = posterior_metrics(output, batch)
                records.append(
                    {
                        "case": case.name,
                        "num_prb": int(case.num_prb),
                        "ebno_db": snr,
                        **coded,
                        **posterior,
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


def train_candidate(
    spec: FactorialCandidate,
    config: dict[str, Any],
    config_path: Path,
    pre: dict[str, Any],
    cases: Sequence[NRCase],
    contexts: Sequence[Any],
    *,
    deadline_epoch: float,
) -> dict[str, Any]:
    checkpoint_dir = OUTPUT_ROOT / "checkpoints" / spec.name
    report_path = ROOT / "outputs/reports" / f"gate1_nr_posterior_factorial_{spec.name}.json"
    log_path = ROOT / "outputs/logs" / f"gate1_nr_posterior_factorial_{spec.name}.csv"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    best_path = checkpoint_dir / "best.pt"
    last_path = checkpoint_dir / "last.pt"
    contract = candidate_contract(spec, config, config_path, pre)

    if report_path.is_file():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report.get("contract", {}).get("signature") != contract["signature"]:
            raise RuntimeError(f"Candidate summary contract mismatch: {spec.name}")
        if report.get("complete") is True:
            return report

    bridges, detectors = candidate_bridges(
        spec,
        cases,
        contexts,
        operator_seed=int(config["operator_seed"]),
    )
    parameters = candidate_parameter_tensors(bridges)
    optimizer = torch.optim.AdamW(
        parameters,
        lr=float(spec.learning_rate),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    start_step = 0
    best_score = float("inf")
    if last_path.is_file():
        state = torch.load(last_path, map_location=contexts[0].device, weights_only=False)
        if state.get("contract") != contract:
            raise RuntimeError(f"Candidate checkpoint contract mismatch: {spec.name}")
        load_candidate_state(spec, bridges[0], state["operator"])
        optimizer.load_state_dict(state["optimizer"])
        start_step = int(state["step"]) + 1
        best_score = float(state["best_score"])
        restore_rng_state(state.get("rng_state"))
        print(f"Resuming {spec.name} at step {start_step}", flush=True)

    fields = [
        "candidate",
        "step",
        "case",
        "num_prb",
        "ebno_db",
        "loss",
        "coded_bit_nll",
        "channel_nll",
        "calibration_penalty",
        "gradient_norm",
        "validation_score",
        "contract_signature",
    ]
    if not log_path.is_file():
        with log_path.open("w", newline="", encoding="utf-8") as stream:
            csv.DictWriter(stream, fieldnames=fields).writeheader()
    elif start_step > 0:
        frame = pd.read_csv(log_path)
        frame = frame[frame["step"] < start_step]
        frame.to_csv(log_path, index=False)

    last_validation: dict[str, Any] | None = None
    training = config["training"]
    for step in range(start_step, spec.steps):
        case_index = step % len(cases)
        case = cases[case_index]
        context = contexts[case_index]
        bridge = bridges[case_index]
        detector = detectors[case_index]
        seed = int(config["seed"]) + 1_000_000 + step
        set_all_seeds(seed)
        fraction = ((step * 2654435761 + case_index * 7919) % 1_000_003) / 1_000_003.0
        snr = float(training["ebno_db_min"]) + fraction * (
            float(training["ebno_db_max"]) - float(training["ebno_db_min"])
        )
        batch = context.sample(int(training["batch_size"]), snr)
        output = repaired_forward(bridge, detector, batch)
        loss, parts = differentiable_loss(
            output,
            batch,
            channel_loss_weight=float(spec.channel_loss_weight),
            calibration_loss_weight=float(spec.calibration_loss_weight),
        )
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss for {spec.name} at step {step}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(
            parameters, float(training["grad_clip"])
        )
        if not torch.isfinite(grad):
            raise RuntimeError(f"Non-finite gradient for {spec.name} at step {step}")
        optimizer.step()

        validation_due = (
            step % int(training["validation_every"]) == 0
            or step == spec.steps - 1
        )
        validation_value = float("nan")
        if validation_due:
            last_validation = validation_score(
                bridges, detectors, contexts, cases, config
            )
            validation_value = float(last_validation["score"])
            if validation_value < best_score:
                best_score = validation_value
                atomic_torch_save(
                    {
                        "version": POSTERIOR_FACTORIAL_VERSION,
                        "candidate": spec.as_dict(),
                        "operator": extract_candidate_state(spec, bridges[0]),
                        "optimizer": optimizer.state_dict(),
                        "step": step,
                        "best_score": best_score,
                        "validation": last_validation,
                        "contract": contract,
                        "rng_state": capture_rng_state(),
                    },
                    best_path,
                )

        save_due = step % int(training["save_every"]) == 0 or step == spec.steps - 1
        if save_due:
            row = {
                "candidate": spec.name,
                "step": step,
                "case": case.name,
                "num_prb": int(case.num_prb),
                "ebno_db": snr,
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
            with log_path.open("a", newline="", encoding="utf-8") as stream:
                csv.DictWriter(stream, fieldnames=fields).writerow(row)
            print(json.dumps(row), flush=True)
            atomic_torch_save(
                {
                    "version": POSTERIOR_FACTORIAL_VERSION,
                    "candidate": spec.as_dict(),
                    "operator": extract_candidate_state(spec, bridges[0]),
                    "optimizer": optimizer.state_dict(),
                    "step": step,
                    "best_score": best_score,
                    "validation": last_validation,
                    "contract": contract,
                    "rng_state": capture_rng_state(),
                },
                last_path,
            )
            if time.time() >= deadline_epoch:
                return {
                    "complete": False,
                    "candidate": spec.name,
                    "step": step,
                    "steps": spec.steps,
                    "best_score": best_score,
                    "contract": contract,
                    "stop_reason": "internal_deadline",
                }

    if not best_path.is_file():
        raise RuntimeError(f"Candidate {spec.name} has no best checkpoint")
    best = torch.load(best_path, map_location=contexts[0].device, weights_only=False)
    if best.get("contract") != contract:
        raise RuntimeError(f"Best checkpoint contract mismatch: {spec.name}")
    log_frame = pd.read_csv(log_path)
    validation_frame = log_frame[
        pd.to_numeric(log_frame["validation_score"], errors="coerce").notna()
    ].copy()
    validation_history = [
        {"step": int(row.step), "score": float(row.validation_score)}
        for row in validation_frame.itertuples(index=False)
    ]
    if len(validation_history) >= 3:
        tail_start = float(validation_history[-3]["score"])
        tail_end = float(validation_history[-1]["score"])
        tail_improvement = tail_start - tail_end
        tail_tolerance = max(0.002, 0.005 * abs(tail_start))
    else:
        tail_improvement = float("inf")
        tail_tolerance = 0.0
    best_step = int(best["step"])
    validation_every = int(config["training"]["validation_every"])
    training_converged = bool(
        best_step <= spec.steps - 2 * validation_every
        or tail_improvement <= tail_tolerance
    )
    summary = {
        "complete": True,
        "version": POSTERIOR_FACTORIAL_VERSION,
        "candidate": spec.as_dict(),
        "steps": spec.steps,
        "updates_per_training_grid": spec.steps // len(cases),
        "best_step": best_step,
        "best_score": float(best["best_score"]),
        "validation": best["validation"],
        "validation_history": validation_history,
        "tail_validation_improvement": float(tail_improvement),
        "tail_validation_tolerance": float(tail_tolerance),
        "training_converged": training_converged,
        "best_checkpoint": str(best_path.relative_to(ROOT)),
        "best_checkpoint_sha256": sha256_file(best_path),
        "model": model_report(spec, bridges[0]),
        "contract": contract,
    }
    save_json(summary, report_path)
    del bridges, detectors, optimizer
    if contexts[0].device.type == "cuda":
        torch.cuda.empty_cache()
    return summary


def load_best_state(summary: dict[str, Any], device: torch.device) -> dict[str, Any]:
    state = torch.load(
        ROOT / str(summary["best_checkpoint"]),
        map_location=device,
        weights_only=False,
    )
    if state.get("contract") != summary["contract"]:
        raise RuntimeError("Best candidate state contract mismatch")
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
            raise RuntimeError("Posterior-factorial CSV column contract mismatch")
        combined = pd.concat([old, new], ignore_index=True)
    else:
        combined = new
    temporary = path.with_suffix(path.suffix + ".tmp")
    combined.to_csv(temporary, index=False)
    temporary.replace(path)


def crc_disagreement(decoded: dict[str, Any], bits: torch.Tensor) -> float:
    block_error = (decoded["b_hat"] != bits).reshape(bits.shape[0], bits.shape[1], -1).any(-1)
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
        "crc_block_disagreement_rate": crc_disagreement(decoded, batch.information_bits),
        **coded_metrics(output, batch),
        **posterior_metrics(output, batch),
        "edge_density": float(output["edge_density"].item()),
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
        "contract_signature": contract_signature,
    }


def evaluation_contract(
    config: dict[str, Any],
    config_path: Path,
    pre: dict[str, Any],
    summaries: Sequence[dict[str, Any]],
    winner: str,
) -> dict[str, Any]:
    payload = {
        "version": POSTERIOR_FACTORIAL_VERSION,
        "config_sha256": sha256_file(config_path),
        "source_sha256": source_hashes(),
        "winner": winner,
        "candidate_checkpoints": {
            item["candidate"]["name"]: item["best_checkpoint_sha256"]
            for item in summaries
        },
        "frozen_checkpoint_sha256": pre["checkpoint_sha256"],
        "evaluation": config["evaluation"],
    }
    payload["signature"] = package_signature(payload)
    return payload


def evaluate(
    config: dict[str, Any],
    config_path: Path,
    pre: dict[str, Any],
    summaries: Sequence[dict[str, Any]],
    winner: str,
    device: torch.device,
    *,
    deadline_epoch: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    candidates = [FactorialCandidate.from_mapping(item) for item in config["candidates"]]
    summary_by_name = {item["candidate"]["name"]: item for item in summaries}
    contract = evaluation_contract(config, config_path, pre, summaries, winner)
    variants = [item.name for item in candidates] + [
        "frozen_old_global",
        "winner_uncertainty_off",
        "ls_estimate_repaired",
        "ls_lmmse",
        "perfect_csi_lmmse",
    ]
    RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
    if RAW_PATH.is_file():
        if not CONTRACT_PATH.is_file():
            raise RuntimeError("Evaluation CSV exists without contract")
        existing_contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
        if existing_contract.get("signature") != contract["signature"]:
            raise RuntimeError("Posterior-factorial evaluation contract mismatch")
    else:
        save_json(contract, CONTRACT_PATH)

    done: set[tuple[str, float, int]] = set()
    if RAW_PATH.is_file():
        frame = pd.read_csv(RAW_PATH)
        keys = ["case", "variant", "ebno_db", "rep"]
        if frame[keys].duplicated().any():
            raise RuntimeError("Posterior-factorial CSV contains duplicate keys")
        counts = frame.groupby(["case", "ebno_db", "rep"])["variant"].nunique()
        partial = counts[counts != len(variants)]
        if len(partial):
            raise RuntimeError("Posterior-factorial CSV contains partial paired batch")
        done = {(str(a), float(b), int(c)) for a, b, c in counts.index}

    old_checkpoint = torch.load(
        ROOT / str(config["frozen_checkpoint_path"]),
        map_location=device,
        weights_only=False,
    )
    old_operators = old_checkpoint.get("operators")
    if not isinstance(old_operators, list) or len(old_operators) != 1:
        raise RuntimeError("Frozen old checkpoint must contain one global operator")
    old_state = {
        "model_type": "single",
        "raw_weights": old_operators[0]["raw_weights"],
        "log_noise_scale": old_operators[0]["log_noise_scale"],
    }
    old_spec = next(item for item in candidates if item.name == "alloc_single_r24")
    winner_spec = next(item for item in candidates if item.name == winner)
    evaluation = config["evaluation"]
    group_by_name = {str(item["name"]): str(item["group"]) for item in evaluation["cases"]}

    for case_index, raw_case in enumerate(evaluation["cases"]):
        case = NRCase.from_mapping(raw_case)
        context = build_nr_context(case, device)
        bridges: dict[str, Any] = {}
        detectors: dict[str, Any] = {}
        for spec in candidates:
            bridge = build_candidate_bridge(
                case, context, spec, operator_seed=int(config["operator_seed"])
            )
            load_candidate_state(
                spec,
                bridge,
                load_best_state(summary_by_name[spec.name], device),
            )
            bridge.eval()
            bridges[spec.name] = bridge
            detectors[spec.name] = make_repaired_detector(
                int(context.grid.bits_per_symbol)
            ).to(device)
        old_bridge = build_candidate_bridge(
            case, context, old_spec, operator_seed=int(config["operator_seed"])
        )
        load_candidate_state(old_spec, old_bridge, old_state)
        old_bridge.eval()
        old_detector = make_repaired_detector(int(context.grid.bits_per_symbol)).to(device)
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
                seed = int(config["seed"]) + 10_000_000 + case_index * 100_000 + snr_index * 1_000 + rep
                set_all_seeds(seed)
                batch = context.sample(int(evaluation["batch_size"]), snr)
                with torch.inference_mode():
                    custom_outputs = {
                        spec.name: repaired_forward(
                            bridges[spec.name], detectors[spec.name], batch
                        )
                        for spec in candidates
                    }
                    custom_outputs["frozen_old_global"] = repaired_forward(
                        old_bridge, old_detector, batch
                    )
                    winner_output = custom_outputs[winner]
                    posterior = winner_output["posterior"]
                    reference_graph = winner_output["reference_graph_mask"]
                    uncertainty_off_raw = detectors[winner](
                        batch.y,
                        posterior.mean,
                        posterior.local_cov,
                        batch.data_idx,
                        batch.noise_var,
                        reference_graph,
                        covariance_mode="none",
                    )
                    uncertainty_off = dict(uncertainty_off_raw)
                    uncertainty_off["posterior"] = posterior
                    uncertainty_off["reference_graph_mask"] = reference_graph
                    uncertainty_off["graph_mask"] = reference_graph
                    uncertainty_off["edge_density"] = uncertainty_off_raw["edge_density"]
                    custom_outputs["winner_uncertainty_off"] = uncertainty_off
                    custom_outputs["ls_estimate_repaired"] = ls_repaired_forward(
                        ls_receiver,
                        context,
                        ls_repaired_detector,
                        batch,
                    )
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
                group = group_by_name[case.name]
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
                        contract_signature=contract["signature"],
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
                            contract_signature=contract["signature"],
                        ),
                        standard_row(
                            case=case,
                            group=group,
                            variant="perfect_csi_lmmse",
                            snr=snr,
                            rep=rep,
                            seed=seed,
                            metrics=perfect_metrics,
                            contract_signature=contract["signature"],
                        ),
                    ]
                )
                if {row["variant"] for row in rows} != set(variants):
                    raise RuntimeError("Posterior-factorial paired variant set mismatch")
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
        del bridges, detectors, old_bridge, old_detector, ls_receiver, perfect_receiver
        if device.type == "cuda":
            torch.cuda.empty_cache()

    frame = pd.read_csv(RAW_PATH)
    unique = int(len(frame.drop_duplicates(["case", "variant", "ebno_db", "rep"])))
    if len(frame) != EXPECTED_ROWS or unique != EXPECTED_ROWS:
        raise RuntimeError(
            f"Posterior-factorial evaluation incomplete: rows={len(frame)}, unique={unique}"
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
) -> dict[str, float]:
    sub = frame if prb is None else frame[frame["num_prb"] == int(prb)]
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


def aggregate(frame: pd.DataFrame) -> pd.DataFrame:
    metrics = [
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
    return frame.groupby(
        ["case", "group", "num_prb", "variant", "ebno_db"], dropna=False
    )[metrics].agg(["mean", "std", "count"]).reset_index()


def mean_metric(
    frame: pd.DataFrame,
    variant: str,
    metric: str,
    *,
    prb: int | None = None,
    snr: float | None = None,
) -> float:
    sub = frame[frame["variant"] == variant]
    if prb is not None:
        sub = sub[sub["num_prb"] == int(prb)]
    if snr is not None:
        sub = sub[sub["ebno_db"] == float(snr)]
    values = pd.to_numeric(sub[metric], errors="coerce").dropna()
    return float(values.mean()) if len(values) else float("nan")


def make_plots(frame: pd.DataFrame, candidates: Sequence[FactorialCandidate], winner: str) -> list[str]:
    output_dir = ROOT / "outputs/plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    display = [winner, "frozen_old_global", "ls_lmmse", "perfect_csi_lmmse"]
    for prb in (4, 8, 12):
        plt.figure(figsize=(7.0, 4.7))
        for variant in display:
            sub = frame[(frame["num_prb"] == prb) & (frame["variant"] == variant)]
            grouped = sub.groupby("ebno_db")["tbler"].mean().sort_index()
            plt.plot(grouped.index, grouped.values, marker="o", label=variant)
        plt.xlabel("$E_b/N_0$ (dB)")
        plt.ylabel("TBLER")
        plt.ylim(-0.02, 1.02)
        plt.grid(True, which="both", alpha=0.3)
        plt.legend(fontsize=7)
        plt.tight_layout()
        path = output_dir / f"gate1_posterior_factorial_{prb}prb_tbler.png"
        plt.savefig(path, dpi=180)
        plt.close()
        paths.append(str(path.relative_to(ROOT)))

    plt.figure(figsize=(8.0, 4.8))
    names = [item.name for item in candidates]
    values = [mean_metric(frame, name, "tbler", prb=12) for name in names]
    plt.bar(np.arange(len(names)), values)
    plt.xticks(np.arange(len(names)), names, rotation=35, ha="right", fontsize=7)
    plt.ylabel("12-PRB mean TBLER")
    plt.tight_layout()
    path = output_dir / "gate1_posterior_factorial_candidate_12prb.png"
    plt.savefig(path, dpi=180)
    plt.close()
    paths.append(str(path.relative_to(ROOT)))
    return paths


def classify(
    frame: pd.DataFrame,
    candidates: Sequence[FactorialCandidate],
    summaries: Sequence[dict[str, Any]],
    winner: str,
) -> dict[str, Any]:
    comparisons = {
        "winner_minus_ls_12prb_tbler": paired_delta(
            frame, winner, "ls_lmmse", "tbler", prb=12
        ),
        "winner_minus_old_12prb_tbler": paired_delta(
            frame, winner, "frozen_old_global", "tbler", prb=12
        ),
        "winner_minus_uncertainty_off_12prb_tbler": paired_delta(
            frame, winner, "winner_uncertainty_off", "tbler", prb=12
        ),
        "ls_repaired_minus_standard_ls_12prb_tbler": paired_delta(
            frame, "ls_estimate_repaired", "ls_lmmse", "tbler", prb=12
        ),
        "winner_minus_ls_repaired_12prb_tbler": paired_delta(
            frame, winner, "ls_estimate_repaired", "tbler", prb=12
        ),
    }
    per_candidate = {
        item.name: {
            "12prb_tbler": mean_metric(frame, item.name, "tbler", prb=12),
            "12prb_channel_nmse": mean_metric(
                frame, item.name, "channel_nmse", prb=12
            ),
            "12prb_coverage95": mean_metric(
                frame, item.name, "coverage95", prb=12
            ),
            "12prb_normalized_error": mean_metric(
                frame, item.name, "normalized_error_mean", prb=12
            ),
            "10db_tbler": mean_metric(frame, item.name, "tbler", prb=12, snr=10.0),
            "14db_tbler": mean_metric(frame, item.name, "tbler", prb=12, snr=14.0),
        }
        for item in candidates
    }
    best_single = min(
        (item for item in candidates if item.model_type == "single"),
        key=lambda item: per_candidate[item.name]["12prb_tbler"],
    ).name
    best_multiscale = min(
        (item for item in candidates if item.model_type == "multiscale"),
        key=lambda item: per_candidate[item.name]["12prb_tbler"],
    ).name
    best_allocation = min(
        (item for item in candidates if item.coordinate_mode == "allocation_normalized"),
        key=lambda item: per_candidate[item.name]["12prb_tbler"],
    ).name
    best_physical = min(
        (item for item in candidates if item.coordinate_mode == "reference_physical"),
        key=lambda item: per_candidate[item.name]["12prb_tbler"],
    ).name
    winner_metrics = per_candidate[winner]
    ls_gap = comparisons["winner_minus_ls_12prb_tbler"]
    winner_summary = next(
        item for item in summaries if item["candidate"]["name"] == winner
    )
    ls_factorized_gap = comparisons[
        "ls_repaired_minus_standard_ls_12prb_tbler"
    ]
    checks = {
        "complete_rows": len(frame) == EXPECTED_ROWS,
        "winner_training_converged": bool(
            winner_summary.get("training_converged", False)
        ),
        "all_metrics_finite": bool(
            np.isfinite(
                frame[frame["variant"].isin([item.name for item in candidates])][
                    ["tbler", "coded_bit_nll", "channel_nmse", "normalized_error_mean", "coverage95"]
                ].to_numpy(dtype=float)
            ).all()
        ),
        "winner_improves_frozen_old": bool(
            comparisons["winner_minus_old_12prb_tbler"]["ci95_high"] < 0.0
        ),
        "uncertainty_helps_winner": bool(
            comparisons["winner_minus_uncertainty_off_12prb_tbler"]["ci95_high"] < 0.0
        ),
        "winner_beats_ls": bool(ls_gap["ci95_high"] < 0.0),
        "winner_within_0p02_of_ls": bool(ls_gap["mean"] <= 0.02),
        "ls_factorized_detector_close_to_standard": bool(
            abs(ls_factorized_gap["mean"]) <= 0.02
        ),
        "winner_beats_ls_estimate_with_same_detector": bool(
            comparisons["winner_minus_ls_repaired_12prb_tbler"]["ci95_high"]
            < 0.0
        ),
        "no_12prb_high_snr_reversal": bool(
            winner_metrics["14db_tbler"] <= winner_metrics["10db_tbler"] + 0.01
        ),
        "12prb_coverage_reasonable": bool(
            0.90 <= winner_metrics["12prb_coverage95"] <= 0.99
        ),
        "12prb_normalized_error_reasonable": bool(
            0.70 <= winner_metrics["12prb_normalized_error"] <= 1.40
        ),
        "multiscale_improves_single": bool(
            per_candidate[best_multiscale]["12prb_tbler"]
            + 0.01
            < per_candidate[best_single]["12prb_tbler"]
        ),
    }
    if not checks["winner_training_converged"]:
        classification = "GATE1_POSTERIOR_FACTORIAL_TRAINING_EXTENSION_REQUIRED"
        next_action = "EXTEND_ONLY_THE_WINNING_CANDIDATE_WITH_FROZEN_HOLDOUT"
    elif not checks["ls_factorized_detector_close_to_standard"]:
        classification = "GATE1_DETECTOR_ESTIMATOR_INTERFACE_REPAIR_REQUIRED"
        next_action = "ALIGN_REPAIRED_DETECTOR_WITH_THE_STANDARD_LS_ESTIMATE"
    elif (
        checks["winner_beats_ls"]
        and checks["no_12prb_high_snr_reversal"]
        and checks["12prb_coverage_reasonable"]
    ):
        classification = "GATE1_POSTERIOR_FACTORIAL_BEATS_LS"
        next_action = "FREEZE_ARCHITECTURE_AND_RUN_PUBLICATION_SCALE_BASELINES"
    elif (
        checks["winner_within_0p02_of_ls"]
        and checks["winner_improves_frozen_old"]
        and checks["no_12prb_high_snr_reversal"]
    ):
        classification = "GATE1_POSTERIOR_FACTORIAL_NEAR_LS"
        next_action = "ADD_ONE_PRINCIPLED_TURBO_POSTERIOR_UPDATE_AND_RETEST"
    elif checks["multiscale_improves_single"] and checks["winner_improves_frozen_old"]:
        classification = "GATE1_TURBO_REFINEMENT_REQUIRED"
        next_action = "ADD_ONE_PRINCIPLED_DATA_AIDED_POSTERIOR_UPDATE"
    elif checks["winner_improves_frozen_old"]:
        classification = "GATE1_POSTERIOR_LOCALIZATION_REQUIRED"
        next_action = "REPLACE_GLOBAL_RFF_PRIOR_WITH_LOCALIZED_DELAY_DOPPLER_BASIS"
    else:
        classification = "GATE1_POSTERIOR_FAMILY_REDESIGN_REQUIRED"
        next_action = "REPLACE_GLOBAL_RFF_PRIOR_WITH_LOCALIZED_DELAY_DOPPLER_BASIS"
    return {
        "classification": classification,
        "next_action": next_action,
        "scientific_checks": checks,
        "paired_comparisons": comparisons,
        "per_candidate_holdout": per_candidate,
        "best_single": best_single,
        "best_multiscale": best_multiscale,
        "best_allocation": best_allocation,
        "best_physical": best_physical,
        "winner": winner,
        "winner_metrics": winner_metrics,
        "ls_12prb_tbler": mean_metric(frame, "ls_lmmse", "tbler", prb=12),
        "perfect_12prb_tbler": mean_metric(
            frame, "perfect_csi_lmmse", "tbler", prb=12
        ),
        "ls_repaired_12prb_tbler": mean_metric(
            frame, "ls_estimate_repaired", "tbler", prb=12
        ),
        "training_summaries": list(summaries),
    }


def write_incomplete(
    training_summaries: Sequence[dict[str, Any]],
    evaluation: dict[str, Any] | None,
) -> None:
    report = {
        "version": POSTERIOR_FACTORIAL_VERSION,
        "complete": False,
        "training_summaries": list(training_summaries),
        "evaluation": evaluation,
        "classification": "GATE1_NR_POSTERIOR_FACTORIAL_INCOMPLETE",
        "next_action": "RESUBMIT_SAME_COMMAND",
        "publication_nr_ready": False,
    }
    save_json(report, REPORT_PATH)
    save_json(report, GATE_JSON)
    GATE_TXT.parent.mkdir(parents=True, exist_ok=True)
    GATE_TXT.write_text(
        "CLASSIFICATION: GATE1_NR_POSTERIOR_FACTORIAL_INCOMPLETE\n"
        "NEXT_ACTION: RESUBMIT_SAME_COMMAND\n"
        "PUBLICATION_NR_READY: NO\n",
        encoding="utf-8",
    )
    print("GATE1_NR_POSTERIOR_FACTORIAL_INCOMPLETE: RESUBMIT")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/gate1_nr_posterior_factorial_screen.yaml",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--deadline-minutes", type=float, default=82.0)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    config_path = ROOT / args.config
    config = load_yaml(config_path)
    if expected_rows(config) != EXPECTED_ROWS:
        raise RuntimeError(
            f"Expected {EXPECTED_ROWS} rows, computed {expected_rows(config)}"
        )
    pre = preconditions(config)
    if not pre["passed"]:
        raise RuntimeError(f"Posterior-factorial preconditions failed: {pre}")
    if args.preflight_only:
        print("GATE1_NR_POSTERIOR_FACTORIAL_SCREEN_PREFLIGHT_PASS")
        print("CANDIDATES", len(config["candidates"]))
        print("EXPECTED_ROWS", EXPECTED_ROWS)
        print("TRAIN_GRIDS", [item["num_prb"] for item in config["training_cases"]])
        print("HOLDOUT_GRID", 12)
        print("FAIR_COORDINATE_RETRAINING YES")
        return

    device = normalize_device(args.device)
    deadline_epoch = time.time() + 60.0 * float(args.deadline_minutes)
    cases = [NRCase.from_mapping(item) for item in config["training_cases"]]
    contexts = [build_nr_context(case, device) for case in cases]
    candidates = [FactorialCandidate.from_mapping(item) for item in config["candidates"]]
    summaries: list[dict[str, Any]] = []
    for spec in candidates:
        summary = train_candidate(
            spec,
            config,
            config_path,
            pre,
            cases,
            contexts,
            deadline_epoch=deadline_epoch,
        )
        summaries.append(summary)
        if not summary.get("complete", False):
            write_incomplete(summaries, None)
            return
        if time.time() >= deadline_epoch:
            write_incomplete(summaries, None)
            return

    winner_summary = min(summaries, key=lambda item: float(item["best_score"]))
    winner = str(winner_summary["candidate"]["name"])
    frame, evaluation = evaluate(
        config,
        config_path,
        pre,
        summaries,
        winner,
        device,
        deadline_epoch=deadline_epoch,
    )
    if not evaluation.get("complete", False):
        write_incomplete(summaries, evaluation)
        return

    aggregate_frame = aggregate(frame)
    AGGREGATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    aggregate_frame.to_csv(AGGREGATE_PATH, index=False)
    scientific = classify(frame, candidates, summaries, winner)
    plots = make_plots(frame, candidates, winner)
    report = {
        "version": POSTERIOR_FACTORIAL_VERSION,
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
        "winner": winner,
        "candidate_summaries": summaries,
        "evaluation": evaluation,
        "aggregate_csv": str(AGGREGATE_PATH.relative_to(ROOT)),
        "plots": plots,
        **scientific,
    }
    save_json(report, REPORT_PATH)
    save_json(report, GATE_JSON)
    checks = report["scientific_checks"]
    lines = [
        "complete_rows: PASS",
        "all_candidate_training_complete: PASS",
        "equal_4prb_8prb_training_exposure: PASS",
        "untouched_12prb_holdout: PASS",
        *(f"{name}: {'PASS' if value else 'FAIL'}" for name, value in checks.items()),
        f"WINNER: {winner}",
        f"CLASSIFICATION: {report['classification']}",
        f"NEXT_ACTION: {report['next_action']}",
        "PUBLICATION_NR_READY: NO",
    ]
    GATE_TXT.parent.mkdir(parents=True, exist_ok=True)
    GATE_TXT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
