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

from bayesroute.config import capture_rng_state, restore_rng_state  # noqa: E402
from bayesroute.nr_gate1 import (  # noqa: E402
    GATE1_NR_VERSION,
    NRBayesRouteBridge,
    NRCase,
    build_nr_context,
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
    atomic_torch_save,
    bind_shared_operator,
    coded_metrics,
    copy_old_operator_if_compatible,
    differentiable_loss,
    edge_density,
    extract_operator_state,
    gradient_report,
    load_operator_state,
    make_repaired_detector,
    package_signature,
    parse_cases,
    posterior_metrics,
    repaired_forward,
    save_json,
    set_all_seeds,
    sha256_file,
    unique_parameters,
)

CAPACITY_VERSION = "gate1_nr_joint_operator_capacity_v1_1"
REQUIRED_CHECKPOINT_SHA256 = (
    "ca3243386e3d0511236a3c2c68f0396df9d05b7dc7f4118a8748d150613d3576"
)
SOURCE_CONTRACT_FILES = (
    "scripts/gate1_nr_joint_operator_common.py",
    "scripts/gate1_nr_joint_operator_capacity.py",
    "configs/gate1_nr_joint_operator_capacity.yaml",
    "src/bayesroute/lmmse_ep.py",
    "src/bayesroute/nr_gate1.py",
    "src/bayesroute/models.py",
    "src/bayesroute/qam.py",
)


def load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected a mapping in {path}")
    return value


def source_hashes() -> dict[str, str]:
    result: dict[str, str] = {}
    for relative in SOURCE_CONTRACT_FILES:
        path = ROOT / relative
        if not path.is_file():
            raise RuntimeError(f"Missing source-contract file: {relative}")
        result[relative] = sha256_file(path)
    return result



def training_schedule_report(config: dict[str, Any]) -> dict[str, Any]:
    """Verify equal per-case exposure for every capacity candidate.

    A global operator shares one parameter set across cases, while the
    case-specific diagnostic has one parameter set per case. To avoid
    undertraining either model, every candidate processes the same number of
    batches from every case. The case-specific model is still only a diagnostic
    upper bound and is not a deployment candidate.
    """
    num_cases = len(config["evaluation"]["cases"])
    updates_per_case = int(config["training"]["updates_per_case"])
    expected_total_steps = num_cases * updates_per_case
    candidate_steps = {
        str(item["name"]): int(item["steps"])
        for item in config["candidates"]
    }
    passed = bool(
        num_cases > 0
        and updates_per_case > 0
        and all(value == expected_total_steps for value in candidate_steps.values())
    )
    return {
        "passed": passed,
        "num_cases": num_cases,
        "updates_per_case": updates_per_case,
        "expected_total_steps_per_candidate": expected_total_steps,
        "candidate_total_steps": candidate_steps,
        "equal_case_exposure": passed,
    }


def verify_preconditions(config: dict[str, Any]) -> dict[str, Any]:
    smoke_path = ROOT / "outputs/gates/GATE1_NR_JOINT_OPERATOR_SMOKE.json"
    screen_path = ROOT / "outputs/reports/gate1_nr_detector_repair_screen.json"
    checkpoint_path = ROOT / str(config["checkpoint_path"])
    revision_path = ROOT / "GATE1_NR_JOINT_OPERATOR_REVISION.json"
    for path in (smoke_path, screen_path, checkpoint_path, revision_path):
        if not path.is_file():
            raise RuntimeError(f"Missing precondition file: {path}")
    smoke = json.loads(smoke_path.read_text(encoding="utf-8"))
    screen = json.loads(screen_path.read_text(encoding="utf-8"))
    revision = json.loads(revision_path.read_text(encoding="utf-8"))
    checkpoint_sha = sha256_file(checkpoint_path)
    schedule = training_schedule_report(config)
    checks = {
        "capacity_workflow_revision": bool(
            config.get("capacity_workflow_revision") == CAPACITY_VERSION
            and revision.get("capacity_workflow_revision") == CAPACITY_VERSION
        ),
        "equal_case_exposure": bool(schedule["passed"]),
        "joint_smoke": bool(
            smoke.get("classification") == "GATE1_NR_JOINT_OPERATOR_SMOKE_PASS"
            and smoke.get("overall_pass") is True
            and smoke.get("capacity_diagnostic_ready") is True
        ),
        "detector_screen": bool(
            screen.get("classification") == "GATE1_DETECTOR_REPAIR_PARTIAL"
            and screen.get("selected_variant") == "delmmse_sparse_i4_d0p7"
            and screen.get("evaluation", {}).get("rows") == 2688
        ),
        "checkpoint": checkpoint_sha == REQUIRED_CHECKPOINT_SHA256,
        "joint_revision": revision.get("revision") == JOINT_OPERATOR_VERSION,
        "gate1_revision": revision.get("gate1_revision") == GATE1_NR_VERSION,
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
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "checkpoint_sha256": checkpoint_sha,
        "training_schedule": schedule,
    }


def build_bridge(
    case: NRCase,
    context: Any,
    spec: CandidateSpec,
    *,
    operator_seed: int,
) -> NRBayesRouteBridge:
    return NRBayesRouteBridge(
        context.grid,
        num_streams=case.num_streams,
        rank=spec.rank,
        bank_rank=spec.bank_rank,
        detector_iterations=4,
        edge_mass=SELECTED_EDGE_MASS,
        length_f=spec.length_f,
        length_t=spec.length_t,
        operator_seed=int(operator_seed),
    ).to(context.device)


def build_candidate_modules(
    spec: CandidateSpec,
    cases: Sequence[NRCase],
    contexts: Sequence[Any],
    old_state_dict: dict[str, torch.Tensor],
    *,
    operator_seed: int,
) -> tuple[list[NRBayesRouteBridge], list[torch.nn.Module], bool]:
    bridges = [
        build_bridge(case, context, spec, operator_seed=operator_seed)
        for case, context in zip(cases, contexts)
    ]
    if spec.mode == "global":
        bind_shared_operator(bridges)
    warm_started = False
    if spec.initialization == "old_checkpoint":
        targets = bridges[:1] if spec.mode == "global" else bridges
        flags = [
            copy_old_operator_if_compatible(item, old_state_dict)
            for item in targets
        ]
        if not all(flags):
            raise RuntimeError(
                f"Candidate {spec.name} requested incompatible warm start"
            )
        warm_started = True
    detectors = [
        make_repaired_detector(int(context.grid.bits_per_symbol)).to(context.device)
        for context in contexts
    ]
    return bridges, detectors, warm_started


def operator_states(
    spec: CandidateSpec,
    bridges: Sequence[NRBayesRouteBridge],
) -> list[dict[str, torch.Tensor]]:
    if spec.mode == "global":
        return [extract_operator_state(bridges[0])]
    return [extract_operator_state(item) for item in bridges]


def load_operator_states(
    spec: CandidateSpec,
    bridges: Sequence[NRBayesRouteBridge],
    states: Sequence[dict[str, torch.Tensor]],
) -> None:
    if spec.mode == "global":
        if len(states) != 1:
            raise RuntimeError("Global candidate checkpoint must contain one state")
        load_operator_state(bridges[0], states[0])
        return
    if len(states) != len(bridges):
        raise RuntimeError("Case-specific candidate checkpoint has wrong state count")
    for bridge, state in zip(bridges, states):
        load_operator_state(bridge, state)


def candidate_contract(
    spec: CandidateSpec,
    config: dict[str, Any],
    config_path: Path,
    preconditions: dict[str, Any],
    cases: Sequence[NRCase],
) -> dict[str, Any]:
    payload = {
        "version": CAPACITY_VERSION,
        "joint_operator_version": JOINT_OPERATOR_VERSION,
        "candidate": spec.__dict__,
        "config_sha256": sha256_file(config_path),
        "source_sha256": source_hashes(),
        "old_checkpoint_sha256": preconditions["checkpoint_sha256"],
        "selected_detector": config["selected_detector"],
        "training": config["training"],
        "cases": [item.__dict__ for item in cases],
    }
    payload["signature"] = package_signature(payload)
    return payload


def validation_score(
    bridges: Sequence[NRBayesRouteBridge],
    detectors: Sequence[torch.nn.Module],
    contexts: Sequence[Any],
    cases: Sequence[NRCase],
    config: dict[str, Any],
    *,
    seed_offset: int,
) -> dict[str, Any]:
    training = config["training"]
    records: list[dict[str, float | str]] = []
    for case_index, (bridge, detector, context, case) in enumerate(
        zip(bridges, detectors, contexts, cases)
    ):
        for snr_index, raw_snr in enumerate(training["validation_ebno_db"]):
            snr = float(raw_snr)
            seed = (
                int(training["validation_seed"])
                + int(seed_offset)
                + 10_000 * case_index
                + 100 * snr_index
            )
            state = capture_rng_state()
            try:
                set_all_seeds(seed)
                batch = context.sample(
                    batch_size=int(training["validation_batch_size"]),
                    ebno_db=snr,
                )
                with torch.no_grad():
                    output = repaired_forward(bridge, detector, batch)
                    coded = coded_metrics(output, batch)
                    posterior = posterior_metrics(output, batch)
                records.append(
                    {
                        "case": case.name,
                        "ebno_db": snr,
                        **coded,
                        **posterior,
                    }
                )
            finally:
                restore_rng_state(state)

    frame = pd.DataFrame(records)
    case_nll = frame.groupby("case")["coded_bit_nll"].mean()
    mean_nll = float(frame["coded_bit_nll"].mean())
    worst_case_nll = float(case_nll.max())
    calibration_abs_log = float(
        np.mean(
            np.abs(
                np.log(
                    np.clip(
                        frame["normalized_error_mean"].to_numpy(dtype=float),
                        0.05,
                        20.0,
                    )
                )
            )
        )
    )
    score = mean_nll + 0.25 * worst_case_nll + 0.05 * calibration_abs_log
    return {
        "score": score,
        "mean_coded_bit_nll": mean_nll,
        "worst_case_coded_bit_nll": worst_case_nll,
        "mean_coded_ber": float(frame["coded_ber"].mean()),
        "mean_channel_nmse": float(frame["channel_nmse"].mean()),
        "mean_coverage95": float(frame["coverage95"].mean()),
        "calibration_abs_log": calibration_abs_log,
        "records": records,
    }


def train_candidate(
    spec: CandidateSpec,
    config: dict[str, Any],
    config_path: Path,
    preconditions: dict[str, Any],
    cases: Sequence[NRCase],
    contexts: Sequence[Any],
    old_state_dict: dict[str, torch.Tensor],
    *,
    operator_seed: int,
    deadline_epoch: float,
) -> dict[str, Any]:
    spec.validate()
    output_root = ROOT / "outputs/gate1_nr_joint_operator"
    checkpoint_dir = output_root / "checkpoints" / spec.name
    report_dir = ROOT / "outputs/reports"
    log_dir = ROOT / "outputs/logs"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    best_path = checkpoint_dir / "best.pt"
    last_path = checkpoint_dir / "last.pt"
    summary_path = report_dir / f"gate1_nr_joint_operator_{spec.name}.json"
    log_path = log_dir / f"gate1_nr_joint_operator_{spec.name}.csv"

    contract = candidate_contract(spec, config, config_path, preconditions, cases)
    if summary_path.is_file():
        existing = json.loads(summary_path.read_text(encoding="utf-8"))
        if existing.get("contract", {}).get("signature") != contract["signature"]:
            raise RuntimeError(f"Candidate summary contract mismatch: {spec.name}")
        if existing.get("complete") is True:
            return existing

    bridges, detectors, warm_started = build_candidate_modules(
        spec,
        cases,
        contexts,
        old_state_dict,
        operator_seed=operator_seed,
    )
    parameters = unique_parameters(bridges)
    expected_parameter_tensors = 2 if spec.mode == "global" else 2 * len(cases)
    if len(parameters) != expected_parameter_tensors:
        raise RuntimeError(
            f"Candidate {spec.name}: expected {expected_parameter_tensors} parameter "
            f"tensors, found {len(parameters)}"
        )
    optimizer = torch.optim.AdamW(
        parameters,
        lr=spec.learning_rate,
        weight_decay=1e-4,
    )
    start_step = 0
    best_score = float("inf")
    if last_path.is_file():
        state = torch.load(last_path, map_location=contexts[0].device, weights_only=False)
        if state.get("contract") != contract:
            raise RuntimeError(f"Candidate checkpoint contract mismatch: {spec.name}")
        load_operator_states(spec, bridges, state["operators"])
        optimizer.load_state_dict(state["optimizer"])
        start_step = int(state["step"]) + 1
        best_score = float(state["best_score"])
        restore_rng_state(state.get("rng_state"))
        print(f"Resuming {spec.name} at step {start_step}", flush=True)

    fields = [
        "candidate",
        "step",
        "case",
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
        old = pd.read_csv(log_path)
        old = old[old["step"] < start_step]
        old.to_csv(log_path, index=False)

    training = config["training"]
    last_validation: dict[str, Any] | None = None
    for step in range(start_step, spec.steps):
        case_index = step % len(cases)
        bridge = bridges[case_index]
        detector = detectors[case_index]
        context = contexts[case_index]
        case = cases[case_index]
        # Every candidate sees the same case/step random stream.
        seed = int(config["seed"]) + 1_000_000 + step
        set_all_seeds(seed)
        fraction = ((step * 2654435761 + case_index * 7919) % 1_000_003) / 1_000_003.0
        snr = float(training["ebno_db_min"]) + fraction * (
            float(training["ebno_db_max"]) - float(training["ebno_db_min"])
        )
        batch = context.sample(
            batch_size=int(training["batch_size"]),
            ebno_db=snr,
        )
        output = repaired_forward(bridge, detector, batch)
        loss, parts = differentiable_loss(
            output,
            batch,
            channel_loss_weight=spec.channel_loss_weight,
            calibration_loss_weight=spec.calibration_loss_weight,
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
        validation_score_value = float("nan")
        if validation_due:
            last_validation = validation_score(
                bridges,
                detectors,
                contexts,
                cases,
                config,
                seed_offset=0,
            )
            validation_score_value = float(last_validation["score"])
            if validation_score_value < best_score:
                best_score = validation_score_value
                atomic_torch_save(
                    {
                        "version": CAPACITY_VERSION,
                        "candidate": spec.__dict__,
                        "operators": operator_states(spec, bridges),
                        "optimizer": optimizer.state_dict(),
                        "step": step,
                        "best_score": best_score,
                        "validation": last_validation,
                        "contract": contract,
                        "rng_state": capture_rng_state(),
                    },
                    best_path,
                )

        save_due = (
            step % int(training["save_every"]) == 0
            or step == spec.steps - 1
        )
        if save_due:
            row = {
                "candidate": spec.name,
                "step": step,
                "case": case.name,
                "ebno_db": snr,
                "loss": float(loss.detach().item()),
                "coded_bit_nll": float(parts["bit_nll"].detach().item()),
                "channel_nll": float(parts["channel_nll"].detach().item()),
                "calibration_penalty": float(
                    parts["calibration_penalty"].detach().item()
                ),
                "gradient_norm": float(grad.detach().item()),
                "validation_score": validation_score_value,
                "contract_signature": contract["signature"],
            }
            with log_path.open("a", newline="", encoding="utf-8") as stream:
                csv.DictWriter(stream, fieldnames=fields).writerow(row)
            print(json.dumps(row), flush=True)
            atomic_torch_save(
                {
                    "version": CAPACITY_VERSION,
                    "candidate": spec.__dict__,
                    "operators": operator_states(spec, bridges),
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

    if not best_path.is_file() or not last_path.is_file():
        raise RuntimeError(f"Candidate {spec.name} completed without checkpoints")
    best = torch.load(best_path, map_location=contexts[0].device, weights_only=False)
    if best.get("contract") != contract:
        raise RuntimeError(f"Best checkpoint contract mismatch: {spec.name}")
    summary = {
        "complete": True,
        "version": CAPACITY_VERSION,
        "candidate": spec.__dict__,
        "warm_started": warm_started,
        "steps": spec.steps,
        "updates_per_case": int(spec.steps // len(cases)),
        "training_schedule": preconditions["training_schedule"],
        "best_score": float(best["best_score"]),
        "best_step": int(best["step"]),
        "validation": best["validation"],
        "best_checkpoint": str(best_path.relative_to(ROOT)),
        "best_checkpoint_sha256": sha256_file(best_path),
        "trainable_parameters": int(sum(p.numel() for p in parameters)),
        "contract": contract,
    }
    save_json(summary, summary_path)
    del bridges, detectors, optimizer
    if contexts[0].device.type == "cuda":
        torch.cuda.empty_cache()
    return summary


def list_candidate_index(config: dict[str, Any], name: str) -> int:
    names = [str(item["name"]) for item in config["candidates"]]
    return names.index(name)


def load_candidate_best_states(
    summary: dict[str, Any],
    device: torch.device,
) -> list[dict[str, torch.Tensor]]:
    path = ROOT / str(summary["best_checkpoint"])
    state = torch.load(path, map_location=device, weights_only=False)
    if state.get("contract") != summary["contract"]:
        raise RuntimeError(f"Candidate best-state contract mismatch: {summary['candidate']['name']}")
    return state["operators"]


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
    candidate_mode: str,
) -> dict[str, Any]:
    return {
        "case": case.name,
        "scenario": case.scenario,
        "num_streams": case.num_streams,
        "variant": variant,
        "candidate_mode": candidate_mode,
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
        "candidate_mode": "standard",
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


def append_rows_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    new = pd.DataFrame(rows)
    if path.is_file():
        old = pd.read_csv(path)
        if list(old.columns) != list(new.columns):
            raise RuntimeError("Joint-operator evaluation CSV column mismatch")
        combined = pd.concat([old, new], ignore_index=True)
    else:
        combined = new
    temporary = path.with_suffix(path.suffix + ".tmp")
    combined.to_csv(temporary, index=False)
    temporary.replace(path)


def evaluation_contract(
    config: dict[str, Any],
    config_path: Path,
    preconditions: dict[str, Any],
    summaries: Sequence[dict[str, Any]],
    variants: Sequence[str],
) -> dict[str, Any]:
    payload = {
        "version": CAPACITY_VERSION,
        "joint_operator_version": JOINT_OPERATOR_VERSION,
        "config_sha256": sha256_file(config_path),
        "source_sha256": source_hashes(),
        "old_checkpoint_sha256": preconditions["checkpoint_sha256"],
        "candidate_checkpoints": {
            item["candidate"]["name"]: item["best_checkpoint_sha256"]
            for item in summaries
        },
        "selected_detector": config["selected_detector"],
        "evaluation": config["evaluation"],
        "variants": list(variants),
    }
    payload["signature"] = package_signature(payload)
    return payload


def evaluate_candidates(
    config: dict[str, Any],
    config_path: Path,
    preconditions: dict[str, Any],
    summaries: Sequence[dict[str, Any]],
    old_state_dict: dict[str, torch.Tensor],
    *,
    operator_seed: int,
    deadline_epoch: float,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    evaluation = config["evaluation"]
    cases = parse_cases(evaluation["cases"])
    summary_by_name = {item["candidate"]["name"]: item for item in summaries}
    global_summaries = [
        item for item in summaries if item["candidate"]["mode"] == "global"
    ]
    case_summaries = [
        item for item in summaries if item["candidate"]["mode"] == "case_specific"
    ]
    best_global = min(global_summaries, key=lambda item: item["best_score"])
    best_case = min(case_summaries, key=lambda item: item["best_score"])
    candidate_variants = [f"joint_{item['candidate']['name']}" for item in summaries]
    control_variants = [
        "old_checkpoint_repaired_detector",
        "best_global_uncertainty_off",
        "best_global_random_graph",
        "best_global_full_graph",
        "ls_lmmse",
        "perfect_csi_lmmse",
    ]
    variants = candidate_variants + control_variants
    expected_rows = (
        len(cases)
        * len(evaluation["ebno_db"])
        * int(evaluation["repetitions"])
        * len(variants)
    )
    contract = evaluation_contract(
        config,
        config_path,
        preconditions,
        summaries,
        variants,
    )
    eval_dir = ROOT / "outputs/eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    raw_path = eval_dir / "gate1_nr_joint_operator_capacity.csv"
    contract_path = eval_dir / "gate1_nr_joint_operator_capacity_contract.json"
    if raw_path.is_file():
        existing_contract = json.loads(contract_path.read_text(encoding="utf-8"))
        if existing_contract.get("signature") != contract["signature"]:
            raise RuntimeError("Joint-operator evaluation resume contract mismatch")
    else:
        save_json(contract, contract_path)

    done: set[tuple[str, str, float, int]] = set()
    if raw_path.is_file():
        old = pd.read_csv(raw_path)
        keys = ["case", "variant", "ebno_db", "rep"]
        if old[keys].duplicated().any():
            raise RuntimeError("Joint-operator evaluation contains duplicate keys")
        for _, row in old.iterrows():
            done.add(
                (
                    str(row["case"]),
                    str(row["variant"]),
                    float(row["ebno_db"]),
                    int(row["rep"]),
                )
            )

    candidate_specs = {
        item["candidate"]["name"]: CandidateSpec.from_mapping(item["candidate"])
        for item in summaries
    }
    candidate_states = {
        item["candidate"]["name"]: load_candidate_best_states(
            item, torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        )
        for item in summaries
    }

    for case_index, case in enumerate(cases):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        context = build_nr_context(case, device)
        candidate_bridges: dict[str, NRBayesRouteBridge] = {}
        detectors: dict[str, torch.nn.Module] = {}
        for summary in summaries:
            name = summary["candidate"]["name"]
            spec = candidate_specs[name]
            bridge = build_bridge(
                case,
                context,
                spec,
                operator_seed=operator_seed,
            )
            states = candidate_states[name]
            state = states[0] if spec.mode == "global" else states[case_index]
            load_operator_state(bridge, state)
            bridge.eval()
            candidate_bridges[name] = bridge
            detectors[name] = make_repaired_detector(
                int(context.grid.bits_per_symbol)
            ).to(device)

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
        old_bridge = build_bridge(
            case,
            context,
            old_spec,
            operator_seed=operator_seed,
        )
        if not copy_old_operator_if_compatible(old_bridge, old_state_dict):
            raise RuntimeError("Old checkpoint cannot initialize evaluation bridge")
        old_bridge.eval()
        old_detector = make_repaired_detector(int(context.grid.bits_per_symbol)).to(device)
        ls_receiver = standard_receiver(context, perfect_csi=False, return_crc=True)
        perfect_receiver = standard_receiver(context, perfect_csi=True, return_crc=True)

        best_global_name = best_global["candidate"]["name"]
        best_global_bridge = candidate_bridges[best_global_name]
        best_global_detector = detectors[best_global_name]
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
                with torch.inference_mode():
                    for summary in summaries:
                        name = summary["candidate"]["name"]
                        variant = f"joint_{name}"
                        if variant in missing:
                            outputs[variant] = repaired_forward(
                                candidate_bridges[name],
                                detectors[name],
                                batch,
                            )
                    if "old_checkpoint_repaired_detector" in missing:
                        outputs["old_checkpoint_repaired_detector"] = repaired_forward(
                            old_bridge,
                            old_detector,
                            batch,
                        )
                    controls_needed = any(
                        name in missing
                        for name in (
                            "best_global_uncertainty_off",
                            "best_global_random_graph",
                            "best_global_full_graph",
                        )
                    )
                    if controls_needed:
                        base = repaired_forward(
                            best_global_bridge,
                            best_global_detector,
                            batch,
                        )
                        posterior = base["posterior"]
                        reference = base["reference_graph_mask"]
                        if "best_global_uncertainty_off" in missing:
                            outputs["best_global_uncertainty_off"] = repaired_forward(
                                best_global_bridge,
                                best_global_detector,
                                batch,
                                covariance_mode="none",
                                posterior=posterior,
                                reference_graph=reference,
                            )
                        if "best_global_random_graph" in missing:
                            outputs["best_global_random_graph"] = repaired_forward(
                                best_global_bridge,
                                best_global_detector,
                                batch,
                                graph_mode="random",
                                random_seed=seed + 900_000,
                                posterior=posterior,
                                reference_graph=reference,
                            )
                        if "best_global_full_graph" in missing:
                            outputs["best_global_full_graph"] = repaired_forward(
                                best_global_bridge,
                                best_global_detector,
                                batch,
                                graph_mode="full",
                                posterior=posterior,
                                reference_graph=reference,
                            )

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
                                variant=name,
                                snr=snr,
                                rep=rep,
                                seed=seed,
                                metrics=metrics,
                                contract_signature=contract["signature"],
                            )
                        )
                        continue
                    if name == "perfect_csi_lmmse":
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
                                variant=name,
                                snr=snr,
                                rep=rep,
                                seed=seed,
                                metrics=metrics,
                                contract_signature=contract["signature"],
                            )
                        )
                        continue
                    if name.startswith("joint_"):
                        candidate_name = name.removeprefix("joint_")
                        mode = summary_by_name[candidate_name]["candidate"]["mode"]
                    elif name.startswith("best_global_"):
                        mode = "global_control"
                    else:
                        mode = "old_checkpoint"
                    rows.append(
                        custom_row(
                            case=case,
                            variant=name,
                            snr=snr,
                            rep=rep,
                            seed=seed,
                            output=outputs[name],
                            batch=batch,
                            decoded=decoded[name],
                            contract_signature=contract["signature"],
                            candidate_mode=mode,
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
                            "expected_rows": expected_rows,
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
                        "expected_rows": expected_rows,
                        "raw_csv": str(raw_path.relative_to(ROOT)),
                        "contract": contract,
                        "stop_reason": "internal_deadline",
                    }, {
                        "best_global": best_global,
                        "best_case_specific": best_case,
                    }

        del (
            context,
            candidate_bridges,
            detectors,
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
    complete = bool(len(frame) == expected_rows and unique == expected_rows)
    if not complete:
        raise RuntimeError(
            f"Joint-operator evaluation incomplete: rows={len(frame)}, "
            f"unique={unique}, expected={expected_rows}"
        )
    return frame, {
        "complete": True,
        "rows": int(len(frame)),
        "unique_rows": unique,
        "expected_rows": expected_rows,
        "raw_csv": str(raw_path.relative_to(ROOT)),
        "contract": contract,
    }, {
        "best_global": best_global,
        "best_case_specific": best_case,
    }


def mean_metric(
    frame: pd.DataFrame,
    *,
    variant: str,
    metric: str,
    multi_stream_only: bool = True,
    high_snr_only: bool = True,
    snr: float | None = None,
) -> float:
    sub = frame[frame["variant"] == variant]
    if multi_stream_only:
        sub = sub[sub["num_streams"] >= 4]
    if high_snr_only:
        sub = sub[sub["ebno_db"].isin([6.0, 10.0])]
    if snr is not None:
        sub = sub[sub["ebno_db"] == float(snr)]
    values = pd.to_numeric(sub[metric], errors="coerce").dropna()
    return float(values.mean()) if not values.empty else float("nan")


def paired_delta(
    frame: pd.DataFrame,
    reference: str,
    comparator: str,
    metric: str,
) -> dict[str, float]:
    keys = ["case", "ebno_db", "rep", "eval_seed"]
    a = frame[(frame["variant"] == reference) & (frame["ebno_db"].isin([6.0, 10.0]))]
    b = frame[(frame["variant"] == comparator) & (frame["ebno_db"].isin([6.0, 10.0]))]
    a = a[a["num_streams"] >= 4]
    b = b[b["num_streams"] >= 4]
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


def classify(
    frame: pd.DataFrame,
    selection: dict[str, Any],
) -> dict[str, Any]:
    best_global_name = selection["best_global"]["candidate"]["name"]
    best_case_name = selection["best_case_specific"]["candidate"]["name"]
    global_variant = f"joint_{best_global_name}"
    case_variant = f"joint_{best_case_name}"
    values = {
        "global_tbler": mean_metric(frame, variant=global_variant, metric="tbler"),
        "case_specific_tbler": mean_metric(frame, variant=case_variant, metric="tbler"),
        "old_checkpoint_tbler": mean_metric(
            frame, variant="old_checkpoint_repaired_detector", metric="tbler"
        ),
        "ls_lmmse_tbler": mean_metric(frame, variant="ls_lmmse", metric="tbler"),
        "perfect_csi_lmmse_tbler": mean_metric(
            frame, variant="perfect_csi_lmmse", metric="tbler"
        ),
        "global_6db_tbler": mean_metric(
            frame,
            variant=global_variant,
            metric="tbler",
            high_snr_only=False,
            snr=6.0,
        ),
        "global_10db_tbler": mean_metric(
            frame,
            variant=global_variant,
            metric="tbler",
            high_snr_only=False,
            snr=10.0,
        ),
        "uncertainty_off_tbler": mean_metric(
            frame, variant="best_global_uncertainty_off", metric="tbler"
        ),
        "random_graph_tbler": mean_metric(
            frame, variant="best_global_random_graph", metric="tbler"
        ),
        "full_graph_tbler": mean_metric(
            frame, variant="best_global_full_graph", metric="tbler"
        ),
    }
    checks = {
        "global_improves_old_checkpoint": bool(
            values["global_tbler"] <= values["old_checkpoint_tbler"] - 0.05
        ),
        "global_within_0p10_of_ls": bool(
            values["global_tbler"] <= values["ls_lmmse_tbler"] + 0.10
        ),
        "case_specific_within_0p10_of_ls": bool(
            values["case_specific_tbler"] <= values["ls_lmmse_tbler"] + 0.10
        ),
        "case_specific_improves_global": bool(
            values["case_specific_tbler"] <= values["global_tbler"] - 0.03
        ),
        "uncertainty_helps_global": bool(
            values["global_tbler"] <= values["uncertainty_off_tbler"] - 0.01
        ),
        "coupling_beats_random": bool(
            values["global_tbler"] <= values["random_graph_tbler"] - 0.005
        ),
        "sparse_not_worse_than_full": bool(
            values["global_tbler"] <= values["full_graph_tbler"] + 0.01
        ),
        "no_high_snr_reversal": bool(
            values["global_10db_tbler"] <= values["global_6db_tbler"] + 0.03
        ),
    }
    if checks["global_within_0p10_of_ls"] and checks["global_improves_old_checkpoint"]:
        classification = "GATE1_JOINT_OPERATOR_SUPPORTED"
        next_action = "RUN_CLEAN_NEW_SEED_ABLATIONS_AND_EXPAND_NR_CASES"
    elif checks["case_specific_within_0p10_of_ls"]:
        classification = "GATE1_CONFIGURATION_CONDITIONED_OPERATOR_REQUIRED"
        next_action = "BUILD_PSD_KERNEL_MIXTURE_WITH_PILOT_CONDITIONED_ROUTING"
    elif checks["case_specific_improves_global"]:
        classification = "GATE1_OPERATOR_BASIS_EXPANSION_REQUIRED"
        next_action = "EXPAND_DELAY_DOPPLER_KERNEL_BANK_BEFORE_CONDITIONING"
    else:
        classification = "GATE1_JOINT_TRAINING_INSUFFICIENT"
        next_action = "REASSESS_POSTERIOR_OPERATOR_MODEL_AND_OBSERVATION_LIKELIHOOD"
    comparisons = {
        comparator: paired_delta(frame, global_variant, comparator, "tbler")
        for comparator in (
            "old_checkpoint_repaired_detector",
            case_variant,
            "ls_lmmse",
            "perfect_csi_lmmse",
            "best_global_uncertainty_off",
            "best_global_random_graph",
            "best_global_full_graph",
        )
    }
    return {
        "classification": classification,
        "next_action": next_action,
        "best_global_candidate": best_global_name,
        "best_case_specific_candidate": best_case_name,
        "high_snr_metrics": values,
        "scientific_checks": checks,
        "paired_comparisons_global_minus_comparator": comparisons,
    }


def make_outputs(
    frame: pd.DataFrame,
    decision: dict[str, Any],
) -> tuple[str, list[str]]:
    eval_dir = ROOT / "outputs/eval"
    report_dir = ROOT / "outputs/reports"
    plot_dir = ROOT / "outputs/plots"
    eval_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)
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
        frame.groupby(["case", "scenario", "variant", "ebno_db"], as_index=False)[numeric]
        .agg(["mean", "std", "count"])
    )
    aggregate.columns = [
        "_".join(str(x) for x in item if str(x))
        for item in aggregate.columns.to_flat_index()
    ]
    aggregate_path = eval_dir / "gate1_nr_joint_operator_capacity_aggregate.csv"
    aggregate.to_csv(aggregate_path, index=False)

    selected_global = f"joint_{decision['best_global_candidate']}"
    selected_case = f"joint_{decision['best_case_specific_candidate']}"
    selected = [
        selected_global,
        selected_case,
        "old_checkpoint_repaired_detector",
        "ls_lmmse",
        "perfect_csi_lmmse",
    ]
    plots: list[str] = []
    for case in sorted(frame["case"].unique()):
        fig, ax = plt.subplots(figsize=(7.4, 4.9))
        sub_case = frame[frame["case"] == case]
        for variant in selected:
            sub = sub_case[sub_case["variant"] == variant]
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
        ax.legend(fontsize=6.5)
        ax.set_title(case)
        fig.tight_layout()
        path = plot_dir / f"gate1_joint_operator_{case}_tbler.png"
        fig.savefig(path, dpi=220)
        plt.close(fig)
        plots.append(str(path.relative_to(ROOT)))
    return str(aggregate_path.relative_to(ROOT)), plots


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/gate1_nr_joint_operator_capacity.yaml",
    )
    args = parser.parse_args()
    config_path = ROOT / args.config
    config = load_yaml(config_path)
    preconditions = verify_preconditions(config)
    if not preconditions["passed"]:
        raise RuntimeError(f"Capacity preconditions failed: {preconditions}")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    deadline_epoch = time.time() + float(config["internal_deadline_seconds"])
    cases = parse_cases(config["evaluation"]["cases"])
    contexts = [build_nr_context(case, device) for case in cases]
    checkpoint = torch.load(
        ROOT / str(config["checkpoint_path"]),
        map_location=device,
        weights_only=False,
    )
    if "model" not in checkpoint:
        raise RuntimeError("Old checkpoint has no model state")
    old_state_dict = checkpoint["model"]
    operator_seed = 58001

    summaries: list[dict[str, Any]] = []
    for raw_candidate in config["candidates"]:
        spec = CandidateSpec.from_mapping(raw_candidate)
        summary = train_candidate(
            spec,
            config,
            config_path,
            preconditions,
            cases,
            contexts,
            old_state_dict,
            operator_seed=operator_seed,
            deadline_epoch=deadline_epoch,
        )
        if summary.get("complete") is not True:
            status = {
                "version": CAPACITY_VERSION,
                "complete": False,
                "stage": "training",
                "active_candidate": spec.name,
                "candidate_status": summary,
                "preconditions": preconditions,
                "publication_nr_ready": False,
            }
            save_json(
                status,
                ROOT / "outputs/reports/gate1_nr_joint_operator_capacity.json",
            )
            print("JOINT_OPERATOR_CAPACITY_INCOMPLETE: RESUBMIT", flush=True)
            return
        summaries.append(summary)

    del contexts
    if device.type == "cuda":
        torch.cuda.empty_cache()
    frame, evaluation, selection = evaluate_candidates(
        config,
        config_path,
        preconditions,
        summaries,
        old_state_dict,
        operator_seed=operator_seed,
        deadline_epoch=deadline_epoch,
    )
    if evaluation.get("complete") is not True:
        status = {
            "version": CAPACITY_VERSION,
            "complete": False,
            "stage": "evaluation",
            "evaluation": evaluation,
            "candidate_summaries": summaries,
            "selection": selection,
            "preconditions": preconditions,
            "publication_nr_ready": False,
        }
        save_json(
            status,
            ROOT / "outputs/reports/gate1_nr_joint_operator_capacity.json",
        )
        print("JOINT_OPERATOR_CAPACITY_INCOMPLETE: RESUBMIT", flush=True)
        return

    decision = classify(frame, selection)
    aggregate_csv, plots = make_outputs(frame, decision)
    software_checks = {
        "complete_rows": bool(evaluation["complete"]),
        "all_metrics_finite": bool(
            np.isfinite(
                frame[["information_ber", "tbler", "crc_failure_rate"]].to_numpy()
            ).all()
        ),
        "all_candidates_complete": all(item["complete"] for item in summaries),
        "equal_case_exposure": bool(preconditions["training_schedule"]["passed"]),
        "candidate_modes_present": bool(
            {item["candidate"]["mode"] for item in summaries}
            == {"global", "case_specific"}
        ),
        "paired_seed_batches": bool(
            frame.groupby(["case", "ebno_db", "rep"])["variant"].nunique().min()
            == frame["variant"].nunique()
        ),
        "source_contract": len(source_hashes()) == len(SOURCE_CONTRACT_FILES),
    }
    report = {
        "version": CAPACITY_VERSION,
        "joint_operator_version": JOINT_OPERATOR_VERSION,
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
        "complete": all(software_checks.values()),
        "preconditions": preconditions,
        "training_schedule": preconditions["training_schedule"],
        "software_checks": software_checks,
        "candidate_summaries": summaries,
        "evaluation": evaluation,
        "selection": {
            "best_global": selection["best_global"]["candidate"]["name"],
            "best_case_specific": selection["best_case_specific"]["candidate"]["name"],
        },
        **decision,
        "aggregate_csv": aggregate_csv,
        "plots": plots,
        "case_specific_scope": (
            "diagnostic upper bound trained and evaluated on the same configuration "
            "families; not a deployment or publication method"
        ),
        "publication_nr_ready": False,
    }
    report_path = ROOT / "outputs/reports/gate1_nr_joint_operator_capacity.json"
    save_json(report, report_path)
    gate_dir = ROOT / "outputs/gates"
    save_json(report, gate_dir / "GATE1_NR_JOINT_OPERATOR_CAPACITY.json")
    lines = [
        *(f"{key}: {'PASS' if value else 'FAIL'}" for key, value in software_checks.items()),
        *(f"{key}: {'PASS' if value else 'FAIL'}" for key, value in decision["scientific_checks"].items()),
        f"BEST_GLOBAL: {decision['best_global_candidate']}",
        f"BEST_CASE_SPECIFIC: {decision['best_case_specific_candidate']}",
        f"CLASSIFICATION: {decision['classification']}",
        f"NEXT_ACTION: {decision['next_action']}",
        "PUBLICATION_NR_READY: NO",
    ]
    (gate_dir / "GATE1_NR_JOINT_OPERATOR_CAPACITY.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print("\n".join(lines), flush=True)
    if not report["complete"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
