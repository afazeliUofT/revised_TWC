#!/usr/bin/env python3
from __future__ import annotations

"""Final, training-free go/no-go ceiling for beating LS+LMMSE.

A bounded-rank localized delay--Doppler subspace is selected using only new
4-/8-PRB cases.  The selected basis is then frozen and tested on a disjoint
12-PRB holdout.  Oracle coefficients fit the true LS residual, so this is an
optimistic upper bound for an LS-anchored learned posterior.  Failure means the
family should be abandoned for the goal of beating LS+LMMSE.
"""

import argparse
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

from bayesroute.localized_delay_doppler import (
    LOCALIZED_DD_VERSION,
    LocalizedDelayDopplerSpec,
    localized_delay_doppler_features,
    mathematical_self_test,
    project_tensor_to_basis,
)
from bayesroute.models import PosteriorOutput
from bayesroute.nr_gate1 import decode_bridge, normalize_device, run_standard_receiver, standard_receiver
from bayesroute.turbo_posterior import posterior_batch_metrics
from gate1_nr_joint_operator_common import (
    coded_metrics,
    make_repaired_detector,
    posterior_graph,
    repaired_forward,
)
from gate1_nr_posterior_factorial_common import (
    ls_posterior_from_receiver,
    ls_repaired_forward,
    save_json,
    sha256_file,
)
from gate1_nr_turbo_posterior_common import (
    build_loaded_bridge,
    experiment_signature,
    initial_detector_output,
    make_case_context,
    pilot_state_and_reference,
    set_all_seeds,
    source_hashes,
)

VERSION = "gate1_nr_localized_ceiling_v1"
PRIOR_REPORT = ROOT / "outputs/reports/gate1_nr_turbo_basis_audit.json"
RAW_SELECTION = ROOT / "outputs/eval/gate1_nr_localized_ceiling_selection.csv"
RAW_HOLDOUT = ROOT / "outputs/eval/gate1_nr_localized_ceiling_holdout.csv"
SELECTION_CONTRACT = ROOT / "outputs/eval/gate1_nr_localized_ceiling_selection_contract.json"
HOLDOUT_CONTRACT = ROOT / "outputs/eval/gate1_nr_localized_ceiling_holdout_contract.json"
AGGREGATE_PATH = ROOT / "outputs/eval/gate1_nr_localized_ceiling_aggregate.csv"
REPORT_PATH = ROOT / "outputs/reports/gate1_nr_localized_ceiling.json"
GATE_JSON = ROOT / "outputs/gates/GATE1_NR_LOCALIZED_CEILING.json"
GATE_TXT = ROOT / "outputs/gates/GATE1_NR_LOCALIZED_CEILING.txt"
EXPECTED_SELECTION_ROWS = 252
EXPECTED_HOLDOUT_ROWS = 288
EXPECTED_ROWS = EXPECTED_SELECTION_ROWS + EXPECTED_HOLDOUT_ROWS
SOURCE_FILES = [
    "configs/gate1_nr_localized_ceiling.yaml",
    "scripts/gate1_nr_localized_ceiling.py",
    "src/bayesroute/localized_delay_doppler.py",
    "scripts/gate1_nr_turbo_posterior_common.py",
    "scripts/gate1_nr_posterior_factorial_common.py",
    "scripts/gate1_nr_joint_operator_common.py",
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


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected JSON object: {path}")
    return value


def preconditions() -> dict[str, Any]:
    if not PRIOR_REPORT.is_file():
        raise RuntimeError(f"Missing prior basis-audit report: {PRIOR_REPORT}")
    report = load_json(PRIOR_REPORT)
    checks = {
        "complete": report.get("complete") is True,
        "classification": report.get("classification") == "GATE1_LOCALIZED_DELAY_DOPPLER_POSTERIOR_JUSTIFIED",
        "next_action": report.get("next_action") == "IMPLEMENT_LOCALIZED_DELAY_DOPPLER_POSTERIOR_AND_RETRAIN_ON_4_8_PRB_HOLD_OUT_12_PRB",
        "rows": report.get("evaluation", {}).get("rows") == 396,
        "unique_rows": report.get("evaluation", {}).get("unique_rows") == 396,
        "true_channel_control": report.get("scientific_checks", {}).get("true_channel_repaired_matches_perfect") is True,
        "global_basis_not_near_ls": report.get("scientific_checks", {}).get("best_global_basis_within_0p01_of_ls_12prb") is False,
    }
    if not all(checks.values()):
        raise RuntimeError(f"Localized ceiling preconditions failed: {checks}")
    return {
        "passed": True,
        "checks": checks,
        "report": str(PRIOR_REPORT.relative_to(ROOT)),
        "classification": report["classification"],
        "checkpoint_sha256": report["evaluation"]["contract"]["checkpoint_sha256"],
        "prior_metrics": report.get("metrics", {}),
    }


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


def ensure_contract(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    payload = dict(payload)
    payload["signature"] = experiment_signature(payload)
    if path.is_file():
        existing = load_json(path)
        if existing.get("signature") != payload["signature"]:
            raise RuntimeError(f"Evaluation contract mismatch: {path}")
    else:
        save_json(payload, path)
    return payload


def completed_batches(path: Path, variants: Sequence[str]) -> set[tuple[str, float, int]]:
    if not path.is_file():
        return set()
    frame = pd.read_csv(path)
    keys = ["case", "variant", "ebno_db", "rep"]
    if frame[keys].duplicated().any():
        raise RuntimeError(f"Duplicate rows: {path}")
    counts = frame.groupby(["case", "ebno_db", "rep"])["variant"].nunique()
    partial = counts[counts != len(variants)]
    if len(partial):
        raise RuntimeError(f"Partial paired batch in {path}: {partial.index.tolist()}")
    return {(str(a), float(b), int(c)) for a, b, c in counts.index}


def diagonal_local_covariance(var_diag: torch.Tensor) -> torch.Tensor:
    if var_diag.ndim != 3:
        raise ValueError("var_diag must have shape [B,N,R]")
    batch, n_layers, n_re = var_diag.shape
    local = torch.zeros(batch, n_layers, n_layers, n_re, dtype=torch.complex64, device=var_diag.device)
    idx = torch.arange(n_layers, device=var_diag.device)
    local[:, idx, idx, :] = var_diag.to(torch.complex64)
    return local


def oracle_projection_posterior(
    *,
    truth: torch.Tensor,
    features: torch.Tensor,
    anchor: PosteriorOutput | None,
    effective_noise: torch.Tensor,
    ridge: float,
) -> tuple[PosteriorOutput, dict[str, float]]:
    target = truth if anchor is None else truth - anchor.mean
    projected, projection = project_tensor_to_basis(target, features, ridge=float(ridge))
    mean = projected if anchor is None else anchor.mean + projected
    residual = truth - mean
    var = residual.abs().square().mean(dim=2).real.clamp_min(1e-8)
    posterior = PosteriorOutput(
        mean=mean,
        var_diag=var,
        local_cov=diagonal_local_covariance(var),
        latent_cov=torch.zeros(1, 1, dtype=torch.complex64, device=truth.device),
        effective_noise=effective_noise,
    )
    truth_power = truth.abs().square().mean().clamp_min(1e-12)
    final_nmse = residual.abs().square().mean() / truth_power
    return posterior, {
        **projection,
        "final_channel_nmse": float(final_nmse.real.item()),
        "anchor": "none" if anchor is None else "sionna_ls",
    }


def true_channel_posterior(batch: Any) -> PosteriorOutput:
    b, n, _, r = batch.h.shape
    var = torch.full((b, n, r), 1e-8, dtype=torch.float32, device=batch.h.device)
    return PosteriorOutput(
        mean=batch.h,
        var_diag=var,
        local_cov=diagonal_local_covariance(var),
        latent_cov=torch.zeros(1, 1, dtype=torch.complex64, device=batch.h.device),
        effective_noise=torch.as_tensor(batch.noise_var, device=batch.h.device),
    )


def output_from_posterior(
    bridge: Any,
    detector: Any,
    batch: Any,
    posterior: PosteriorOutput,
    diagnostics: dict[str, Any],
) -> dict[str, Any]:
    _, graph = posterior_graph(posterior, batch)
    output = repaired_forward(
        bridge,
        detector,
        batch,
        posterior=posterior,
        reference_graph=graph,
    )
    output["posterior_metrics"] = posterior_batch_metrics(posterior, batch.h, batch.data_idx)
    output["ceiling_diagnostics"] = diagnostics
    return output


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


def crc_disagreement(decoded: dict[str, Any], bits: torch.Tensor) -> float:
    block_error = (decoded["b_hat"] != bits).reshape(bits.shape[0], bits.shape[1], -1).any(-1)
    crc_failure = ~decoded["crc"].bool()
    if crc_failure.shape != block_error.shape:
        crc_failure = crc_failure.reshape(block_error.shape)
    return float((block_error != crc_failure).float().mean().item())


def custom_row(
    *,
    phase: str,
    case: Any,
    variant: str,
    snr: float,
    rep: int,
    seed: int,
    output: dict[str, Any],
    batch: Any,
    decoded: dict[str, Any],
    signature: str,
    spec: LocalizedDelayDopplerSpec | None,
) -> dict[str, Any]:
    diagnostics = dict(output.get("ceiling_diagnostics", {}))
    posterior_metrics = output.get("posterior_metrics") or posterior_batch_metrics(output["posterior"], batch.h, batch.data_idx)
    return {
        "phase": phase,
        "case": case.name,
        "group": getattr(case, "group", phase),
        "num_prb": int(case.num_prb),
        "variant": variant,
        "ebno_db": float(snr),
        "rep": int(rep),
        "eval_seed": int(seed),
        "information_ber": float(decoded["information_ber"]),
        "tbler": float(decoded["tbler"]),
        "crc_failure_rate": float(decoded["crc_failure_rate"]),
        "crc_block_disagreement_rate": crc_disagreement(decoded, batch.information_bits),
        **coded_metrics(output, batch),
        **posterior_metrics,
        "edge_density": float(output["edge_density"].item()),
        "basis_name": "none" if spec is None else spec.name,
        "basis_nominal_rank": 0 if spec is None else int(spec.nominal_rank),
        "basis_effective_rank": int(diagnostics.get("effective_rank", 0)),
        "basis_projection_nmse": float(diagnostics.get("projection_nmse", float("nan"))),
        "final_channel_nmse": float(diagnostics.get("final_channel_nmse", float("nan"))),
        "anchor": str(diagnostics.get("anchor", "none")),
        "contract_signature": signature,
    }


def standard_row(
    *, phase: str, case: Any, variant: str, snr: float, rep: int, seed: int,
    metrics: dict[str, Any], signature: str,
) -> dict[str, Any]:
    return {
        "phase": phase,
        "case": case.name,
        "group": getattr(case, "group", phase),
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
        "basis_name": "none",
        "basis_nominal_rank": 0,
        "basis_effective_rank": 0,
        "basis_projection_nmse": float("nan"),
        "final_channel_nmse": float("nan"),
        "anchor": "none",
        "contract_signature": signature,
    }


def selection_variants(specs: Sequence[LocalizedDelayDopplerSpec]) -> list[str]:
    return [f"localized_residual_{item.name}" for item in specs] + [
        "global_full_oracle", "ls_lmmse", "perfect_csi_lmmse"
    ]


def holdout_variants() -> list[str]:
    return [
        "localized_residual_oracle",
        "localized_full_oracle",
        "global_full_oracle",
        "pilot_only_current",
        "ls_estimate_repaired",
        "true_channel_repaired",
        "ls_lmmse",
        "perfect_csi_lmmse",
    ]


def make_basis(context: Any, spec: LocalizedDelayDopplerSpec, device: torch.device) -> tuple[torch.Tensor, dict[str, Any]]:
    return localized_delay_doppler_features(
        num_symbols=int(context.grid.num_ofdm_symbols),
        num_subcarriers=int(context.grid.num_effective_subcarriers),
        subcarrier_spacing_khz=float(context.case.subcarrier_spacing_khz),
        spec=spec,
        device=device,
    )


def global_oracle(state: Any, batch: Any, ridge: float) -> tuple[PosteriorOutput, dict[str, float]]:
    return oracle_projection_posterior(
        truth=batch.h,
        features=state.features.to(batch.h.dtype),
        anchor=None,
        effective_noise=state.posterior.effective_noise,
        ridge=float(ridge),
    )


def run_selection(
    config: dict[str, Any], config_path: Path, device: torch.device, pre: dict[str, Any],
    specs: Sequence[LocalizedDelayDopplerSpec], *, deadline: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    variants = selection_variants(specs)
    section = config["selection"]
    payload = {
        "version": VERSION,
        "phase": "selection_4prb_8prb",
        "config_sha256": sha256_file(config_path),
        "source_sha256": source_hashes(SOURCE_FILES),
        "checkpoint_sha256": pre["checkpoint_sha256"],
        "cases": section["cases"],
        "ebno_db": section["ebno_db"],
        "repetitions": section["repetitions"],
        "specs": [item.as_dict() for item in specs],
        "variants": variants,
        "holdout_used_for_selection": False,
        "training_required": False,
    }
    contract = ensure_contract(SELECTION_CONTRACT, payload)
    done = completed_batches(RAW_SELECTION, variants)
    ridge = float(config["projection_ridge"])
    for case_index, raw_case in enumerate(section["cases"]):
        case, context = make_case_context(raw_case, device)
        bridge = build_loaded_bridge(case, context, operator_seed=int(config["operator_seed"]))
        detector = make_repaired_detector(int(context.grid.bits_per_symbol)).to(device)
        ls_receiver = standard_receiver(context, perfect_csi=False, return_crc=True)
        perfect_receiver = standard_receiver(context, perfect_csi=True, return_crc=True)
        basis_cache = {item.name: make_basis(context, item, device) for item in specs}
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
                    state, _, _ = pilot_state_and_reference(bridge, batch)
                    ls_posterior, _ = ls_posterior_from_receiver(ls_receiver, context, batch)
                    outputs: dict[str, dict[str, Any]] = {}
                    spec_lookup: dict[str, LocalizedDelayDopplerSpec | None] = {}
                    for spec in specs:
                        features, basis_report = basis_cache[spec.name]
                        posterior, diagnostics = oracle_projection_posterior(
                            truth=batch.h,
                            features=features,
                            anchor=ls_posterior,
                            effective_noise=state.posterior.effective_noise,
                            ridge=ridge,
                        )
                        diagnostics.update({
                            "effective_rank": basis_report["effective_rank"],
                            "nominal_rank": basis_report["nominal_rank"],
                        })
                        name = f"localized_residual_{spec.name}"
                        outputs[name] = output_from_posterior(bridge, detector, batch, posterior, diagnostics)
                        spec_lookup[name] = spec
                    global_posterior, global_diag = global_oracle(state, batch, ridge)
                    outputs["global_full_oracle"] = output_from_posterior(
                        bridge, detector, batch, global_posterior, global_diag
                    )
                    spec_lookup["global_full_oracle"] = None
                    decoded = decode_outputs(
                        context, batch, outputs,
                        bp_iterations=int(config["bp_iterations"]), device=device,
                    )
                    ls_metrics = run_standard_receiver(ls_receiver, batch, batch.information_bits, perfect_csi=False)
                    perfect_metrics = run_standard_receiver(perfect_receiver, batch, batch.information_bits, perfect_csi=True)
                rows = [
                    custom_row(
                        phase="selection", case=case, variant=name, snr=snr,
                        rep=rep, seed=seed, output=output, batch=batch,
                        decoded=decoded[name], signature=contract["signature"],
                        spec=spec_lookup[name],
                    )
                    for name, output in outputs.items()
                ]
                rows.extend([
                    standard_row(phase="selection", case=case, variant="ls_lmmse", snr=snr, rep=rep, seed=seed, metrics=ls_metrics, signature=contract["signature"]),
                    standard_row(phase="selection", case=case, variant="perfect_csi_lmmse", snr=snr, rep=rep, seed=seed, metrics=perfect_metrics, signature=contract["signature"]),
                ])
                if {row["variant"] for row in rows} != set(variants):
                    raise RuntimeError("Selection variant set mismatch")
                append_rows_atomic(RAW_SELECTION, rows)
                done.add(key)
                print(json.dumps({
                    "phase": "selection", "case": case.name, "ebno_db": snr,
                    "rep": rep, "rows_committed": len(rows),
                    "completed_rows": len(done) * len(variants),
                    "expected_rows": EXPECTED_SELECTION_ROWS,
                }), flush=True)
                if time.time() >= deadline:
                    frame = pd.read_csv(RAW_SELECTION)
                    return frame, {"complete": False, "rows": len(frame), "expected_rows": EXPECTED_SELECTION_ROWS, "contract": contract}
        del bridge, detector, ls_receiver, perfect_receiver
        if device.type == "cuda":
            torch.cuda.empty_cache()
    frame = pd.read_csv(RAW_SELECTION)
    unique = len(frame.drop_duplicates(["case", "variant", "ebno_db", "rep"]))
    if len(frame) != EXPECTED_SELECTION_ROWS or unique != EXPECTED_SELECTION_ROWS:
        raise RuntimeError(f"Selection incomplete: rows={len(frame)}, unique={unique}")
    return frame, {"complete": True, "rows": len(frame), "unique_rows": unique, "expected_rows": EXPECTED_SELECTION_ROWS, "contract": contract}


def choose_winner(frame: pd.DataFrame, specs: Sequence[LocalizedDelayDopplerSpec]) -> tuple[LocalizedDelayDopplerSpec, list[dict[str, Any]]]:
    ranking: list[dict[str, Any]] = []
    for spec in specs:
        variant = f"localized_residual_{spec.name}"
        sub = frame[frame["variant"] == variant]
        condition = sub.groupby(["case", "ebno_db"])["tbler"].mean()
        objective = float(sub["tbler"].mean()) + 0.25 * float(condition.max()) + 1e-5 * float(spec.nominal_rank)
        ranking.append({
            "basis": spec.name,
            "variant": variant,
            "selection_objective": objective,
            "selection_tbler": float(sub["tbler"].mean()),
            "selection_worst_condition_tbler": float(condition.max()),
            "nominal_rank": int(spec.nominal_rank),
            "effective_rank_mean": float(sub["basis_effective_rank"].mean()),
            "projection_nmse_mean": float(sub["basis_projection_nmse"].mean()),
            "final_channel_nmse_mean": float(sub["final_channel_nmse"].mean()),
        })
    ranking.sort(key=lambda item: (item["selection_objective"], item["nominal_rank"], item["basis"]))
    lookup = {item.name: item for item in specs}
    return lookup[str(ranking[0]["basis"])], ranking


def run_holdout(
    config: dict[str, Any], config_path: Path, device: torch.device, pre: dict[str, Any],
    winner: LocalizedDelayDopplerSpec, selection_contract: dict[str, Any], *, deadline: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    variants = holdout_variants()
    section = config["holdout"]
    payload = {
        "version": VERSION,
        "phase": "fresh_12prb_holdout",
        "config_sha256": sha256_file(config_path),
        "source_sha256": source_hashes(SOURCE_FILES),
        "checkpoint_sha256": pre["checkpoint_sha256"],
        "winner": winner.as_dict(),
        "selection_contract_signature": selection_contract["signature"],
        "selection_csv_sha256": sha256_file(RAW_SELECTION),
        "cases": section["cases"],
        "ebno_db": section["ebno_db"],
        "repetitions": section["repetitions"],
        "variants": variants,
        "holdout_used_for_selection": False,
        "training_required": False,
    }
    contract = ensure_contract(HOLDOUT_CONTRACT, payload)
    done = completed_batches(RAW_HOLDOUT, variants)
    ridge = float(config["projection_ridge"])
    for case_index, raw_case in enumerate(section["cases"]):
        case, context = make_case_context(raw_case, device)
        bridge = build_loaded_bridge(case, context, operator_seed=int(config["operator_seed"]))
        detector = make_repaired_detector(int(context.grid.bits_per_symbol)).to(device)
        ls_detector = make_repaired_detector(int(context.grid.bits_per_symbol)).to(device)
        ls_receiver = standard_receiver(context, perfect_csi=False, return_crc=True)
        perfect_receiver = standard_receiver(context, perfect_csi=True, return_crc=True)
        local_features, local_basis_report = make_basis(context, winner, device)
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
                    initial["posterior_metrics"] = posterior_batch_metrics(state.posterior, batch.h, batch.data_idx)
                    initial["ceiling_diagnostics"] = {}
                    ls_posterior, _ = ls_posterior_from_receiver(ls_receiver, context, batch)
                    residual_posterior, residual_diag = oracle_projection_posterior(
                        truth=batch.h, features=local_features, anchor=ls_posterior,
                        effective_noise=state.posterior.effective_noise, ridge=ridge,
                    )
                    residual_diag.update({
                        "effective_rank": local_basis_report["effective_rank"],
                        "nominal_rank": local_basis_report["nominal_rank"],
                    })
                    full_posterior, full_diag = oracle_projection_posterior(
                        truth=batch.h, features=local_features, anchor=None,
                        effective_noise=state.posterior.effective_noise, ridge=ridge,
                    )
                    full_diag.update({
                        "effective_rank": local_basis_report["effective_rank"],
                        "nominal_rank": local_basis_report["nominal_rank"],
                    })
                    global_posterior, global_diag = global_oracle(state, batch, ridge)
                    outputs = {
                        "localized_residual_oracle": output_from_posterior(bridge, detector, batch, residual_posterior, residual_diag),
                        "localized_full_oracle": output_from_posterior(bridge, detector, batch, full_posterior, full_diag),
                        "global_full_oracle": output_from_posterior(bridge, detector, batch, global_posterior, global_diag),
                        "pilot_only_current": initial,
                        "ls_estimate_repaired": ls_repaired_forward(ls_receiver, context, ls_detector, batch),
                        "true_channel_repaired": output_from_posterior(bridge, detector, batch, true_channel_posterior(batch), {}),
                    }
                    decoded = decode_outputs(
                        context, batch, outputs,
                        bp_iterations=int(config["bp_iterations"]), device=device,
                    )
                    ls_metrics = run_standard_receiver(ls_receiver, batch, batch.information_bits, perfect_csi=False)
                    perfect_metrics = run_standard_receiver(perfect_receiver, batch, batch.information_bits, perfect_csi=True)
                spec_lookup = {
                    "localized_residual_oracle": winner,
                    "localized_full_oracle": winner,
                    "global_full_oracle": None,
                    "pilot_only_current": None,
                    "ls_estimate_repaired": None,
                    "true_channel_repaired": None,
                }
                rows = [
                    custom_row(
                        phase="holdout", case=case, variant=name, snr=snr,
                        rep=rep, seed=seed, output=output, batch=batch,
                        decoded=decoded[name], signature=contract["signature"],
                        spec=spec_lookup[name],
                    )
                    for name, output in outputs.items()
                ]
                rows.extend([
                    standard_row(phase="holdout", case=case, variant="ls_lmmse", snr=snr, rep=rep, seed=seed, metrics=ls_metrics, signature=contract["signature"]),
                    standard_row(phase="holdout", case=case, variant="perfect_csi_lmmse", snr=snr, rep=rep, seed=seed, metrics=perfect_metrics, signature=contract["signature"]),
                ])
                if {row["variant"] for row in rows} != set(variants):
                    raise RuntimeError("Holdout variant set mismatch")
                append_rows_atomic(RAW_HOLDOUT, rows)
                done.add(key)
                print(json.dumps({
                    "phase": "holdout", "case": case.name, "ebno_db": snr,
                    "rep": rep, "rows_committed": len(rows),
                    "completed_rows": len(done) * len(variants),
                    "expected_rows": EXPECTED_HOLDOUT_ROWS,
                }), flush=True)
                if time.time() >= deadline:
                    frame = pd.read_csv(RAW_HOLDOUT)
                    return frame, {"complete": False, "rows": len(frame), "expected_rows": EXPECTED_HOLDOUT_ROWS, "contract": contract}
        del bridge, detector, ls_detector, ls_receiver, perfect_receiver
        if device.type == "cuda":
            torch.cuda.empty_cache()
    frame = pd.read_csv(RAW_HOLDOUT)
    unique = len(frame.drop_duplicates(["case", "variant", "ebno_db", "rep"]))
    if len(frame) != EXPECTED_HOLDOUT_ROWS or unique != EXPECTED_HOLDOUT_ROWS:
        raise RuntimeError(f"Holdout incomplete: rows={len(frame)}, unique={unique}")
    return frame, {"complete": True, "rows": len(frame), "unique_rows": unique, "expected_rows": EXPECTED_HOLDOUT_ROWS, "contract": contract}


def paired_delta(frame: pd.DataFrame, reference: str, comparator: str, metric: str) -> dict[str, float]:
    keys = ["case", "ebno_db", "rep", "eval_seed"]
    a = frame[frame["variant"] == reference]
    b = frame[frame["variant"] == comparator]
    merged = a.merge(b, on=keys, suffixes=("_a", "_b"))
    values = (pd.to_numeric(merged[f"{metric}_a"], errors="coerce") - pd.to_numeric(merged[f"{metric}_b"], errors="coerce")).dropna()
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
    return {"pairs": int(len(values)), "mean": mean, "ci95_low": mean - half, "ci95_high": mean + half}


def mean_metric(frame: pd.DataFrame, variant: str, metric: str, *, snr: float | None = None) -> float:
    sub = frame[frame["variant"] == variant]
    if snr is not None:
        sub = sub[sub["ebno_db"] == float(snr)]
    values = pd.to_numeric(sub[metric], errors="coerce").dropna()
    return float(values.mean()) if len(values) else float("nan")


def classify(holdout: pd.DataFrame) -> dict[str, Any]:
    comparisons = {
        "localized_minus_ls_tbler": paired_delta(holdout, "localized_residual_oracle", "ls_lmmse", "tbler"),
        "localized_minus_global_tbler": paired_delta(holdout, "localized_residual_oracle", "global_full_oracle", "tbler"),
        "localized_full_minus_residual_tbler": paired_delta(holdout, "localized_full_oracle", "localized_residual_oracle", "tbler"),
        "global_minus_ls_tbler": paired_delta(holdout, "global_full_oracle", "ls_lmmse", "tbler"),
        "ls_repaired_minus_ls_tbler": paired_delta(holdout, "ls_estimate_repaired", "ls_lmmse", "tbler"),
        "true_minus_perfect_tbler": paired_delta(holdout, "true_channel_repaired", "perfect_csi_lmmse", "tbler"),
    }
    local_tbler = mean_metric(holdout, "localized_residual_oracle", "tbler")
    ls_tbler = mean_metric(holdout, "ls_lmmse", "tbler")
    local_nmse = mean_metric(holdout, "localized_residual_oracle", "final_channel_nmse")
    local_projection_nmse = mean_metric(holdout, "localized_residual_oracle", "basis_projection_nmse")
    local14 = mean_metric(holdout, "localized_residual_oracle", "tbler", snr=14.0)
    local10 = mean_metric(holdout, "localized_residual_oracle", "tbler", snr=10.0)
    ls_match = abs(comparisons["ls_repaired_minus_ls_tbler"]["mean"]) <= 0.005
    true_match = abs(comparisons["true_minus_perfect_tbler"]["mean"]) <= 0.005
    beats_ls = comparisons["localized_minus_ls_tbler"]["ci95_high"] < 0.0
    possible_beats_ls = comparisons["localized_minus_ls_tbler"]["mean"] < 0.0
    improves_global = comparisons["localized_minus_global_tbler"]["ci95_high"] < 0.0
    checks = {
        "ls_factorized_matches_standard": ls_match,
        "true_channel_repaired_matches_perfect": true_match,
        "localized_oracle_beats_global_basis": improves_global,
        "localized_oracle_mean_beats_ls": possible_beats_ls,
        "localized_oracle_statistically_beats_ls": beats_ls,
        "localized_final_channel_nmse_le_0p02": local_nmse <= 0.02,
        "no_12prb_high_snr_reversal": local14 <= local10 + 0.01,
    }
    if not ls_match or not true_match:
        classification = "GATE1_LOCALIZED_CEILING_INTERFACE_REPAIR_REQUIRED"
        next_action = "REPAIR_CONTROL_PATH_BEFORE_DECISION"
    elif beats_ls:
        classification = "GATE1_LOCALIZED_CEILING_BEATS_LS"
        next_action = "TRAIN_ONE_LS_ANCHORED_LOCALIZED_POSTERIOR_THEN_RUN_FINAL_HOLDOUT"
    elif possible_beats_ls and improves_global and local_nmse <= 0.02:
        classification = "GATE1_LOCALIZED_CEILING_POSSIBLY_BEATS_LS"
        next_action = "TRAIN_ONE_LS_ANCHORED_LOCALIZED_POSTERIOR_WITH_HARD_FINAL_STOP"
    else:
        classification = "GATE1_ABANDON_BAYESROUTE_FOR_LS_BEATING"
        next_action = "STOP_ARCHITECTURE_SEARCH_AND_PIVOT_OR_REFRAME_THE_PAPER"
    return {
        "classification": classification,
        "next_action": next_action,
        "scientific_checks": checks,
        "paired_comparisons": comparisons,
        "metrics": {
            "localized_oracle_12prb_tbler": local_tbler,
            "global_oracle_12prb_tbler": mean_metric(holdout, "global_full_oracle", "tbler"),
            "pilot_current_12prb_tbler": mean_metric(holdout, "pilot_only_current", "tbler"),
            "ls_12prb_tbler": ls_tbler,
            "perfect_12prb_tbler": mean_metric(holdout, "perfect_csi_lmmse", "tbler"),
            "localized_final_channel_nmse": local_nmse,
            "localized_residual_projection_nmse": local_projection_nmse,
            "localized_10db_tbler": local10,
            "localized_14db_tbler": local14,
        },
    }


def aggregate(selection: pd.DataFrame, holdout: pd.DataFrame) -> pd.DataFrame:
    frame = pd.concat([selection, holdout], ignore_index=True)
    metrics = [
        "information_ber", "tbler", "crc_failure_rate", "coded_ber",
        "coded_bit_nll", "channel_nmse", "normalized_error_mean", "coverage95",
        "edge_density", "basis_nominal_rank", "basis_effective_rank",
        "basis_projection_nmse", "final_channel_nmse",
    ]
    return frame.groupby(["phase", "case", "group", "num_prb", "variant", "ebno_db"], dropna=False)[metrics].agg(["mean", "std", "count"]).reset_index()


def make_plots(selection: pd.DataFrame, holdout: pd.DataFrame, ranking: list[dict[str, Any]]) -> list[str]:
    out = ROOT / "outputs/plots"
    out.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    plt.figure(figsize=(8, 4.8))
    names = [item["basis"] for item in ranking]
    values = [item["selection_objective"] for item in ranking]
    plt.bar(np.arange(len(names)), values)
    plt.xticks(np.arange(len(names)), names, rotation=30, ha="right", fontsize=8)
    plt.ylabel("Selection objective")
    plt.tight_layout()
    path = out / "gate1_localized_ceiling_selection.png"
    plt.savefig(path, dpi=180); plt.close(); paths.append(str(path.relative_to(ROOT)))

    plt.figure(figsize=(7.2, 4.8))
    for variant in [
        "localized_residual_oracle", "global_full_oracle", "pilot_only_current",
        "ls_lmmse", "perfect_csi_lmmse",
    ]:
        sub = holdout[holdout["variant"] == variant]
        grouped = sub.groupby("ebno_db")["tbler"].mean().sort_index()
        plt.plot(grouped.index, grouped.values, marker="o", label=variant)
    plt.xlabel("$E_b/N_0$ (dB)")
    plt.ylabel("TBLER")
    plt.ylim(-0.01, 0.15)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=7)
    plt.tight_layout()
    path = out / "gate1_localized_ceiling_12prb_tbler.png"
    plt.savefig(path, dpi=180); plt.close(); paths.append(str(path.relative_to(ROOT)))
    return paths


def write_incomplete(selection: dict[str, Any] | None, holdout: dict[str, Any] | None, winner: str | None) -> None:
    report = {
        "version": VERSION,
        "complete": False,
        "selection": selection,
        "holdout": holdout,
        "winner": winner,
        "classification": "GATE1_NR_LOCALIZED_CEILING_INCOMPLETE",
        "next_action": "RESUBMIT_SAME_COMMAND",
        "publication_nr_ready": False,
    }
    save_json(report, REPORT_PATH); save_json(report, GATE_JSON)
    GATE_TXT.parent.mkdir(parents=True, exist_ok=True)
    GATE_TXT.write_text(
        "CLASSIFICATION: GATE1_NR_LOCALIZED_CEILING_INCOMPLETE\n"
        "NEXT_ACTION: RESUBMIT_SAME_COMMAND\nPUBLICATION_NR_READY: NO\n",
        encoding="utf-8",
    )
    print("GATE1_NR_LOCALIZED_CEILING_INCOMPLETE: RESUBMIT")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/gate1_nr_localized_ceiling.yaml")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--deadline-minutes", type=float, default=25.0)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    config_path = ROOT / args.config
    config = load_yaml(config_path)
    pre = preconditions()
    specs = [LocalizedDelayDopplerSpec.from_mapping(item) for item in config["candidates"]]
    selection_expected = len(config["selection"]["cases"]) * len(config["selection"]["ebno_db"]) * int(config["selection"]["repetitions"]) * len(selection_variants(specs))
    holdout_expected = len(config["holdout"]["cases"]) * len(config["holdout"]["ebno_db"]) * int(config["holdout"]["repetitions"]) * len(holdout_variants())
    if selection_expected != EXPECTED_SELECTION_ROWS or holdout_expected != EXPECTED_HOLDOUT_ROWS:
        raise RuntimeError((selection_expected, holdout_expected, EXPECTED_SELECTION_ROWS, EXPECTED_HOLDOUT_ROWS))
    math_report = mathematical_self_test("cpu")
    if math_report.get("passed") is not True:
        raise RuntimeError(f"Localized basis self-test failed: {math_report}")
    if args.preflight_only:
        basis_reports = []
        for prb in (4, 8, 12):
            for spec in specs:
                _, basis_report = localized_delay_doppler_features(
                    num_symbols=14,
                    num_subcarriers=12 * prb,
                    subcarrier_spacing_khz=30.0,
                    spec=spec,
                    device="cpu",
                )
                if basis_report["effective_rank"] <= 0 or not basis_report["finite"]:
                    raise RuntimeError(f"Invalid preflight basis: {basis_report}")
                basis_reports.append({
                    "prb": prb,
                    "basis": spec.name,
                    "nominal_rank": basis_report["nominal_rank"],
                    "effective_rank": basis_report["effective_rank"],
                })
        print("GATE1_NR_LOCALIZED_CEILING_PREFLIGHT_PASS")
        print("PRIOR_CLASSIFICATION", pre["classification"])
        print("LOCALIZED_DD_VERSION", LOCALIZED_DD_VERSION)
        print("CANDIDATES", len(specs))
        print("BASIS_PREFLIGHTS", len(basis_reports))
        print("MAX_NOMINAL_RANK", max(item.nominal_rank for item in specs))
        print("SELECTION_ROWS", EXPECTED_SELECTION_ROWS)
        print("HOLDOUT_ROWS", EXPECTED_HOLDOUT_ROWS)
        print("EXPECTED_ROWS", EXPECTED_ROWS)
        print("TRAINING_REQUIRED NO")
        print("HARD_ABANDON_GATE YES")
        return

    device = normalize_device(args.device)
    deadline = time.time() + 60.0 * float(args.deadline_minutes)
    selection_frame, selection_report = run_selection(
        config, config_path, device, pre, specs, deadline=deadline
    )
    if not selection_report["complete"]:
        write_incomplete(selection_report, None, None); return
    winner, ranking = choose_winner(selection_frame, specs)
    if time.time() >= deadline:
        write_incomplete(selection_report, None, winner.name); return
    holdout_frame, holdout_report = run_holdout(
        config, config_path, device, pre, winner, selection_report["contract"], deadline=deadline
    )
    if not holdout_report["complete"]:
        write_incomplete(selection_report, holdout_report, winner.name); return

    combined = pd.concat([selection_frame, holdout_frame], ignore_index=True)
    unique = len(combined.drop_duplicates(["phase", "case", "variant", "ebno_db", "rep"]))
    if len(combined) != EXPECTED_ROWS or unique != EXPECTED_ROWS:
        raise RuntimeError("Combined localized ceiling row count mismatch")
    aggregate_frame = aggregate(selection_frame, holdout_frame)
    AGGREGATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    aggregate_frame.to_csv(AGGREGATE_PATH, index=False)
    scientific = classify(holdout_frame)
    plots = make_plots(selection_frame, holdout_frame, ranking)
    software_checks = {
        "complete_rows": len(combined) == EXPECTED_ROWS,
        "unique_rows": unique == EXPECTED_ROWS,
        "selection_holdout_disjoint": set(selection_frame["eval_seed"]).isdisjoint(set(holdout_frame["eval_seed"])),
        "holdout_unused_for_selection": holdout_report["contract"].get("holdout_used_for_selection") is False,
        "all_core_metrics_finite": bool(np.isfinite(pd.to_numeric(combined["tbler"], errors="coerce")).all()),
        "rank_cap_respected": max(item.nominal_rank for item in specs) <= 128,
        "training_not_required": True,
    }
    report = {
        "version": VERSION,
        "complete": True,
        "classification": scientific["classification"],
        "next_action": scientific["next_action"],
        "publication_nr_ready": False,
        "preconditions": pre,
        "winner": winner.name,
        "winner_spec": winner.as_dict(),
        "selection_ranking": ranking,
        "selection": selection_report,
        "holdout": holdout_report,
        "evaluation": {"complete": True, "rows": EXPECTED_ROWS, "unique_rows": unique, "expected_rows": EXPECTED_ROWS},
        "software_checks": software_checks,
        "aggregate_csv": str(AGGREGATE_PATH.relative_to(ROOT)),
        "plots": plots,
        "environment": {
            "device": str(device),
            "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "torch": torch.__version__,
            "python": platform.python_version(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        },
        **scientific,
    }
    save_json(report, REPORT_PATH); save_json(report, GATE_JSON)
    lines = [
        *(f"{name}: {'PASS' if value else 'FAIL'}" for name, value in software_checks.items()),
        *(f"{name}: {'PASS' if value else 'FAIL'}" for name, value in report["scientific_checks"].items()),
        f"WINNER: {winner.name}",
        f"CLASSIFICATION: {report['classification']}",
        f"NEXT_ACTION: {report['next_action']}",
        "PUBLICATION_NR_READY: NO",
    ]
    GATE_TXT.parent.mkdir(parents=True, exist_ok=True)
    GATE_TXT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
