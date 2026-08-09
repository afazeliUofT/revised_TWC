#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
import copy
import hashlib
import json
from types import SimpleNamespace
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import optuna
from optuna.samplers import RandomSampler
from optuna.trial import TrialState
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bayesroute.config import (
    AttrDict,
    capture_rng_state,
    get_device,
    load_config,
    restore_rng_state,
    save_json,
    set_seed,
)
from bayesroute.losses import bit_bce_loss, bit_metrics, channel_marginal_nll
from bayesroute.models import BayesRouteReceiver
from bayesroute.simulator import UplinkToySimulator

SEARCH_SPACE_VERSION = "gate0_v2_4_search_v1"
DESIGN_NAME = "space_filling_12"
RANK_CANDIDATES = [8, 12, 16, 24]
SEARCH_GRID = {
    "rank": RANK_CANDIDATES,
    "detector_iterations": [2, 3, 4],
    "lr": [5e-4, 1e-3, 2e-3, 4e-3],
    "channel_loss_weight": [0.0, 0.05, 0.10, 0.15],
}
PARAMETER_NAMES = tuple(SEARCH_GRID)

# A deterministic 12-point space-filling screening design. Every pair of
# hyperparameter columns has twelve distinct level pairs. This is not claimed
# to be a full factorial design or a statistically orthogonal array.
_SPACE_FILLING_INDEX_DESIGN = [
    (0, 0, 2, 0), (1, 0, 0, 1), (2, 0, 1, 3), (3, 0, 3, 2),
    (0, 1, 1, 2), (1, 1, 3, 3), (2, 1, 2, 1), (3, 1, 0, 0),
    (0, 2, 0, 3), (1, 2, 2, 2), (2, 2, 3, 0), (3, 2, 1, 1),
]
SPACE_FILLING_TRIALS = [
    {
        "rank": SEARCH_GRID["rank"][rank_i],
        "detector_iterations": SEARCH_GRID["detector_iterations"][iter_i],
        "lr": SEARCH_GRID["lr"][lr_i],
        "channel_loss_weight": SEARCH_GRID["channel_loss_weight"][loss_i],
    }
    for rank_i, iter_i, lr_i, loss_i in _SPACE_FILLING_INDEX_DESIGN
]
OPTIMIZER_WEIGHT_DECAY = 1e-4


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _parameter_key(params: dict[str, Any]) -> str | None:
    if set(params) != set(PARAMETER_NAMES):
        return None
    normalized = {name: params[name] for name in PARAMETER_NAMES}
    return _canonical_json(normalized)


DESIGN_KEY_TO_INDEX = {
    _parameter_key(params): index
    for index, params in enumerate(SPACE_FILLING_TRIALS)
}
DESIGN_SIGNATURE = hashlib.sha256(
    _canonical_json(SPACE_FILLING_TRIALS).encode("utf-8")
).hexdigest()


def _design_index(params: dict[str, Any]) -> int | None:
    key = _parameter_key(params)
    return None if key is None else DESIGN_KEY_TO_INDEX.get(key)


def _design_report() -> dict[str, Any]:
    names = list(PARAMETER_NAMES)
    rows = [tuple(row[name] for name in names) for row in SPACE_FILLING_TRIALS]
    pair_counts: dict[str, int] = {}
    for left in range(len(names)):
        for right in range(left + 1, len(names)):
            pair_counts[f"{names[left]}__{names[right]}"] = len(
                {(row[left], row[right]) for row in rows}
            )
    levels_valid = all(
        row[name] in SEARCH_GRID[name]
        for row in SPACE_FILLING_TRIALS
        for name in names
    )
    passed = bool(
        len(SPACE_FILLING_TRIALS) == 12
        and len(set(rows)) == 12
        and levels_valid
        and all(count == 12 for count in pair_counts.values())
        and len(DESIGN_KEY_TO_INDEX) == 12
    )
    return {
        "name": DESIGN_NAME,
        "passed": passed,
        "rows": len(SPACE_FILLING_TRIALS),
        "unique_rows": len(set(rows)),
        "pairwise_distinct_level_pairs": pair_counts,
        "signature": DESIGN_SIGNATURE,
        "full_factorial_size": 4 * 3 * 4 * 4,
        "claim_scope": "deterministic space-filling screening design; not a full factorial or orthogonal array",
    }


def _search_contract(cfg: AttrDict) -> dict[str, Any]:
    opt_cfg = cfg.optuna
    return {
        "search_space_version": SEARCH_SPACE_VERSION,
        "package_revision": cfg.get("package_revision"),
        "base_seed": int(cfg.seed),
        "system": cfg.system.to_dict(),
        "fixed_model": {
            "use_uncertainty": bool(cfg.model.get("use_uncertainty", True)),
            "edge_mass": float(cfg.model.get("edge_mass", 1.0)),
            "operator_seed": int(cfg.model.get("operator_seed", int(cfg.seed) + 1009)),
            "operator_bank_rank": int(cfg.model.get("operator_bank_rank", cfg.model.rank)),
        },
        "search_space": {key: list(values) for key, values in SEARCH_GRID.items()},
        "design": {
            "name": DESIGN_NAME,
            "report": _design_report(),
            "trials": SPACE_FILLING_TRIALS,
        },
        "training": {
            "batch_size": int(cfg.training.batch_size),
            "grad_clip": float(cfg.training.grad_clip),
            "save_every": int(cfg.training.save_every),
            "weight_decay": OPTIMIZER_WEIGHT_DECAY,
        },
        "evaluation": {
            "snr_grid_db": [float(x) for x in cfg.evaluation.snr_grid_db],
            "batch_size": int(cfg.evaluation.batch_size),
        },
        "optuna": {
            "train_steps_per_trial": int(opt_cfg.train_steps_per_trial),
            "eval_batches_per_snr": int(opt_cfg.eval_batches_per_snr),
            "validation_seed": int(opt_cfg.validation_seed),
            "training_seed": int(opt_cfg.training_seed),
            "sampler_seed": int(opt_cfg.sampler_seed),
            "max_failed_attempts_per_design": int(
                opt_cfg.get("max_failed_attempts_per_design", 2)
            ),
        },
    }


def _contract_signature(cfg: AttrDict) -> str:
    return hashlib.sha256(_canonical_json(_search_contract(cfg)).encode("utf-8")).hexdigest()


def _trial_cache_key(contract_signature: str, params: dict[str, Any]) -> str:
    signature = {"contract_signature": contract_signature, "params": params}
    return hashlib.sha256(_canonical_json(signature).encode("utf-8")).hexdigest()[:24]


def _atomic_torch_save(obj: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)


def _fixed_bit_nll(
    model: BayesRouteReceiver,
    simulator: UplinkToySimulator,
    *,
    seed: int,
    snr_grid_db: list[float],
    batches_per_snr: int,
    batch_size: int,
) -> tuple[float, float]:
    rng_state = capture_rng_state()
    was_training = model.training
    values: list[float] = []
    edge_densities: list[float] = []
    try:
        set_seed(int(seed))
        model.eval()
        with torch.no_grad():
            for snr in snr_grid_db:
                for _ in range(int(batches_per_snr)):
                    batch = simulator.sample(batch_size=int(batch_size), snr_db=float(snr))
                    out = model(batch)
                    values.append(bit_metrics(out["bit_logits"], batch.data_bits)["bit_nll"])
                    edge_densities.append(float(out["edge_density"]))
    finally:
        restore_rng_state(rng_state)
        model.train(was_training)
    if not values:
        raise RuntimeError("Validation stream produced no values")
    return float(sum(values) / len(values)), float(sum(edge_densities) / len(edge_densities))


def _params_from_trial(trial: Any) -> dict[str, Any]:
    fixed = dict(getattr(trial, "system_attrs", {}).get("fixed_params", {}))
    return fixed if fixed else dict(getattr(trial, "params", {}))


def _analyze_trial_collection(
    trials: Iterable[Any],
    *,
    contract_signature: str,
    target_complete: int,
    max_failed_attempts_per_design: int,
) -> tuple[dict[str, Any], dict[int, Any]]:
    required = set(range(int(target_complete)))
    complete_by_index: dict[int, Any] = {}
    waiting: set[int] = set()
    running: set[int] = set()
    failed_counts: Counter[int] = Counter()
    attempt_counts: Counter[int] = Counter()
    unexpected: list[int] = []
    duplicate_complete: set[int] = set()

    for trial in trials:
        params = _params_from_trial(trial)
        index = _design_index(params)
        trial_number = int(getattr(trial, "number", -1))
        trial_contract = getattr(trial, "user_attrs", {}).get("contract_signature")
        if index is None or trial_contract != contract_signature:
            unexpected.append(trial_number)
            continue
        if index not in required:
            # A valid design point outside a deliberately smaller target is ignored.
            continue
        attempt_counts[index] += 1
        state = getattr(trial, "state")
        if state == TrialState.COMPLETE:
            current = complete_by_index.get(index)
            value = getattr(trial, "value", None)
            if value is None:
                unexpected.append(trial_number)
                continue
            if current is None:
                complete_by_index[index] = trial
            else:
                duplicate_complete.add(index)
                if float(value) < float(current.value):
                    complete_by_index[index] = trial
        elif state == TrialState.WAITING:
            waiting.add(index)
        elif state == TrialState.RUNNING:
            running.add(index)
        elif state == TrialState.FAIL:
            failed_counts[index] += 1
        elif state == TrialState.PRUNED:
            # NopPruner is used. A PRUNED state is therefore unexpected.
            unexpected.append(trial_number)

    completed = set(complete_by_index)
    missing = required - completed
    enqueue_candidates = missing - waiting
    blocked = {
        index
        for index in enqueue_candidates
        if failed_counts[index] >= int(max_failed_attempts_per_design)
    }
    report = {
        "required_design_indices": sorted(required),
        "completed_design_indices": sorted(completed),
        "missing_design_indices": sorted(missing),
        "waiting_design_indices": sorted(waiting),
        "stale_running_design_indices": sorted(running & missing),
        "enqueue_candidate_design_indices": sorted(enqueue_candidates - blocked),
        "blocked_failed_design_indices": sorted(blocked),
        "failed_attempts_by_design": {
            str(index): int(count) for index, count in sorted(failed_counts.items())
        },
        "attempt_counts_by_design": {
            str(index): int(count) for index, count in sorted(attempt_counts.items())
        },
        "complete_trial_numbers_by_design": {
            str(index): int(trial.number)
            for index, trial in sorted(complete_by_index.items())
        },
        "duplicate_complete_design_indices": sorted(duplicate_complete),
        "unexpected_trial_numbers": sorted(set(unexpected)),
        "all_required_design_points_complete": bool(completed == required),
    }
    return report, complete_by_index


def _workflow_self_test() -> dict[str, Any]:
    contract = "self-test-contract"

    def fake(number: int, state: TrialState, index: int, value: float | None = None):
        params = dict(SPACE_FILLING_TRIALS[index])
        return SimpleNamespace(
            number=number,
            state=state,
            params=params if state != TrialState.WAITING else {},
            system_attrs={"fixed_params": params} if state == TrialState.WAITING else {},
            user_attrs={"contract_signature": contract},
            value=value,
        )

    trials = [
        fake(0, TrialState.COMPLETE, 0, 0.5),
        fake(1, TrialState.FAIL, 1),
        fake(2, TrialState.RUNNING, 2),
        fake(3, TrialState.WAITING, 3),
    ]
    first, _ = _analyze_trial_collection(
        trials,
        contract_signature=contract,
        target_complete=12,
        max_failed_attempts_per_design=2,
    )
    trials.append(fake(4, TrialState.FAIL, 1))
    second, _ = _analyze_trial_collection(
        trials,
        contract_signature=contract,
        target_complete=12,
        max_failed_attempts_per_design=2,
    )
    # Exercise Optuna's real queue semantics as well. In particular, a FAIL
    # and a stale RUNNING trial must be re-enqueued even though Optuna's
    # skip_if_exists=True would suppress both states.
    integration_contract = "integration-contract"
    previous_verbosity = optuna.logging.get_verbosity()
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        sampler=RandomSampler(seed=7),
        pruner=optuna.pruners.NopPruner(),
    )
    initial_events = _enqueue_missing_design_trials(
        study,
        contract_signature=integration_contract,
        target_complete=12,
        max_failed_attempts_per_design=2,
    )

    def populate(trial: optuna.Trial) -> None:
        for name, values in SEARCH_GRID.items():
            trial.suggest_categorical(name, values)

    complete_trial = study.ask()
    populate(complete_trial)
    study.tell(complete_trial, 0.5)
    failed_trial = study.ask()
    populate(failed_trial)
    study.tell(failed_trial, state=TrialState.FAIL)
    running_trial = study.ask()
    populate(running_trial)
    integration_before, _ = _analyze_trial_collection(
        study.trials,
        contract_signature=integration_contract,
        target_complete=12,
        max_failed_attempts_per_design=2,
    )
    recovery_events = _enqueue_missing_design_trials(
        study,
        contract_signature=integration_contract,
        target_complete=12,
        max_failed_attempts_per_design=2,
    )
    recovered_indices = {event["design_index"] for event in recovery_events}
    optuna.logging.set_verbosity(previous_verbosity)
    integration_passed = bool(
        len(initial_events) == 12
        and integration_before["completed_design_indices"] == [0]
        and integration_before["failed_attempts_by_design"] == {"1": 1}
        and integration_before["stale_running_design_indices"] == [2]
        and recovered_indices == {1, 2}
        and sum(t.state == TrialState.WAITING for t in study.trials) == 11
    )

    passed = bool(
        first["completed_design_indices"] == [0]
        and 1 in first["enqueue_candidate_design_indices"]
        and 2 in first["enqueue_candidate_design_indices"]
        and 3 not in first["enqueue_candidate_design_indices"]
        and first["stale_running_design_indices"] == [2]
        and second["blocked_failed_design_indices"] == [1]
        and 1 not in second["enqueue_candidate_design_indices"]
        and not first["unexpected_trial_numbers"]
        and _design_report()["passed"]
        and integration_passed
    )
    return {
        "passed": passed,
        "design_report": _design_report(),
        "single_failure_requeued": 1 in first["enqueue_candidate_design_indices"],
        "stale_running_requeued": 2 in first["enqueue_candidate_design_indices"],
        "waiting_not_duplicated": 3 not in first["enqueue_candidate_design_indices"],
        "retry_limit_blocks_repeated_failure": second["blocked_failed_design_indices"] == [1],
        "actual_optuna_queue_integration_passed": integration_passed,
        "actual_optuna_recovered_design_indices": sorted(recovered_indices),
    }


def _preflight_report(cfg: AttrDict) -> dict[str, Any]:
    design = _design_report()
    workflow = _workflow_self_test()
    passed = bool(
        str(cfg.get("package_revision", "")) == "gate0_v2_4_20260809"
        and str(cfg.optuna.get("design", "")) == DESIGN_NAME
        and int(cfg.optuna.n_trials) == len(SPACE_FILLING_TRIALS)
        and int(cfg.model.get("operator_bank_rank", cfg.model.rank)) >= max(RANK_CANDIDATES)
        and int(cfg.optuna.get("max_failed_attempts_per_design", 0)) >= 1
        and design["passed"]
        and workflow["passed"]
    )
    return {
        "passed": passed,
        "package_revision": str(cfg.get("package_revision", "unknown")),
        "search_space_version": SEARCH_SPACE_VERSION,
        "contract_signature": _contract_signature(cfg),
        "design_report": design,
        "workflow_self_test": workflow,
    }


def _bind_study_contract(
    study: optuna.Study,
    *,
    cfg: AttrDict,
    contract_signature: str,
) -> None:
    existing = study.user_attrs.get("contract_signature")
    if existing is None:
        if study.trials:
            raise RuntimeError(
                "Existing Optuna study has trials but no contract signature. "
                "Move or delete the stale outputs/optuna directory."
            )
        study.set_user_attr("contract_signature", contract_signature)
        study.set_user_attr("package_revision", str(cfg.get("package_revision", "unknown")))
        study.set_user_attr("search_space_version", SEARCH_SPACE_VERSION)
        study.set_user_attr("design_name", DESIGN_NAME)
        study.set_user_attr("design_signature", DESIGN_SIGNATURE)
        return
    if str(existing) != contract_signature:
        raise RuntimeError(
            "Optuna study/search-contract mismatch. Move or delete the stale "
            "outputs/optuna directory before starting a new search."
        )
    required_attrs = {
        "package_revision": str(cfg.get("package_revision", "unknown")),
        "search_space_version": SEARCH_SPACE_VERSION,
        "design_name": DESIGN_NAME,
        "design_signature": DESIGN_SIGNATURE,
    }
    for name, expected in required_attrs.items():
        if study.user_attrs.get(name) != expected:
            raise RuntimeError(f"Optuna study attribute mismatch: {name}")


def _enqueue_missing_design_trials(
    study: optuna.Study,
    *,
    contract_signature: str,
    target_complete: int,
    max_failed_attempts_per_design: int,
) -> list[dict[str, Any]]:
    state, _ = _analyze_trial_collection(
        study.trials,
        contract_signature=contract_signature,
        target_complete=target_complete,
        max_failed_attempts_per_design=max_failed_attempts_per_design,
    )
    if state["unexpected_trial_numbers"]:
        raise RuntimeError(
            f"Unexpected Optuna trials: {state['unexpected_trial_numbers']}"
        )
    events: list[dict[str, Any]] = []
    failed = {int(k): int(v) for k, v in state["failed_attempts_by_design"].items()}
    stale = set(state["stale_running_design_indices"])
    for index in state["enqueue_candidate_design_indices"]:
        if failed.get(index, 0) > 0:
            reason = "retry_failed_trial"
        elif index in stale:
            reason = "recover_stale_running_trial"
        else:
            reason = "initial_design_enqueue"
        study.enqueue_trial(
            dict(SPACE_FILLING_TRIALS[index]),
            user_attrs={
                "contract_signature": contract_signature,
                "space_filling_design": True,
                "design_name": DESIGN_NAME,
                "design_signature": DESIGN_SIGNATURE,
                "design_index": int(index),
                "enqueue_reason": reason,
            },
            # Deliberately False: Optuna's skip_if_exists checks all states,
            # including FAIL and stale RUNNING trials. We already prevent
            # duplicate WAITING or COMPLETE design points above.
            skip_if_exists=False,
        )
        events.append({"design_index": int(index), "reason": reason})
    return events


def _write_status(
    study: optuna.Study,
    outdir: Path,
    target_complete: int,
    *,
    cfg: AttrDict,
    contract_signature: str,
    max_failed_attempts_per_design: int,
    recovery_events: list[dict[str, Any]] | None = None,
    stop_reason: str | None = None,
) -> dict[str, Any]:
    design_state, complete_by_index = _analyze_trial_collection(
        study.trials,
        contract_signature=contract_signature,
        target_complete=target_complete,
        max_failed_attempts_per_design=max_failed_attempts_per_design,
    )
    states = {state.name: 0 for state in TrialState}
    for trial in study.trials:
        states[trial.state.name] = states.get(trial.state.name, 0) + 1

    status: dict[str, Any] = {
        "package_revision": str(cfg.get("package_revision", "unknown")),
        "search_space_version": SEARCH_SPACE_VERSION,
        "contract_signature": contract_signature,
        "study_name": study.study_name,
        "objective_metric": "fixed_validation_bit_nll",
        "pruning_metric": "none_all_required_design_points_are_run",
        "sampler": "Optuna RandomSampler used only as a guarded fallback; all trials are explicitly enqueued",
        "sampler_seed": int(cfg.optuna.sampler_seed),
        "common_training_seed": int(cfg.optuna.training_seed),
        "design_name": DESIGN_NAME,
        "design_signature": DESIGN_SIGNATURE,
        "design_report": _design_report(),
        "fixed_edge_mass": float(cfg.model.get("edge_mass", 1.0)),
        "target_complete_trials": int(target_complete),
        "complete_trials": len(complete_by_index),
        "target_reached": bool(design_state["all_required_design_points_complete"]),
        "all_required_design_points_complete": bool(
            design_state["all_required_design_points_complete"]
        ),
        "design_state": design_state,
        "state_counts": states,
        "study_db": str(outdir / "study.db"),
        "updated": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "recovery_events_this_invocation": recovery_events or [],
        "max_failed_attempts_per_design": int(max_failed_attempts_per_design),
        "stop_reason": stop_reason,
    }

    if complete_by_index:
        best_index, best = min(
            complete_by_index.items(), key=lambda item: float(item[1].value)
        )
        status["best_value"] = float(best.value)
        status["best_trial_number"] = int(best.number)
        status["best_design_index"] = int(best_index)
        status["best_params"] = dict(best.params)
        status["best_trial_user_attrs"] = dict(best.user_attrs)

    best_path = outdir / "best_params.json"
    if status["target_reached"]:
        best_index, best = min(
            complete_by_index.items(), key=lambda item: float(item[1].value)
        )
        save_json(
            {
                "package_revision": str(cfg.get("package_revision", "unknown")),
                "search_space_version": SEARCH_SPACE_VERSION,
                "contract_signature": contract_signature,
                "objective_metric": "fixed_validation_bit_nll",
                "design_name": DESIGN_NAME,
                "design_signature": DESIGN_SIGNATURE,
                "design_report": _design_report(),
                "fixed_edge_mass": float(cfg.model.get("edge_mass", 1.0)),
                "best_value": float(best.value),
                "best_trial_number": int(best.number),
                "best_design_index": int(best_index),
                "best_params": dict(best.params),
                "best_trial_user_attrs": dict(best.user_attrs),
                "n_complete_trials": len(complete_by_index),
                "target_complete_trials": int(target_complete),
                "all_required_design_points_complete": True,
                "completed_design_indices": design_state["completed_design_indices"],
                "missing_design_indices": design_state["missing_design_indices"],
                "unexpected_trial_numbers": design_state["unexpected_trial_numbers"],
                "study_db": str(outdir / "study.db"),
            },
            best_path,
        )
    else:
        best_path.unlink(missing_ok=True)

    save_json(status, outdir / "OPTUNA_STATUS.json")
    study.trials_dataframe().to_csv(outdir / "trials.csv", index=False)
    return status


def objective(
    trial: optuna.Trial,
    base_cfg: AttrDict,
    outdir: Path,
    contract_signature: str,
) -> float:
    cfg = AttrDict(copy.deepcopy(base_cfg.to_dict()))
    cfg.model.rank = trial.suggest_categorical("rank", SEARCH_GRID["rank"])
    cfg.model.detector_iterations = trial.suggest_categorical(
        "detector_iterations", SEARCH_GRID["detector_iterations"]
    )
    cfg.training.lr = trial.suggest_categorical("lr", SEARCH_GRID["lr"])
    cfg.training.channel_loss_weight = trial.suggest_categorical(
        "channel_loss_weight", SEARCH_GRID["channel_loss_weight"]
    )
    cfg.training.steps = int(cfg.optuna.train_steps_per_trial)

    params = dict(trial.params)
    design_index = _design_index(params)
    expected_index = trial.user_attrs.get("design_index")
    if (
        design_index is None
        or expected_index is None
        or int(expected_index) != int(design_index)
        or trial.user_attrs.get("contract_signature") != contract_signature
        or trial.user_attrs.get("design_signature") != DESIGN_SIGNATURE
    ):
        trial.set_user_attr("unexpected_sampled_trial", True)
        raise RuntimeError(
            "Optuna attempted a trial outside the explicit 12-point design"
        )

    cache_key = _trial_cache_key(contract_signature, params)
    trial.set_user_attr("cache_key", cache_key)
    trial.set_user_attr("common_training_seed", int(cfg.optuna.training_seed))
    cache_dir = outdir / "trial_cache" / cache_key
    checkpoint = cache_dir / "last.pt"
    summary_path = cache_dir / "summary.json"
    cache_dir.mkdir(parents=True, exist_ok=True)

    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("contract_signature") != contract_signature:
            raise RuntimeError(f"Stale trial summary contract for cache {cache_key}")
        if dict(summary.get("params", {})) != params:
            raise RuntimeError(f"Stale trial summary parameters for cache {cache_key}")
        trial.set_user_attr("resumed_from_completed_cache", True)
        trial.set_user_attr("validation_edge_density", float(summary["edge_density"]))
        return float(summary["score"])

    set_seed(int(cfg.optuna.training_seed))
    device = get_device(cfg)
    simulator = UplinkToySimulator(cfg, device)
    model = BayesRouteReceiver(cfg, simulator.coords, simulator.pilot_idx).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(cfg.training.lr), weight_decay=OPTIMIZER_WEIGHT_DECAY
    )

    start_step = 0
    if checkpoint.exists():
        state = torch.load(checkpoint, map_location=device, weights_only=False)
        if state.get("contract_signature") != contract_signature:
            raise RuntimeError(f"Checkpoint contract mismatch for cache {cache_key}")
        if dict(state.get("params", {})) != params:
            raise RuntimeError(f"Checkpoint parameter mismatch for cache {cache_key}")
        model.load_state_dict(state["model"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        restore_rng_state(state.get("rng_state"))
        start_step = int(state["step"]) + 1
        trial.set_user_attr("resumed_from_step", start_step)

    for step in range(start_step, int(cfg.training.steps)):
        model.train()
        batch = simulator.sample(batch_size=int(cfg.training.batch_size))
        out = model(batch)
        bce = bit_bce_loss(out["bit_logits"], batch.data_bits)
        loss = bce
        if float(cfg.training.channel_loss_weight) > 0.0:
            loss = loss + float(cfg.training.channel_loss_weight) * channel_marginal_nll(
                out["posterior"].mean[..., batch.data_idx],
                out["posterior"].var_diag[:, batch.data_idx],
                batch.h[..., batch.data_idx],
            )
        if not torch.isfinite(loss):
            raise RuntimeError(
                f"Non-finite Optuna loss at trial {trial.number}, step {step}"
            )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), float(cfg.training.grad_clip)
        )
        if not torch.isfinite(grad_norm):
            raise RuntimeError(
                f"Non-finite Optuna gradient at trial {trial.number}, step {step}"
            )
        optimizer.step()

        report_due = (
            step % int(cfg.training.save_every) == 0
            or step == int(cfg.training.steps) - 1
        )
        if report_due:
            _atomic_torch_save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "step": step,
                    "params": params,
                    "config": cfg.to_dict(),
                    "contract_signature": contract_signature,
                    "design_index": int(design_index),
                    "design_signature": DESIGN_SIGNATURE,
                    "rng_state": capture_rng_state(),
                },
                checkpoint,
            )
            trial.set_user_attr("last_checkpoint_step", int(step))

    score, density = _fixed_bit_nll(
        model,
        simulator,
        seed=int(cfg.optuna.validation_seed),
        snr_grid_db=[float(x) for x in cfg.evaluation.snr_grid_db],
        batches_per_snr=int(cfg.optuna.eval_batches_per_snr),
        batch_size=int(cfg.evaluation.batch_size),
    )
    trial.set_user_attr("validation_bit_nll", score)
    trial.set_user_attr("validation_edge_density", density)
    save_json(
        {
            "package_revision": str(cfg.get("package_revision", "unknown")),
            "search_space_version": SEARCH_SPACE_VERSION,
            "contract_signature": contract_signature,
            "design_name": DESIGN_NAME,
            "design_signature": DESIGN_SIGNATURE,
            "design_index": int(design_index),
            "cache_key": cache_key,
            "last_optuna_trial_number": trial.number,
            "score": score,
            "params": params,
            "edge_density": density,
            "completed_steps": int(cfg.training.steps),
        },
        summary_path,
    )
    return score


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", default="outputs/optuna")
    parser.add_argument("--target-trials", type=int, default=None)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    preflight = _preflight_report(cfg)
    save_json(preflight, outdir / "OPTUNA_PREFLIGHT.json")
    if not preflight["passed"]:
        raise RuntimeError(f"Optuna workflow preflight failed: {preflight}")
    if args.preflight_only:
        print(json.dumps(preflight, indent=2))
        return

    target = int(args.target_trials or cfg.optuna.n_trials)
    if target <= 0 or target > len(SPACE_FILLING_TRIALS):
        raise RuntimeError(
            f"target-trials must be in 1..{len(SPACE_FILLING_TRIALS)}"
        )
    contract_signature = _contract_signature(cfg)
    storage = f"sqlite:///{outdir / 'study.db'}"
    sampler = RandomSampler(seed=int(cfg.optuna.sampler_seed))
    study = optuna.create_study(
        direction="minimize",
        study_name=str(cfg.optuna.study_name),
        storage=storage,
        load_if_exists=True,
        sampler=sampler,
        pruner=optuna.pruners.NopPruner(),
    )
    _bind_study_contract(study, cfg=cfg, contract_signature=contract_signature)

    max_failed = int(cfg.optuna.get("max_failed_attempts_per_design", 2))
    deadline = time.monotonic() + float(cfg.optuna.timeout_minutes) * 60.0
    recovery_events: list[dict[str, Any]] = []
    stop_reason: str | None = None

    while True:
        status = _write_status(
            study,
            outdir,
            target,
            cfg=cfg,
            contract_signature=contract_signature,
            max_failed_attempts_per_design=max_failed,
            recovery_events=recovery_events,
            stop_reason=stop_reason,
        )
        design_state = status["design_state"]
        if status["target_reached"]:
            stop_reason = "all_required_design_points_complete"
            break
        if design_state["unexpected_trial_numbers"]:
            raise RuntimeError(
                f"Unexpected Optuna trials: {design_state['unexpected_trial_numbers']}"
            )
        if time.monotonic() >= deadline:
            stop_reason = "timeout_before_all_design_points_completed"
            break

        events = _enqueue_missing_design_trials(
            study,
            contract_signature=contract_signature,
            target_complete=target,
            max_failed_attempts_per_design=max_failed,
        )
        recovery_events.extend(events)
        queued_state, _ = _analyze_trial_collection(
            study.trials,
            contract_signature=contract_signature,
            target_complete=target,
            max_failed_attempts_per_design=max_failed,
        )
        if not queued_state["waiting_design_indices"]:
            if queued_state["blocked_failed_design_indices"]:
                stop_reason = "failed_retry_limit_reached"
            else:
                stop_reason = "no_waiting_trial_available_for_missing_design_points"
            break

        remaining_seconds = max(1.0, deadline - time.monotonic())
        study.optimize(
            lambda trial: objective(trial, cfg, outdir, contract_signature),
            n_trials=1,
            timeout=remaining_seconds,
            callbacks=[
                lambda current_study, _: _write_status(
                    current_study,
                    outdir,
                    target,
                    cfg=cfg,
                    contract_signature=contract_signature,
                    max_failed_attempts_per_design=max_failed,
                    recovery_events=recovery_events,
                    stop_reason=None,
                )
            ],
            gc_after_trial=True,
            catch=(RuntimeError,),
        )

    final_status = _write_status(
        study,
        outdir,
        target,
        cfg=cfg,
        contract_signature=contract_signature,
        max_failed_attempts_per_design=max_failed,
        recovery_events=recovery_events,
        stop_reason=stop_reason,
    )
    print(json.dumps(final_status, indent=2))


if __name__ == "__main__":
    main()
