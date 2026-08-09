#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import optuna
from optuna.samplers import TPESampler
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

SEARCH_SPACE_VERSION = "gate0_v2_2_search_v1"
OPTIMIZER_WEIGHT_DECAY = 1e-4


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _search_contract(cfg: AttrDict) -> dict[str, Any]:
    """Fields that determine trial training, validation, and cache validity."""
    opt_cfg = cfg.optuna
    return {
        "search_space_version": SEARCH_SPACE_VERSION,
        "package_revision": cfg.get("package_revision"),
        "base_seed": int(cfg.seed),
        "system": cfg.system.to_dict(),
        "fixed_model": {
            "use_uncertainty": bool(cfg.model.get("use_uncertainty", True)),
            "operator_seed": int(cfg.model.get("operator_seed", int(cfg.seed) + 1009)),
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
            "pruning_validation_seed": int(opt_cfg.pruning_validation_seed),
            "pruning_validation_snr_db": float(opt_cfg.pruning_validation_snr_db),
            "pruning_validation_batches": int(opt_cfg.pruning_validation_batches),
            "pruning_validation_batch_size": int(opt_cfg.pruning_validation_batch_size),
            "sampler_seed": int(opt_cfg.sampler_seed),
        },
    }


def _contract_signature(cfg: AttrDict) -> str:
    return hashlib.sha256(_canonical_json(_search_contract(cfg)).encode("utf-8")).hexdigest()


def _trial_cache_key(contract_signature: str, params: dict[str, Any]) -> str:
    signature = {
        "contract_signature": contract_signature,
        "params": params,
    }
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
    """Evaluate a fixed validation stream without changing the training RNG state."""
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


def _recover_interrupted_parameter_sets(study: optuna.Study) -> list[int]:
    """Enqueue each stale RUNNING parameter set once.

    The replacement trial uses the same hash-keyed checkpoint, so the numerical
    work resumes near the interruption even though Optuna assigns a new trial ID.
    """
    complete = {
        _canonical_json(t.params)
        for t in study.trials
        if t.state == TrialState.COMPLETE and t.params
    }
    waiting = {
        _canonical_json(t.params)
        for t in study.trials
        if t.state == TrialState.WAITING and t.params
    }
    recovered: list[int] = []
    for trial in study.trials:
        if trial.state != TrialState.RUNNING or not trial.params:
            continue
        key = _canonical_json(trial.params)
        if key in complete or key in waiting:
            continue
        study.enqueue_trial(
            dict(trial.params),
            user_attrs={"resume_of_trial": int(trial.number)},
        )
        waiting.add(key)
        recovered.append(int(trial.number))
    return recovered


def _write_status(
    study: optuna.Study,
    outdir: Path,
    target_complete: int,
    *,
    cfg: AttrDict,
    contract_signature: str,
    recovered: list[int] | None = None,
) -> None:
    states = {state.name: 0 for state in TrialState}
    for trial in study.trials:
        states[trial.state.name] = states.get(trial.state.name, 0) + 1
    complete = [t for t in study.trials if t.state == TrialState.COMPLETE]
    status: dict[str, Any] = {
        "package_revision": str(cfg.get("package_revision", "unknown")),
        "search_space_version": SEARCH_SPACE_VERSION,
        "contract_signature": contract_signature,
        "study_name": study.study_name,
        "objective_metric": "fixed_validation_bit_nll",
        "pruning_metric": "fixed_validation_bit_nll",
        "sampler": "TPESampler",
        "sampler_seed": int(cfg.optuna.sampler_seed),
        "common_training_seed": int(cfg.optuna.training_seed),
        "target_complete_trials": int(target_complete),
        "state_counts": states,
        "complete_trials": len(complete),
        "target_reached": len(complete) >= int(target_complete),
        "study_db": str(outdir / "study.db"),
        "updated": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "recovery_enqueued_from_trials": recovered or [],
    }
    if complete:
        best = study.best_trial
        status["best_value"] = float(best.value)
        status["best_trial_number"] = int(best.number)
        status["best_params"] = dict(best.params)
        status["best_trial_user_attrs"] = dict(best.user_attrs)
        save_json(
            {
                "package_revision": str(cfg.get("package_revision", "unknown")),
                "search_space_version": SEARCH_SPACE_VERSION,
                "contract_signature": contract_signature,
                "objective_metric": "fixed_validation_bit_nll",
                "best_value": float(best.value),
                "best_trial_number": int(best.number),
                "best_params": dict(best.params),
                "best_trial_user_attrs": dict(best.user_attrs),
                "n_complete_trials": len(complete),
                "target_complete_trials": int(target_complete),
                "study_db": str(outdir / "study.db"),
            },
            outdir / "best_params.json",
        )
    save_json(status, outdir / "OPTUNA_STATUS.json")
    study.trials_dataframe().to_csv(outdir / "trials.csv", index=False)


def objective(
    trial: optuna.Trial,
    base_cfg: AttrDict,
    outdir: Path,
    contract_signature: str,
) -> float:
    cfg = AttrDict(copy.deepcopy(base_cfg.to_dict()))
    cfg.model.rank = trial.suggest_categorical("rank", [8, 12, 16, 24])
    cfg.model.detector_iterations = trial.suggest_int("detector_iterations", 2, 4)
    cfg.model.edge_mass = trial.suggest_float("edge_mass", 0.65, 1.0)
    cfg.training.lr = trial.suggest_float("lr", 5e-4, 4e-3, log=True)
    cfg.training.channel_loss_weight = trial.suggest_float(
        "channel_loss_weight", 0.0, 0.15
    )
    cfg.training.steps = int(cfg.optuna.train_steps_per_trial)

    params = dict(trial.params)
    cache_key = _trial_cache_key(contract_signature, params)
    trial.set_user_attr("cache_key", cache_key)
    trial.set_user_attr("contract_signature", contract_signature)
    trial.set_user_attr("common_training_seed", int(cfg.optuna.training_seed))
    cache_dir = outdir / "trial_cache" / cache_key
    checkpoint = cache_dir / "last.pt"
    summary_path = cache_dir / "summary.json"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # A job may have ended after writing the summary but before Optuna committed.
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("contract_signature") != contract_signature:
            raise RuntimeError(f"Stale trial summary contract for cache {cache_key}")
        if dict(summary.get("params", {})) != params:
            raise RuntimeError(f"Stale trial summary parameters for cache {cache_key}")
        trial.set_user_attr("resumed_from_completed_cache", True)
        trial.set_user_attr("validation_edge_density", float(summary["edge_density"]))
        return float(summary["score"])

    # All hyperparameter sets see the same ordered training stream. This removes
    # an avoidable trial-to-trial random-seed confound in the short Gate-0 search.
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
                    "rng_state": capture_rng_state(),
                },
                checkpoint,
            )
            pruning_nll, pruning_density = _fixed_bit_nll(
                model,
                simulator,
                seed=int(cfg.optuna.pruning_validation_seed),
                snr_grid_db=[float(cfg.optuna.pruning_validation_snr_db)],
                batches_per_snr=int(cfg.optuna.pruning_validation_batches),
                batch_size=int(cfg.optuna.pruning_validation_batch_size),
            )
            trial.set_user_attr("last_pruning_validation_bit_nll", pruning_nll)
            trial.set_user_attr("last_pruning_edge_density", pruning_density)
            # This metric is independent of the tuned channel-loss weight.
            trial.report(pruning_nll, step)
            if trial.should_prune():
                raise optuna.TrialPruned()

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
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", default="outputs/optuna")
    ap.add_argument("--target-trials", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    target = int(args.target_trials or cfg.optuna.n_trials)
    contract_signature = _contract_signature(cfg)
    storage = f"sqlite:///{outdir / 'study.db'}"
    sampler = TPESampler(seed=int(cfg.optuna.sampler_seed))
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=int(cfg.optuna.n_startup_trials),
        n_warmup_steps=int(cfg.optuna.n_warmup_steps),
        interval_steps=int(cfg.optuna.interval_steps),
    )
    study = optuna.create_study(
        direction="minimize",
        study_name=str(cfg.optuna.study_name),
        storage=storage,
        load_if_exists=True,
        sampler=sampler,
        pruner=pruner,
    )
    recovered = _recover_interrupted_parameter_sets(study)
    complete_before = sum(t.state == TrialState.COMPLETE for t in study.trials)
    remaining = max(0, target - complete_before)
    _write_status(
        study,
        outdir,
        target,
        cfg=cfg,
        contract_signature=contract_signature,
        recovered=recovered,
    )
    if remaining > 0:
        study.optimize(
            lambda trial: objective(trial, cfg, outdir, contract_signature),
            n_trials=remaining,
            timeout=int(float(cfg.optuna.timeout_minutes) * 60),
            callbacks=[
                lambda current_study, _: _write_status(
                    current_study,
                    outdir,
                    target,
                    cfg=cfg,
                    contract_signature=contract_signature,
                    recovered=recovered,
                )
            ],
            gc_after_trial=True,
            catch=(RuntimeError,),
        )
    _write_status(
        study,
        outdir,
        target,
        cfg=cfg,
        contract_signature=contract_signature,
        recovered=recovered,
    )
    status = json.loads((outdir / "OPTUNA_STATUS.json").read_text(encoding="utf-8"))
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
