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
from optuna.samplers import GridSampler
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

SEARCH_SPACE_VERSION = "gate0_v2_3_search_v2"
RANK_CANDIDATES = [8, 12, 16, 24]
SEARCH_GRID = {
    "rank": RANK_CANDIDATES,
    "detector_iterations": [2, 3, 4],
    "lr": [5e-4, 1e-3, 2e-3, 4e-3],
    "channel_loss_weight": [0.0, 0.05, 0.10, 0.15],
}

# Twelve configurations forming a pairwise-balanced discrete design. Every pair
# of hyperparameter columns appears in twelve distinct level combinations.
_BALANCED_INDEX_DESIGN = [
    (0, 0, 2, 0), (1, 0, 0, 1), (2, 0, 1, 3), (3, 0, 3, 2),
    (0, 1, 1, 2), (1, 1, 3, 3), (2, 1, 2, 1), (3, 1, 0, 0),
    (0, 2, 0, 3), (1, 2, 2, 2), (2, 2, 3, 0), (3, 2, 1, 1),
]
BALANCED_TRIALS = [
    {
        "rank": SEARCH_GRID["rank"][rank_i],
        "detector_iterations": SEARCH_GRID["detector_iterations"][iter_i],
        "lr": SEARCH_GRID["lr"][lr_i],
        "channel_loss_weight": SEARCH_GRID["channel_loss_weight"][loss_i],
    }
    for rank_i, iter_i, lr_i, loss_i in _BALANCED_INDEX_DESIGN
]
OPTIMIZER_WEIGHT_DECAY = 1e-4


def _balanced_design_report() -> dict[str, Any]:
    """Validate the fixed twelve-run screening design before any GPU work."""
    names = ["rank", "detector_iterations", "lr", "channel_loss_weight"]
    rows = [tuple(row[name] for name in names) for row in BALANCED_TRIALS]
    pair_counts: dict[str, int] = {}
    for left in range(len(names)):
        for right in range(left + 1, len(names)):
            pair_counts[f"{names[left]}__{names[right]}"] = len(
                {(row[left], row[right]) for row in rows}
            )
    levels_valid = all(
        row[name] in SEARCH_GRID[name]
        for row in BALANCED_TRIALS
        for name in names
    )
    passed = bool(
        len(BALANCED_TRIALS) == 12
        and len(set(rows)) == 12
        and levels_valid
        and all(count == 12 for count in pair_counts.values())
    )
    signature = hashlib.sha256(
        json.dumps(BALANCED_TRIALS, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "passed": passed,
        "rows": len(BALANCED_TRIALS),
        "unique_rows": len(set(rows)),
        "pairwise_unique_counts": pair_counts,
        "signature": signature,
    }


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
            "edge_mass": float(cfg.model.get("edge_mass", 1.0)),
            "operator_seed": int(cfg.model.get("operator_seed", int(cfg.seed) + 1009)),
            "operator_bank_rank": int(cfg.model.get("operator_bank_rank", cfg.model.rank)),
        },
        "search_space": {
            key: list(values) for key, values in SEARCH_GRID.items()
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
            "design": str(opt_cfg.get("design", "pairwise_balanced_12")),
            "balanced_design_report": _balanced_design_report(),
            "balanced_trials": BALANCED_TRIALS,
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


def _bind_study_contract(
    study: optuna.Study,
    *,
    cfg: AttrDict,
    contract_signature: str,
) -> None:
    """Bind a persistent Optuna study to one immutable search contract."""
    existing = study.user_attrs.get("contract_signature")
    if existing is None:
        if study.trials:
            raise RuntimeError(
                "Existing Optuna study has trials but no contract signature. "
                "Move or delete the stale outputs/optuna directory."
            )
        study.set_user_attr("contract_signature", contract_signature)
        study.set_user_attr(
            "package_revision", str(cfg.get("package_revision", "unknown"))
        )
        study.set_user_attr("search_space_version", SEARCH_SPACE_VERSION)
        return
    if str(existing) != contract_signature:
        raise RuntimeError(
            "Optuna study/search-contract mismatch. Move or delete the stale "
            "outputs/optuna directory before starting a new search."
        )
    if study.user_attrs.get("package_revision") != str(
        cfg.get("package_revision", "unknown")
    ):
        raise RuntimeError("Optuna study/package revision mismatch")
    if study.user_attrs.get("search_space_version") != SEARCH_SPACE_VERSION:
        raise RuntimeError("Optuna study/search-space version mismatch")



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
    complete = [
        t for t in study.trials
        if t.state == TrialState.COMPLETE
        and t.user_attrs.get("contract_signature") == contract_signature
    ]
    status: dict[str, Any] = {
        "package_revision": str(cfg.get("package_revision", "unknown")),
        "search_space_version": SEARCH_SPACE_VERSION,
        "contract_signature": contract_signature,
        "study_name": study.study_name,
        "objective_metric": "fixed_validation_bit_nll",
        "pruning_metric": "none_all_balanced_trials_complete",
        "sampler": "GridSampler_with_enqueued_pairwise_balanced_design",
        "balanced_design_size": len(BALANCED_TRIALS),
        "balanced_design_report": _balanced_design_report(),
        "sampler_seed": int(cfg.optuna.sampler_seed),
        "common_training_seed": int(cfg.optuna.training_seed),
        "fixed_edge_mass": float(cfg.model.get("edge_mass", 1.0)),
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
                "balanced_design_report": _balanced_design_report(),
                "fixed_edge_mass": float(cfg.model.get("edge_mass", 1.0)),
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
    cfg.model.rank = trial.suggest_categorical(
        "rank", SEARCH_GRID["rank"]
    )
    cfg.model.detector_iterations = trial.suggest_categorical(
        "detector_iterations", SEARCH_GRID["detector_iterations"]
    )
    cfg.training.lr = trial.suggest_categorical(
        "lr", SEARCH_GRID["lr"]
    )
    cfg.training.channel_loss_weight = trial.suggest_categorical(
        "channel_loss_weight", SEARCH_GRID["channel_loss_weight"]
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
    design_report = _balanced_design_report()
    if not design_report["passed"]:
        raise RuntimeError(f"Invalid pairwise-balanced Optuna design: {design_report}")
    if str(cfg.optuna.get("design", "")) != "pairwise_balanced_12":
        raise RuntimeError("configs/optuna.yaml must request design=pairwise_balanced_12")
    if int(cfg.optuna.n_trials) != len(BALANCED_TRIALS):
        raise RuntimeError(
            "The Gate-0 v2.3 screening contract requires exactly twelve trials"
        )
    if target <= 0 or target > len(BALANCED_TRIALS):
        raise RuntimeError(
            f"target-trials must be in 1..{len(BALANCED_TRIALS)}"
        )
    contract_signature = _contract_signature(cfg)
    storage = f"sqlite:///{outdir / 'study.db'}"
    sampler = GridSampler(SEARCH_GRID, seed=int(cfg.optuna.sampler_seed))
    pruner = optuna.pruners.NopPruner()
    if int(cfg.model.get("operator_bank_rank", cfg.model.rank)) < max(RANK_CANDIDATES):
        raise RuntimeError(
            "operator_bank_rank must cover every Optuna rank candidate"
        )
    study = optuna.create_study(
        direction="minimize",
        study_name=str(cfg.optuna.study_name),
        storage=storage,
        load_if_exists=True,
        sampler=sampler,
        pruner=pruner,
    )
    _bind_study_contract(
        study, cfg=cfg, contract_signature=contract_signature
    )
    for design_index, params in enumerate(BALANCED_TRIALS):
        study.enqueue_trial(
            dict(params),
            user_attrs={
                "balanced_design": True,
                "balanced_design_index": int(design_index),
            },
            skip_if_exists=True,
        )
    recovered = _recover_interrupted_parameter_sets(study)
    complete_before = sum(
        t.state == TrialState.COMPLETE
        and t.user_attrs.get("contract_signature") == contract_signature
        for t in study.trials
    )
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
