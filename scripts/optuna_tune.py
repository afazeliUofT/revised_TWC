#!/usr/bin/env python3
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import sys
import time
from pathlib import Path
import optuna
from optuna.trial import TrialState
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bayesroute.config import (
    load_config,
    AttrDict,
    set_seed,
    get_device,
    save_json,
    capture_rng_state,
    restore_rng_state,
)
from bayesroute.simulator import UplinkToySimulator
from bayesroute.models import BayesRouteReceiver
from bayesroute.losses import bit_bce_loss, bit_metrics, channel_marginal_nll


def _canonical_params(params: dict) -> str:
    return json.dumps(params, sort_keys=True, separators=(",", ":"), default=str)


def _trial_cache_key(cfg: AttrDict, params: dict) -> str:
    signature = {
        "package_revision": cfg.get("package_revision"),
        "system": cfg.system.to_dict(),
        "model_seed": cfg.model.get("operator_seed"),
        "training_steps": int(cfg.optuna.train_steps_per_trial),
        "params": params,
    }
    return hashlib.sha256(_canonical_params(signature).encode("utf-8")).hexdigest()[:20]


def _atomic_torch_save(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)


def _recover_interrupted_parameter_sets(study: optuna.Study) -> list[int]:
    """Enqueue stale RUNNING parameter sets once.

    The replacement trial uses the same hash-keyed checkpoint, so training resumes
    near the interruption even though Optuna assigns a new trial number.
    """
    complete = {
        _canonical_params(t.params)
        for t in study.trials
        if t.state == TrialState.COMPLETE and t.params
    }
    waiting = {
        _canonical_params(t.params)
        for t in study.trials
        if t.state == TrialState.WAITING and t.params
    }
    recovered: list[int] = []
    for trial in study.trials:
        if trial.state != TrialState.RUNNING or not trial.params:
            continue
        key = _canonical_params(trial.params)
        if key in complete or key in waiting:
            continue
        study.enqueue_trial(
            dict(trial.params),
            user_attrs={"resume_of_trial": int(trial.number)},
        )
        waiting.add(key)
        recovered.append(int(trial.number))
    return recovered


def _write_status(study: optuna.Study, outdir: Path, target_complete: int,
                  recovered: list[int] | None = None) -> None:
    states = {state.name: 0 for state in TrialState}
    for trial in study.trials:
        states[trial.state.name] = states.get(trial.state.name, 0) + 1
    complete = [t for t in study.trials if t.state == TrialState.COMPLETE]
    status = {
        "study_name": study.study_name,
        "target_complete_trials": int(target_complete),
        "state_counts": states,
        "complete_trials": len(complete),
        "target_reached": len(complete) >= int(target_complete),
        "study_db": str(outdir / "study.db"),
        "updated": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "recovery_enqueued_from_trials": recovered or [],
    }
    if complete:
        status["best_value"] = float(study.best_value)
        status["best_params"] = dict(study.best_params)
        save_json(
            {
                "best_value": float(study.best_value),
                "best_params": dict(study.best_params),
                "n_complete_trials": len(complete),
                "target_complete_trials": int(target_complete),
                "study_db": str(outdir / "study.db"),
            },
            outdir / "best_params.json",
        )
    save_json(status, outdir / "OPTUNA_STATUS.json")
    study.trials_dataframe().to_csv(outdir / "trials.csv", index=False)


def objective(trial: optuna.Trial, base_cfg: AttrDict, outdir: Path) -> float:
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
    cache_key = _trial_cache_key(base_cfg, params)
    trial.set_user_attr("cache_key", cache_key)
    cache_dir = outdir / "trial_cache" / cache_key
    checkpoint = cache_dir / "last.pt"
    summary_path = cache_dir / "summary.json"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # A job may have died after writing the summary but before Optuna committed the return.
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        trial.set_user_attr("resumed_from_completed_cache", True)
        trial.set_user_attr("validation_edge_density", summary["edge_density"])
        return float(summary["score"])

    stable_seed = int(base_cfg.seed) ^ int(cache_key[:8], 16)
    set_seed(stable_seed)
    device = get_device(cfg)
    sim = UplinkToySimulator(cfg, device)
    model = BayesRouteReceiver(cfg, sim.coords, sim.pilot_idx).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(cfg.training.lr), weight_decay=1e-4
    )

    start_step = 0
    if checkpoint.exists():
        state = torch.load(checkpoint, map_location=device, weights_only=False)
        if dict(state.get("params", {})) != params:
            raise RuntimeError(f"Checkpoint parameter mismatch for cache {cache_key}")
        model.load_state_dict(state["model"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        restore_rng_state(state.get("rng_state"))
        start_step = int(state["step"]) + 1
        trial.set_user_attr("resumed_from_step", start_step)

    for step in range(start_step, int(cfg.training.steps)):
        model.train()
        batch = sim.sample(batch_size=int(cfg.training.batch_size))
        out = model(batch)
        loss = bit_bce_loss(out["bit_logits"], batch.data_bits)
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

        if step % int(cfg.training.save_every) == 0 or step == int(cfg.training.steps) - 1:
            _atomic_torch_save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "step": step,
                    "params": params,
                    "config": cfg.to_dict(),
                    "rng_state": capture_rng_state(),
                },
                checkpoint,
            )
            trial.report(float(loss.item()), step)
            if trial.should_prune():
                raise optuna.TrialPruned()

    # Every parameter set sees the same validation stream.
    set_seed(int(cfg.optuna.get("validation_seed", int(base_cfg.seed) + 90000)))
    model.eval()
    nll_values: list[float] = []
    edge_densities: list[float] = []
    with torch.no_grad():
        for snr in cfg.evaluation.snr_grid_db:
            for _ in range(int(cfg.optuna.eval_batches_per_snr)):
                batch = sim.sample(
                    batch_size=int(cfg.evaluation.batch_size), snr_db=float(snr)
                )
                out = model(batch)
                nll_values.append(
                    bit_metrics(out["bit_logits"], batch.data_bits)["bit_nll"]
                )
                edge_densities.append(float(out["edge_density"]))
    score = float(sum(nll_values) / len(nll_values))
    density = float(sum(edge_densities) / len(edge_densities))
    trial.set_user_attr("validation_bit_nll", score)
    trial.set_user_attr("validation_edge_density", density)
    save_json(
        {
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
    storage = f"sqlite:///{outdir / 'study.db'}"
    study = optuna.create_study(
        direction="minimize",
        study_name=str(cfg.optuna.study_name),
        storage=storage,
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=40),
    )
    recovered = _recover_interrupted_parameter_sets(study)
    complete_before = sum(t.state == TrialState.COMPLETE for t in study.trials)
    remaining = max(0, target - complete_before)
    _write_status(study, outdir, target, recovered)
    if remaining > 0:
        study.optimize(
            lambda trial: objective(trial, cfg, outdir),
            n_trials=remaining,
            timeout=int(float(cfg.optuna.timeout_minutes) * 60),
            callbacks=[lambda st, _: _write_status(st, outdir, target, recovered)],
            gc_after_trial=True,
            catch=(RuntimeError,),
        )
    _write_status(study, outdir, target, recovered)
    status = json.loads((outdir / "OPTUNA_STATUS.json").read_text(encoding="utf-8"))
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
