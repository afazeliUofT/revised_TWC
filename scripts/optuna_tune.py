#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path
import copy
import optuna
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bayesroute.config import load_config, AttrDict, set_seed, get_device, save_json
from bayesroute.simulator import UplinkToySimulator
from bayesroute.models import BayesRouteReceiver
from bayesroute.losses import bit_bce_loss, bit_metrics


def objective(trial, base_cfg, outdir: Path):
    cfg = AttrDict(copy.deepcopy(base_cfg.to_dict()))
    cfg.model.rank = trial.suggest_categorical("rank", [8, 12, 16, 24])
    cfg.model.detector_iterations = trial.suggest_int("detector_iterations", 2, 4)
    cfg.model.edge_mass = trial.suggest_float("edge_mass", 0.85, 0.98)
    cfg.training.lr = trial.suggest_float("lr", 5e-4, 4e-3, log=True)
    cfg.training.channel_loss_weight = trial.suggest_float("channel_loss_weight", 0.0, 0.15)
    cfg.training.steps = int(cfg.optuna.train_steps_per_trial)
    set_seed(int(cfg.seed) + trial.number)
    device = get_device(cfg)
    sim = UplinkToySimulator(cfg, device)
    model = BayesRouteReceiver(cfg, sim.coords, sim.pilot_idx).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg.training.lr), weight_decay=1e-4)
    for step in range(int(cfg.training.steps)):
        batch = sim.sample(batch_size=int(cfg.training.batch_size))
        out = model(batch)
        loss = bit_bce_loss(out["bit_logits"], batch.data_bits)
        if cfg.training.channel_loss_weight > 0:
            loss = loss + float(cfg.training.channel_loss_weight) * model.posterior.channel_nmse(out["posterior"], batch.h, batch.data_idx)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.training.grad_clip))
        opt.step()
        if step % 40 == 0:
            trial.report(float(loss.item()), step)
            if trial.should_prune():
                raise optuna.TrialPruned()
    model.eval()
    vals = []
    with torch.no_grad():
        for snr in cfg.evaluation.snr_grid_db:
            for _ in range(int(cfg.optuna.eval_batches_per_snr)):
                batch = sim.sample(batch_size=int(cfg.evaluation.batch_size), snr_db=float(snr))
                out = model(batch)
                vals.append(bit_metrics(out["bit_logits"], batch.data_bits)["bit_nll"])
    score = float(sum(vals) / len(vals))
    trial.set_user_attr("score", score)
    return score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", default="outputs/optuna")
    ap.add_argument("--n-trials", type=int, default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    outdir = Path(args.out); outdir.mkdir(parents=True, exist_ok=True)
    n_trials = int(args.n_trials or cfg.optuna.n_trials)
    storage = f"sqlite:///{outdir / 'study.db'}"
    study = optuna.create_study(direction="minimize", study_name=str(cfg.optuna.study_name), storage=storage, load_if_exists=True,
                                pruner=optuna.pruners.MedianPruner(n_warmup_steps=40))
    study.optimize(lambda trial: objective(trial, cfg, outdir), n_trials=n_trials,
                   timeout=int(float(cfg.optuna.timeout_minutes) * 60))
    best = {"best_value": float(study.best_value), "best_params": study.best_params,
            "n_trials_total": len(study.trials), "study_db": str(outdir / 'study.db')}
    save_json(best, outdir / "best_params.json")
    print(json.dumps(best, indent=2))

if __name__ == "__main__":
    main()
