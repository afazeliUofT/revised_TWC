#!/usr/bin/env python3
from __future__ import annotations
import argparse
import csv
import json
import math
import sys
from pathlib import Path
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bayesroute.config import (
    load_config,
    set_seed,
    get_device,
    save_json,
    count_parameters,
    apply_optuna_best,
    capture_rng_state,
    restore_rng_state,
)
from bayesroute.simulator import UplinkToySimulator
from bayesroute.models import BayesRouteReceiver
from bayesroute.losses import bit_bce_loss, bit_metrics, channel_nmse, channel_marginal_nll


def save_ckpt(path: Path, model, optimizer, step: int, best_metric: float, cfg) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": int(step),
            "best_metric": float(best_metric),
            "config": cfg.to_dict(),
            "rng_state": capture_rng_state(),
        },
        tmp,
    )
    tmp.replace(path)


def fixed_validation_nll(model, simulator, cfg) -> float:
    """Evaluate on a fixed stream without changing the training RNG state."""
    state = capture_rng_state()
    was_training = model.training
    try:
        set_seed(int(cfg.training.get("validation_seed", int(cfg.seed) + 700000)))
        model.eval()
        values = []
        with torch.no_grad():
            for _ in range(int(cfg.training.get("validation_batches", 2))):
                batch = simulator.sample(
                    batch_size=int(cfg.training.get("validation_batch_size", cfg.training.batch_size)),
                    snr_db=float(cfg.training.get("validation_snr_db", 10.0)),
                )
                out = model(batch)
                values.append(bit_metrics(out["bit_logits"], batch.data_bits)["bit_nll"])
        return float(sum(values) / len(values))
    finally:
        restore_rng_state(state)
        model.train(was_training)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--run-name", default="initial")
    resume = ap.add_mutually_exclusive_group()
    resume.add_argument("--resume", dest="resume", action="store_true")
    resume.add_argument("--no-resume", dest="resume", action="store_false")
    ap.set_defaults(resume=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    cfg, optuna_meta = apply_optuna_best(cfg)
    set_seed(int(cfg.seed))
    device = get_device(cfg)
    sim = UplinkToySimulator(cfg, device)
    model = BayesRouteReceiver(cfg, sim.coords, sim.pilot_idx).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(cfg.training.lr), weight_decay=1e-4
    )

    ckpt_dir = Path("outputs/checkpoints") / args.run_name
    log_dir = Path("outputs/logs")
    report_dir = Path("outputs/reports")
    log_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = log_dir / f"{args.run_name}_train_metrics.csv"
    start_step = 0
    best_metric = float("inf")
    last_path = ckpt_dir / "last.pt"
    best_path = ckpt_dir / "best.pt"

    if args.resume and last_path.exists():
        state = torch.load(last_path, map_location=device, weights_only=False)
        saved_cfg = state.get("config")
        if saved_cfg is not None:
            current_contract = {
                "package_revision": cfg.get("package_revision"),
                "system": cfg.system.to_dict(),
                "model": cfg.model.to_dict(),
                "lr": float(cfg.training.lr),
                "channel_loss_weight": float(cfg.training.channel_loss_weight),
            }
            saved_contract = {
                "package_revision": saved_cfg.get("package_revision"),
                "system": saved_cfg.get("system"),
                "model": saved_cfg.get("model"),
                "lr": float(saved_cfg.get("training", {}).get("lr")),
                "channel_loss_weight": float(saved_cfg.get("training", {}).get("channel_loss_weight")),
            }
            if saved_contract != current_contract:
                raise RuntimeError(
                    "Checkpoint/effective-config mismatch. Use a new run name or remove the stale checkpoint."
                )
        model.load_state_dict(state["model"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        start_step = int(state.get("step", -1)) + 1
        best_metric = float(state.get("best_metric", best_metric))
        restore_rng_state(state.get("rng_state"))
        print(f"Resuming {args.run_name} from step {start_step}")
        # Remove any log rows beyond the checkpoint in case a prior write raced the save.
        if metrics_path.exists():
            old = pd.read_csv(metrics_path)
            old = old[old["step"] < start_step]
            old.to_csv(metrics_path, index=False)

    fields = [
        "step", "loss", "bce", "nmse", "ber", "tblER_proxy", "bit_nll",
        "brier", "snr_db", "lr", "params", "edge_density", "validation_bit_nll", "channel_marginal_nll",
    ]
    if not metrics_path.exists():
        with metrics_path.open("w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=fields).writeheader()

    save_json(
        {
            "run_name": args.run_name,
            "source_config": args.config,
            "effective_config": cfg.to_dict(),
            "optuna": optuna_meta,
            "device": str(device),
        },
        report_dir / f"{args.run_name}_effective_config.json",
    )

    steps = int(cfg.training.steps)
    for step in range(start_step, steps):
        model.train()
        batch = sim.sample(batch_size=int(cfg.training.batch_size))
        out = model(batch)
        bce = bit_bce_loss(out["bit_logits"], batch.data_bits)
        channel_nll_t = channel_marginal_nll(
            out["posterior"].mean[..., batch.data_idx],
            out["posterior"].var_diag[:, batch.data_idx],
            batch.h[..., batch.data_idx],
        )
        loss = bce + float(cfg.training.channel_loss_weight) * channel_nll_t
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss at step {step}: {loss}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), float(cfg.training.grad_clip)
        )
        if not torch.isfinite(grad_norm):
            raise RuntimeError(f"Non-finite gradient norm at step {step}")
        optimizer.step()

        validation_every = int(cfg.training.get("validation_every", cfg.training.save_every))
        validation_due = (step % validation_every == 0 or step == steps - 1)
        validation_nll = fixed_validation_nll(model, sim, cfg) if validation_due else float("nan")
        if validation_due and validation_nll < best_metric:
            best_metric = validation_nll
            save_ckpt(best_path, model, optimizer, step, best_metric, cfg)

        if step % int(cfg.training.log_every) == 0 or step == steps - 1:
            model.eval()
            with torch.no_grad():
                met = bit_metrics(out["bit_logits"], batch.data_bits)
                nmse = channel_nmse(
                    out["posterior"].mean[..., batch.data_idx],
                    batch.h[..., batch.data_idx],
                )
            row = {
                "step": step,
                "loss": float(loss.item()),
                "bce": float(bce.item()),
                "nmse": nmse,
                "channel_marginal_nll": float(channel_nll_t.item()),
                "snr_db": batch.snr_db,
                "lr": float(optimizer.param_groups[0]["lr"]),
                "params": count_parameters(model),
                "edge_density": float(out["edge_density"]),
                "validation_bit_nll": validation_nll,
                **met,
            }
            with metrics_path.open("a", newline="", encoding="utf-8") as f:
                csv.DictWriter(f, fieldnames=fields).writerow(row)
            print(json.dumps(row), flush=True)

        if step % int(cfg.training.save_every) == 0 or step == steps - 1:
            save_ckpt(last_path, model, optimizer, step, best_metric, cfg)

    save_json(
        {
            "run_name": args.run_name,
            "steps": steps,
            "best_metric": best_metric,
            "checkpoint": str(best_path),
            "last_checkpoint": str(last_path),
            "trainable_params": count_parameters(model),
            "optuna": optuna_meta,
        },
        report_dir / f"{args.run_name}_train_summary.json",
    )


if __name__ == "__main__":
    main()
