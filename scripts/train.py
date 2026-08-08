#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, json, math, os, sys, time
from pathlib import Path
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bayesroute.config import load_config, set_seed, get_device, save_json, count_parameters
from bayesroute.simulator import UplinkToySimulator
from bayesroute.models import BayesRouteReceiver
from bayesroute.losses import bit_bce_loss, bit_metrics, channel_nmse


def save_ckpt(path, model, opt, step, best_metric, cfg):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(), "step": step,
                "best_metric": best_metric, "config": cfg.to_dict()}, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--run-name", default="initial")
    ap.add_argument("--resume", action="store_true", default=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    set_seed(int(cfg.seed))
    device = get_device(cfg)
    sim = UplinkToySimulator(cfg, device)
    model = BayesRouteReceiver(cfg, sim.coords, sim.pilot_idx).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg.training.lr), weight_decay=1e-4)
    ckpt_dir = Path("outputs/checkpoints") / args.run_name
    log_dir = Path("outputs/logs"); log_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = log_dir / f"{args.run_name}_train_metrics.csv"
    start_step = 0
    best_metric = float("inf")
    last_path = ckpt_dir / "last.pt"
    best_path = ckpt_dir / "best.pt"
    if args.resume and last_path.exists():
        state = torch.load(last_path, map_location=device)
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["optimizer"])
        start_step = int(state.get("step", 0)) + 1
        best_metric = float(state.get("best_metric", best_metric))
        print(f"Resuming {args.run_name} from step {start_step}")

    if not metrics_path.exists():
        with metrics_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["step","loss","bce","nmse","ber","tblER_proxy","bit_nll","brier","snr_db","lr","params"])
            writer.writeheader()

    steps = int(cfg.training.steps)
    for step in range(start_step, steps):
        model.train()
        batch = sim.sample(batch_size=int(cfg.training.batch_size))
        out = model(batch)
        bce = bit_bce_loss(out["bit_logits"], batch.data_bits)
        nmse_t = model.posterior.channel_nmse(out["posterior"], batch.h, batch.data_idx)
        loss = bce + float(cfg.training.channel_loss_weight) * nmse_t
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss at step {step}: {loss}")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.training.grad_clip))
        opt.step()

        if step % int(cfg.training.log_every) == 0 or step == steps - 1:
            model.eval()
            with torch.no_grad():
                met = bit_metrics(out["bit_logits"], batch.data_bits)
                nmse = channel_nmse(out["posterior"].mean[..., batch.data_idx], batch.h[..., batch.data_idx])
            row = {"step": step, "loss": float(loss.item()), "bce": float(bce.item()), "nmse": nmse,
                   "snr_db": batch.snr_db, "lr": float(opt.param_groups[0]["lr"]), "params": count_parameters(model), **met}
            with metrics_path.open("a", newline="", encoding="utf-8") as f:
                csv.DictWriter(f, fieldnames=list(row.keys())).writerow(row)
            print(json.dumps(row), flush=True)
            if row["bit_nll"] < best_metric:
                best_metric = row["bit_nll"]
                save_ckpt(best_path, model, opt, step, best_metric, cfg)
        if step % int(cfg.training.save_every) == 0 or step == steps - 1:
            save_ckpt(last_path, model, opt, step, best_metric, cfg)

    save_json({"run_name": args.run_name, "steps": steps, "best_metric": best_metric, "checkpoint": str(best_path),
               "trainable_params": count_parameters(model)}, Path("outputs/reports") / f"{args.run_name}_train_summary.json")

if __name__ == "__main__":
    main()
