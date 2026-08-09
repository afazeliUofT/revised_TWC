#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bayesroute.config import (
    apply_optuna_best,
    capture_rng_state,
    count_parameters,
    get_device,
    load_config,
    restore_rng_state,
    save_json,
    set_seed,
)
from bayesroute.losses import (
    bit_bce_loss,
    bit_metrics,
    channel_marginal_nll,
    channel_nmse,
)
from bayesroute.models import BayesRouteReceiver
from bayesroute.simulator import UplinkToySimulator

TRAINING_CONTRACT_VERSION = "gate0_v2_4_train_v1"
OPTIMIZER_WEIGHT_DECAY = 1e-4


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _training_contract(cfg, optuna_meta: dict, run_name: str) -> dict[str, Any]:
    """Immutable numerical contract for an exactly resumable training run."""
    contract: dict[str, Any] = {
        "contract_version": TRAINING_CONTRACT_VERSION,
        "package_revision": cfg.get("package_revision"),
        "run_name": str(run_name),
        "seed": int(cfg.seed),
        "system": cfg.system.to_dict(),
        "model": cfg.model.to_dict(),
        "training": cfg.training.to_dict(),
        "optimizer": {
            "name": "AdamW",
            "weight_decay": OPTIMIZER_WEIGHT_DECAY,
        },
        "optuna": {
            "applied": bool(optuna_meta.get("applied", False)),
            "package_revision": optuna_meta.get("package_revision"),
            "search_space_version": optuna_meta.get("search_space_version"),
            "contract_signature": optuna_meta.get("contract_signature"),
            "best_trial_number": optuna_meta.get("best_trial_number"),
            "best_params": optuna_meta.get("best_params"),
        },
    }
    contract["signature"] = hashlib.sha256(
        _canonical_json(contract).encode("utf-8")
    ).hexdigest()
    return contract


def save_checkpoint(
    path: Path,
    model,
    optimizer,
    step: int,
    best_metric: float,
    cfg,
    training_contract: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": int(step),
            "best_metric": float(best_metric),
            "config": cfg.to_dict(),
            "training_contract": training_contract,
            "rng_state": capture_rng_state(),
        },
        tmp,
    )
    tmp.replace(path)


def fixed_validation_nll(model, simulator, cfg) -> float:
    """Evaluate a fixed validation stream without changing training RNG state."""
    state = capture_rng_state()
    was_training = model.training
    try:
        set_seed(
            int(cfg.training.get("validation_seed", int(cfg.seed) + 700000))
        )
        model.eval()
        values: list[float] = []
        with torch.no_grad():
            for _ in range(int(cfg.training.get("validation_batches", 2))):
                batch = simulator.sample(
                    batch_size=int(
                        cfg.training.get(
                            "validation_batch_size", cfg.training.batch_size
                        )
                    ),
                    snr_db=float(cfg.training.get("validation_snr_db", 10.0)),
                )
                output = model(batch)
                values.append(
                    bit_metrics(output["bit_logits"], batch.data_bits)[
                        "bit_nll"
                    ]
                )
        if not values:
            raise RuntimeError("Validation stream produced no batches")
        return float(sum(values) / len(values))
    finally:
        restore_rng_state(state)
        model.train(was_training)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-name", default="initial")
    resume = parser.add_mutually_exclusive_group()
    resume.add_argument("--resume", dest="resume", action="store_true")
    resume.add_argument("--no-resume", dest="resume", action="store_false")
    parser.set_defaults(resume=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg, optuna_meta = apply_optuna_best(cfg)
    if cfg.get("optuna_best_path") and not optuna_meta.get("applied", False):
        raise RuntimeError(
            "Configured Optuna result was not applied. Complete Optuna before training."
        )

    set_seed(int(cfg.seed))
    device = get_device(cfg)
    simulator = UplinkToySimulator(cfg, device)
    model = BayesRouteReceiver(cfg, simulator.coords, simulator.pilot_idx).to(
        device
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.training.lr),
        weight_decay=OPTIMIZER_WEIGHT_DECAY,
    )

    checkpoint_dir = Path("outputs/checkpoints") / args.run_name
    log_dir = Path("outputs/logs")
    report_dir = Path("outputs/reports")
    log_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = log_dir / f"{args.run_name}_train_metrics.csv"
    last_path = checkpoint_dir / "last.pt"
    best_path = checkpoint_dir / "best.pt"

    if not args.resume:
        shutil.rmtree(checkpoint_dir, ignore_errors=True)
        metrics_path.unlink(missing_ok=True)
        for suffix in ("effective_config", "train_summary"):
            (report_dir / f"{args.run_name}_{suffix}.json").unlink(
                missing_ok=True
            )
    elif metrics_path.exists() and not last_path.exists():
        raise RuntimeError(
            "Training metrics exist without a resumable last checkpoint. "
            "Use --no-resume or a new run name."
        )

    contract = _training_contract(cfg, optuna_meta, args.run_name)
    start_step = 0
    best_metric = float("inf")

    if args.resume and last_path.exists():
        state = torch.load(last_path, map_location=device, weights_only=False)
        saved_contract = state.get("training_contract")
        if not isinstance(saved_contract, dict):
            raise RuntimeError(
                "Checkpoint has no v2.4 training contract. Use a new run name."
            )
        if saved_contract != contract:
            raise RuntimeError(
                "Checkpoint/effective-training-contract mismatch. "
                "Use a new run name or remove the stale checkpoint."
            )
        model.load_state_dict(state["model"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        start_step = int(state.get("step", -1)) + 1
        best_metric = float(state.get("best_metric", best_metric))
        restore_rng_state(state.get("rng_state"))
        print(f"Resuming {args.run_name} from step {start_step}")
        if metrics_path.exists():
            old = pd.read_csv(metrics_path)
            old = old[old["step"] < start_step]
            old.to_csv(metrics_path, index=False)

    fields = [
        "step",
        "loss",
        "bce",
        "channel_marginal_nll",
        "nmse",
        "ber",
        "tblER_proxy",
        "bit_nll",
        "brier",
        "snr_db",
        "lr",
        "grad_norm",
        "params",
        "edge_density",
        "validation_bit_nll",
        "training_contract_signature",
    ]
    if not metrics_path.exists():
        with metrics_path.open("w", newline="", encoding="utf-8") as stream:
            csv.DictWriter(stream, fieldnames=fields).writeheader()

    save_json(
        {
            "run_name": args.run_name,
            "source_config": args.config,
            "effective_config": cfg.to_dict(),
            "optuna": optuna_meta,
            "device": str(device),
            "training_contract": contract,
        },
        report_dir / f"{args.run_name}_effective_config.json",
    )

    steps = int(cfg.training.steps)
    for step in range(start_step, steps):
        model.train()
        batch = simulator.sample(batch_size=int(cfg.training.batch_size))
        output = model(batch)
        bce = bit_bce_loss(output["bit_logits"], batch.data_bits)
        channel_nll = channel_marginal_nll(
            output["posterior"].mean[..., batch.data_idx],
            output["posterior"].var_diag[:, batch.data_idx],
            batch.h[..., batch.data_idx],
        )
        loss = bce + float(cfg.training.channel_loss_weight) * channel_nll
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss at step {step}: {loss}")

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm_tensor = torch.nn.utils.clip_grad_norm_(
            model.parameters(), float(cfg.training.grad_clip)
        )
        if not torch.isfinite(grad_norm_tensor):
            raise RuntimeError(f"Non-finite gradient norm at step {step}")
        grad_norm = float(grad_norm_tensor.item())
        optimizer.step()

        validation_every = int(
            cfg.training.get("validation_every", cfg.training.save_every)
        )
        validation_due = step % validation_every == 0 or step == steps - 1
        validation_nll = (
            fixed_validation_nll(model, simulator, cfg)
            if validation_due
            else float("nan")
        )
        if validation_due and validation_nll < best_metric:
            best_metric = validation_nll
            save_checkpoint(
                best_path,
                model,
                optimizer,
                step,
                best_metric,
                cfg,
                contract,
            )

        if step % int(cfg.training.log_every) == 0 or step == steps - 1:
            with torch.no_grad():
                metrics = bit_metrics(output["bit_logits"], batch.data_bits)
                nmse = channel_nmse(
                    output["posterior"].mean[..., batch.data_idx],
                    batch.h[..., batch.data_idx],
                )
            row = {
                "step": step,
                "loss": float(loss.item()),
                "bce": float(bce.item()),
                "channel_marginal_nll": float(channel_nll.item()),
                "nmse": nmse,
                "snr_db": batch.snr_db,
                "lr": float(optimizer.param_groups[0]["lr"]),
                "grad_norm": grad_norm,
                "params": count_parameters(model),
                "edge_density": float(output["edge_density"]),
                "validation_bit_nll": validation_nll,
                "training_contract_signature": contract["signature"],
                **metrics,
            }
            with metrics_path.open("a", newline="", encoding="utf-8") as stream:
                csv.DictWriter(stream, fieldnames=fields).writerow(row)
            print(json.dumps(row), flush=True)

        if step % int(cfg.training.save_every) == 0 or step == steps - 1:
            save_checkpoint(
                last_path,
                model,
                optimizer,
                step,
                best_metric,
                cfg,
                contract,
            )

    if not best_path.is_file() or not last_path.is_file():
        raise RuntimeError("Training completed without both best and last checkpoints")

    save_json(
        {
            "run_name": args.run_name,
            "steps": steps,
            "best_metric": best_metric,
            "checkpoint": str(best_path),
            "last_checkpoint": str(last_path),
            "trainable_params": count_parameters(model),
            "optuna": optuna_meta,
            "training_contract": contract,
            "complete": True,
        },
        report_dir / f"{args.run_name}_train_summary.json",
    )


if __name__ == "__main__":
    main()
