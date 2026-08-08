from __future__ import annotations
from pathlib import Path
import time
import pandas as pd
import torch
from .models import BayesRouteReceiver, LSReceiver, OracleReceiver
from .losses import bit_metrics, calibration_ece, channel_nmse
from .config import count_parameters


def make_receiver(name: str, cfg, simulator, checkpoint: str | None = None):
    if name in {"bayesroute", "bayesroute_uncertainty", "bayesroute_mean"}:
        model = BayesRouteReceiver(cfg, simulator.coords, simulator.pilot_idx).to(simulator.device)
        if checkpoint and Path(checkpoint).exists():
            state = torch.load(checkpoint, map_location=simulator.device)
            model.load_state_dict(state["model"], strict=True)
        model.eval()
        return model
    if name == "ls":
        return LSReceiver(cfg).to(simulator.device).eval()
    if name == "oracle":
        return OracleReceiver(cfg).to(simulator.device).eval()
    raise ValueError(f"Unknown receiver baseline: {name}")


def evaluate_baseline(name: str, cfg, simulator, snr_db: float, n_batches: int, batch_size: int,
                      checkpoint: str | None = None) -> dict:
    model = make_receiver(name, cfg, simulator, checkpoint)
    use_unc = None
    if name == "bayesroute_mean":
        use_unc = False
    elif name == "bayesroute_uncertainty":
        use_unc = True
    rows = []
    t0 = time.time()
    with torch.no_grad():
        for _ in range(n_batches):
            batch = simulator.sample(batch_size=batch_size, snr_db=snr_db)
            if isinstance(model, BayesRouteReceiver):
                out = model(batch, use_uncertainty=use_unc)
                nmse = channel_nmse(out["posterior"].mean[..., batch.data_idx], batch.h[..., batch.data_idx])
                kappa = out["kappa"]
                edge_mean = float(kappa.mean().item()) if kappa is not None else 0.0
            else:
                out = model(batch)
                nmse = float("nan")
                edge_mean = float("nan")
            met = bit_metrics(out["bit_logits"], batch.data_bits)
            met["ece"] = calibration_ece(out["bit_logits"], batch.data_bits)
            met["channel_nmse"] = nmse
            met["edge_coupling_mean"] = edge_mean
            rows.append(met)
    df = pd.DataFrame(rows)
    out = {k: float(df[k].mean()) for k in df.columns}
    out.update({
        "baseline": name,
        "snr_db": float(snr_db),
        "n_batches": int(n_batches),
        "batch_size": int(batch_size),
        "elapsed_sec": float(time.time() - t0),
        "trainable_params": count_parameters(model),
    })
    return out
