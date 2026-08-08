from __future__ import annotations
from pathlib import Path
import time
import torch
from .models import BayesRouteReceiver, LSReceiver, OracleReceiver
from .losses import bit_metrics, calibration_ece, channel_nmse, channel_marginal_nll, channel_coverage95
from .config import count_parameters


def make_receiver(name: str, cfg, simulator, checkpoint: str | None = None):
    if name in {
        "bayesroute", "bayesroute_uncertainty", "bayesroute_mean",
        "bayesroute_graph_off", "bayesroute_full_graph",
    }:
        model = BayesRouteReceiver(cfg, simulator.coords, simulator.pilot_idx).to(simulator.device)
        if checkpoint:
            path = Path(checkpoint)
            if not path.exists():
                raise FileNotFoundError(f"BayesRoute checkpoint not found: {path}")
            state = torch.load(path, map_location=simulator.device, weights_only=False)
            model.load_state_dict(state["model"], strict=True)
        return model.eval()
    if name == "ls":
        return LSReceiver(cfg).to(simulator.device).eval()
    if name == "oracle":
        return OracleReceiver(cfg).to(simulator.device).eval()
    raise ValueError(f"Unknown receiver baseline: {name}")


def evaluate_one_batch(name: str, model, simulator, snr_db: float,
                       batch_size: int) -> dict:
    use_unc = None
    edge_mass = None
    if name == "bayesroute_mean":
        use_unc = False
    elif name in {
        "bayesroute", "bayesroute_uncertainty",
        "bayesroute_graph_off", "bayesroute_full_graph",
    }:
        use_unc = True
    if name == "bayesroute_graph_off":
        edge_mass = 0.0
    elif name == "bayesroute_full_graph":
        edge_mass = 1.0

    if simulator.device.type == "cuda":
        torch.cuda.synchronize(simulator.device)
    t0 = time.perf_counter()
    with torch.no_grad():
        batch = simulator.sample(batch_size=int(batch_size), snr_db=float(snr_db))
        if isinstance(model, BayesRouteReceiver):
            out = model(
                batch, use_uncertainty=use_unc, edge_mass=edge_mass
            )
            nmse = channel_nmse(
                out["posterior"].mean[..., batch.data_idx],
                batch.h[..., batch.data_idx],
            )
            edge_mean = float(out["kappa"].mean().item())
            density = float(out["edge_density"])
            ch_nll = float(channel_marginal_nll(
                out["posterior"].mean[..., batch.data_idx],
                out["posterior"].var_diag[:, batch.data_idx],
                batch.h[..., batch.data_idx],
            ).item())
            ch_cov95 = channel_coverage95(
                out["posterior"].mean[..., batch.data_idx],
                out["posterior"].var_diag[:, batch.data_idx],
                batch.h[..., batch.data_idx],
            )
        else:
            out = model(batch)
            nmse = float("nan")
            edge_mean = float("nan")
            density = float(out.get("edge_density", float("nan")))
            ch_nll = float("nan")
            ch_cov95 = float("nan")
    if simulator.device.type == "cuda":
        torch.cuda.synchronize(simulator.device)
    elapsed = time.perf_counter() - t0

    metrics = bit_metrics(out["bit_logits"], batch.data_bits)
    metrics.update({
        "ece": calibration_ece(out["bit_logits"], batch.data_bits),
        "channel_nmse": nmse,
        "edge_coupling_mean": edge_mean,
        "edge_density": density,
        "channel_marginal_nll": ch_nll,
        "channel_coverage95": ch_cov95,
        "baseline": name,
        "snr_db": float(snr_db),
        "batch_size": int(batch_size),
        "elapsed_sec": float(elapsed),
        "trainable_params": count_parameters(model),
    })
    return metrics
