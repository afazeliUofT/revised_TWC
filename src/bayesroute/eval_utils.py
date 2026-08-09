from __future__ import annotations

from pathlib import Path
import time

import torch

from .config import count_parameters, set_seed
from .losses import (
    bit_metrics,
    calibration_ece,
    channel_coverage95,
    channel_marginal_nll,
    channel_nmse,
)
from .models import BayesRouteReceiver, LSReceiver, OracleReceiver


def _checkpoint_contract(cfg) -> dict:
    return {
        "package_revision": cfg.get("package_revision"),
        "system": cfg.system.to_dict(),
        "model": cfg.model.to_dict(),
    }


def _validate_checkpoint_contract(state: dict, cfg, path: Path) -> None:
    saved_cfg = state.get("config")
    if saved_cfg is None:
        raise RuntimeError(f"Checkpoint has no saved configuration contract: {path}")
    saved = {
        "package_revision": saved_cfg.get("package_revision"),
        "system": saved_cfg.get("system"),
        "model": saved_cfg.get("model"),
    }
    expected = _checkpoint_contract(cfg)
    if saved != expected:
        raise RuntimeError(
            "Checkpoint/evaluation-config mismatch. Use the matching checkpoint "
            f"or a new run name. checkpoint={path}"
        )


def make_receiver(name: str, cfg, simulator, checkpoint: str | None = None):
    if name in {
        "bayesroute",
        "bayesroute_uncertainty",
        "bayesroute_uncertainty_off",
        "bayesroute_graph_off",
        "bayesroute_full_graph",
    }:
        model = BayesRouteReceiver(cfg, simulator.coords, simulator.pilot_idx).to(
            simulator.device
        )
        if checkpoint:
            path = Path(checkpoint)
            if not path.exists():
                raise FileNotFoundError(f"BayesRoute checkpoint not found: {path}")
            state = torch.load(path, map_location=simulator.device, weights_only=False)
            _validate_checkpoint_contract(state, cfg, path)
            model.load_state_dict(state["model"], strict=True)
        return model.eval()
    if name == "ls":
        return LSReceiver(cfg).to(simulator.device).eval()
    if name == "oracle":
        return OracleReceiver(cfg).to(simulator.device).eval()
    raise ValueError(f"Unknown receiver baseline: {name}")


def _forward(name: str, model, batch):
    use_uncertainty = None
    edge_mass = None
    if name == "bayesroute_uncertainty_off":
        use_uncertainty = False
    elif name in {
        "bayesroute",
        "bayesroute_uncertainty",
        "bayesroute_graph_off",
        "bayesroute_full_graph",
    }:
        use_uncertainty = True
    if name == "bayesroute_graph_off":
        edge_mass = 0.0
    elif name == "bayesroute_full_graph":
        edge_mass = 1.0

    if isinstance(model, BayesRouteReceiver):
        return model(
            batch,
            use_uncertainty=use_uncertainty,
            edge_mass=edge_mass,
        )
    return model(batch)


def warmup_receiver(
    name: str,
    model,
    simulator,
    *,
    snr_db: float,
    batch_size: int,
    seed: int,
) -> None:
    """Run one unmeasured batch so CUDA initialization is outside latency data."""
    set_seed(int(seed))
    batch = simulator.sample(batch_size=int(batch_size), snr_db=float(snr_db))
    with torch.no_grad():
        _forward(name, model, batch)
    if simulator.device.type == "cuda":
        torch.cuda.synchronize(simulator.device)


def evaluate_one_batch(
    name: str,
    model,
    simulator,
    snr_db: float,
    batch_size: int,
) -> dict:
    # Batch generation is deliberately outside the timer. The reported latency
    # is the receiver forward pass only and is therefore comparable across methods.
    batch = simulator.sample(batch_size=int(batch_size), snr_db=float(snr_db))

    allocated_before = float("nan")
    if simulator.device.type == "cuda":
        torch.cuda.synchronize(simulator.device)
        torch.cuda.reset_peak_memory_stats(simulator.device)
        allocated_before = float(torch.cuda.memory_allocated(simulator.device))
    t0 = time.perf_counter()
    with torch.no_grad():
        out = _forward(name, model, batch)
    if simulator.device.type == "cuda":
        torch.cuda.synchronize(simulator.device)
    receiver_elapsed = time.perf_counter() - t0

    if simulator.device.type == "cuda":
        peak_allocated = float(torch.cuda.max_memory_allocated(simulator.device))
        incremental_peak = max(0.0, peak_allocated - allocated_before)
    else:
        peak_allocated = float("nan")
        incremental_peak = float("nan")

    # Diagnostics are outside the timed region.
    if isinstance(model, BayesRouteReceiver):
        nmse = channel_nmse(
            out["posterior"].mean[..., batch.data_idx],
            batch.h[..., batch.data_idx],
        )
        edge_mean = float(out["kappa"].mean().item())
        density = float(out["edge_density"])
        channel_nll = float(
            channel_marginal_nll(
                out["posterior"].mean[..., batch.data_idx],
                out["posterior"].var_diag[:, batch.data_idx],
                batch.h[..., batch.data_idx],
            ).item()
        )
        coverage95 = channel_coverage95(
            out["posterior"].mean[..., batch.data_idx],
            out["posterior"].var_diag[:, batch.data_idx],
            batch.h[..., batch.data_idx],
        )
    else:
        nmse = float("nan")
        edge_mean = float("nan")
        density = float(out.get("edge_density", float("nan")))
        channel_nll = float("nan")
        coverage95 = float("nan")

    metrics = bit_metrics(out["bit_logits"], batch.data_bits)
    metrics.update(
        {
            "ece": calibration_ece(out["bit_logits"], batch.data_bits),
            "channel_nmse": nmse,
            "edge_coupling_mean": edge_mean,
            "edge_density": density,
            "channel_marginal_nll": channel_nll,
            "channel_coverage95": coverage95,
            "baseline": name,
            "snr_db": float(snr_db),
            "batch_size": int(batch_size),
            "receiver_elapsed_sec": float(receiver_elapsed),
            "receiver_ms_per_sample": float(1000.0 * receiver_elapsed / int(batch_size)),
            "receiver_samples_per_sec": float(int(batch_size) / max(receiver_elapsed, 1e-12)),
            "receiver_peak_memory_mib": float(peak_allocated / (1024.0**2)),
            "receiver_incremental_peak_memory_mib": float(
                incremental_peak / (1024.0**2)
            ),
            "trainable_params": count_parameters(model),
        }
    )
    return metrics
