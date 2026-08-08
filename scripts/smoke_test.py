#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, math, os, sys, time
from pathlib import Path
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bayesroute.config import load_config, set_seed, get_device, save_json, count_parameters
from bayesroute.simulator import UplinkToySimulator
from bayesroute.pilots import pilot_orthogonality_report
from bayesroute.models import BayesRouteReceiver, LSReceiver, OracleReceiver, coupling_matrix
from bayesroute.losses import bit_bce_loss, bit_metrics, channel_nmse
from bayesroute.sionna_check import check_sionna


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/smoke.yaml")
    ap.add_argument("--out", default="outputs/smoke")
    args = ap.parse_args()
    cfg = load_config(args.config)
    set_seed(int(cfg.seed))
    device = get_device(cfg)
    outdir = Path(args.out); outdir.mkdir(parents=True, exist_ok=True)
    report = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "device": str(device)}

    report["sionna"] = check_sionna()
    sim = UplinkToySimulator(cfg, device)
    report["port_metadata"] = sim.port_meta
    report["pilot_orthogonality"] = pilot_orthogonality_report(sim.phi)

    batch = sim.sample(batch_size=int(cfg.training.batch_size), snr_db=10.0)
    report["shapes"] = {
        "y": list(batch.y.shape), "h": list(batch.h.shape), "x": list(batch.x.shape),
        "data_bits": list(batch.data_bits.shape), "phi": list(batch.phi.shape),
        "pilot_idx": list(batch.pilot_idx.shape), "data_idx": list(batch.data_idx.shape)
    }

    model = BayesRouteReceiver(cfg, sim.coords, sim.pilot_idx).to(device)
    out = model(batch)
    loss = bit_bce_loss(out["bit_logits"], batch.data_bits)
    loss = loss + float(cfg.training.channel_loss_weight) * model.posterior.channel_nmse(out["posterior"], batch.h, batch.data_idx)
    loss.backward()
    grad_norm = 0.0
    bad_grad = False
    for p in model.parameters():
        if p.grad is not None:
            if not torch.isfinite(p.grad).all():
                bad_grad = True
            grad_norm += float(torch.sum(p.grad.detach() ** 2).item())
    grad_norm = math.sqrt(grad_norm)
    report["gradient_flow"] = {"loss": float(loss.item()), "grad_norm": grad_norm, "all_finite": not bad_grad}
    report["trainable_params"] = count_parameters(model)

    # One optimizer step
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg.training.lr))
    opt.step(); opt.zero_grad(set_to_none=True)
    out2 = model(batch)
    report["post_step_metrics"] = bit_metrics(out2["bit_logits"], batch.data_bits)
    report["post_step_channel_nmse"] = channel_nmse(out2["posterior"].mean[..., batch.data_idx], batch.h[..., batch.data_idx])

    # Baseline forward checks
    for name, cls in [("ls", LSReceiver), ("oracle", OracleReceiver)]:
        baseline = cls(cfg).to(device)
        bout = baseline(batch)
        report[f"{name}_metrics"] = bit_metrics(bout["bit_logits"], batch.data_bits)

    kappa = out2["kappa"]
    report["coupling"] = {
        "shape": list(kappa.shape),
        "finite": bool(torch.isfinite(kappa).all().item()),
        "mean": float(kappa.mean().item()),
        "max": float(kappa.max().item()),
    }

    checks = {
        "sionna_import_and_mapping_api": bool(report["sionna"].get("passed", False)),
        "pilot_orthogonality": bool(report["pilot_orthogonality"].get("passed", False)),
        "tensor_shapes_nonempty": all(v for v in report["shapes"].values()),
        "gradient_nonzero_finite": bool(grad_norm > 0 and not bad_grad),
        "loss_finite": bool(math.isfinite(report["gradient_flow"]["loss"])),
        "coupling_finite": bool(report["coupling"]["finite"]),
    }
    report["checks"] = checks
    report["overall_pass"] = bool(all(checks.values()))
    save_json(report, outdir / "SMOKE_HEALTH.json")
    lines = [f"{k}: {'PASS' if v else 'FAIL'}" for k, v in checks.items()]
    lines.append(f"OVERALL: {'PASS' if report['overall_pass'] else 'FAIL'}")
    (outdir / "SMOKE_HEALTH.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    if not report["overall_pass"]:
        raise SystemExit(2)

if __name__ == "__main__":
    main()
