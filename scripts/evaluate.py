#!/usr/bin/env python3
from __future__ import annotations
import argparse, os, sys, json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bayesroute.config import load_config, set_seed, get_device, save_json
from bayesroute.simulator import UplinkToySimulator
from bayesroute.eval_utils import evaluate_baseline


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--run-name", default="initial")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--resume", action="store_true", default=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    set_seed(int(cfg.seed) + 900)
    device = get_device(cfg)
    sim = UplinkToySimulator(cfg, device)
    eval_dir = Path("outputs/eval"); eval_dir.mkdir(parents=True, exist_ok=True)
    out_csv = eval_dir / f"{args.run_name}_eval.csv"
    done = set()
    if args.resume and out_csv.exists():
        old = pd.read_csv(out_csv)
        for _, r in old.iterrows():
            done.add((str(r["baseline"]), float(r["snr_db"]), int(r.get("rep", 0))))
    rows = []
    baselines = [str(x) for x in cfg.evaluation.baselines]
    for baseline in baselines:
        for snr in cfg.evaluation.snr_grid_db:
            key = (baseline, float(snr), 0)
            if key in done:
                continue
            ckpt = args.checkpoint if baseline.startswith("bayesroute") else None
            row = evaluate_baseline(baseline, cfg, sim, float(snr), int(cfg.evaluation.batches_per_snr), int(cfg.evaluation.batch_size), ckpt)
            row["rep"] = 0
            rows.append(row)
            pd.DataFrame(rows).to_csv(out_csv, mode="a", header=not out_csv.exists(), index=False)
            rows = []
            print(json.dumps(row), flush=True)
    df = pd.read_csv(out_csv)
    summary = {
        "run_name": args.run_name,
        "csv": str(out_csv),
        "rows": int(len(df)),
        "baselines": sorted(df["baseline"].unique().tolist()),
        "snr_grid_db": sorted([float(x) for x in df["snr_db"].unique().tolist()]),
    }
    for b in summary["baselines"]:
        sub = df[df["baseline"] == b]
        summary[f"{b}_mean_ber"] = float(sub["ber"].mean())
        summary[f"{b}_mean_tblER_proxy"] = float(sub["tblER_proxy"].mean())
    save_json(summary, Path("outputs/reports") / f"{args.run_name}_eval_summary.json")

if __name__ == "__main__":
    main()
