#!/usr/bin/env python3
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bayesroute.config import (
    load_config, set_seed, get_device, save_json, apply_optuna_best
)
from bayesroute.simulator import UplinkToySimulator
from bayesroute.eval_utils import make_receiver, evaluate_one_batch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--run-name", default="initial")
    ap.add_argument("--checkpoint", default=None)
    resume = ap.add_mutually_exclusive_group()
    resume.add_argument("--resume", dest="resume", action="store_true")
    resume.add_argument("--no-resume", dest="resume", action="store_false")
    ap.set_defaults(resume=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    cfg, optuna_meta = apply_optuna_best(cfg)
    device = get_device(cfg)
    set_seed(int(cfg.seed) + 900)
    simulator = UplinkToySimulator(cfg, device)
    eval_dir = Path("outputs/eval")
    report_dir = Path("outputs/reports")
    eval_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    out_csv = eval_dir / f"{args.run_name}_eval.csv"

    done: set[tuple[str, float, int]] = set()
    if args.resume and out_csv.exists():
        old = pd.read_csv(out_csv)
        for _, row in old.iterrows():
            done.add((str(row["baseline"]), float(row["snr_db"]), int(row["rep"])))
    elif out_csv.exists():
        out_csv.unlink()

    baselines = [str(x) for x in cfg.evaluation.baselines]
    models = {}
    for baseline in baselines:
        ckpt = args.checkpoint if baseline.startswith("bayesroute") else None
        models[baseline] = make_receiver(baseline, cfg, simulator, ckpt)

    n_reps = int(cfg.evaluation.batches_per_snr)
    batch_size = int(cfg.evaluation.batch_size)
    for baseline in baselines:
        for snr_index, snr in enumerate(cfg.evaluation.snr_grid_db):
            for rep in range(n_reps):
                key = (baseline, float(snr), rep)
                if key in done:
                    continue
                # Baseline-independent seed gives paired batches across receivers.
                eval_seed = int(cfg.seed) + 1_000_000 + 10_000 * snr_index + rep
                set_seed(eval_seed)
                row = evaluate_one_batch(
                    baseline, models[baseline], simulator, float(snr), batch_size
                )
                row["rep"] = rep
                row["eval_seed"] = eval_seed
                pd.DataFrame([row]).to_csv(
                    out_csv, mode="a", header=not out_csv.exists(), index=False
                )
                done.add(key)
                print(json.dumps(row), flush=True)

    df = pd.read_csv(out_csv)
    grouped = (
        df.groupby(["baseline", "snr_db"], as_index=False)
        .agg(
            ber_mean=("ber", "mean"),
            ber_std=("ber", "std"),
            bit_nll_mean=("bit_nll", "mean"),
            brier_mean=("brier", "mean"),
            ece_mean=("ece", "mean"),
            channel_nmse_mean=("channel_nmse", "mean"),
            channel_marginal_nll_mean=("channel_marginal_nll", "mean"),
            channel_coverage95_mean=("channel_coverage95", "mean"),
            edge_density_mean=("edge_density", "mean"),
            elapsed_sec_mean=("elapsed_sec", "mean"),
            reps=("rep", "count"),
        )
    )
    grouped.to_csv(eval_dir / f"{args.run_name}_eval_aggregate.csv", index=False)

    paired_path = eval_dir / f"{args.run_name}_paired_deltas.csv"
    paired_rows = []
    proposed_name = "bayesroute_uncertainty"
    if proposed_name in set(df["baseline"]):
        proposed = df[df["baseline"] == proposed_name]
        keys = ["snr_db", "rep", "eval_seed"]
        for comparator in [b for b in baselines if b != proposed_name]:
            other = df[df["baseline"] == comparator]
            merged = proposed.merge(other, on=keys, suffixes=("_proposed", "_comparator"))
            for _, row in merged.iterrows():
                paired_rows.append({
                    "comparator": comparator,
                    "snr_db": float(row["snr_db"]),
                    "rep": int(row["rep"]),
                    "eval_seed": int(row["eval_seed"]),
                    "ber_delta_proposed_minus_comparator": float(row["ber_proposed"] - row["ber_comparator"]),
                    "bit_nll_delta_proposed_minus_comparator": float(row["bit_nll_proposed"] - row["bit_nll_comparator"]),
                    "ece_delta_proposed_minus_comparator": float(row["ece_proposed"] - row["ece_comparator"]),
                    "elapsed_sec_delta_proposed_minus_comparator": float(row["elapsed_sec_proposed"] - row["elapsed_sec_comparator"]),
                })
    paired_df = pd.DataFrame(paired_rows)
    if not paired_df.empty:
        paired_df.to_csv(paired_path, index=False)

    summary = {
        "run_name": args.run_name,
        "scope": "GATE0_PRINCIPLE_ONLY_NOT_PUBLICATION_TBLER",
        "csv": str(out_csv),
        "aggregate_csv": str(eval_dir / f"{args.run_name}_eval_aggregate.csv"),
        "rows": int(len(df)),
        "expected_rows": int(len(baselines) * len(cfg.evaluation.snr_grid_db) * n_reps),
        "complete": bool(len(df) == len(baselines) * len(cfg.evaluation.snr_grid_db) * n_reps),
        "baselines": baselines,
        "snr_grid_db": [float(x) for x in cfg.evaluation.snr_grid_db],
        "optuna": optuna_meta,
        "paired_delta_csv": str(paired_path) if not paired_df.empty else None,
        "delta_sign_convention": "negative means BayesRoute uncertainty is better/lower",
    }
    if not paired_df.empty:
        for comparator, sub in paired_df.groupby("comparator"):
            summary[f"mean_bit_nll_delta_vs_{comparator}"] = float(
                sub["bit_nll_delta_proposed_minus_comparator"].mean()
            )
            summary[f"mean_ber_delta_vs_{comparator}"] = float(
                sub["ber_delta_proposed_minus_comparator"].mean()
            )
    save_json(summary, report_dir / f"{args.run_name}_eval_summary.json")


if __name__ == "__main__":
    main()
