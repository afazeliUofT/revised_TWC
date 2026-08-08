#!/usr/bin/env python3
from __future__ import annotations
import argparse, sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from bayesroute.config import load_config, save_json


def line_plot(df, y, ylabel, title, out_path):
    fig = plt.figure(figsize=(7.2, 4.8))
    ax = fig.add_subplot(111)
    for name, sub in df.groupby("baseline"):
        sub = sub.sort_values("snr_db")
        ax.semilogy(sub["snr_db"], sub[y], marker="o", label=name)
    ax.set_xlabel("SNR [dB]")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def normal_plot(df, y, ylabel, title, out_path):
    fig = plt.figure(figsize=(7.2, 4.8))
    ax = fig.add_subplot(111)
    for name, sub in df.groupby("baseline"):
        sub = sub.sort_values("snr_db")
        ax.plot(sub["snr_db"], sub[y], marker="o", label=name)
    ax.set_xlabel("SNR [dB]")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--run-name", default="initial")
    args = ap.parse_args()
    cfg = load_config(args.config)
    csv_path = Path("outputs/eval") / f"{args.run_name}_eval.csv"
    if not csv_path.exists():
        raise SystemExit(f"Missing evaluation CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    plot_dir = Path("outputs/plots"); plot_dir.mkdir(parents=True, exist_ok=True)
    line_plot(df, "ber", "BER", "Bit error rate versus SNR", plot_dir / f"{args.run_name}_ber_vs_snr.png")
    line_plot(df, "tblER_proxy", "TBLER proxy", "Block error proxy versus SNR", plot_dir / f"{args.run_name}_tbler_proxy_vs_snr.png")
    if "channel_nmse" in df.columns:
        sub = df[df["channel_nmse"].notna()]
        if not sub.empty:
            line_plot(sub, "channel_nmse", "Channel NMSE", "Posterior channel NMSE versus SNR", plot_dir / f"{args.run_name}_channel_nmse_vs_snr.png")
    if "bit_nll" in df.columns:
        normal_plot(df, "bit_nll", "Bit NLL", "Bit negative log-likelihood versus SNR", plot_dir / f"{args.run_name}_bit_nll_vs_snr.png")
    if "ece" in df.columns:
        normal_plot(df, "ece", "ECE", "Bit calibration error versus SNR", plot_dir / f"{args.run_name}_ece_vs_snr.png")
    # Compact ablation table.
    summary = df.groupby("baseline").agg({"ber":"mean", "tblER_proxy":"mean", "bit_nll":"mean", "brier":"mean", "trainable_params":"max"}).reset_index()
    table_path = Path("outputs/reports") / f"{args.run_name}_ablation_table.csv"
    table_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(table_path, index=False)
    save_json({"plots": sorted(str(p) for p in plot_dir.glob(f"{args.run_name}_*.png")), "ablation_table": str(table_path)},
              Path("outputs/reports") / f"{args.run_name}_plot_summary.json")

if __name__ == "__main__":
    main()
