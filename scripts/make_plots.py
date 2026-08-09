#!/usr/bin/env python3
from __future__ import annotations
import argparse
import math
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from bayesroute.config import load_config, save_json


def _aggregate(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    out = (
        df[df[metric].notna()]
        .groupby(["baseline", "snr_db"], as_index=False)[metric]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    out["std"] = out["std"].fillna(0.0)
    out["half_ci95"] = 1.96 * out["std"] / out["count"].clip(lower=1).pow(0.5)
    return out


def _plot_metric(df: pd.DataFrame, metric: str, ylabel: str, title: str,
                 out_path: Path, log_y: bool = False) -> None:
    agg = _aggregate(df, metric)
    if agg.empty:
        return
    fig = plt.figure(figsize=(7.2, 4.8))
    ax = fig.add_subplot(111)
    for name, sub in agg.groupby("baseline"):
        sub = sub.sort_values("snr_db")
        y = sub["mean"].clip(lower=1e-8) if log_y else sub["mean"]
        lower = (sub["mean"] - sub["half_ci95"])
        upper = (sub["mean"] + sub["half_ci95"])
        if log_y:
            lower = lower.clip(lower=1e-8)
        ax.plot(sub["snr_db"], y, marker="o", label=name)
        ax.fill_between(sub["snr_db"], lower, upper, alpha=0.15)
    if log_y:
        ax.set_yscale("log")
    ax.set_xlabel("SNR [dB]")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=240)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--run-name", default="initial")
    args = ap.parse_args()
    load_config(args.config)  # validates YAML and records the intended run contract
    csv_path = Path("outputs/eval") / f"{args.run_name}_eval.csv"
    if not csv_path.exists():
        raise SystemExit(f"Missing evaluation CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    plot_dir = Path("outputs/plots")
    report_dir = Path("outputs/reports")
    plot_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    plot_specs = [
        ("ber", "Uncoded BER", "Gate-0 uncoded BER versus SNR", True),
        ("bit_nll", "Bit NLL", "Gate-0 bit negative log-likelihood", False),
        ("brier", "Brier score", "Gate-0 bit-probability Brier score", False),
        ("ece", "Expected calibration error", "Gate-0 bit calibration", False),
        ("channel_nmse", "Channel NMSE", "Gate-0 posterior channel NMSE", True),
        ("channel_marginal_nll", "Channel marginal NLL", "Gate-0 posterior channel marginal NLL", False),
        ("channel_coverage95", "Empirical 95% coverage", "Gate-0 posterior marginal coverage", False),
        ("edge_density", "Retained directed-edge fraction", "Posterior coupling graph density", False),
        ("receiver_ms_per_sample", "Receiver latency [ms/sample]", "Measured receiver-only latency", False),
        ("receiver_samples_per_sec", "Receiver throughput [samples/s]", "Measured receiver throughput", False),
        ("receiver_incremental_peak_memory_mib", "Incremental peak memory [MiB]", "Receiver incremental peak GPU memory", False),
    ]
    made = []
    for metric, ylabel, title, log_y in plot_specs:
        if metric not in df.columns or not df[metric].notna().any():
            continue
        out = plot_dir / f"{args.run_name}_{metric}_vs_snr.png"
        _plot_metric(df, metric, ylabel, title, out, log_y=log_y)
        made.append(str(out))

    paired_path = Path("outputs/eval") / f"{args.run_name}_paired_deltas.csv"
    if paired_path.exists():
        paired = pd.read_csv(paired_path)
        for metric, ylabel, title in [
            (
                "bit_nll_delta_proposed_minus_comparator",
                "Bit-NLL delta: proposed - comparator",
                "Paired Gate-0 bit-NLL differences",
            ),
            (
                "ber_delta_proposed_minus_comparator",
                "BER delta: proposed - comparator",
                "Paired Gate-0 BER differences",
            ),
            (
                "receiver_ms_per_sample_delta_proposed_minus_comparator",
                "Receiver-latency delta [ms/sample]",
                "Paired receiver-only latency differences",
            ),
        ]:
            agg = (
                paired.groupby(["comparator", "snr_db"], as_index=False)[metric]
                .agg(["mean", "std", "count"])
                .reset_index()
            )
            agg["std"] = agg["std"].fillna(0.0)
            agg["half_ci95"] = 1.96 * agg["std"] / agg["count"].clip(lower=1).pow(0.5)
            fig = plt.figure(figsize=(7.2, 4.8))
            ax = fig.add_subplot(111)
            for comparator, sub in agg.groupby("comparator"):
                sub = sub.sort_values("snr_db")
                ax.plot(sub["snr_db"], sub["mean"], marker="o", label=comparator)
                ax.fill_between(
                    sub["snr_db"],
                    sub["mean"] - sub["half_ci95"],
                    sub["mean"] + sub["half_ci95"],
                    alpha=0.15,
                )
            ax.axhline(0.0, linewidth=1.0)
            ax.set_xlabel("SNR [dB]")
            ax.set_ylabel(ylabel)
            ax.set_title(title + " (negative favors proposed)")
            ax.grid(True, alpha=0.3)
            ax.legend()
            fig.tight_layout()
            out = plot_dir / f"{args.run_name}_{metric}_vs_snr.png"
            fig.savefig(out, dpi=240)
            plt.close(fig)
            made.append(str(out))

    numeric = [
        c for c in [
            "ber", "bit_nll", "brier", "ece", "channel_nmse",
            "channel_marginal_nll", "channel_coverage95", "edge_density",
            "receiver_ms_per_sample", "receiver_samples_per_sec",
            "receiver_peak_memory_mib", "receiver_incremental_peak_memory_mib",
            "trainable_params",
        ] if c in df.columns
    ]
    summary = df.groupby("baseline", as_index=False)[numeric].mean(numeric_only=True)
    table_path = report_dir / f"{args.run_name}_ablation_table.csv"
    summary.to_csv(table_path, index=False)

    save_json(
        {
            "scope": "GATE0_PRINCIPLE_ONLY_NOT_CODED_TBLER",
            "source_csv": str(csv_path),
            "plots": made,
            "ablation_table": str(table_path),
            "note": "tblER_proxy is intentionally excluded. Latency is receiver-forward-only; batch generation and diagnostics are excluded.",
        },
        report_dir / f"{args.run_name}_plot_summary.json",
    )


if __name__ == "__main__":
    main()
