#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bayesroute.config import (
    apply_optuna_best,
    get_device,
    load_config,
    save_json,
    set_seed,
)
from bayesroute.eval_utils import evaluate_one_batch, make_receiver, warmup_receiver
from bayesroute.simulator import UplinkToySimulator

EVALUATION_CONTRACT_VERSION = "gate0_v2_4_eval_v1"


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _contract(cfg, run_name: str, checkpoint: Path, baselines: list[str]) -> dict[str, Any]:
    contract: dict[str, Any] = {
        "contract_version": EVALUATION_CONTRACT_VERSION,
        "package_revision": cfg.get("package_revision"),
        "run_name": run_name,
        "system": cfg.system.to_dict(),
        "model": cfg.model.to_dict(),
        "evaluation": cfg.evaluation.to_dict(),
        "baselines": baselines,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256_file(checkpoint),
    }
    contract["signature"] = hashlib.sha256(
        _canonical_json(contract).encode("utf-8")
    ).hexdigest()
    return contract


def _load_contract(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _remove_run_outputs(eval_dir: Path, run_name: str) -> None:
    for path in eval_dir.glob(f"{run_name}_*"):
        if path.is_file():
            path.unlink()


def main() -> None:
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
    baselines = [str(x) for x in cfg.evaluation.baselines]
    if any(name.startswith("bayesroute") for name in baselines) and not args.checkpoint:
        raise SystemExit("BayesRoute evaluation requires --checkpoint")
    checkpoint = Path(str(args.checkpoint)).resolve()
    if not checkpoint.is_file():
        raise SystemExit(f"Evaluation checkpoint is missing: {checkpoint}")

    set_seed(int(cfg.seed) + 900)
    simulator = UplinkToySimulator(cfg, device)
    eval_dir = Path("outputs/eval")
    report_dir = Path("outputs/reports")
    eval_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    out_csv = eval_dir / f"{args.run_name}_eval.csv"
    contract_path = eval_dir / f"{args.run_name}_eval_contract.json"
    current_contract = _contract(cfg, args.run_name, checkpoint, baselines)

    if args.resume and out_csv.exists():
        if not contract_path.exists():
            raise RuntimeError(
                "Existing evaluation CSV has no contract. Use --no-resume or a new run name."
            )
        saved_contract = _load_contract(contract_path)
        if saved_contract.get("signature") != current_contract["signature"]:
            raise RuntimeError(
                "Evaluation resume contract mismatch. Use --no-resume or a new run name."
            )
    else:
        _remove_run_outputs(eval_dir, args.run_name)
        save_json(current_contract, contract_path)

    done: set[tuple[str, float, int]] = set()
    if args.resume and out_csv.exists():
        old = pd.read_csv(out_csv)
        required_columns = {"baseline", "snr_db", "rep", "eval_seed"}
        missing = required_columns - set(old.columns)
        if missing:
            raise RuntimeError(f"Evaluation CSV is missing columns: {sorted(missing)}")
        keys = old[["baseline", "snr_db", "rep"]].astype(
            {"baseline": str, "snr_db": float, "rep": int}
        )
        if keys.duplicated().any():
            raise RuntimeError("Evaluation CSV contains duplicate baseline/SNR/rep rows")
        for _, row in old.iterrows():
            done.add((str(row["baseline"]), float(row["snr_db"]), int(row["rep"])))
    elif out_csv.exists():
        out_csv.unlink()

    # Evaluate one receiver at a time. This avoids keeping every ablation model
    # resident on the GPU and makes the peak-memory numbers interpretable.
    n_reps = int(cfg.evaluation.batches_per_snr)
    batch_size = int(cfg.evaluation.batch_size)
    warmup_snr = float(cfg.evaluation.snr_grid_db[len(cfg.evaluation.snr_grid_db) // 2])
    warmup_batch_size = min(16, batch_size)
    for baseline_index, baseline in enumerate(baselines):
        ckpt = str(checkpoint) if baseline.startswith("bayesroute") else None
        model = make_receiver(baseline, cfg, simulator, ckpt)
        warmup_receiver(
            baseline,
            model,
            simulator,
            snr_db=warmup_snr,
            batch_size=warmup_batch_size,
            seed=int(cfg.seed) + 800000 + baseline_index,
        )
        for snr_index, snr in enumerate(cfg.evaluation.snr_grid_db):
            for rep in range(n_reps):
                key = (baseline, float(snr), rep)
                if key in done:
                    continue
                # Baseline-independent seed gives paired batches across receivers.
                eval_seed = int(cfg.seed) + 1_000_000 + 10_000 * snr_index + rep
                set_seed(eval_seed)
                row = evaluate_one_batch(
                    baseline,
                    model,
                    simulator,
                    float(snr),
                    batch_size,
                )
                row["rep"] = rep
                row["eval_seed"] = eval_seed
                row["evaluation_contract_signature"] = current_contract["signature"]
                pd.DataFrame([row]).to_csv(
                    out_csv,
                    mode="a",
                    header=not out_csv.exists(),
                    index=False,
                )
                done.add(key)
                print(json.dumps(row), flush=True)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    df = pd.read_csv(out_csv)
    key_columns = ["baseline", "snr_db", "rep"]
    unique_rows = int(len(df.drop_duplicates(key_columns)))
    expected_rows = int(len(baselines) * len(cfg.evaluation.snr_grid_db) * n_reps)

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
            receiver_ms_per_sample_mean=("receiver_ms_per_sample", "mean"),
            receiver_samples_per_sec_mean=("receiver_samples_per_sec", "mean"),
            receiver_peak_memory_mib_mean=("receiver_peak_memory_mib", "mean"),
            receiver_incremental_peak_memory_mib_mean=(
                "receiver_incremental_peak_memory_mib", "mean"
            ),
            reps=("rep", "count"),
        )
    )
    aggregate_path = eval_dir / f"{args.run_name}_eval_aggregate.csv"
    grouped.to_csv(aggregate_path, index=False)

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
                paired_rows.append(
                    {
                        "comparator": comparator,
                        "snr_db": float(row["snr_db"]),
                        "rep": int(row["rep"]),
                        "eval_seed": int(row["eval_seed"]),
                        "ber_delta_proposed_minus_comparator": float(
                            row["ber_proposed"] - row["ber_comparator"]
                        ),
                        "bit_nll_delta_proposed_minus_comparator": float(
                            row["bit_nll_proposed"] - row["bit_nll_comparator"]
                        ),
                        "ece_delta_proposed_minus_comparator": float(
                            row["ece_proposed"] - row["ece_comparator"]
                        ),
                        "receiver_ms_per_sample_delta_proposed_minus_comparator": float(
                            row["receiver_ms_per_sample_proposed"]
                            - row["receiver_ms_per_sample_comparator"]
                        ),
                    }
                )
    paired_df = pd.DataFrame(paired_rows)
    if not paired_df.empty:
        paired_df.to_csv(paired_path, index=False)
    elif paired_path.exists():
        paired_path.unlink()

    summary: dict[str, Any] = {
        "run_name": args.run_name,
        "scope": "GATE0_PRINCIPLE_ONLY_NOT_PUBLICATION_TBLER",
        "evaluation_contract_version": EVALUATION_CONTRACT_VERSION,
        "evaluation_contract_signature": current_contract["signature"],
        "checkpoint_sha256": current_contract["checkpoint_sha256"],
        "csv": str(out_csv),
        "aggregate_csv": str(aggregate_path),
        "rows": int(len(df)),
        "unique_rows": unique_rows,
        "expected_rows": expected_rows,
        "complete": bool(len(df) == expected_rows and unique_rows == expected_rows),
        "baselines": baselines,
        "snr_grid_db": [float(x) for x in cfg.evaluation.snr_grid_db],
        "optuna": optuna_meta,
        "paired_delta_csv": str(paired_path) if not paired_df.empty else None,
        "delta_sign_convention": "negative means BayesRoute uncertainty is better/lower",
        "latency_definition": "receiver forward pass only; batch generation and diagnostics excluded",
    }
    if not paired_df.empty:
        for comparator, sub in paired_df.groupby("comparator"):
            summary[f"mean_bit_nll_delta_vs_{comparator}"] = float(
                sub["bit_nll_delta_proposed_minus_comparator"].mean()
            )
            summary[f"mean_ber_delta_vs_{comparator}"] = float(
                sub["ber_delta_proposed_minus_comparator"].mean()
            )
            summary[f"mean_receiver_ms_per_sample_delta_vs_{comparator}"] = float(
                sub[
                    "receiver_ms_per_sample_delta_proposed_minus_comparator"
                ].mean()
            )
    save_json(summary, report_dir / f"{args.run_name}_eval_summary.json")


if __name__ == "__main__":
    main()
