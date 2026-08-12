#!/usr/bin/env python3
from __future__ import annotations

"""Conservative reanalysis of the completed clean-generalization campaign.

This script never changes the original 4,800-row evidence or its source contract.
It adds a supplemental report that fixes subgroup labels, reports per-case failure
modes, and records the UMi/UMa delay-spread configuration limitation.
"""

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "outputs/eval/gate1_nr_clean_generalization.csv"
CONFIG_PATH = ROOT / "configs/gate1_nr_clean_generalization.yaml"
NR_SOURCE_PATH = ROOT / "src/bayesroute/nr_gate1.py"
REPORT_PATH = ROOT / "outputs/reports/gate1_nr_clean_generalization_reanalysis.json"
CASE_CSV_PATH = ROOT / "outputs/reports/gate1_nr_clean_generalization_reanalysis_cases.csv"
GATE_JSON_PATH = ROOT / "outputs/gates/GATE1_NR_CLEAN_REANALYSIS.json"
GATE_TXT_PATH = ROOT / "outputs/gates/GATE1_NR_CLEAN_REANALYSIS.txt"
EXPECTED_ROWS = 4800
HIGH_SNR = (6.0, 10.0, 14.0)


def save_json(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def paired_delta(
    frame: pd.DataFrame,
    reference: str,
    comparator: str,
    metric: str,
    *,
    group: str | None = None,
    holdout_only: bool = True,
    multi_stream: bool = True,
) -> dict[str, float]:
    sub = frame[frame["ebno_db"].isin(HIGH_SNR)].copy()
    if holdout_only:
        sub = sub[sub["group"] != "seen_new_seed"]
    if multi_stream:
        sub = sub[sub["num_streams"] >= 4]
    if group is not None:
        sub = sub[sub["group"] == group]
    keys = ["case", "ebno_db", "rep", "eval_seed"]
    a = sub[sub["variant"] == reference]
    b = sub[sub["variant"] == comparator]
    merged = a.merge(b, on=keys, suffixes=("_a", "_b"))
    values = (
        pd.to_numeric(merged[f"{metric}_a"], errors="coerce")
        - pd.to_numeric(merged[f"{metric}_b"], errors="coerce")
    ).dropna()
    mean = float(values.mean()) if len(values) else float("nan")
    std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    if len(values) > 1:
        try:
            from scipy.stats import t as student_t

            critical = float(student_t.ppf(0.975, len(values) - 1))
        except Exception:
            critical = 1.96
    else:
        critical = 0.0
    half = critical * std / math.sqrt(max(len(values), 1))
    return {
        "pairs": int(len(values)),
        "mean": mean,
        "ci95_low": mean - half,
        "ci95_high": mean + half,
    }


def metric_mean(sub: pd.DataFrame, variant: str, metric: str) -> float:
    values = pd.to_numeric(
        sub.loc[sub["variant"] == variant, metric], errors="coerce"
    ).dropna()
    return float(values.mean()) if len(values) else float("nan")


def metric_at_snr(
    sub: pd.DataFrame, variant: str, metric: str, snr: float
) -> float:
    values = pd.to_numeric(
        sub.loc[
            (sub["variant"] == variant) & (sub["ebno_db"] == float(snr)),
            metric,
        ],
        errors="coerce",
    ).dropna()
    return float(values.mean()) if len(values) else float("nan")


def case_table(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    holdout = frame[
        (frame["group"] != "seen_new_seed")
        & (frame["num_streams"] >= 4)
        & (frame["ebno_db"].isin(HIGH_SNR))
    ]
    for case_name in sorted(holdout["case"].unique()):
        sub = holdout[holdout["case"] == case_name]
        frozen = metric_mean(sub, "frozen_global", "tbler")
        ls = metric_mean(sub, "ls_lmmse", "tbler")
        perfect = metric_mean(sub, "perfect_csi_lmmse", "tbler")
        at10 = metric_at_snr(sub, "frozen_global", "tbler", 10.0)
        at14 = metric_at_snr(sub, "frozen_global", "tbler", 14.0)
        posterior = sub[sub["variant"] == "frozen_global"]
        coverage = pd.to_numeric(posterior["coverage95"], errors="coerce").dropna()
        normalized = pd.to_numeric(
            posterior["normalized_error_mean"], errors="coerce"
        ).dropna()
        rows.append(
            {
                "case": case_name,
                "group": str(sub["group"].iloc[0]),
                "num_streams": int(sub["num_streams"].iloc[0]),
                "num_prb": int(sub["num_prb"].iloc[0]),
                "frozen_high_snr_tbler": frozen,
                "ls_high_snr_tbler": ls,
                "perfect_high_snr_tbler": perfect,
                "frozen_minus_ls": frozen - ls,
                "frozen_10db_tbler": at10,
                "frozen_14db_tbler": at14,
                "reversal_14_minus_10": at14 - at10,
                "coverage95_min": float(coverage.min()) if len(coverage) else float("nan"),
                "normalized_error_max": (
                    float(normalized.max()) if len(normalized) else float("nan")
                ),
                "within_0p05_of_ls": bool(frozen - ls <= 0.05),
                "strict_no_reversal": bool(at14 <= at10 + 0.01),
                "coverage_ge_0p85": bool(len(coverage) and coverage.min() >= 0.85),
                "normalized_error_le_2": bool(
                    len(normalized) and normalized.max() <= 2.0
                ),
            }
        )
    return pd.DataFrame(rows)


def semantic_warnings(config: dict[str, Any], nr_source: str) -> list[dict[str, Any]]:
    warnings: list[dict[str, Any]] = []
    system_level_delay_unused = (
        'if scenario == "umi"' in nr_source
        and 'elif scenario == "uma"' in nr_source
        and 'delay_spread=float(case.delay_spread_s)' in nr_source
        and 'elif scenario.startswith("cdl-")' in nr_source
    )
    for case in config["evaluation"]["cases"]:
        if str(case.get("group")) == "delay_holdout" and str(
            case.get("scenario", "")
        ).lower() in {"umi", "uma"}:
            warnings.append(
                {
                    "case": str(case["name"]),
                    "issue": (
                        "delay_spread_s is not passed to Sionna UMi/UMa in "
                        "build_nr_context; this is not a controlled delay-spread holdout"
                    ),
                    "configured_delay_spread_s": float(case["delay_spread_s"]),
                    "confirmed_by_source_audit": bool(system_level_delay_unused),
                    "recommended_label": "new-seed/port-permutation UMa holdout",
                }
            )
    return warnings


def run() -> dict[str, Any]:
    if not RAW_PATH.is_file():
        raise RuntimeError(f"Missing clean-generalization CSV: {RAW_PATH}")
    if not CONFIG_PATH.is_file() or not NR_SOURCE_PATH.is_file():
        raise RuntimeError("Missing clean-generalization config or NR source")
    frame = pd.read_csv(RAW_PATH)
    keys = ["case", "variant", "ebno_db", "rep"]
    unique_rows = int(len(frame.drop_duplicates(keys)))
    if len(frame) != EXPECTED_ROWS or unique_rows != EXPECTED_ROWS:
        raise RuntimeError(
            f"Clean evidence integrity failure: rows={len(frame)}, "
            f"unique={unique_rows}, expected={EXPECTED_ROWS}"
        )
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    per_case = case_table(frame)
    CASE_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    per_case.to_csv(CASE_CSV_PATH, index=False)

    comparisons = {
        "holdout_frozen_minus_old_tbler": paired_delta(
            frame, "frozen_global", "old_checkpoint_repaired_detector", "tbler"
        ),
        "holdout_frozen_minus_uncertainty_off_tbler": paired_delta(
            frame, "frozen_global", "uncertainty_off_fixed_graph", "tbler"
        ),
        "holdout_frozen_minus_uncertainty_off_nll": paired_delta(
            frame,
            "frozen_global",
            "uncertainty_off_fixed_graph",
            "coded_bit_nll",
        ),
        "all_multistream_holdout_frozen_minus_random_tbler": paired_delta(
            frame, "frozen_global", "random_graph_fixed_cardinality", "tbler"
        ),
        "loaded_only_frozen_minus_random_tbler": paired_delta(
            frame,
            "frozen_global",
            "random_graph_fixed_cardinality",
            "tbler",
            group="loaded_holdout",
        ),
        "holdout_frozen_minus_ls_tbler": paired_delta(
            frame, "frozen_global", "ls_lmmse", "tbler"
        ),
    }

    worst_ls = per_case.sort_values("frozen_minus_ls", ascending=False).iloc[0]
    worst_reversal = per_case.sort_values(
        "reversal_14_minus_10", ascending=False
    ).iloc[0]
    worst_coverage = per_case.sort_values("coverage95_min", ascending=True).iloc[0]
    warnings = semantic_warnings(
        config, NR_SOURCE_PATH.read_text(encoding="utf-8")
    )

    expected_variants = len(config["evaluation"]["variants"])
    batch_groups = frame.groupby(["case", "ebno_db", "rep"], dropna=False)
    software_checks = {
        "complete_rows": len(frame) == EXPECTED_ROWS,
        "unique_rows": unique_rows == EXPECTED_ROWS,
        "all_variants_present": set(config["evaluation"]["variants"]).issubset(
            set(frame["variant"].unique())
        ),
        "paired_seed_batches": bool(
            batch_groups["eval_seed"].nunique().eq(1).all()
            and batch_groups["variant"].nunique().eq(expected_variants).all()
        ),
    }
    scientific_checks = {
        "frozen_improves_old_checkpoint": bool(
            comparisons["holdout_frozen_minus_old_tbler"]["ci95_high"] < 0.0
        ),
        "uncertainty_improves_tbler": bool(
            comparisons["holdout_frozen_minus_uncertainty_off_tbler"][
                "ci95_high"
            ]
            < 0.0
        ),
        "uncertainty_improves_nll": bool(
            comparisons["holdout_frozen_minus_uncertainty_off_nll"][
                "ci95_high"
            ]
            < 0.0
        ),
        "loaded_only_coupling_beats_random": bool(
            comparisons["loaded_only_frozen_minus_random_tbler"]["ci95_high"]
            < 0.0
        ),
        "all_cases_within_0p05_of_ls": bool(per_case["within_0p05_of_ls"].all()),
        "strict_no_case_high_snr_reversal": bool(
            per_case["strict_no_reversal"].all()
        ),
        "all_cases_coverage_ge_0p85": bool(per_case["coverage_ge_0p85"].all()),
        "all_cases_normalized_error_le_2": bool(
            per_case["normalized_error_le_2"].all()
        ),
    }
    grid_rows = per_case[per_case["group"] == "grid_holdout"]
    grid_exception = bool(
        len(grid_rows)
        and (
            (~grid_rows["within_0p05_of_ls"]).any()
            or (~grid_rows["strict_no_reversal"]).any()
            or (~grid_rows["coverage_ge_0p85"]).any()
            or (~grid_rows["normalized_error_le_2"]).any()
        )
    )
    scientific_checks["grid_scale_exception_detected"] = grid_exception
    if (
        all(software_checks.values())
        and scientific_checks["frozen_improves_old_checkpoint"]
        and scientific_checks["uncertainty_improves_tbler"]
        and grid_exception
    ):
        classification = "CLEAN_GENERALIZATION_SUPPORTED_WITH_GRID_SCALE_EXCEPTION"
        next_action = "RUN_GRID_SCALE_COORDINATE_AUDIT_BEFORE_PUBLICATION_SCALE"
    elif all(software_checks.values()):
        classification = "CLEAN_GENERALIZATION_REANALYSIS_COMPLETE"
        next_action = "REVIEW_PER_CASE_RESULTS_BEFORE_PUBLICATION_SCALE"
    else:
        classification = "CLEAN_GENERALIZATION_REANALYSIS_BLOCKED"
        next_action = "REPAIR_EVIDENCE_INTEGRITY"

    report = {
        "version": "gate1_nr_clean_reanalysis_v1",
        "complete": True,
        "classification": classification,
        "next_action": next_action,
        "publication_nr_ready": False,
        "raw_evidence": {
            "path": str(RAW_PATH.relative_to(ROOT)),
            "rows": int(len(frame)),
            "unique_rows": unique_rows,
            "expected_rows": EXPECTED_ROWS,
        },
        "software_checks": software_checks,
        "scientific_checks": scientific_checks,
        "paired_comparisons": comparisons,
        "worst_cases": {
            "largest_ls_gap": worst_ls.to_dict(),
            "largest_14db_reversal": worst_reversal.to_dict(),
            "lowest_coverage": worst_coverage.to_dict(),
        },
        "semantic_warnings": warnings,
        "case_table_csv": str(CASE_CSV_PATH.relative_to(ROOT)),
        "original_report_qualifications": {
            "loaded_only_label_was_not_actually_loaded_only": True,
            "within_0p05_of_ls_was_a_pooled_mean_check": True,
            "high_snr_reversal_check_allowed_plus_0p03": True,
            "posterior_check_used_pooled_mean_or_median": True,
        },
    }
    save_json(report, REPORT_PATH)
    save_json(report, GATE_JSON_PATH)
    gate_lines = [
        f"complete_rows: {'PASS' if software_checks['complete_rows'] else 'FAIL'}",
        f"unique_rows: {'PASS' if software_checks['unique_rows'] else 'FAIL'}",
        f"frozen_improves_old_checkpoint: {'PASS' if scientific_checks['frozen_improves_old_checkpoint'] else 'FAIL'}",
        f"uncertainty_improves_tbler: {'PASS' if scientific_checks['uncertainty_improves_tbler'] else 'FAIL'}",
        f"loaded_only_coupling_beats_random: {'PASS' if scientific_checks['loaded_only_coupling_beats_random'] else 'FAIL'}",
        f"all_cases_within_0p05_of_ls: {'PASS' if scientific_checks['all_cases_within_0p05_of_ls'] else 'FAIL'}",
        f"strict_no_case_high_snr_reversal: {'PASS' if scientific_checks['strict_no_case_high_snr_reversal'] else 'FAIL'}",
        f"all_cases_coverage_ge_0p85: {'PASS' if scientific_checks['all_cases_coverage_ge_0p85'] else 'FAIL'}",
        f"grid_scale_exception_detected: {'YES' if grid_exception else 'NO'}",
        f"umi_uma_delay_spread_field_not_applied: {'YES' if warnings else 'NO'}",
        f"CLASSIFICATION: {classification}",
        f"NEXT_ACTION: {next_action}",
        "PUBLICATION_NR_READY: NO",
    ]
    GATE_TXT_PATH.parent.mkdir(parents=True, exist_ok=True)
    GATE_TXT_PATH.write_text("\n".join(gate_lines) + "\n", encoding="utf-8")
    print("\n".join(gate_lines))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--print-json", action="store_true")
    args = parser.parse_args()
    report = run()
    if args.print_json:
        print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
