#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bayesroute.nr_gate1 import (
    GATE1_NR_VERSION,
    NRCase,
    build_transmitter,
    extract_nr_grid,
    pilot_orthogonality_report,
)


def load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected mapping in {path}")
    return value


def gate0_classification() -> str | None:
    path = ROOT / "outputs/gates/GATE0_MECHANISM_DIAGNOSTIC.txt"
    if not path.is_file():
        return None
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("CLASSIFICATION:"):
            return line.split(":", 1)[1].strip()
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-config", default="configs/gate1_nr_smoke.yaml")
    parser.add_argument("--evidence-config", default="configs/gate1_nr_evidence.yaml")
    parser.add_argument("--out", default="outputs/gates/GATE1_NR_PREFLIGHT.json")
    args = parser.parse_args()

    import sionna
    import sionna.phy

    if str(getattr(sionna, "__version__", "")) != "2.0.1":
        raise RuntimeError(f"Gate-1 requires Sionna 2.0.1, found {sionna.__version__}")
    sionna.phy.config.device = "cpu"
    smoke = load_yaml(Path(args.smoke_config))
    evidence = load_yaml(Path(args.evidence_config))
    if smoke.get("gate1_revision") != GATE1_NR_VERSION:
        raise RuntimeError("Smoke config/revision mismatch")
    if evidence.get("gate1_revision") != GATE1_NR_VERSION:
        raise RuntimeError("Evidence config/revision mismatch")

    all_case_mappings: list[dict[str, Any]] = []
    all_case_mappings.extend(smoke["mapping_cases"])
    all_case_mappings.extend(smoke["channel_cases"])
    all_case_mappings.append(evidence["training"]["source_case"])
    all_case_mappings.extend(evidence["evaluation"]["cases"])
    parsed: list[NRCase] = []
    seen: set[str] = set()
    for raw in all_case_mappings:
        case = NRCase.from_mapping(raw)
        case.validate()
        if case.name not in seen:
            parsed.append(case)
            seen.add(case.name)

    mapping_results: dict[str, Any] = {}
    for raw in smoke["mapping_cases"]:
        case = NRCase.from_mapping(raw)
        transmitter, configs = build_transmitter(case, torch.device("cpu"))
        grid = extract_nr_grid(transmitter, configs, torch.device("cpu"))
        orthogonality = pilot_orthogonality_report(grid.phi)
        mapping_results[case.name] = {
            "passed": bool(
                orthogonality["passed"]
                and grid.num_streams == case.num_streams
                and len(grid.port_metadata) == case.num_streams
                and [int(x["dmrs_port"]) for x in grid.port_metadata]
                == list(case.dmrs_ports)
                and grid.num_data_symbols > 0
                and grid.num_pilot_observations > 0
            ),
            "num_users": case.num_users,
            "num_layers_per_user": case.num_layers_per_user,
            "num_streams": case.num_streams,
            "ports": list(case.dmrs_ports),
            "phi_shape": list(grid.phi.shape),
            "data_symbols_per_stream": grid.num_data_symbols,
            "orthogonality": orthogonality,
        }

    base_revision = json.loads((ROOT / "PACKAGE_REVISION.json").read_text(encoding="utf-8"))
    gate1_revision = json.loads((ROOT / "GATE1_NR_REVISION.json").read_text(encoding="utf-8"))
    gate0 = gate0_classification()
    checks = {
        "base_revision": base_revision.get("revision") == "gate0_v2_4_20260809",
        "gate1_revision": gate1_revision.get("revision") == GATE1_NR_VERSION,
        "gate0_mechanism_supported": gate0 == "GATE0_MECHANISM_SUPPORTED",
        "sionna_2_0_1": str(getattr(sionna, "__version__", "")) == "2.0.1",
        "all_cases_parse": len(parsed) == len(seen),
        "all_mapping_configs_construct": all(x["passed"] for x in mapping_results.values()),
        "pgca_agmp_excluded": gate1_revision.get("pgca_agmp_baseline_included") is False,
    }
    report = {
        "passed": all(checks.values()),
        "gate1_revision": GATE1_NR_VERSION,
        "checks": checks,
        "gate0_classification": gate0,
        "parsed_cases": [case.__dict__ | {"num_streams": case.num_streams} for case in parsed],
        "mapping_results": mapping_results,
        "pgca_agmp_baseline_included": False,
    }
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    if not report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
