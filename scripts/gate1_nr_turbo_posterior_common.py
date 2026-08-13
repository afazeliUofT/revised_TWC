#!/usr/bin/env python3
from __future__ import annotations

"""Common integration helpers for the one-step turbo posterior gate."""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]

from bayesroute.nr_gate1 import NRCase, build_nr_context
from bayesroute.turbo_posterior import (
    LatentPosteriorState,
    latent_posterior_from_pilots,
    posterior_batch_metrics,
    soft_data_posterior_update,
)
from gate1_nr_joint_operator_common import (
    make_repaired_detector,
    posterior_graph,
    repaired_forward,
)
from gate1_nr_posterior_factorial_common import (
    FactorialCandidate,
    build_candidate_bridge,
    load_candidate_state,
    package_signature,
    set_all_seeds,
    sha256_file,
)


TURBO_GATE_VERSION = "gate1_nr_turbo_posterior_v1"
REQUIRED_EXTENSION_CLASSIFICATION = "GATE1_TURBO_REFINEMENT_REQUIRED"
REQUIRED_EXTENSION_NEXT_ACTION = "ADD_ONE_PRINCIPLED_DATA_AIDED_POSTERIOR_UPDATE"
REQUIRED_WINNER = "physical_context_multiscale_r64"
REQUIRED_EXTENSION_CHECKPOINT_SHA256 = (
    "45b8439c9aa3be7d9ee2d04c77352522129eee75ab2dfd47ad0bbf988da89e2a"
)
EXTENSION_REPORT_PATH = ROOT / "outputs/reports/gate1_nr_posterior_extension.json"
EXTENSION_CHECKPOINT_PATH = (
    ROOT / "outputs/gate1_nr_posterior_extension/checkpoints/best.pt"
)


@dataclass(frozen=True)
class TurboSetting:
    name: str
    information_damping: float
    data_fraction: float
    min_observations: int
    max_observations: int

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> "TurboSetting":
        result = cls(
            name=str(value["name"]),
            information_damping=float(value["information_damping"]),
            data_fraction=float(value["data_fraction"]),
            min_observations=int(value.get("min_observations", 16)),
            max_observations=int(value["max_observations"]),
        )
        result.validate()
        return result

    def validate(self) -> None:
        if not self.name:
            raise ValueError("Turbo setting name must be nonempty")
        if not 0.0 < self.information_damping <= 1.0:
            raise ValueError("information_damping must lie in (0,1]")
        if not 0.0 < self.data_fraction <= 1.0:
            raise ValueError("data_fraction must lie in (0,1]")
        if self.min_observations <= 0:
            raise ValueError("min_observations must be positive")
        if self.max_observations < self.min_observations:
            raise ValueError("max_observations must be at least min_observations")

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "information_damping": self.information_damping,
            "data_fraction": self.data_fraction,
            "min_observations": self.min_observations,
            "max_observations": self.max_observations,
        }


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected JSON object: {path}")
    return value


def sha256_bytes(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def extension_preconditions() -> dict[str, Any]:
    if not EXTENSION_REPORT_PATH.is_file():
        raise RuntimeError(f"Missing extension report: {EXTENSION_REPORT_PATH}")
    if not EXTENSION_CHECKPOINT_PATH.is_file():
        raise RuntimeError(f"Missing extension checkpoint: {EXTENSION_CHECKPOINT_PATH}")
    report = load_json(EXTENSION_REPORT_PATH)
    checkpoint_sha = sha256_bytes(EXTENSION_CHECKPOINT_PATH)
    checks = {
        "extension_complete": report.get("complete") is True,
        "extension_classification": (
            report.get("classification") == REQUIRED_EXTENSION_CLASSIFICATION
        ),
        "extension_next_action": (
            report.get("next_action") == REQUIRED_EXTENSION_NEXT_ACTION
        ),
        "extension_rows": (
            report.get("evaluation", {}).get("rows") == 648
            and report.get("evaluation", {}).get("unique_rows") == 648
        ),
        "winner": (
            report.get("training", {}).get("winner") == REQUIRED_WINNER
        ),
        "training_converged": (
            report.get("training", {}).get("training_converged") is True
        ),
        "fresh_holdout": (
            report.get("training", {}).get(
                "fresh_12prb_used_for_training_or_selection"
            )
            is False
        ),
        "checkpoint_hash": checkpoint_sha == REQUIRED_EXTENSION_CHECKPOINT_SHA256,
        "recorded_checkpoint_hash": (
            report.get("training", {}).get("best_checkpoint_sha256")
            == REQUIRED_EXTENSION_CHECKPOINT_SHA256
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(f"Turbo posterior preconditions failed: {checks}")
    return {
        "passed": True,
        "checks": checks,
        "report": str(EXTENSION_REPORT_PATH.relative_to(ROOT)),
        "checkpoint": str(EXTENSION_CHECKPOINT_PATH.relative_to(ROOT)),
        "checkpoint_sha256": checkpoint_sha,
        "classification": report["classification"],
        "next_action": report["next_action"],
        "extension_metrics": report.get("metrics", {}),
    }


def winner_spec_from_report() -> FactorialCandidate:
    report = load_json(EXTENSION_REPORT_PATH)
    mapping = report.get("training", {}).get("winner_spec")
    if not isinstance(mapping, dict):
        raise RuntimeError("Extension report does not contain winner_spec")
    spec = FactorialCandidate.from_mapping(mapping)
    if spec.name != REQUIRED_WINNER:
        raise RuntimeError("Extension winner identity mismatch")
    return spec


def build_loaded_bridge(case: NRCase, context: Any, *, operator_seed: int) -> Any:
    spec = winner_spec_from_report()
    bridge = build_candidate_bridge(
        case,
        context,
        spec,
        operator_seed=int(operator_seed),
    )
    checkpoint = torch.load(
        EXTENSION_CHECKPOINT_PATH,
        map_location=context.device,
        weights_only=False,
    )
    if checkpoint.get("winner") != REQUIRED_WINNER:
        raise RuntimeError("Extension checkpoint winner mismatch")
    load_candidate_state(spec, bridge, checkpoint["operator"])
    bridge.eval()
    return bridge


def pilot_state_and_reference(
    bridge: Any,
    batch: Any,
) -> tuple[LatentPosteriorState, torch.Tensor, dict[str, Any]]:
    state = latent_posterior_from_pilots(
        bridge.posterior,
        batch.y[..., batch.pilot_idx],
        batch.phi,
        batch.noise_var,
    )
    public = bridge.posterior(
        batch.y[..., batch.pilot_idx],
        batch.phi,
        batch.noise_var,
    )
    mean_error = torch.max(torch.abs(state.posterior.mean - public.mean))
    covariance_error = torch.max(
        torch.abs(state.posterior.latent_cov - public.latent_cov)
    )
    if float(mean_error.item()) > 2e-5 or float(covariance_error.item()) > 2e-5:
        raise RuntimeError(
            "Exposed latent posterior does not match the public operator forward"
        )
    _, reference = posterior_graph(state.posterior, batch)
    return state, reference, {
        "public_mean_max_abs_error": float(mean_error.item()),
        "public_latent_cov_max_abs_error": float(covariance_error.item()),
        "passed": True,
    }


def true_data_symbols(batch: Any) -> torch.Tensor:
    grid = batch.x_grid.reshape(
        batch.x_grid.shape[0], batch.x_grid.shape[1], -1
    )
    return grid[..., batch.data_idx].contiguous()


def initial_detector_output(
    bridge: Any,
    detector: Any,
    batch: Any,
    state: LatentPosteriorState,
    reference_graph: torch.Tensor,
) -> dict[str, Any]:
    return repaired_forward(
        bridge,
        detector,
        batch,
        posterior=state.posterior,
        reference_graph=reference_graph,
    )


def turbo_forward(
    bridge: Any,
    detector: Any,
    batch: Any,
    setting: TurboSetting,
    *,
    state: LatentPosteriorState | None = None,
    reference_graph: torch.Tensor | None = None,
    initial_output: dict[str, Any] | None = None,
    oracle_symbols: bool = False,
) -> dict[str, Any]:
    if state is None or reference_graph is None:
        state, reference_graph, _ = pilot_state_and_reference(bridge, batch)
    if initial_output is None:
        initial_output = initial_detector_output(
            bridge, detector, batch, state, reference_graph
        )
    if oracle_symbols:
        symbol_mean = true_data_symbols(batch)
        symbol_var = torch.zeros_like(symbol_mean.real)
    else:
        symbol_mean = initial_output["x_mean"]
        symbol_var = initial_output["x_var"]
    refined, diagnostics, selected = soft_data_posterior_update(
        state,
        y=batch.y,
        data_idx=batch.data_idx,
        noise_var=batch.noise_var,
        symbol_mean=symbol_mean,
        symbol_var=symbol_var,
        information_damping=float(setting.information_damping),
        data_fraction=float(setting.data_fraction),
        min_observations=int(setting.min_observations),
        max_observations=int(setting.max_observations),
    )
    output = repaired_forward(
        bridge,
        detector,
        batch,
        posterior=refined,
        reference_graph=reference_graph,
    )
    if not torch.equal(output["graph_mask"], reference_graph):
        raise RuntimeError("Turbo refinement changed the fixed routing graph")
    output["turbo_diagnostics"] = diagnostics.__dict__
    output["selected_data_indices"] = selected
    output["turbo_setting"] = setting.as_dict()
    output["oracle_symbols"] = bool(oracle_symbols)
    output["initial_output"] = initial_output
    output["posterior_metrics"] = posterior_batch_metrics(
        refined, batch.h, batch.data_idx
    )
    return output


def experiment_signature(payload: dict[str, Any]) -> str:
    return package_signature(payload)


def source_hashes(paths: list[str]) -> dict[str, str]:
    return {path: sha256_file(ROOT / path) for path in paths}


def make_case_context(raw_case: dict[str, Any], device: torch.device) -> tuple[NRCase, Any]:
    case = NRCase.from_mapping(raw_case)
    case.validate()
    return case, build_nr_context(case, device)
