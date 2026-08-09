from __future__ import annotations
from pathlib import Path
from typing import Any, Mapping
import json
import random
import numpy as np
import torch
import yaml

EXPECTED_OPTUNA_SEARCH_SPACE_VERSION = "gate0_v2_4_search_v1"
EXPECTED_OPTUNA_COMPLETE_TRIALS = 12
EXPECTED_OPTUNA_DESIGN_NAME = "space_filling_12"
EXPECTED_OPTUNA_DESIGN_SIGNATURE = "53f3a2614ac172c6ec39515b8bee71f1567da7a42587c42bd93fa9dc54bc1c74"


class AttrDict(dict):
    """Dictionary with attribute access for simple configs."""

    def __getattr__(self, key: str) -> Any:
        try:
            value = self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc
        if isinstance(value, Mapping) and not isinstance(value, AttrDict):
            value = AttrDict(value)
            self[key] = value
        return value

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value

    def to_dict(self) -> dict[str, Any]:
        out = {}
        for k, v in self.items():
            if isinstance(v, AttrDict):
                out[k] = v.to_dict()
            elif isinstance(v, dict):
                out[k] = AttrDict(v).to_dict()
            else:
                out[k] = v
        return out


def load_config(path: str | Path) -> AttrDict:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return AttrDict(data)


def save_json(obj: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True, default=str)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict[str, Any] | None) -> None:
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


def canonical_torch_device(device: torch.device | str) -> torch.device:
    """Return an explicit device accepted by both PyTorch and Sionna.

    Sionna 2.0 enumerates CUDA devices as ``cuda:0``, ``cuda:1``, and so on.
    PyTorch also accepts the shorthand ``cuda``. We normalize the shorthand so
    every component receives the same unambiguous device.
    """
    dev = torch.device(device)
    if dev.type == "cuda" and dev.index is None:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false.")
        dev = torch.device(f"cuda:{torch.cuda.current_device()}")
    return dev


def get_device(cfg: AttrDict | None = None) -> torch.device:
    requested = "auto" if cfg is None else str(cfg.get("device", "auto"))
    if requested == "auto":
        requested = "cuda:0" if torch.cuda.is_available() else "cpu"
    return canonical_torch_device(requested)


def count_parameters(module: torch.nn.Module) -> int:
    return int(sum(p.numel() for p in module.parameters() if p.requires_grad))


def apply_optuna_best(cfg: AttrDict, path: str | Path | None = None) -> tuple[AttrDict, dict]:
    """Apply a completed, revision-matched Optuna result to known fields only."""
    configured = path or cfg.get("optuna_best_path", None)
    if not configured:
        return cfg, {"applied": False, "reason": "no_path_configured"}
    best_path = Path(str(configured))
    if not best_path.exists():
        return cfg, {"applied": False, "reason": "file_missing", "path": str(best_path)}
    data = json.loads(best_path.read_text(encoding="utf-8"))
    expected_revision = str(cfg.get("package_revision", "unknown"))
    result_revision = str(data.get("package_revision", "unknown"))
    if result_revision != expected_revision:
        raise RuntimeError(
            "Optuna/package revision mismatch: "
            f"result={result_revision}, config={expected_revision}"
        )
    complete = int(data.get("n_complete_trials", 0))
    target = int(data.get("target_complete_trials", 0))
    if target != EXPECTED_OPTUNA_COMPLETE_TRIALS or complete < target:
        raise RuntimeError(
            "Optuna result is incomplete or does not contain the required "
            f"{EXPECTED_OPTUNA_COMPLETE_TRIALS}-point design: "
            f"complete={complete}, target={target}"
        )
    if data.get("search_space_version") != EXPECTED_OPTUNA_SEARCH_SPACE_VERSION:
        raise RuntimeError(
            "Unexpected Optuna search-space version: "
            f"{data.get('search_space_version')}"
        )
    if data.get("objective_metric") != "fixed_validation_bit_nll":
        raise RuntimeError(
            f"Unexpected Optuna objective metric: {data.get('objective_metric')}"
        )
    design = data.get("design_report", {})
    if (
        data.get("design_name") != EXPECTED_OPTUNA_DESIGN_NAME
        or data.get("design_signature") != EXPECTED_OPTUNA_DESIGN_SIGNATURE
        or design.get("passed") is not True
        or design.get("signature") != EXPECTED_OPTUNA_DESIGN_SIGNATURE
        or int(design.get("unique_rows", 0)) != 12
    ):
        raise RuntimeError("Optuna result lacks the validated exact 12-point design")
    completed_indices = [int(x) for x in data.get("completed_design_indices", [])]
    if (
        data.get("all_required_design_points_complete") is not True
        or completed_indices != list(range(EXPECTED_OPTUNA_COMPLETE_TRIALS))
        or data.get("missing_design_indices") not in ([], None)
        or data.get("unexpected_trial_numbers") not in ([], None)
    ):
        raise RuntimeError(
            "Optuna result does not contain one successful result for every required design index"
        )
    configured_mass = float(cfg.model.get("edge_mass", 1.0))
    result_mass = float(data.get("fixed_edge_mass", float("nan")))
    if not np.isfinite(result_mass) or abs(result_mass - configured_mass) > 1e-12:
        raise RuntimeError(
            f"Optuna fixed-edge-mass mismatch: result={result_mass}, config={configured_mass}"
        )
    params = dict(data.get("best_params", {}))
    model_fields = {"rank", "detector_iterations"}
    training_fields = {"lr", "channel_loss_weight"}
    expected_fields = model_fields | training_fields
    unknown = set(params) - expected_fields
    missing = expected_fields - set(params)
    if unknown or missing:
        raise RuntimeError(
            "Invalid Optuna parameter fields: "
            f"unknown={sorted(unknown)}, missing={sorted(missing)}"
        )
    for key, value in params.items():
        if key in model_fields:
            cfg.model[key] = value
        elif key in training_fields:
            cfg.training[key] = value
    return cfg, {
        "applied": True,
        "path": str(best_path),
        "package_revision": result_revision,
        "search_space_version": data.get("search_space_version"),
        "contract_signature": data.get("contract_signature"),
        "objective_metric": data.get("objective_metric"),
        "best_value": data.get("best_value"),
        "best_trial_number": data.get("best_trial_number"),
        "best_params": params,
        "n_complete_trials": complete,
        "target_complete_trials": target,
    }
