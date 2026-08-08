from __future__ import annotations
from pathlib import Path
from typing import Any, Mapping
import json
import random
import numpy as np
import torch
import yaml


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


def get_device(cfg: AttrDict | None = None) -> torch.device:
    requested = "auto" if cfg is None else str(cfg.get("device", "auto"))
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def count_parameters(module: torch.nn.Module) -> int:
    return int(sum(p.numel() for p in module.parameters() if p.requires_grad))


def apply_optuna_best(cfg: AttrDict, path: str | Path | None = None) -> tuple[AttrDict, dict]:
    """Apply only known Optuna fields and return the effective config plus metadata."""
    configured = path or cfg.get("optuna_best_path", None)
    if not configured:
        return cfg, {"applied": False, "reason": "no_path_configured"}
    best_path = Path(str(configured))
    if not best_path.exists():
        return cfg, {"applied": False, "reason": "file_missing", "path": str(best_path)}
    data = json.loads(best_path.read_text(encoding="utf-8"))
    params = dict(data.get("best_params", {}))
    model_fields = {"rank", "detector_iterations", "edge_mass"}
    training_fields = {"lr", "channel_loss_weight"}
    for key, value in params.items():
        if key in model_fields:
            cfg.model[key] = value
        elif key in training_fields:
            cfg.training[key] = value
    return cfg, {
        "applied": True,
        "path": str(best_path),
        "best_value": data.get("best_value"),
        "best_params": params,
    }
