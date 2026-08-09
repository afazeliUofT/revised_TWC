#!/usr/bin/env python3
from __future__ import annotations

import json
import shutil
from pathlib import Path

MAX_SLURM_FILES = 20
MAX_SLURM_BYTES_PER_FILE = 512 * 1024


def copy_tree_files(src: Path, dst: Path, patterns: list[str]) -> None:
    if not src.exists():
        return
    dst.mkdir(parents=True, exist_ok=True)
    seen: set[Path] = set()
    for pattern in patterns:
        for path in src.glob(pattern):
            if path.is_file() and path not in seen:
                shutil.copy2(path, dst / path.name)
                seen.add(path)


def copy_recent_slurm_logs(src: Path, dst: Path) -> list[dict]:
    """Copy compact tails of recent Slurm stdout/stderr files for remote diagnosis."""
    if not src.exists():
        return []
    candidates = [
        p for pattern in ("*.out", "*.err") for p in src.glob(pattern) if p.is_file()
    ]
    candidates = sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)
    dst.mkdir(parents=True, exist_ok=True)
    index: list[dict] = []
    for path in candidates[:MAX_SLURM_FILES]:
        stat = path.stat()
        raw = path.read_bytes()
        truncated = len(raw) > MAX_SLURM_BYTES_PER_FILE
        if truncated:
            raw = raw[-MAX_SLURM_BYTES_PER_FILE:]
            header = (
                f"[truncated to final {MAX_SLURM_BYTES_PER_FILE} bytes; "
                f"original size {stat.st_size} bytes]\n"
            ).encode("utf-8")
            raw = header + raw
        target = dst / path.name
        target.write_bytes(raw)
        index.append(
            {
                "name": path.name,
                "original_size_bytes": int(stat.st_size),
                "copied_size_bytes": int(len(raw)),
                "truncated": bool(truncated),
                "mtime_epoch": float(stat.st_mtime),
            }
        )
    return index


def main() -> None:
    root = Path.cwd()
    dest = root / "outputs/github_artifacts"
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)

    copy_tree_files(root / "outputs/smoke", dest / "smoke", ["*.json", "*.txt"])
    copy_tree_files(root / "outputs/setup", dest / "setup", ["*.json", "*.txt"])
    copy_tree_files(
        root / "outputs/optuna",
        dest / "optuna",
        ["best_params.json", "OPTUNA_STATUS.json", "trials.csv"],
    )
    copy_tree_files(root / "outputs/reports", dest / "reports", ["*.json", "*.csv"])
    copy_tree_files(root / "outputs/eval", dest / "eval", ["*.json", "*.csv"])
    copy_tree_files(root / "outputs/plots", dest / "plots", ["*.png", "*.pdf"])
    copy_tree_files(root / "outputs/logs", dest / "logs", ["*_train_metrics.csv", "*.txt"])
    copy_tree_files(root / "outputs/gates", dest / "gates", ["*.json", "*.txt"])

    slurm_index = copy_recent_slurm_logs(root / "outputs/slurm", dest / "slurm")
    if slurm_index:
        (dest / "slurm" / "SLURM_LOG_INDEX.json").write_text(
            json.dumps(slurm_index, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    (dest / "README.md").write_text(
        "Compact BayesRoute-Rx Gate-0 evidence. Heavy checkpoints and study.db are excluded. "
        "Recent Slurm log tails are included for diagnosis.\n",
        encoding="utf-8",
    )
    print(f"Prepared {dest}")


if __name__ == "__main__":
    main()
