#!/usr/bin/env python3
from __future__ import annotations
import shutil
from pathlib import Path


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
    copy_tree_files(root / "outputs/eval", dest / "eval", ["*.csv"])
    copy_tree_files(root / "outputs/plots", dest / "plots", ["*.png", "*.pdf"])
    copy_tree_files(root / "outputs/logs", dest / "logs", ["*_train_metrics.csv", "*.txt"])
    (dest / "README.md").write_text(
        "Compact BayesRoute-Rx Gate-0 evidence. Heavy checkpoints and study.db are excluded.\n",
        encoding="utf-8",
    )
    print(f"Prepared {dest}")


if __name__ == "__main__":
    main()
