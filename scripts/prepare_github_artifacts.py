#!/usr/bin/env python3
from __future__ import annotations
import shutil
from pathlib import Path


def copy_tree_files(src: Path, dst: Path, patterns):
    if not src.exists():
        return
    dst.mkdir(parents=True, exist_ok=True)
    for pat in patterns:
        for p in src.glob(pat):
            if p.is_file():
                shutil.copy2(p, dst / p.name)


def main():
    root = Path.cwd()
    dest = root / "outputs/github_artifacts"
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    copy_tree_files(root / "outputs/smoke", dest / "smoke", ["*.json", "*.txt"])
    copy_tree_files(root / "outputs/setup", dest / "setup", ["*.json", "*.txt"])
    copy_tree_files(root / "outputs/optuna", dest / "optuna", ["best_params.json"])
    copy_tree_files(root / "outputs/reports", dest / "reports", ["*.json", "*.csv"])
    copy_tree_files(root / "outputs/eval", dest / "eval", ["*.csv"])
    copy_tree_files(root / "outputs/plots", dest / "plots", ["*.png", "*.pdf"])
    copy_tree_files(root / "outputs/logs", dest / "logs", ["*_train_metrics.csv", "*.txt"])
    (dest / "README.md").write_text("Compact BayesRoute-Rx artifacts for review. Heavy checkpoints are intentionally excluded.\n", encoding="utf-8")
    print(f"Prepared {dest}")

if __name__ == "__main__":
    main()
