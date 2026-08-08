#!/usr/bin/env python3
from __future__ import annotations
import argparse, shutil
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["light", "hard"], default="light")
    args = ap.parse_args()
    root = Path("outputs")
    if args.mode == "light":
        for pat in ["tmp", "raw", "__pycache__"]:
            for p in root.rglob(pat):
                if p.is_dir(): shutil.rmtree(p, ignore_errors=True)
        print("Light cleanup complete. Checkpoints and compact reports kept.")
    else:
        if root.exists(): shutil.rmtree(root)
        root.mkdir()
        (root / ".keep").write_text("", encoding="utf-8")
        print("Hard cleanup complete. outputs/ reset.")

if __name__ == "__main__":
    main()
