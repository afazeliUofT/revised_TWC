from __future__ import annotations
import argparse, os, shutil, subprocess, sys, zipfile
from pathlib import Path

REMOTE = "rsadve1@rorqual.alliancecan.ca"
REMOTE_ROOT = "/home/rsadve1/links/scratch/revised_TWC"
GITHUB_REMOTE = "git@github.com:afazeliUofT/revised_TWC.git"
PACKAGE_NAME = "BayesRoute_TWC_Impl_Package.zip"
DOWNLOADS = Path("/mnt/c/Users/alifa/Downloads")

def run(cmd, cwd=None, check=True):
    print("+", " ".join(str(x) for x in cmd), flush=True)
    p = subprocess.run([str(x) for x in cmd], cwd=cwd, text=True)
    if check and p.returncode != 0:
        raise SystemExit(p.returncode)
    return p

def ssh(command, check=True):
    return run(["ssh", REMOTE, command], check=check)

def ensure_local_repo(root: Path):
    if not (root / ".git").exists():
        run(["git", "init"], cwd=root)
        run(["git", "branch", "-M", "main"], cwd=root)
        run(["git", "remote", "add", "origin", GITHUB_REMOTE], cwd=root, check=False)
    else:
        run(["git", "remote", "set-url", "origin", GITHUB_REMOTE], cwd=root, check=False)

def git_commit_push(root: Path, msg: str):
    run(["git", "add", "-A"], cwd=root)
    p = subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=root)
    if p.returncode != 0:
        run(["git", "commit", "-m", msg], cwd=root)
    run(["git", "push", "-u", "origin", "main"], cwd=root, check=False)

def find_package() -> Path:
    candidates = [Path.cwd() / PACKAGE_NAME, DOWNLOADS / PACKAGE_NAME]
    for p in candidates:
        if p.exists(): return p
    raise SystemExit(f"Could not find {PACKAGE_NAME} in current dir or {DOWNLOADS}")


def main():
    global REMOTE_ROOT
    ap = argparse.ArgumentParser()
    ap.add_argument("--remote-root", default=REMOTE_ROOT)
    args = ap.parse_args()
    REMOTE_ROOT = args.remote_root
    ssh(f"cd {REMOTE_ROOT} && mkdir -p outputs/slurm && sbatch slurm/full_train_eval.sbatch")
    print("Submitted full training/evaluation. It resumes from outputs/checkpoints/full/last.pt.")
if __name__ == "__main__": main()
