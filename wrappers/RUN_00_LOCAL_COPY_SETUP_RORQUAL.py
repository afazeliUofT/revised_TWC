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
    global REMOTE, REMOTE_ROOT
    ap = argparse.ArgumentParser(description="Unpack package locally, initialize GitHub repo, copy clean tree to Rorqual, and build .venv.")
    ap.add_argument("--remote", default=REMOTE)
    ap.add_argument("--remote-root", default=REMOTE_ROOT)
    args = ap.parse_args()
    REMOTE = args.remote; REMOTE_ROOT = args.remote_root
    local_root = Path.cwd().resolve()
    pkg = find_package()
    stage = local_root / "_package_stage"
    if stage.exists(): shutil.rmtree(stage)
    stage.mkdir()
    with zipfile.ZipFile(pkg, "r") as z:
        z.extractall(stage)
    extracted = stage / "bayesroute_rx_twc"
    if not extracted.exists():
        # fallback: first directory
        dirs = [p for p in stage.iterdir() if p.is_dir()]
        if len(dirs) != 1: raise SystemExit("Unexpected ZIP layout.")
        extracted = dirs[0]
    # Copy package source into current repo root, excluding local artifacts.
    for item in extracted.iterdir():
        dest = local_root / item.name
        if dest.name in {".git", ".venv", "_package_stage"}: continue
        if dest.exists():
            if dest.is_dir(): shutil.rmtree(dest)
            else: dest.unlink()
        if item.is_dir(): shutil.copytree(item, dest)
        else: shutil.copy2(item, dest)
    shutil.rmtree(stage)
    (local_root / "outputs").mkdir(exist_ok=True)
    ensure_local_repo(local_root)
    git_commit_push(local_root, "first commit")
    ssh(f"mkdir -p {REMOTE_ROOT}")
    run(["rsync", "-av", "--delete", "--exclude", ".git/", "--exclude", ".venv/", "--exclude", "outputs/checkpoints/", "--exclude", "outputs/raw/", str(local_root) + "/", f"{REMOTE}:{REMOTE_ROOT}/"])
    ssh(f"cd {REMOTE_ROOT} && python3 scripts/rorqual_setup_env.py --venv .venv --out outputs/setup")
    print("\nDONE: package copied and remote .venv setup completed. Check outputs/setup/VENV_OK.txt on Rorqual.")

if __name__ == "__main__":
    main()
