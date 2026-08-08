from __future__ import annotations
import argparse
import shlex
import subprocess
from pathlib import Path

REMOTE = "rsadve1@rorqual.alliancecan.ca"
REMOTE_ROOT = "/home/rsadve1/links/scratch/revised_TWC"


def run(cmd: list[str], cwd: Path | None = None, check: bool = True) -> subprocess.CompletedProcess:
    print("+", " ".join(str(x) for x in cmd), flush=True)
    proc = subprocess.run([str(x) for x in cmd], cwd=cwd, text=True)
    if check and proc.returncode != 0:
        raise SystemExit(proc.returncode)
    return proc


def ssh(command: str, check: bool = True) -> subprocess.CompletedProcess:
    return run(["ssh", REMOTE, command], check=check)

GITHUB_REMOTE = "git@github.com:afazeliUofT/revised_TWC.git"


def ensure_local_repo(root: Path) -> None:
    if not (root / ".git").exists():
        run(["git", "init"], cwd=root)
        run(["git", "branch", "-M", "main"], cwd=root)
        run(["git", "remote", "add", "origin", GITHUB_REMOTE], cwd=root)
    else:
        run(["git", "remote", "set-url", "origin", GITHUB_REMOTE], cwd=root)


def commit_push(root: Path) -> None:
    run(["git", "add", "-A"], cwd=root)
    changed = subprocess.run(
        ["git", "diff", "--cached", "--quiet"], cwd=root
    ).returncode != 0
    if changed:
        run(["git", "commit", "-m", "sync compact Rorqual evidence"], cwd=root)
    run(["git", "push", "-u", "origin", "main"], cwd=root)


def main() -> None:
    global REMOTE_ROOT
    ap = argparse.ArgumentParser()
    ap.add_argument("--remote-root", default=REMOTE_ROOT)
    args = ap.parse_args()
    REMOTE_ROOT = args.remote_root
    qroot = shlex.quote(REMOTE_ROOT)
    local_root = Path.cwd().resolve()
    ensure_local_repo(local_root)
    ssh(
        f"cd {qroot} && source .venv/bin/activate && "
        "python scripts/prepare_github_artifacts.py"
    )
    dest = local_root / "results" / "from_rorqual"
    dest.mkdir(parents=True, exist_ok=True)
    run([
        "rsync", "-av", "--delete",
        f"{REMOTE}:{REMOTE_ROOT}/outputs/github_artifacts/",
        str(dest) + "/",
    ])
    commit_push(local_root)
    print("RESULTS_SYNCED_AND_PUSHED")


if __name__ == "__main__":
    main()
