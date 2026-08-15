#!/usr/bin/env python3
from __future__ import annotations
import subprocess
from pathlib import Path

REMOTE = "rsadve1@rorqual.alliancecan.ca"
REMOTE_ROOT = "/home/rsadve1/links/scratch/revised_TWC"
LOCAL_ROOT = Path("/home/afazeli2006/revised_TWC")


def run(command: list[str], *, cwd: Path | None = None, check: bool = True) -> subprocess.CompletedProcess:
    print("+", " ".join(command), flush=True)
    return subprocess.run(command, cwd=cwd, check=check, text=True)


def main() -> None:
    run(["git", "remote", "set-url", "origin", "git@github.com:afazeliUofT/revised_TWC.git"], cwd=LOCAL_ROOT)
    run([
        "ssh", REMOTE,
        f"cd {REMOTE_ROOT} && source .venv/bin/activate && python scripts/prepare_github_artifacts.py",
    ])
    destination = LOCAL_ROOT / "results/from_rorqual"
    destination.mkdir(parents=True, exist_ok=True)
    run([
        "rsync", "-av", "--delete",
        f"{REMOTE}:{REMOTE_ROOT}/outputs/github_artifacts/",
        str(destination) + "/",
    ])
    run(["git", "add", "-A"], cwd=LOCAL_ROOT)
    status = subprocess.run(
        ["git", "diff", "--cached", "--quiet"], cwd=LOCAL_ROOT
    )
    if status.returncode != 0:
        run(["git", "commit", "-m", "sync Gate-2 evidence-mixture Rorqual evidence"], cwd=LOCAL_ROOT)
        run(["git", "push", "-u", "origin", "main"], cwd=LOCAL_ROOT)
    else:
        print("NO_NEW_GATE2_EVIDENCE_TO_COMMIT", flush=True)
    head = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=LOCAL_ROOT, text=True).strip()
    print("GATE2_RESULTS_SYNCED_AND_PUSHED", flush=True)
    print("GITHUB_COMMIT", head, flush=True)


if __name__ == "__main__":
    main()
