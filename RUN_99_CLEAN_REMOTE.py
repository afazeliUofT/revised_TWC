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


def main() -> None:
    global REMOTE_ROOT
    ap = argparse.ArgumentParser()
    ap.add_argument("--remote-root", default=REMOTE_ROOT)
    ap.add_argument("--hard", action="store_true")
    args = ap.parse_args()
    REMOTE_ROOT = args.remote_root
    mode = "hard" if args.hard else "light"
    qroot = shlex.quote(REMOTE_ROOT)
    ssh(
        f"cd {qroot} && source .venv/bin/activate && "
        f"python scripts/clean_outputs.py --mode {mode}"
    )


if __name__ == "__main__":
    main()
