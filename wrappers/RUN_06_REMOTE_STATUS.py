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
    args = ap.parse_args()
    REMOTE_ROOT = args.remote_root
    qroot = shlex.quote(REMOTE_ROOT)
    command = f"""set +e
squeue -u rsadve1
cd {qroot} || exit 0
echo '--- newest Slurm logs ---'
find outputs/slurm -maxdepth 1 -type f -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -12 | cut -d' ' -f2-
echo '--- gate summaries ---'
for f in outputs/setup/VENV_OK.txt outputs/smoke/SMOKE_HEALTH.txt outputs/optuna/OPTUNA_STATUS.json outputs/reports/initial_eval_summary.json; do
  if [ -f "$f" ]; then echo "### $f"; tail -40 "$f"; fi
done
"""
    ssh(command, check=False)


if __name__ == "__main__":
    main()
