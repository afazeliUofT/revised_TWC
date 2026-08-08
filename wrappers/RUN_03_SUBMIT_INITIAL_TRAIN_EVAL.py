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
    command = f"""set -euo pipefail
cd {qroot}
.venv/bin/python - <<'REMOTE_PY'
import json
from pathlib import Path
p = Path('outputs/optuna/OPTUNA_STATUS.json')
b = Path('outputs/optuna/best_params.json')
if not p.exists() or not b.exists():
    raise SystemExit('BLOCKED: Optuna status/best parameters are missing')
s = json.loads(p.read_text())
if s.get('target_reached') is not True:
    raise SystemExit('BLOCKED: Optuna target complete-trial count has not been reached')
print('INITIAL_PREFLIGHT_PASS')
REMOTE_PY
if squeue -h -u rsadve1 -n brx_initial | grep -q .; then
  echo 'A brx_initial job is already queued or running.' >&2
  exit 3
fi
mkdir -p outputs/slurm
sbatch slurm/initial_train_eval.sbatch
"""
    ssh(command)
    print("INITIAL_GATE0_TRAIN_EVAL_SUBMITTED_OR_RESUMED")


if __name__ == "__main__":
    main()
