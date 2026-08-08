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
p = Path('outputs/smoke/SMOKE_HEALTH.json')
if not p.exists():
    raise SystemExit('BLOCKED: missing outputs/smoke/SMOKE_HEALTH.json')
h = json.loads(p.read_text())
required = [
    h.get('overall_pass') is True,
    h.get('optuna_ready') is True,
    str(h.get('package_revision','')).startswith('gate0_v2_'),
    h.get('checks',{{}}).get('sionna_mapper_demapper_executed') is True,
    h.get('checks',{{}}).get('uncertainty_and_routing_paths_active') is True,
    h.get('checks',{{}}).get('posterior_psd_and_finite') is True,
    h.get('checks',{{}}).get('posterior_coverage_metric_valid') is True,
    h.get('checks',{{}}).get('checkpoint_roundtrip') is True,
]
if not all(required):
    raise SystemExit('BLOCKED: strengthened Gate-0 v2 smoke has not passed')
print('OPTUNA_PREFLIGHT_PASS')
REMOTE_PY
if squeue -h -u rsadve1 -n brx_optuna | grep -q .; then
  echo 'A brx_optuna job is already queued or running.' >&2
  exit 3
fi
mkdir -p outputs/slurm
sbatch slurm/optuna.sbatch
"""
    ssh(command)
    print("OPTUNA_SUBMITTED_OR_RESUMED")


if __name__ == "__main__":
    main()
