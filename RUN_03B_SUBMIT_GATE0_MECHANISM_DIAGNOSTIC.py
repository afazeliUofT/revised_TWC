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
    parser = argparse.ArgumentParser()
    parser.add_argument("--remote-root", default=REMOTE_ROOT)
    args = parser.parse_args()
    REMOTE_ROOT = args.remote_root
    qroot = shlex.quote(REMOTE_ROOT)
    command = f"""set -euo pipefail
cd {qroot}
.venv/bin/python - <<'REMOTE_PY'
import json
from pathlib import Path
import yaml
train = Path('outputs/reports/initial_train_summary.json')
eval_summary = Path('outputs/reports/initial_eval_summary.json')
checkpoint = Path('outputs/checkpoints/initial/best.pt')
config = Path('configs/gate0_mechanism_diagnostic.yaml')
for path in (train, eval_summary, checkpoint, config):
    if not path.exists():
        raise SystemExit(f'BLOCKED: missing {{path}}')
t = json.loads(train.read_text())
e = json.loads(eval_summary.read_text())
d = yaml.safe_load(config.read_text())
required = [
    t.get('complete') is True,
    int(t.get('steps', 0)) == 1200,
    e.get('complete') is True,
    int(e.get('rows', 0)) == int(e.get('expected_rows', -1)) == 960,
    d.get('package_revision') == 'gate0_v2_4_20260809',
]
if not all(required):
    raise SystemExit('BLOCKED: initial Gate-0 evidence is incomplete or incompatible')
print('GATE0_DIAGNOSTIC_PREFLIGHT_PASS')
REMOTE_PY
if squeue -h -u rsadve1 -n brx_diag | grep -q .; then
  echo 'A brx_diag job is already queued or running.' >&2
  exit 3
fi
mkdir -p outputs/slurm
sbatch slurm/gate0_mechanism_diagnostic.sbatch
"""
    ssh(command)
    print("GATE0_MECHANISM_DIAGNOSTIC_SUBMITTED_OR_RESUMED")


if __name__ == "__main__":
    main()
