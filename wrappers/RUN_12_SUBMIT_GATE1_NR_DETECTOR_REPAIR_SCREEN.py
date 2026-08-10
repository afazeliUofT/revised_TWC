#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shlex
import subprocess

REMOTE = "rsadve1@rorqual.alliancecan.ca"
REMOTE_ROOT = "/home/rsadve1/links/scratch/revised_TWC"
EXPECTED_ROWS = 2688


def run(command: list[str]) -> None:
    print("+", " ".join(str(x) for x in command), flush=True)
    completed = subprocess.run([str(x) for x in command], text=True)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--remote", default=REMOTE)
    parser.add_argument("--remote-root", default=REMOTE_ROOT)
    args = parser.parse_args()
    root = shlex.quote(args.remote_root)
    command = f'''set -euo pipefail
cd {root}
source .venv/bin/activate
export PYTHONPATH="$PWD/src:${{PYTHONPATH:-}}"
python - <<'REMOTE_PY'
import json, yaml
from pathlib import Path
smoke = json.loads(Path('outputs/gates/GATE1_NR_DETECTOR_REPAIR_SMOKE.json').read_text())
if not (
    smoke.get('classification') == 'GATE1_NR_DETECTOR_REPAIR_SMOKE_PASS'
    and smoke.get('overall_pass') is True
    and smoke.get('screen_ready') is True
):
    raise SystemExit('BLOCKED: detector-repair smoke has not passed')
config = yaml.safe_load(Path('configs/gate1_nr_detector_repair_screen.yaml').read_text())
variants = len(config['screen']['iterations']) * len(config['screen']['damping']) + 12
rows = (
    len(config['evaluation']['cases'])
    * len(config['evaluation']['ebno_db'])
    * int(config['evaluation']['repetitions'])
    * variants
)
if rows != {EXPECTED_ROWS}:
    raise SystemExit(f'BLOCKED: expected {EXPECTED_ROWS} rows, computed {{rows}}')
if set(config['screen']['selection_reps']) & set(config['screen']['holdout_reps']):
    raise SystemExit('BLOCKED: selection and holdout repetitions overlap')
print('GATE1_NR_DETECTOR_REPAIR_SCREEN_PREFLIGHT_PASS')
print('EXPECTED_ROWS', rows)
print('VARIANTS', variants)
print('SELECTION_HOLDOUT_DISJOINT YES')
REMOTE_PY
if squeue -h -u rsadve1 -n brx_lmep_screen | grep -q .; then
  echo 'A brx_lmep_screen job is already queued or running.' >&2
  exit 3
fi
mkdir -p outputs/slurm outputs/gates outputs/reports outputs/eval outputs/plots
sbatch slurm/gate1_nr_detector_repair_screen.sbatch
'''
    run(["ssh", args.remote, command])
    print("GATE1_NR_DETECTOR_REPAIR_SCREEN_SUBMITTED_OR_RESUMED")


if __name__ == "__main__":
    main()
