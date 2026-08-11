#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shlex
import subprocess

REMOTE = "rsadve1@rorqual.alliancecan.ca"
REMOTE_ROOT = "/home/rsadve1/links/scratch/revised_TWC"
EXPECTED_ROWS = 704


def run(command: list[str]) -> None:
    print("+", " ".join(str(x) for x in command), flush=True)
    result = subprocess.run([str(x) for x in command], text=True)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--remote", default=REMOTE)
    parser.add_argument("--remote-root", default=REMOTE_ROOT)
    args = parser.parse_args()
    root = shlex.quote(args.remote_root)
    command = f'''set -euo pipefail
cd {root}
source .venv/bin/activate
export PYTHONPATH="$PWD/src:$PWD/scripts:${{PYTHONPATH:-}}"
python - <<'REMOTE_PY'
import json, yaml
from pathlib import Path
smoke=json.loads(Path('outputs/gates/GATE1_NR_JOINT_OPERATOR_SMOKE.json').read_text())
assert smoke.get('classification') == 'GATE1_NR_JOINT_OPERATOR_SMOKE_PASS', smoke
assert smoke.get('overall_pass') is True, smoke
assert smoke.get('capacity_diagnostic_ready') is True, smoke
config=yaml.safe_load(Path('configs/gate1_nr_joint_operator_capacity.yaml').read_text())
variants=len(config['candidates']) + 6
rows=(
    len(config['evaluation']['cases'])
    * len(config['evaluation']['ebno_db'])
    * int(config['evaluation']['repetitions'])
    * variants
)
assert rows == {EXPECTED_ROWS}, rows
assert {{item['mode'] for item in config['candidates']}} == {{'global','case_specific'}}
print('GATE1_NR_JOINT_OPERATOR_CAPACITY_PREFLIGHT_PASS')
print('CANDIDATES', len(config['candidates']))
print('VARIANTS', variants)
print('EXPECTED_ROWS', rows)
REMOTE_PY
if squeue -h -u rsadve1 -n brx_joint_cap | grep -q .; then
  echo 'A brx_joint_cap job is already queued or running.' >&2
  exit 3
fi
mkdir -p outputs/slurm outputs/gates outputs/reports outputs/eval outputs/plots outputs/logs outputs/gate1_nr_joint_operator
sbatch slurm/gate1_nr_joint_operator_capacity.sbatch
'''
    run(["ssh", args.remote, command])
    print("GATE1_NR_JOINT_OPERATOR_CAPACITY_SUBMITTED_OR_RESUMED")


if __name__ == "__main__":
    main()
