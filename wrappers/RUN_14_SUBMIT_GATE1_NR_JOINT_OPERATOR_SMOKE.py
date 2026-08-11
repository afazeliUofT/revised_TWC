#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shlex
import subprocess

REMOTE = "rsadve1@rorqual.alliancecan.ca"
REMOTE_ROOT = "/home/rsadve1/links/scratch/revised_TWC"


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
import json
from pathlib import Path
revision=json.loads(Path('GATE1_NR_JOINT_OPERATOR_REVISION.json').read_text())
screen=json.loads(Path('outputs/reports/gate1_nr_detector_repair_screen.json').read_text())
assert revision.get('revision') == 'gate1_nr_joint_operator_v1', revision
assert screen.get('classification') == 'GATE1_DETECTOR_REPAIR_PARTIAL', screen
assert screen.get('selected_variant') == 'delmmse_sparse_i4_d0p7', screen
assert screen.get('evaluation',{{}}).get('rows') == 2688, screen
print('GATE1_NR_JOINT_OPERATOR_SMOKE_PREFLIGHT_PASS')
print('DETECTOR', revision['fixed_detector'])
print('CAPACITY_EXPECTED_ROWS', revision['capacity_expected_rows'])
REMOTE_PY
if squeue -h -u rsadve1 -n brx_joint_smoke | grep -q .; then
  echo 'A brx_joint_smoke job is already queued or running.' >&2
  exit 3
fi
mkdir -p outputs/slurm outputs/gates outputs/reports outputs/logs outputs/gate1_nr_joint_operator
sbatch slurm/gate1_nr_joint_operator_smoke.sbatch
'''
    run(["ssh", args.remote, command])
    print("GATE1_NR_JOINT_OPERATOR_SMOKE_SUBMITTED")


if __name__ == "__main__":
    main()
