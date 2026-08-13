#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shlex
import subprocess

REMOTE = "rsadve1@rorqual.alliancecan.ca"
REMOTE_ROOT = "/home/rsadve1/links/scratch/revised_TWC"
EXPECTED_ROWS = 792


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
from pathlib import Path
import json
smoke=json.loads(Path('outputs/gates/GATE1_NR_POSTERIOR_FACTORIAL_SMOKE.json').read_text())
assert smoke.get('classification') == 'GATE1_NR_POSTERIOR_FACTORIAL_SMOKE_PASS', smoke
assert smoke.get('overall_pass') is True, smoke
assert smoke.get('screen_ready') is True, smoke
assert smoke.get('checks',{{}}).get('ls_alignment_self_test') is True, smoke
assert smoke.get('checks',{{}}).get('ls_estimator_effective_grid_alignment') is True, smoke
print('GATE1_NR_POSTERIOR_FACTORIAL_SMOKE_PASS')
print('LS_ESTIMATOR_ALIGNMENT_PASS')
REMOTE_PY
python scripts/gate1_nr_posterior_factorial_screen.py \
  --config configs/gate1_nr_posterior_factorial_screen.yaml \
  --preflight-only --device cpu
if squeue -h -u rsadve1 -n brx_post_fact | grep -q .; then
  echo 'A brx_post_fact job is already queued or running.' >&2
  exit 3
fi
mkdir -p outputs/slurm outputs/gates outputs/reports outputs/eval outputs/logs outputs/plots outputs/gate1_nr_posterior_factorial
sbatch slurm/gate1_nr_posterior_factorial_screen.sbatch
'''
    run(["ssh", args.remote, command])
    print("GATE1_NR_POSTERIOR_FACTORIAL_SCREEN_SUBMITTED_OR_RESUMED")
    print("EXPECTED_ROWS", EXPECTED_ROWS)
    print("TRAINING_GRIDS [4, 8]")
    print("UNTOUCHED_HOLDOUT_GRID 12")
    print("Next: python3 RUN_23_GATE1_NR_POSTERIOR_FACTORIAL_STATUS.py")


if __name__ == "__main__":
    main()
