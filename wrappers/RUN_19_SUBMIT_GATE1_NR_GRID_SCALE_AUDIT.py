#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shlex
import subprocess

REMOTE = "rsadve1@rorqual.alliancecan.ca"
REMOTE_ROOT = "/home/rsadve1/links/scratch/revised_TWC"
EXPECTED_ROWS = 360


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
python scripts/gate1_nr_clean_reanalysis.py
python scripts/gate1_nr_grid_scale_audit.py \
  --config configs/gate1_nr_grid_scale_audit.yaml \
  --preflight-only --device cpu
if squeue -h -u rsadve1 -n brx_grid_audit | grep -q .; then
  echo 'A brx_grid_audit job is already queued or running.' >&2
  exit 3
fi
mkdir -p outputs/slurm outputs/gates outputs/reports outputs/eval outputs/plots
sbatch slurm/gate1_nr_grid_scale_audit.sbatch
'''
    run(["ssh", args.remote, command])
    print("GATE1_NR_GRID_SCALE_AUDIT_SUBMITTED_OR_RESUMED")
    print("EXPECTED_ROWS", EXPECTED_ROWS)
    print("RETRAINING NO")


if __name__ == "__main__":
    main()
