#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shlex
import subprocess

REMOTE = "rsadve1@rorqual.alliancecan.ca"
REMOTE_ROOT = "/home/rsadve1/links/scratch/revised_TWC"


def run(command: list[str]) -> None:
    print("+", " ".join(command), flush=True)
    result = subprocess.run(command, text=True)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def main() -> None:
    global REMOTE_ROOT
    parser = argparse.ArgumentParser()
    parser.add_argument("--remote-root", default=REMOTE_ROOT)
    args = parser.parse_args()
    REMOTE_ROOT = args.remote_root
    root = shlex.quote(REMOTE_ROOT)
    command = f'''set -euo pipefail
cd {root}
source .venv/bin/activate
export PYTHONPATH="$PWD/src:${{PYTHONPATH:-}}"
python scripts/gate1_nr_failure_mode_diagnostic.py \
  --config configs/gate1_nr_failure_mode_diagnostic.yaml \
  --preflight-only
if squeue -h -u rsadve1 -n brx_nr_fdiag | grep -q .; then
  echo 'A brx_nr_fdiag job is already queued or running.' >&2
  exit 3
fi
mkdir -p outputs/slurm outputs/gates outputs/reports outputs/eval outputs/plots
sbatch slurm/gate1_nr_failure_mode_diagnostic.sbatch
'''
    run(["ssh", REMOTE, command])
    print("GATE1_NR_FAILURE_DIAGNOSTIC_SUBMITTED_OR_RESUMED")


if __name__ == "__main__":
    main()
