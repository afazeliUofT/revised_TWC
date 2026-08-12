#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shlex
import subprocess

REMOTE = "rsadve1@rorqual.alliancecan.ca"
REMOTE_ROOT = "/home/rsadve1/links/scratch/revised_TWC"
EXPECTED_ROWS = 4800


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
python scripts/gate1_nr_clean_generalization.py \
  --config configs/gate1_nr_clean_generalization.yaml \
  --preflight-only
if squeue -h -u rsadve1 -n brx_clean_gen | grep -q .; then
  echo 'A brx_clean_gen job is already queued or running.' >&2
  exit 3
fi
mkdir -p outputs/slurm outputs/gates outputs/reports outputs/eval outputs/plots
sbatch slurm/gate1_nr_clean_generalization.sbatch
'''
    run(["ssh", args.remote, command])
    print("GATE1_NR_CLEAN_GENERALIZATION_SUBMITTED_OR_RESUMED")
    print("EXPECTED_ROWS", EXPECTED_ROWS)


if __name__ == "__main__":
    main()
