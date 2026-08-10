#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shlex
import subprocess

REMOTE = "rsadve1@rorqual.alliancecan.ca"
REMOTE_ROOT = "/home/rsadve1/links/scratch/revised_TWC"


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
python scripts/gate1_nr_detector_repair_smoke.py \
  --config configs/gate1_nr_detector_repair_smoke.yaml \
  --preflight-only --device cpu
if squeue -h -u rsadve1 -n brx_lmep_smoke | grep -q .; then
  echo 'A brx_lmep_smoke job is already queued or running.' >&2
  exit 3
fi
rm -f outputs/gates/GATE1_NR_DETECTOR_REPAIR_SMOKE.json \
      outputs/gates/GATE1_NR_DETECTOR_REPAIR_SMOKE.txt
rm -f outputs/slurm/brx_lmep_smoke-*.out outputs/slurm/brx_lmep_smoke-*.err
mkdir -p outputs/slurm outputs/gates
sbatch slurm/gate1_nr_detector_repair_smoke.sbatch
'''
    run(["ssh", args.remote, command])
    print("GATE1_NR_DETECTOR_REPAIR_SMOKE_SUBMITTED")


if __name__ == "__main__":
    main()
