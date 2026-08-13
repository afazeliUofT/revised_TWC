#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shlex
import subprocess

REMOTE = "rsadve1@rorqual.alliancecan.ca"
REMOTE_ROOT = "/home/rsadve1/links/scratch/revised_TWC"
EXPECTED_ROWS = 648


def run(command: list[str]) -> None:
    print("+", " ".join(str(x) for x in command), flush=True)
    subprocess.run([str(x) for x in command], text=True, check=True)


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
python scripts/gate1_nr_posterior_extension.py \\
  --config configs/gate1_nr_posterior_extension.yaml \\
  --preflight-only --device cpu
if squeue -h -u rsadve1 -n brx_post_ext | grep -q .; then
  echo 'A brx_post_ext job is already queued or running.' >&2
  exit 3
fi
mkdir -p outputs/slurm outputs/gates outputs/reports outputs/eval outputs/logs outputs/plots outputs/gate1_nr_posterior_extension
sbatch slurm/gate1_nr_posterior_extension.sbatch
'''
    run(["ssh", args.remote, command])
    print("GATE1_NR_POSTERIOR_EXTENSION_SUBMITTED_OR_RESUMED")
    print("EXPECTED_ROWS", EXPECTED_ROWS)
    print("WINNER physical_context_multiscale_r64")
    print("EXTENSION_STEPS 1600")
    print("FRESH_12PRB_HOLDOUT_USED_FOR_SELECTION NO")
    print("Next: python3 RUN_25_GATE1_NR_POSTERIOR_EXTENSION_STATUS.py")


if __name__ == "__main__":
    main()
