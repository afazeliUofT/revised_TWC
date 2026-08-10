#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shlex
import subprocess

REMOTE = "rsadve1@rorqual.alliancecan.ca"
REMOTE_ROOT = "/home/rsadve1/links/scratch/revised_TWC"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--remote", default=REMOTE)
    parser.add_argument("--remote-root", default=REMOTE_ROOT)
    args = parser.parse_args()
    root = shlex.quote(args.remote_root)
    command = fr'''set +e
squeue -u rsadve1
cd {root} || exit 0
echo '--- detector-repair Slurm logs ---'
for f in $(find outputs/slurm -maxdepth 1 -type f \
  \( -name 'brx_lmep_smoke-*.out' -o -name 'brx_lmep_smoke-*.err' \
     -o -name 'brx_lmep_screen-*.out' -o -name 'brx_lmep_screen-*.err' \) \
  -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -6 | cut -d' ' -f2-); do
  echo "### $f"
  tail -80 "$f"
done
echo '--- detector-repair gates ---'
for f in \
  outputs/gates/GATE1_NR_DETECTOR_REPAIR_SMOKE.txt \
  outputs/gates/GATE1_NR_DETECTOR_REPAIR_SCREEN.txt \
  outputs/reports/gate1_nr_detector_repair_screen.json; do
  if [ -f "$f" ]; then echo "### $f"; tail -100 "$f"; fi
done
'''
    print("+ ssh", args.remote, "<detector-repair-status>", flush=True)
    completed = subprocess.run(["ssh", args.remote, command], text=True)
    raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
