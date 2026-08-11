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
echo '--- joint-operator Slurm logs ---'
for f in $(find outputs/slurm -maxdepth 1 -type f \
  \( -name 'brx_joint_smoke-*.out' -o -name 'brx_joint_smoke-*.err' \
     -o -name 'brx_joint_cap-*.out' -o -name 'brx_joint_cap-*.err' \) \
  -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -8 | cut -d' ' -f2-); do
  echo "### $f"
  tail -100 "$f"
done
echo '--- joint-operator gates/reports ---'
for f in \
  outputs/gates/GATE1_NR_JOINT_OPERATOR_SMOKE.txt \
  outputs/gates/GATE1_NR_JOINT_OPERATOR_CAPACITY.txt \
  outputs/reports/gate1_nr_joint_operator_capacity.json; do
  if [ -f "$f" ]; then echo "### $f"; tail -140 "$f"; fi
done
'''
    print("+ ssh", args.remote, "<joint-operator-status>", flush=True)
    result = subprocess.run(["ssh", args.remote, command], text=True)
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
