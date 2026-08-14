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
    script = f'''set +e
squeue -u rsadve1 -n brx_local_ceil
cd {root} || exit 0
echo '--- localized-ceiling Slurm logs ---'
for f in $(find outputs/slurm -maxdepth 1 -type f -name 'brx_local_ceil-*' -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -4 | cut -d' ' -f2-); do
  echo "### $f"
  tail -120 "$f"
done
echo '--- localized-ceiling gates/reports ---'
for f in outputs/gates/GATE1_NR_LOCALIZED_CEILING.txt outputs/reports/gate1_nr_localized_ceiling.json; do
  if [ -f "$f" ]; then echo "### $f"; tail -160 "$f"; fi
done
'''
    print("+ ssh", args.remote, "<localized-ceiling-status>", flush=True)
    subprocess.run(["ssh", args.remote, script], check=False)


if __name__ == "__main__":
    main()
