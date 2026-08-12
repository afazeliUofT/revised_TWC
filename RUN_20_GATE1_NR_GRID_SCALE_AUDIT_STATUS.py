#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess

REMOTE = "rsadve1@rorqual.alliancecan.ca"
REMOTE_ROOT = "/home/rsadve1/links/scratch/revised_TWC"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--remote", default=REMOTE)
    parser.add_argument("--remote-root", default=REMOTE_ROOT)
    args = parser.parse_args()
    command = f'''set +e
squeue -u rsadve1
cd {args.remote_root} || exit 0
echo '--- grid-scale Slurm logs ---'
for f in $(find outputs/slurm -maxdepth 1 -type f -name 'brx_grid_audit-*' -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -4 | cut -d' ' -f2-); do
  echo "### $f"
  tail -100 "$f"
done
echo '--- supplemental clean reanalysis ---'
for f in \
  outputs/gates/GATE1_NR_CLEAN_REANALYSIS.txt \
  outputs/reports/gate1_nr_clean_generalization_reanalysis.json; do
  if [ -f "$f" ]; then echo "### $f"; tail -100 "$f"; fi
done
echo '--- grid-scale gates/reports ---'
for f in \
  outputs/gates/GATE1_NR_GRID_SCALE_AUDIT.txt \
  outputs/reports/gate1_nr_grid_scale_audit.json; do
  if [ -f "$f" ]; then echo "### $f"; tail -120 "$f"; fi
done
'''
    print(f"+ ssh {args.remote} <grid-scale-audit-status>", flush=True)
    raise SystemExit(subprocess.run(["ssh", args.remote, command], text=True).returncode)


if __name__ == "__main__":
    main()
