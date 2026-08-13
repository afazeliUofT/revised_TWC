#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shlex
import subprocess

REMOTE = "rsadve1@rorqual.alliancecan.ca"
REMOTE_ROOT = "/home/rsadve1/links/scratch/revised_TWC"


def run(command: list[str]) -> None:
    print("+", " ".join(str(x) for x in command), flush=True)
    subprocess.run([str(x) for x in command], text=True, check=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--remote", default=REMOTE)
    parser.add_argument("--remote-root", default=REMOTE_ROOT)
    args = parser.parse_args()
    root = shlex.quote(args.remote_root)
    command = rf'''set +e
squeue -u rsadve1 -n brx_post_ext
cd {root} || exit 0
echo '--- posterior-extension Slurm logs ---'
for f in $(find outputs/slurm -maxdepth 1 -type f -name 'brx_post_ext-*' \
  -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -6 | cut -d' ' -f2-); do
  echo "### $f"
  tail -100 "$f"
done
echo '--- posterior-extension gates/reports ---'
for f in \
  outputs/gates/GATE1_NR_POSTERIOR_EXTENSION.txt \
  outputs/reports/gate1_nr_posterior_extension_train.json \
  outputs/reports/gate1_nr_posterior_extension.json; do
  if [ -f "$f" ]; then echo "### $f"; tail -160 "$f"; fi
done
'''
    run(["ssh", args.remote, command])


if __name__ == "__main__":
    main()
