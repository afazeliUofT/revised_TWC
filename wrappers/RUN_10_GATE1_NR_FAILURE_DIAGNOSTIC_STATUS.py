#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shlex
import subprocess

REMOTE = "rsadve1@rorqual.alliancecan.ca"
REMOTE_ROOT = "/home/rsadve1/links/scratch/revised_TWC"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--remote-root", default=REMOTE_ROOT)
    args = parser.parse_args()
    root = shlex.quote(args.remote_root)
    command = f'''set +e
squeue -u rsadve1 -n brx_nr_fdiag
cd {root} || exit 0
echo '--- newest diagnostic logs ---'
for f in $(find outputs/slurm -maxdepth 1 -type f -name 'brx_nr_fdiag-*' -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -4 | cut -d' ' -f2-); do
  echo "### $f"
  tail -80 "$f"
done
echo '--- diagnostic gate ---'
if [ -f outputs/gates/GATE1_NR_FAILURE_MODE_DIAGNOSTIC.txt ]; then
  cat outputs/gates/GATE1_NR_FAILURE_MODE_DIAGNOSTIC.txt
fi
if [ -f outputs/reports/gate1_nr_failure_mode_diagnostic.json ]; then
python - <<'REMOTE_PY'
import json
from pathlib import Path
p = Path('outputs/reports/gate1_nr_failure_mode_diagnostic.json')
r = json.loads(p.read_text())
print('classification:', r.get('classification'))
print('next_action:', r.get('next_action'))
print('rows:', r.get('evaluation', {{}}).get('rows'))
print('expected_rows:', r.get('evaluation', {{}}).get('expected_rows'))
print('complete:', r.get('evaluation', {{}}).get('complete'))
REMOTE_PY
fi
'''
    result = subprocess.run(["ssh", REMOTE, command], text=True)
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
