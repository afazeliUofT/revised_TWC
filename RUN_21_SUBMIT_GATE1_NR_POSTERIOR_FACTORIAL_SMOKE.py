#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shlex
import subprocess

REMOTE = "rsadve1@rorqual.alliancecan.ca"
REMOTE_ROOT = "/home/rsadve1/links/scratch/revised_TWC"


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
python - <<'REMOTE_PY'
from pathlib import Path
import hashlib, json

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()
revision=json.loads(Path('GATE1_NR_POSTERIOR_FACTORIAL_REVISION.json').read_text())
assert revision.get('revision') == 'gate1_nr_posterior_factorial_v1', revision
checked=0
for raw in Path('GATE1_NR_POSTERIOR_FACTORIAL_MANIFEST.sha256').read_text().splitlines():
    if not raw.strip():
        continue
    expected, relative=raw.split(None,1)
    path=Path(relative.strip().lstrip('*').lstrip('./'))
    assert path.is_file(), path
    assert digest(path) == expected, path
    checked += 1
assert checked == 16, checked
grid=json.loads(Path('outputs/reports/gate1_nr_grid_scale_audit.json').read_text())
assert grid.get('complete') is True, grid
assert grid.get('classification') == 'GRID_SCALE_COORDINATE_HYPOTHESIS_NOT_SUPPORTED', grid
assert grid.get('evaluation',{{}}).get('rows') == 360, grid
checkpoint=Path('outputs/gate1_nr_joint_operator/checkpoints/global_r24_cold_lf1_lt0p5/best.pt')
assert checkpoint.is_file(), checkpoint
assert digest(checkpoint) == '4f71c7a0a925005d676687e90c5a241668cfcfed21503e2874c3528721c66980'
print('GATE1_NR_POSTERIOR_FACTORIAL_SMOKE_PREFLIGHT_PASS')
print('MANIFEST_FILES', checked)
print('GRID_AUDIT', grid['classification'])
print('FROZEN_CHECKPOINT', digest(checkpoint))
REMOTE_PY
if squeue -h -u rsadve1 -n brx_post_smoke | grep -q .; then
  echo 'A brx_post_smoke job is already queued or running.' >&2
  exit 3
fi
mkdir -p outputs/slurm outputs/gates outputs/reports outputs/gate1_nr_posterior_factorial
sbatch slurm/gate1_nr_posterior_factorial_smoke.sbatch
'''
    run(["ssh", args.remote, command])
    print("GATE1_NR_POSTERIOR_FACTORIAL_SMOKE_SUBMITTED")
    print("Next: python3 RUN_23_GATE1_NR_POSTERIOR_FACTORIAL_STATUS.py")


if __name__ == "__main__":
    main()
