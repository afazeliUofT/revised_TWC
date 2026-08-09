#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shlex
import subprocess

REMOTE = "rsadve1@rorqual.alliancecan.ca"
REMOTE_ROOT = "/home/rsadve1/links/scratch/revised_TWC"


def run(command: list[str], check: bool = True) -> subprocess.CompletedProcess:
    print("+", " ".join(str(x) for x in command), flush=True)
    result = subprocess.run([str(x) for x in command], text=True)
    if check and result.returncode != 0:
        raise SystemExit(result.returncode)
    return result


def main() -> None:
    global REMOTE_ROOT
    parser = argparse.ArgumentParser()
    parser.add_argument("--remote-root", default=REMOTE_ROOT)
    args = parser.parse_args()
    REMOTE_ROOT = args.remote_root
    root = shlex.quote(REMOTE_ROOT)
    command = f'''set -euo pipefail
cd {root}
source .venv/bin/activate
export PYTHONPATH="$PWD/src:${{PYTHONPATH:-}}"
python - <<'REMOTE_PY'
from pathlib import Path
import hashlib, json
revision = json.loads(Path('GATE1_NR_REVISION.json').read_text())
assert revision.get('revision') == 'gate1_nr_integration_v1', revision
assert revision.get('base_package_revision') == 'gate0_v2_4_20260809', revision
assert revision.get('pgca_agmp_baseline_included') is False, revision
assert revision.get('topology_batch_resize_patch') == 'sionna_2_0_1_topology_reset_v1', revision
assert revision.get('topology_shape_tracking') is True, revision
assert revision.get('topology_resize_smoke_required') is True, revision
checked = 0
for raw in Path('GATE1_NR_MANIFEST.sha256').read_text().splitlines():
    if not raw.strip():
        continue
    expected, relative = raw.split(None, 1)
    path = Path(relative.strip().lstrip('*'))
    assert path.is_file(), path
    assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, path
    checked += 1
assert checked == 19, checked
from bayesroute.sionna_kbest_compat import configure_sionna_kbest_compat
compat = configure_sionna_kbest_compat(force_eager=True)
assert compat.get('passed') is True, compat
assert compat.get('backend') == 'eager_exact', compat
assert compat.get('active_semantics_exact') is True, compat
gate0 = Path('outputs/gates/GATE0_MECHANISM_DIAGNOSTIC.txt').read_text()
assert 'CLASSIFICATION: GATE0_MECHANISM_SUPPORTED' in gate0
print('GATE1_NR_MANIFEST_PREFLIGHT_PASS', checked)
print('GATE1_NR_KBEST_COMPAT_PREFLIGHT_PASS', compat['compat_version'], compat['backend'])
print('GATE1_NR_TOPOLOGY_RESIZE_PATCH_PREFLIGHT_PASS', revision['topology_batch_resize_patch'])
print('PGCA_AGMP_BASELINE_INCLUDED NO')
REMOTE_PY
python scripts/gate1_nr_preflight.py \
  --smoke-config configs/gate1_nr_smoke.yaml \
  --evidence-config configs/gate1_nr_evidence.yaml \
  --out outputs/gates/GATE1_NR_PREFLIGHT.json
if squeue -h -u rsadve1 -n brx_nr_smoke | grep -q .; then
  echo 'A brx_nr_smoke job is already queued or running.' >&2
  exit 3
fi
rm -f outputs/gates/GATE1_NR_SMOKE.json outputs/gates/GATE1_NR_SMOKE.txt
rm -f outputs/slurm/brx_nr_smoke-*.out outputs/slurm/brx_nr_smoke-*.err
mkdir -p outputs/slurm outputs/gates
sbatch slurm/gate1_nr_smoke.sbatch
'''
    run(["ssh", REMOTE, command])
    print("GATE1_NR_SMOKE_SUBMITTED")


if __name__ == "__main__":
    main()
