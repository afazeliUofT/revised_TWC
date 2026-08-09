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
import json
path = Path('outputs/gates/GATE1_NR_SMOKE.json')
if not path.is_file():
    raise SystemExit('BLOCKED: missing Gate-1 NR smoke report')
report = json.loads(path.read_text())
required = [
    report.get('classification') == 'GATE1_NR_SMOKE_PASS',
    report.get('overall_pass') is True,
    report.get('evidence_ready') is True,
    report.get('gate1_revision') == 'gate1_nr_integration_v1',
    report.get('pgca_agmp_baseline_included') is False,
    report.get('checks', {{}}).get('all_dmrs_mapping_cases') is True,
    report.get('checks', {{}}).get('explicit_user_layer_port_mapping') is True,
    report.get('checks', {{}}).get('nr_tb_ldpc_roundtrip') is True,
    report.get('checks', {{}}).get('all_38901_channel_cases') is True,
    report.get('checks', {{}}).get('kbest_path') is True,
    report.get('checks', {{}}).get('bayesroute_nr_bridge') is True,
]
if not all(required):
    raise SystemExit('BLOCKED: Gate-1 NR smoke contract has not passed')
print('GATE1_NR_EVIDENCE_PREFLIGHT_PASS')
print('PGCA_AGMP_BASELINE_INCLUDED NO')
REMOTE_PY
if squeue -h -u rsadve1 -n brx_nr_evid | grep -q .; then
  echo 'A brx_nr_evid job is already queued or running.' >&2
  exit 3
fi
mkdir -p outputs/slurm outputs/gates outputs/reports
sbatch slurm/gate1_nr_evidence.sbatch
'''
    run(["ssh", REMOTE, command])
    print("GATE1_NR_EVIDENCE_SUBMITTED_OR_RESUMED")


if __name__ == "__main__":
    main()
