#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shlex
import subprocess

REMOTE = "rsadve1@rorqual.alliancecan.ca"
REMOTE_ROOT = "/home/rsadve1/links/scratch/revised_TWC"
WORKFLOW_PATCH_VERSION = "gate1_nr_evidence_workflow_v1"
EXPECTED_MANIFEST_FILES = 19
EXPECTED_ROWS = 1140


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
    command = f"""set -euo pipefail
cd {root}
source .venv/bin/activate
export PYTHONPATH="$PWD/src:${{PYTHONPATH:-}}"
python - <<'REMOTE_PY'
from pathlib import Path
import hashlib, json, yaml
smoke_path = Path('outputs/gates/GATE1_NR_SMOKE.json')
if not smoke_path.is_file():
    raise SystemExit('BLOCKED: missing Gate-1 NR smoke report')
report = json.loads(smoke_path.read_text())
revision = json.loads(Path('GATE1_NR_REVISION.json').read_text())
required = [
    report.get('classification') == 'GATE1_NR_SMOKE_PASS',
    report.get('overall_pass') is True,
    report.get('evidence_ready') is True,
    report.get('gate1_revision') == 'gate1_nr_integration_v1',
    report.get('topology_batch_resize_patch') == 'sionna_2_0_1_topology_reset_v1',
    report.get('pgca_agmp_baseline_included') is False,
    report.get('checks', {{}}).get('all_dmrs_mapping_cases') is True,
    report.get('checks', {{}}).get('explicit_user_layer_port_mapping') is True,
    report.get('checks', {{}}).get('nr_tb_ldpc_roundtrip') is True,
    report.get('checks', {{}}).get('all_38901_channel_cases') is True,
    report.get('checks', {{}}).get('sionna_topology_batch_resize') is True,
    report.get('checks', {{}}).get('kbest_eager_exact_compatibility') is True,
    report.get('checks', {{}}).get('kbest_path') is True,
    report.get('checks', {{}}).get('bayesroute_nr_bridge') is True,
    report.get('checks', {{}}).get('learned_operator_transfer_across_dmrs') is True,
    revision.get('evidence_workflow_patch') == 'gate1_nr_evidence_workflow_v1',
    revision.get('evidence_contract_source_hashing') is True,
    revision.get('github_training_log_included') is True,
    revision.get('topology_batch_resize_patch') == 'sionna_2_0_1_topology_reset_v1',
    revision.get('topology_shape_tracking') is True,
    revision.get('topology_resize_smoke_required') is True,
]
if not all(required):
    raise SystemExit('BLOCKED: Gate-1 NR evidence preflight contract has not passed')
checked = 0
for raw in Path('GATE1_NR_MANIFEST.sha256').read_text().splitlines():
    if not raw.strip():
        continue
    expected, relative = raw.split(None, 1)
    path = Path(relative.strip().lstrip('*'))
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise SystemExit(f'BLOCKED: Gate-1 manifest mismatch: {{path}}')
    checked += 1
if checked != 19:
    raise SystemExit(f'BLOCKED: expected 19 Gate-1 manifest files, found {{checked}}')
config = yaml.safe_load(Path('configs/gate1_nr_evidence.yaml').read_text())
variants = [str(x) for x in config['evaluation']['variants']]
for forbidden in ('pgca', 'agmp'):
    if any(forbidden in value.lower() for value in variants):
        raise SystemExit(f'BLOCKED: forbidden historical baseline found: {{forbidden}}')
expected_rows = 0
for case in config['evaluation']['cases']:
    expected_rows += len(variants) * len(config['evaluation']['ebno_db']) * int(config['evaluation']['repetitions'])
    if bool(case.get('run_kbest', False)):
        expected_rows += len(config['evaluation']['ebno_db']) * int(config['evaluation']['repetitions'])
if expected_rows != 1140:
    raise SystemExit(f'BLOCKED: expected 1140 evidence rows, computed {{expected_rows}}')
if int(config['training']['steps']) != 500:
    raise SystemExit('BLOCKED: expected 500 preliminary training steps')
print('GATE1_NR_EVIDENCE_PREFLIGHT_PASS')
print('GATE1_NR_EVIDENCE_WORKFLOW', revision['evidence_workflow_patch'])
print('GATE1_NR_MANIFEST_FILES', checked)
print('GATE1_NR_EVIDENCE_EXPECTED_ROWS', expected_rows)
print('GATE1_NR_TOPOLOGY_RESIZE_SMOKE_PASS')
print('PGCA_AGMP_BASELINE_INCLUDED NO')
REMOTE_PY
if squeue -h -u rsadve1 -n brx_nr_evid | grep -q .; then
  echo 'A brx_nr_evid job is already queued or running.' >&2
  exit 3
fi
mkdir -p outputs/slurm outputs/gates outputs/reports
sbatch slurm/gate1_nr_evidence.sbatch
"""
    run(["ssh", REMOTE, command])
    print("GATE1_NR_EVIDENCE_SUBMITTED_OR_RESUMED")

if __name__ == "__main__":
    main()
