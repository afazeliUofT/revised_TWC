from __future__ import annotations

import argparse
import shlex
import subprocess
from pathlib import Path

REMOTE = "rsadve1@rorqual.alliancecan.ca"
REMOTE_ROOT = "/home/rsadve1/links/scratch/revised_TWC"


def run(
    cmd: list[str], cwd: Path | None = None, check: bool = True
) -> subprocess.CompletedProcess:
    print("+", " ".join(str(x) for x in cmd), flush=True)
    proc = subprocess.run([str(x) for x in cmd], cwd=cwd, text=True)
    if check and proc.returncode != 0:
        raise SystemExit(proc.returncode)
    return proc


def ssh(command: str, check: bool = True) -> subprocess.CompletedProcess:
    return run(["ssh", REMOTE, command], check=check)


def main() -> None:
    global REMOTE_ROOT
    ap = argparse.ArgumentParser()
    ap.add_argument("--remote-root", default=REMOTE_ROOT)
    args = ap.parse_args()
    REMOTE_ROOT = args.remote_root
    qroot = shlex.quote(REMOTE_ROOT)
    command = f"""set -euo pipefail
cd {qroot}
.venv/bin/python - <<'REMOTE_PY'
import json
from pathlib import Path
import yaml
status_path = Path('outputs/optuna/OPTUNA_STATUS.json')
best_path = Path('outputs/optuna/best_params.json')
config_path = Path('configs/initial.yaml')
if not status_path.exists() or not best_path.exists():
    raise SystemExit('BLOCKED: Optuna status/best parameters are missing')
status = json.loads(status_path.read_text())
best = json.loads(best_path.read_text())
cfg = yaml.safe_load(config_path.read_text())
revision = str(cfg.get('package_revision', ''))
required = [
    status.get('target_reached') is True,
    int(status.get('complete_trials', 0)) >= int(status.get('target_complete_trials', 1)),
    revision == 'gate0_v2_4_20260809',
    status.get('package_revision') == revision,
    best.get('package_revision') == revision,
    status.get('contract_signature') == best.get('contract_signature'),
    status.get('search_space_version') == 'gate0_v2_4_search_v1',
    best.get('search_space_version') == 'gate0_v2_4_search_v1',
    best.get('objective_metric') == 'fixed_validation_bit_nll',
    int(status.get('target_complete_trials', 0)) == 12,
    int(status.get('complete_trials', 0)) >= 12,
    int(best.get('target_complete_trials', 0)) == 12,
    int(best.get('n_complete_trials', 0)) >= 12,
    status.get('design_name') == 'space_filling_12',
    best.get('design_name') == 'space_filling_12',
    status.get('design_report', {{}}).get('passed') is True,
    best.get('design_report', {{}}).get('passed') is True,
    status.get('design_signature') == best.get('design_signature'),
    status.get('all_required_design_points_complete') is True,
    best.get('all_required_design_points_complete') is True,
    status.get('design_state', {{}}).get('completed_design_indices') == list(range(12)),
    best.get('completed_design_indices') == list(range(12)),
    status.get('design_state', {{}}).get('missing_design_indices') == [],
    best.get('missing_design_indices') == [],
    status.get('design_state', {{}}).get('unexpected_trial_numbers') == [],
    best.get('unexpected_trial_numbers') == [],
    abs(float(status.get('fixed_edge_mass')) - float(cfg.get('model', {{}}).get('edge_mass'))) < 1e-12,
    abs(float(best.get('fixed_edge_mass')) - float(cfg.get('model', {{}}).get('edge_mass'))) < 1e-12,
]
if not all(required):
    raise SystemExit('BLOCKED: Optuna completion/revision/contract validation failed')
print('INITIAL_PREFLIGHT_PASS', revision, best.get('best_trial_number'))
REMOTE_PY
if squeue -h -u rsadve1 -n brx_initial | grep -q .; then
  echo 'A brx_initial job is already queued or running.' >&2
  exit 3
fi
mkdir -p outputs/slurm
sbatch slurm/initial_train_eval.sbatch
"""
    ssh(command)
    print("INITIAL_GATE0_TRAIN_EVAL_SUBMITTED_OR_RESUMED")


if __name__ == "__main__":
    main()
