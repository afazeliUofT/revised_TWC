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
import hashlib
import json
from pathlib import Path
import yaml
summary_path = Path('outputs/reports/initial_eval_summary.json')
approval_path = Path('outputs/gates/GATE0_FULL_APPROVED.json')
config = yaml.safe_load(Path('configs/full.yaml').read_text())
revision = str(config.get('package_revision', ''))
if not summary_path.exists():
    raise SystemExit('BLOCKED: initial Gate-0 evaluation is missing')
summary = json.loads(summary_path.read_text())
if summary.get('complete') is not True:
    raise SystemExit('BLOCKED: initial Gate-0 evaluation is incomplete')
if not approval_path.exists():
    raise SystemExit('BLOCKED: scientific approval file is missing')
approval = json.loads(approval_path.read_text())
summary_hash = hashlib.sha256(summary_path.read_bytes()).hexdigest()
required = [
    approval.get('approved') is True,
    approval.get('package_revision') == revision,
    approval.get('initial_eval_summary_sha256') == summary_hash,
]
if not all(required):
    raise SystemExit('BLOCKED: approval does not match this revision and initial evidence')
print('GATE0_STRESS_PREFLIGHT_PASS', revision, summary_hash)
REMOTE_PY
if squeue -h -u rsadve1 -n brx_full | grep -q .; then
  echo 'A brx_full job is already queued or running.' >&2
  exit 3
fi
mkdir -p outputs/slurm
sbatch slurm/full_train_eval.sbatch
"""
    ssh(command)
    print("LARGE_GATE0_STRESS_SUBMITTED_OR_RESUMED")
    print("NOTE: this remains Gate-0 and is not the publication-level NR campaign.")


if __name__ == "__main__":
    main()
