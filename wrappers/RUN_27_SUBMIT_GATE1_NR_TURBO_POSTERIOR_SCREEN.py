#!/usr/bin/env python3
from __future__ import annotations
import argparse, shlex, subprocess
REMOTE="rsadve1@rorqual.alliancecan.ca"
REMOTE_ROOT="/home/rsadve1/links/scratch/revised_TWC"
def run(cmd):
    print("+", " ".join(map(str,cmd)), flush=True)
    subprocess.run([str(x) for x in cmd], check=True)
def main():
    p=argparse.ArgumentParser(); p.add_argument("--remote",default=REMOTE); p.add_argument("--remote-root",default=REMOTE_ROOT); a=p.parse_args(); root=shlex.quote(a.remote_root)
    script=f'''set -euo pipefail
cd {root}
source .venv/bin/activate
export PYTHONPATH="$PWD/src:$PWD/scripts:${{PYTHONPATH:-}}"
python scripts/gate1_nr_turbo_posterior_screen.py --config configs/gate1_nr_turbo_posterior_screen.yaml --preflight-only --device cpu
if squeue -h -u rsadve1 -n brx_turbo_screen | grep -q .; then echo 'A brx_turbo_screen job is already queued or running.' >&2; exit 3; fi
mkdir -p outputs/slurm outputs/gates outputs/reports outputs/eval outputs/plots outputs/gate1_nr_turbo_posterior
sbatch slurm/gate1_nr_turbo_posterior_screen.sbatch
'''
    run(["ssh",a.remote,script])
    print("GATE1_NR_TURBO_POSTERIOR_SCREEN_SUBMITTED_OR_RESUMED")
    print("EXPECTED_ROWS 492")
    print("TRAINING_REQUIRED NO")
    print("Next: python3 RUN_28_GATE1_NR_TURBO_POSTERIOR_STATUS.py")
if __name__=="__main__": main()
