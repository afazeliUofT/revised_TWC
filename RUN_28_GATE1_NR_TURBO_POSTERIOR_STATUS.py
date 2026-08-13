#!/usr/bin/env python3
from __future__ import annotations
import argparse, shlex, subprocess
REMOTE="rsadve1@rorqual.alliancecan.ca"
REMOTE_ROOT="/home/rsadve1/links/scratch/revised_TWC"
def main():
    p=argparse.ArgumentParser(); p.add_argument("--remote",default=REMOTE); p.add_argument("--remote-root",default=REMOTE_ROOT); a=p.parse_args(); root=shlex.quote(a.remote_root)
    script=rf'''set +e
squeue -u rsadve1 -n brx_turbo_smoke,brx_turbo_screen
cd {root} || exit 0
echo '--- turbo-posterior Slurm logs ---'
for f in $(find outputs/slurm -maxdepth 1 -type f \( -name 'brx_turbo_smoke-*' -o -name 'brx_turbo_screen-*' \) -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -8 | cut -d' ' -f2-); do echo "### $f"; tail -100 "$f"; done
echo '--- turbo-posterior gates/reports ---'
for f in outputs/gates/GATE1_NR_TURBO_POSTERIOR_SMOKE.txt outputs/gates/GATE1_NR_TURBO_POSTERIOR.txt outputs/reports/gate1_nr_turbo_posterior.json; do if [ -f "$f" ]; then echo "### $f"; tail -160 "$f"; fi; done
'''
    print("+ ssh",a.remote,"<turbo-posterior-status>",flush=True)
    subprocess.run(["ssh",a.remote,script],check=False)
if __name__=="__main__": main()
