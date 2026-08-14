#!/usr/bin/env python3
from __future__ import annotations
import argparse, shlex, subprocess
REMOTE="rsadve1@rorqual.alliancecan.ca"
REMOTE_ROOT="/home/rsadve1/links/scratch/revised_TWC"
def main():
    p=argparse.ArgumentParser(); p.add_argument("--remote",default=REMOTE); p.add_argument("--remote-root",default=REMOTE_ROOT); a=p.parse_args(); root=shlex.quote(a.remote_root)
    script=rf'''set +e
squeue -u rsadve1 -n brx_turbo_audit
cd {root} || exit 0
echo '--- turbo basis-audit Slurm logs ---'
for f in $(find outputs/slurm -maxdepth 1 -type f -name 'brx_turbo_audit-*' -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -4 | cut -d' ' -f2-); do echo "### $f"; tail -120 "$f"; done
echo '--- turbo basis-audit gate/report ---'
for f in outputs/gates/GATE1_NR_TURBO_BASIS_AUDIT.txt outputs/reports/gate1_nr_turbo_basis_audit.json; do if [ -f "$f" ]; then echo "### $f"; tail -220 "$f"; fi; done
'''
    print("+ ssh",a.remote,"<turbo-basis-audit-status>",flush=True)
    subprocess.run(["ssh",a.remote,script],check=False)
if __name__=="__main__": main()
