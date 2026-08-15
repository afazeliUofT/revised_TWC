#!/usr/bin/env python3
from __future__ import annotations
import subprocess
import textwrap

REMOTE = "rsadve1@rorqual.alliancecan.ca"
REMOTE_ROOT = "/home/rsadve1/links/scratch/revised_TWC"


def main() -> None:
    script = textwrap.dedent(
        f"""
        set +e
        squeue -u rsadve1
        cd {REMOTE_ROOT} || exit 0
        echo '--- Gate-2 evidence-mixture Slurm logs ---'
        for f in $(ls -1t outputs/slurm/brx_emix_smoke-* outputs/slurm/brx_emix_screen-* 2>/dev/null | head -6); do
          echo "### $f"
          tail -120 "$f"
        done
        echo '--- Gate-2 evidence-mixture gates/reports ---'
        for f in \\
          outputs/gates/GATE2_NR_EVIDENCE_MIXTURE_SMOKE.txt \\
          outputs/gates/GATE2_NR_EVIDENCE_MIXTURE.txt \\
          outputs/reports/gate2_nr_evidence_mixture_train.json \\
          outputs/reports/gate2_nr_evidence_mixture.json; do
          if [ -f "$f" ]; then
            echo "### $f"
            tail -180 "$f"
          fi
        done
        """
    ).strip()
    print(f"+ ssh {REMOTE} <gate2-evidence-mixture-status>", flush=True)
    subprocess.run(["ssh", REMOTE, script], check=False)


if __name__ == "__main__":
    main()
