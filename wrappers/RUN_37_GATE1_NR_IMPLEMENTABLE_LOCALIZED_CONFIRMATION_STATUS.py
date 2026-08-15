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
        echo '--- implementable-localized confirmation Slurm logs ---'
        for f in $(ls -1t outputs/slurm/brx_lalp_conf-* 2>/dev/null | head -4); do
          echo "### $f"
          tail -160 "$f"
        done
        echo '--- implementable-localized confirmation gates/reports ---'
        for f in \\
          outputs/gates/GATE1_NR_IMPLEMENTABLE_LOCALIZED_CONFIRMATION.txt \\
          outputs/reports/gate1_nr_implementable_localized_confirmation.json; do
          if [ -f "$f" ]; then
            echo "### $f"
            tail -220 "$f"
          fi
        done
        """
    ).strip()
    print(f"+ ssh {REMOTE} <implementable-localized-confirmation-status>", flush=True)
    subprocess.run(["ssh", REMOTE, script], check=False)


if __name__ == "__main__":
    main()
