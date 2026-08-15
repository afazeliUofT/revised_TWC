#!/usr/bin/env python3
from __future__ import annotations
import subprocess
import textwrap

REMOTE = "rsadve1@rorqual.alliancecan.ca"
REMOTE_ROOT = "/home/rsadve1/links/scratch/revised_TWC"


def main() -> None:
    script = textwrap.dedent(
        f"""
        set -euo pipefail
        cd {REMOTE_ROOT}
        source .venv/bin/activate
        export PYTHONPATH="$PWD/src:$PWD/scripts:${{PYTHONPATH:-}}"
        python scripts/gate2_nr_evidence_mixture_smoke.py \\
          --config configs/gate2_nr_evidence_mixture_smoke.yaml \\
          --preflight-only --device cpu
        if squeue -h -u rsadve1 -n brx_emix_smoke,brx_emix_screen | grep -q .; then
          echo 'A Gate-2 evidence-mixture job is already queued or running.' >&2
          exit 3
        fi
        rm -rf outputs/gate2_nr_evidence_mixture
        rm -f outputs/gates/GATE2_NR_EVIDENCE_MIXTURE*.json \\
          outputs/gates/GATE2_NR_EVIDENCE_MIXTURE*.txt \\
          outputs/reports/gate2_nr_evidence_mixture* \\
          outputs/eval/gate2_nr_evidence_mixture* \\
          outputs/logs/gate2_nr_evidence_mixture* \\
          outputs/plots/gate2_evidence_mixture* \\
          outputs/slurm/brx_emix_smoke-* \\
          outputs/slurm/brx_emix_screen-*
        mkdir -p outputs/slurm outputs/gates outputs/reports outputs/eval \\
          outputs/logs outputs/plots outputs/gate2_nr_evidence_mixture
        sbatch slurm/gate2_nr_evidence_mixture_smoke.sbatch
        """
    ).strip()
    print(f"+ ssh {REMOTE} <gate2-evidence-mixture-smoke-submit>", flush=True)
    subprocess.run(["ssh", REMOTE, script], check=True)
    print("GATE2_NR_EVIDENCE_MIXTURE_SMOKE_SUBMITTED", flush=True)
    print("SMOKE_WALLTIME 00:15:00", flush=True)
    print("Next: python3 RUN_40_GATE2_NR_EVIDENCE_MIXTURE_STATUS.py", flush=True)


if __name__ == "__main__":
    main()
