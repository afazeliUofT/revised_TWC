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
        python - <<'REMOTE_PY'
        import json
        from pathlib import Path
        path=Path('outputs/gates/GATE2_NR_EVIDENCE_MIXTURE_SMOKE.json')
        if not path.is_file():
            raise SystemExit('BLOCKED: missing Gate-2 evidence-mixture smoke report')
        report=json.loads(path.read_text())
        required=[
            report.get('classification') == 'GATE2_NR_EVIDENCE_MIXTURE_SMOKE_PASS',
            report.get('overall_pass') is True,
            report.get('screen_ready') is True,
            report.get('model_version') == 'evidence_mixture_lmmse_ce_v1',
            report.get('checks',dict()).get('actual_lmmse_ce_k1_path') is True,
            report.get('checks',dict()).get('moment_matched_lmmse_ce_path') is True,
            report.get('checks',dict()).get('inference_observability_contract') is True,
        ]
        if not all(required):
            raise SystemExit('BLOCKED: Gate-2 smoke contract has not passed')
        print('GATE2_NR_EVIDENCE_MIXTURE_SMOKE_PASS')
        print('TRAINABLE_PARAMETERS', report['parameter_report']['trainable_parameters'])
        REMOTE_PY
        python scripts/gate2_nr_evidence_mixture_screen.py \\
          --config configs/gate2_nr_evidence_mixture_screen.yaml \\
          --preflight-only --device cpu
        if squeue -h -u rsadve1 -n brx_emix_screen | grep -q .; then
          echo 'A brx_emix_screen job is already queued or running.' >&2
          exit 3
        fi
        mkdir -p outputs/slurm outputs/gates outputs/reports outputs/eval \\
          outputs/logs outputs/plots outputs/gate2_nr_evidence_mixture
        sbatch slurm/gate2_nr_evidence_mixture_screen.sbatch
        """
    ).strip()
    print(f"+ ssh {REMOTE} <gate2-evidence-mixture-screen-submit>", flush=True)
    subprocess.run(["ssh", REMOTE, script], check=True)
    print("GATE2_NR_EVIDENCE_MIXTURE_SCREEN_SUBMITTED_OR_RESUMED", flush=True)
    print("MAX_TRAINING_STEPS 6000", flush=True)
    print("EXPECTED_ROWS 2304", flush=True)
    print("SCREEN_WALLTIME 01:00:00", flush=True)
    print("Next: python3 RUN_40_GATE2_NR_EVIDENCE_MIXTURE_STATUS.py", flush=True)


if __name__ == "__main__":
    main()
