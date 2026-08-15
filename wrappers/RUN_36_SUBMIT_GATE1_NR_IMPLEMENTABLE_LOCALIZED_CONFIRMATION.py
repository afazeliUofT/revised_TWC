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
        import hashlib, json
        from pathlib import Path

        checkpoint = Path('outputs/gate1_nr_implementable_localized/checkpoints/best.pt')
        report_path = Path('outputs/reports/gate1_nr_implementable_localized.json')
        if not checkpoint.is_file() or not report_path.is_file():
            raise SystemExit('BLOCKED: missing frozen checkpoint or source report')
        digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        report = json.loads(report_path.read_text())
        required = [
            digest == 'cfb1afd665d6df89e6590b611badd68a77b9ada43de07daeee7b8d8a27dd70aa',
            report.get('complete') is True,
            report.get('classification') == 'GATE1_IMPLEMENTABLE_LOCALIZED_POSSIBLY_BEATS_LS',
            report.get('next_action') == 'RUN_ONE_LARGER_FIXED_CONFIRMATION_WITHOUT_RETUNING',
            report.get('evaluation', dict()).get('rows') == 2016,
            report.get('training', dict()).get('training_converged') is True,
            report.get('training', dict()).get('inference_uses_true_channel') is False,
        ]
        if not all(required):
            raise SystemExit('BLOCKED: source result/checkpoint contract has not passed')
        print('GATE1_NR_IMPLEMENTABLE_LOCALIZED_SOURCE_RESULT_PASS')
        print('SOURCE_CLASSIFICATION', report['classification'])
        print('FROZEN_CHECKPOINT', digest)
        REMOTE_PY
        python scripts/gate1_nr_implementable_localized_confirmation.py \\
          --config configs/gate1_nr_implementable_localized_confirmation.yaml \\
          --preflight-only --device cpu
        if squeue -h -u rsadve1 -n brx_lalp_conf | grep -q .; then
          echo 'A brx_lalp_conf job is already queued or running.' >&2
          exit 3
        fi
        mkdir -p outputs/slurm outputs/gates outputs/reports outputs/eval \\
          outputs/plots outputs/gate1_nr_implementable_localized_confirmation
        sbatch slurm/gate1_nr_implementable_localized_confirmation.sbatch
        """
    ).strip()
    print(f"+ ssh {REMOTE} <implementable-localized-confirmation-submit>", flush=True)
    subprocess.run(["ssh", REMOTE, script], check=True)
    print("GATE1_NR_IMPLEMENTABLE_LOCALIZED_CONFIRMATION_SUBMITTED_OR_RESUMED")
    print("EXPECTED_ROWS 2304")
    print("TRANSPORT_BLOCKS_PER_SNR_PER_RECEIVER 32768")
    print("TRAINING_REQUIRED NO")
    print("RETUNING_ALLOWED NO")
    print("Next: python3 RUN_37_GATE1_NR_IMPLEMENTABLE_LOCALIZED_CONFIRMATION_STATUS.py")


if __name__ == "__main__":
    main()
