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
        revision=json.loads(Path('GATE1_NR_IMPLEMENTABLE_LOCALIZED_REVISION.json').read_text())
        assert revision.get('revision') == 'gate1_nr_implementable_localized_v1', revision
        assert revision.get('hard_final_stop') is True, revision
        print('GATE1_NR_IMPLEMENTABLE_LOCALIZED_REVISION_PASS')
        REMOTE_PY
        python scripts/gate1_nr_implementable_localized_smoke.py \\
          --config configs/gate1_nr_implementable_localized_smoke.yaml \\
          --preflight-only --device cpu
        if squeue -h -u rsadve1 -n brx_lalp_smoke | grep -q .; then
          echo 'A brx_lalp_smoke job is already queued or running.' >&2
          exit 3
        fi
        mkdir -p outputs/slurm outputs/gates outputs/reports outputs/logs \\
          outputs/eval outputs/plots outputs/gate1_nr_implementable_localized
        sbatch slurm/gate1_nr_implementable_localized_smoke.sbatch
        """
    ).strip()
    print(f"+ ssh {REMOTE} <implementable-localized-smoke-submit>", flush=True)
    subprocess.run(["ssh", REMOTE, script], check=True)
    print("GATE1_NR_IMPLEMENTABLE_LOCALIZED_SMOKE_SUBMITTED", flush=True)
    print("Next: python3 RUN_35_GATE1_NR_IMPLEMENTABLE_LOCALIZED_STATUS.py", flush=True)


if __name__ == "__main__":
    main()
