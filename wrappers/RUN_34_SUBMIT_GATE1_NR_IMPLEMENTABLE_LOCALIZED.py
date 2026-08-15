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
        import yaml
        smoke_path=Path('outputs/gates/GATE1_NR_IMPLEMENTABLE_LOCALIZED_SMOKE.json')
        if not smoke_path.is_file():
            raise SystemExit('BLOCKED: missing implementable-localized smoke report')
        smoke=json.loads(smoke_path.read_text())
        required=[
            smoke.get('classification') == 'GATE1_NR_IMPLEMENTABLE_LOCALIZED_SMOKE_PASS',
            smoke.get('overall_pass') is True,
            smoke.get('screen_ready') is True,
            smoke.get('model_version') == 'ls_anchored_localized_residual_v1',
            smoke.get('checks',dict()).get('inference_observability_contract') is True,
            smoke.get('checks',dict()).get('checkpoint_roundtrip') is True,
            smoke.get('checks',dict()).get('ls_alignment_paths') is True,
        ]
        if not all(required):
            raise SystemExit('BLOCKED: implementable-localized smoke contract has not passed')
        revision=json.loads(
            Path('GATE1_NR_IMPLEMENTABLE_LOCALIZED_REVISION.json').read_text()
        )
        config=yaml.safe_load(
            Path('configs/gate1_nr_implementable_localized.yaml').read_text()
        )
        training=config['training']
        rng_required=[
            revision.get('training_rng_patch') == 'deterministic_per_step_seed_v1',
            revision.get('validation_rng_isolated_from_training') is True,
            revision.get('resume_training_seed_is_step_derived') is True,
            training.get('rng_patch_version') == 'deterministic_per_step_seed_v1',
            training.get('deterministic_step_seeding') is True,
            int(training.get('step_seed_offset', 0)) == 20000000,
        ]
        if not all(rng_required):
            raise SystemExit('BLOCKED: deterministic training RNG patch has not passed')
        print('GATE1_NR_IMPLEMENTABLE_LOCALIZED_SMOKE_PASS')
        print('GATE1_NR_IMPLEMENTABLE_LOCALIZED_TRAINING_RNG_PASS', revision['training_rng_patch'])
        print('TRAINING_STEP_SEED_OFFSET', training['step_seed_offset'])
        print('INFERENCE_USES_TRUE_CHANNEL NO')
        REMOTE_PY
        python scripts/gate1_nr_implementable_localized.py \\
          --config configs/gate1_nr_implementable_localized.yaml \\
          --preflight-only --device cpu
        if squeue -h -u rsadve1 -n brx_lalp_final | grep -q .; then
          echo 'A brx_lalp_final job is already queued or running.' >&2
          exit 3
        fi
        mkdir -p outputs/slurm outputs/gates outputs/reports outputs/logs \\
          outputs/eval outputs/plots outputs/gate1_nr_implementable_localized
        sbatch slurm/gate1_nr_implementable_localized.sbatch
        """
    ).strip()
    print(f"+ ssh {REMOTE} <implementable-localized-final-submit>", flush=True)
    subprocess.run(["ssh", REMOTE, script], check=True)
    print("GATE1_NR_IMPLEMENTABLE_LOCALIZED_SUBMITTED_OR_RESUMED", flush=True)
    print("EXPECTED_ROWS 2016", flush=True)
    print("MAX_TRAINING_STEPS 12000", flush=True)
    print("HARD_FINAL_STOP YES", flush=True)
    print("Next: python3 RUN_35_GATE1_NR_IMPLEMENTABLE_LOCALIZED_STATUS.py", flush=True)


if __name__ == "__main__":
    main()
