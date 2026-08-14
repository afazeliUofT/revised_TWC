#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shlex
import subprocess

REMOTE = "rsadve1@rorqual.alliancecan.ca"
REMOTE_ROOT = "/home/rsadve1/links/scratch/revised_TWC"


def run(command: list[str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--remote", default=REMOTE)
    parser.add_argument("--remote-root", default=REMOTE_ROOT)
    args = parser.parse_args()
    root = shlex.quote(args.remote_root)
    script = f'''set -euo pipefail
cd {root}
source .venv/bin/activate
export PYTHONPATH="$PWD/src:$PWD/scripts:${{PYTHONPATH:-}}"
python - <<'REMOTE_PY'
import json
from pathlib import Path
revision=json.loads(Path("GATE1_NR_LOCALIZED_CEILING_REVISION.json").read_text())
assert revision.get("batch_dependent_graph_patch") == "gate1_nr_localized_ceiling_batch_graph_v1", revision
print("GATE1_NR_LOCALIZED_CEILING_GRAPH_PATCH_PASS", revision["batch_dependent_graph_patch"])
REMOTE_PY
python scripts/gate1_nr_localized_ceiling.py --config configs/gate1_nr_localized_ceiling.yaml --preflight-only --device cpu
if squeue -h -u rsadve1 -n brx_local_ceil | grep -q .; then
  echo 'A brx_local_ceil job is already queued or running.' >&2
  exit 3
fi
mkdir -p outputs/slurm outputs/gates outputs/reports outputs/eval outputs/plots outputs/gate1_nr_localized_ceiling
sbatch slurm/gate1_nr_localized_ceiling.sbatch
'''
    run(["ssh", args.remote, script])
    print("GATE1_NR_LOCALIZED_CEILING_SUBMITTED_OR_RESUMED")
    print("BATCH_GRAPH_PATCH gate1_nr_localized_ceiling_batch_graph_v1")
    print("EXPECTED_ROWS 540")
    print("TRAINING_REQUIRED NO")
    print("HARD_ABANDON_GATE YES")
    print("Next: python3 RUN_32_GATE1_NR_LOCALIZED_CEILING_STATUS.py")


if __name__ == "__main__":
    main()
