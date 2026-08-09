from __future__ import annotations

import argparse
import shlex
import subprocess

REMOTE = "rsadve1@rorqual.alliancecan.ca"
REMOTE_ROOT = "/home/rsadve1/links/scratch/revised_TWC"


def run(command: list[str], check: bool = True) -> subprocess.CompletedProcess:
    print("+", " ".join(str(x) for x in command), flush=True)
    result = subprocess.run([str(x) for x in command], text=True)
    if check and result.returncode != 0:
        raise SystemExit(result.returncode)
    return result


def main() -> None:
    global REMOTE_ROOT
    parser = argparse.ArgumentParser()
    parser.add_argument("--remote-root", default=REMOTE_ROOT)
    args = parser.parse_args()
    REMOTE_ROOT = args.remote_root
    root = shlex.quote(REMOTE_ROOT)
    command = f'''set +e
squeue -u rsadve1
cd {root} || exit 0
echo '--- newest Slurm logs ---'
find outputs/slurm -maxdepth 1 -type f -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -14 | cut -d' ' -f2-
echo '--- tails of six newest Slurm logs ---'
for f in $(find outputs/slurm -maxdepth 1 -type f -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -6 | cut -d' ' -f2-); do
  echo "### $f"
  tail -60 "$f"
done
echo '--- gate summaries ---'
for f in \
  outputs/setup/VENV_OK.txt \
  outputs/gates/GATE0_MECHANISM_DIAGNOSTIC.txt \
  outputs/gates/GATE1_NR_PREFLIGHT.json \
  outputs/gates/GATE1_NR_SMOKE.txt \
  outputs/gates/GATE1_NR_PRELIMINARY_EVIDENCE.txt \
  outputs/reports/gate1_nr_preliminary_train_summary.json \
  outputs/reports/gate1_nr_preliminary_summary.json; do
  if [ -f "$f" ]; then echo "### $f"; tail -80 "$f"; fi
done
'''
    run(["ssh", REMOTE, command], check=False)


if __name__ == "__main__":
    main()
