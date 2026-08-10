#!/bin/bash
#SBATCH --job-name=brx_lmep_smoke
#SBATCH --time=00:15:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --account=def-rsadve_gpu
#SBATCH --output=outputs/slurm/%x-%j.out
#SBATCH --error=outputs/slurm/%x-%j.err
set -euo pipefail
cd "${PROJECT_ROOT:-/home/rsadve1/links/scratch/revised_TWC}"
mkdir -p outputs/slurm outputs/gates outputs/reports
source .venv/bin/activate
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
status=0
python scripts/gate1_nr_detector_repair_smoke.py \
  --config configs/gate1_nr_detector_repair_smoke.yaml || status=$?
python scripts/prepare_github_artifacts.py || true
exit "$status"
