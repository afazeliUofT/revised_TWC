#!/bin/bash
#SBATCH --job-name=brx_grid_audit
#SBATCH --time=00:35:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --account=def-rsadve_gpu
#SBATCH --output=outputs/slurm/%x-%j.out
#SBATCH --error=outputs/slurm/%x-%j.err
set -euo pipefail
cd "${PROJECT_ROOT:-/home/rsadve1/links/scratch/revised_TWC}"
mkdir -p outputs/slurm outputs/gates outputs/reports outputs/eval outputs/plots
source .venv/bin/activate
export PYTHONPATH="$PWD/src:$PWD/scripts:${PYTHONPATH:-}"
status=0
python scripts/gate1_nr_grid_scale_audit.py \
  --config configs/gate1_nr_grid_scale_audit.yaml || status=$?
python scripts/prepare_github_artifacts.py || true
exit "$status"
