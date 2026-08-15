#!/bin/bash
#SBATCH --job-name=brx_lalp_smoke
#SBATCH --output=outputs/slurm/brx_lalp_smoke-%j.out
#SBATCH --error=outputs/slurm/brx_lalp_smoke-%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --account=def-rsadve_gpu
#SBATCH --time=00:20:00
set -euo pipefail
cd "${PROJECT_ROOT:-/home/rsadve1/links/scratch/revised_TWC}"
source .venv/bin/activate
export PYTHONPATH="$PWD/src:$PWD/scripts:${PYTHONPATH:-}"
export CUDA_LAUNCH_BLOCKING=1
python scripts/gate1_nr_implementable_localized_smoke.py \
  --config configs/gate1_nr_implementable_localized_smoke.yaml \
  --device cuda:0
python scripts/prepare_github_artifacts.py || true
