#!/bin/bash
#SBATCH --job-name=brx_local_ceil
#SBATCH --output=outputs/slurm/brx_local_ceil-%j.out
#SBATCH --error=outputs/slurm/brx_local_ceil-%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --account=def-rsadve_gpu
#SBATCH --time=00:30:00
set -euo pipefail
cd "${PROJECT_ROOT:-/home/rsadve1/links/scratch/revised_TWC}"
source .venv/bin/activate
export PYTHONPATH="$PWD/src:$PWD/scripts:${PYTHONPATH:-}"
export CUDA_LAUNCH_BLOCKING=1
set +e
python scripts/gate1_nr_localized_ceiling.py \
  --config configs/gate1_nr_localized_ceiling.yaml \
  --device cuda:0 \
  --deadline-minutes 25
status=$?
python scripts/prepare_github_artifacts.py || true
exit $status
