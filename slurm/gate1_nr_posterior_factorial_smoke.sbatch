#!/bin/bash
#SBATCH --job-name=brx_post_smoke
#SBATCH --output=outputs/slurm/brx_post_smoke-%j.out
#SBATCH --error=outputs/slurm/brx_post_smoke-%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --account=def-rsadve_gpu
#SBATCH --time=00:20:00

set -euo pipefail
cd "${PROJECT_ROOT:-/home/rsadve1/links/scratch/revised_TWC}"
source .venv/bin/activate
export PYTHONPATH="$PWD/src:$PWD/scripts:${PYTHONPATH:-}"
# Make any remaining CUDA indexing failure point to its exact source line.
export CUDA_LAUNCH_BLOCKING=1
set +e
python scripts/gate1_nr_posterior_factorial_smoke.py \
  --config configs/gate1_nr_posterior_factorial_smoke.yaml \
  --device cuda:0
status=$?
python scripts/prepare_github_artifacts.py || true
exit $status
