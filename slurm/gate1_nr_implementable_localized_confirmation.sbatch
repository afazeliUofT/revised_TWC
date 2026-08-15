#!/bin/bash
#SBATCH --job-name=brx_lalp_conf
#SBATCH --output=outputs/slurm/brx_lalp_conf-%j.out
#SBATCH --error=outputs/slurm/brx_lalp_conf-%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --account=def-rsadve_gpu
#SBATCH --time=00:45:00
set -euo pipefail
cd "${PROJECT_ROOT:-/home/rsadve1/links/scratch/revised_TWC}"
source .venv/bin/activate
export PYTHONPATH="$PWD/src:$PWD/scripts:${PYTHONPATH:-}"
set +e
python scripts/gate1_nr_implementable_localized_confirmation.py \
  --config configs/gate1_nr_implementable_localized_confirmation.yaml \
  --device cuda:0 \
  --deadline-minutes 40
status=$?
python scripts/prepare_github_artifacts.py || true
exit $status
