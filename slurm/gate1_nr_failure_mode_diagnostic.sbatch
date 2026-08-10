#!/bin/bash
#SBATCH --job-name=brx_nr_fdiag
#SBATCH --time=00:45:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --account=def-rsadve_gpu
#SBATCH --output=outputs/slurm/%x-%j.out
#SBATCH --error=outputs/slurm/%x-%j.err
set -euo pipefail
cd "${PROJECT_ROOT:-/home/rsadve1/links/scratch/revised_TWC}"
mkdir -p outputs/slurm outputs/gates outputs/reports outputs/eval outputs/plots
source .venv/bin/activate
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
status=0
python scripts/gate1_nr_failure_mode_diagnostic.py \
  --config configs/gate1_nr_failure_mode_diagnostic.yaml || status=$?
python scripts/prepare_github_artifacts.py || true
exit "$status"
