#!/bin/bash
#SBATCH --job-name=brx_nr_evid
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --account=def-rsadve_gpu
#SBATCH --output=outputs/slurm/%x-%j.out
#SBATCH --error=outputs/slurm/%x-%j.err
set -euo pipefail
cd "${PROJECT_ROOT:-/home/rsadve1/links/scratch/revised_TWC}"
mkdir -p outputs/slurm outputs/gates outputs/reports
source .venv/bin/activate
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
status=0
python scripts/gate1_nr_evidence.py --config configs/gate1_nr_evidence.yaml || status=$?
python scripts/prepare_github_artifacts.py || true
exit "$status"
