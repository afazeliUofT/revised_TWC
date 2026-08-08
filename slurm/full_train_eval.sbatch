#!/bin/bash
#SBATCH --job-name=brx_full
#SBATCH --time=10:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=124G
#SBATCH --account=def-rsadve_gpu
#SBATCH --output=outputs/slurm/%x-%j.out
#SBATCH --error=outputs/slurm/%x-%j.err
set -euo pipefail
cd "${PROJECT_ROOT:-/home/rsadve1/links/scratch/revised_TWC}"
mkdir -p outputs/slurm
source .venv/bin/activate
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
python scripts/train.py --config configs/full.yaml --run-name full --resume
python scripts/evaluate.py --config configs/full.yaml --run-name full --checkpoint outputs/checkpoints/full/best.pt --resume
python scripts/make_plots.py --config configs/full.yaml --run-name full
python scripts/prepare_github_artifacts.py
