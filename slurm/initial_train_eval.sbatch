#!/bin/bash
#SBATCH --job-name=brx_initial
#SBATCH --time=03:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --account=def-rsadve_gpu
#SBATCH --output=outputs/slurm/%x-%j.out
#SBATCH --error=outputs/slurm/%x-%j.err
set -euo pipefail
cd "${PROJECT_ROOT:-/home/rsadve1/links/scratch/revised_TWC}"
mkdir -p outputs/slurm
source .venv/bin/activate
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
python scripts/train.py --config configs/initial.yaml --run-name initial --resume
python scripts/evaluate.py --config configs/initial.yaml --run-name initial --checkpoint outputs/checkpoints/initial/best.pt --resume
python scripts/make_plots.py --config configs/initial.yaml --run-name initial
python scripts/prepare_github_artifacts.py
