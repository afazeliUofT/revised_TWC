#!/bin/bash
#SBATCH --job-name=brx_turbo_audit
#SBATCH --output=outputs/slurm/brx_turbo_audit-%j.out
#SBATCH --error=outputs/slurm/brx_turbo_audit-%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --account=def-rsadve_gpu
#SBATCH --time=00:30:00
set -euo pipefail
cd "${PROJECT_ROOT:-/home/rsadve1/links/scratch/revised_TWC}"
source .venv/bin/activate
export PYTHONPATH="$PWD/src:$PWD/scripts:${PYTHONPATH:-}"
set +e
python scripts/gate1_nr_turbo_basis_audit.py --config configs/gate1_nr_turbo_basis_audit.yaml --device cuda:0 --deadline-minutes 25
status=$?
python scripts/prepare_github_artifacts.py || true
exit $status
