# BayesRoute-Rx TWC Implementation Package

This package implements the first simulation path for the revised TWC receiver idea:

**Learn a positive-semidefinite stochastic channel operator from DMRS observations, propagate channel uncertainty into detection, and build layer coupling from posterior interference geometry.**

The package is designed for Rorqual under `/home/rsadve1/links/scratch/revised_TWC` and for local WSL orchestration from `/home/afazeli2006/revised_TWC`.

## What is included

- PyTorch implementation of a low-rank Bayesian channel posterior operator.
- Abstract NR-like DMRS port/OCC/CDM pilot model with exact port orthogonality checks.
- Uncertainty-aware multi-layer detector.
- Baselines and ablations:
  - LS point-channel receiver.
  - BayesRoute posterior-mean-only receiver.
  - BayesRoute posterior-mean-plus-covariance receiver.
  - Oracle-CSI receiver.
- Smoke tests for Sionna import/API, pilot orthogonality, tensor shapes, gradient flow, and pipeline health.
- Resume-friendly Optuna tuning, training, evaluation, and plotting.
- Slurm scripts with rational wall-time requests.
- Local Python wrappers for copy/setup, job submission, status, GitHub sync, and cleanup.

## Important scope note

This is a clean research implementation for validating the new receiver principle. It does not claim to be a full 3GPP PUSCH link-level implementation. The pilot model is an NR-like orthogonal DMRS port abstraction with explicit port, CDM group, and OCC metadata. The smoke test verifies that the assumed orthogonality holds exactly and that Sionna is installed and usable.

## Remote layout

After running the setup wrapper, the remote directory is:

```text
/home/rsadve1/links/scratch/revised_TWC/
  src/bayesroute/
  scripts/
  configs/
  slurm/
  outputs/
  .venv/
```

Only compact summaries, CSVs, plots, and health reports are prepared for GitHub. Heavy checkpoints and raw generated data are kept out of Git.

## Resume behavior

- Optuna uses SQLite storage in `outputs/optuna/study.db`.
- Training checkpoints are saved in `outputs/checkpoints/<run_name>/last.pt` and `best.pt`.
- Evaluation writes partial CSV files after each SNR/baseline/repetition and skips completed rows when resumed.
- Plotting can be re-run any time from CSV outputs.

## Main commands on Rorqual

These are called by the wrappers, but can also be run manually after activation:

```bash
source .venv/bin/activate
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
python scripts/smoke_test.py --config configs/smoke.yaml --out outputs/smoke
python scripts/optuna_tune.py --config configs/optuna.yaml --out outputs/optuna --n-trials 12
python scripts/train.py --config configs/initial.yaml --run-name initial
python scripts/evaluate.py --config configs/initial.yaml --run-name initial --checkpoint outputs/checkpoints/initial/best.pt
python scripts/make_plots.py --config configs/initial.yaml --run-name initial
```

## Files to inspect after runs

```text
outputs/smoke/SMOKE_HEALTH.json
outputs/optuna/best_params.json
outputs/reports/initial_eval_summary.json
outputs/reports/full_eval_summary.json
outputs/plots/*.png
outputs/eval/*.csv
```
