# BayesRoute-Rx Gate-0 v2.3

This package tests one receiver principle:

> Learn a positive-semidefinite channel prior from pilot observations, propagate calibrated posterior channel uncertainty into detection, and route explicit soft interference cancellation through a posterior coupling graph.

The package runs on Rorqual under `/home/rsadve1/links/scratch/revised_TWC` and is controlled from WSL under `/home/afazeli2006/revised_TWC`.

## Scope

This is **Gate 0: principle validation**. It is not yet a publication-level 3GPP PUSCH simulator.

- QAM mapping is taken from Sionna PHY and checked against an executed Sionna Mapper/Demapper path.
- The channel and resource grid are compact PyTorch research models.
- Pilot separation uses an explicitly labelled orthogonal DFT codebook. The port, CDM-group, and OCC fields are bookkeeping labels for this Gate-0 abstraction; they are not claimed to reproduce TS 38.211 DMRS mapping.
- No LDPC coding is used. `tblER_proxy` is only an uncoded sample-error flag and is excluded from the main plots.

A later **NR integration gate** must add exact PUSCH/DMRS mapping, channel coding, 3GPP channel models, impairments, and publication baselines. It must also add data-aided channel-posterior refinement, iteration-dependent effective-noise coupling, spatial covariance, and a sparse execution kernel. Gate 0 uses a pilot-only posterior, white-noise coupling, and dense tensor operations; its retained-edge fraction is structural and does not by itself prove wall-clock savings.

## v2.3 mathematical and workflow corrections

- The posterior covariance is projected with the correct complex contraction `u C u^H` for the model `h=u^T z`.
- Full local cross-layer posterior covariance is retained in both the uncertainty-aware detector and the posterior coupling graph.
- The detector uses the correct channel-error term after soft cancellation: `v|mu|^2 + E|x|^2 Sigma`.
- Optuna rank candidates use nested modes from one fixed random Fourier feature bank. Rank is no longer confounded with a different random basis realization.
- The graph threshold `edge_mass` remains an interpretable fixed retained-coupling-mass parameter. It is evaluated through routing-off/default/full-graph ablations rather than optimized by a performance-only objective.
- The smoke gate now checks matched-model posterior calibration, the exact covariance projection, detector moments, and the coupling formula against Monte Carlo simulation.
- Optuna evaluates a seeded pairwise-balanced 12-point discrete design using fixed validation bit NLL, a common training stream, revision-bound storage, and hash-keyed trial checkpoints.
- Training and evaluation use immutable contracts. Evaluation resumes only when the configuration and checkpoint hash match.
- Reported latency times the receiver forward pass only; data generation and diagnostic metrics are excluded.
- Compact tails of recent Slurm stdout/stderr files are synchronized to GitHub.

## Gate-0 ablations

- `bayesroute_uncertainty_off`: posterior covariance is removed from both the detector metric and the coupling graph.
- `bayesroute_graph_off`: posterior uncertainty is retained, but no interferer is explicitly soft-cancelled.
- `bayesroute_uncertainty`: proposed receiver with the fixed 0.80 retained-coupling-mass rule.
- `bayesroute_full_graph`: posterior uncertainty with all inter-layer edges retained.

## Required order

```text
v2.3 deployment -> v2.3 smoke -> Optuna -> initial Gate-0 evidence -> scientific review -> larger Gate-0 stress
```

Do not run Optuna unless `outputs/smoke/SMOKE_HEALTH.json` reports:

```json
"overall_pass": true,
"optuna_ready": true
```

The same report will still say `publication_nr_ready: false`. This is intentional.

## Resume behavior

- Optuna stores the study in `outputs/optuna/study.db` and resumes interrupted parameter sets from hash-keyed checkpoints.
- Training saves atomic `last.pt` and `best.pt` checkpoints with optimizer, RNG, effective configuration, and training contract.
- Evaluation writes one paired-batch row at a time, validates its saved contract, and skips completed `(baseline, SNR, repetition)` rows.
- Plotting is reproducible from the compact CSV outputs.

## Compact evidence synchronized to GitHub

```text
results/from_rorqual/setup/
results/from_rorqual/smoke/
results/from_rorqual/optuna/
results/from_rorqual/reports/
results/from_rorqual/eval/
results/from_rorqual/plots/
results/from_rorqual/slurm/
```

Heavy checkpoints and the Optuna database remain only on Rorqual.
