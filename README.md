# BayesRoute-Rx Gate-0 v2.2

This package tests one receiver principle:

> Learn a positive-semidefinite channel prior from pilot observations, propagate posterior channel uncertainty into detection, and route explicit soft interference cancellation through a posterior coupling graph.

The package runs on Rorqual under `/home/rsadve1/links/scratch/revised_TWC` and is controlled from WSL under `/home/afazeli2006/revised_TWC`.

## Scope

This is **Gate 0: principle validation**. It is not yet a publication-level 3GPP PUSCH simulator.

- QAM mapping is taken from Sionna PHY.
- The smoke test executes real Sionna `Mapper` and `Demapper` blocks on the allocated device.
- The channel and resource grid are compact PyTorch research models.
- Pilot separation uses an explicitly labelled orthogonal DFT codebook. The port, CDM-group, and OCC fields are bookkeeping labels for this Gate-0 abstraction; they are not claimed to reproduce TS 38.211 DMRS mapping.
- No LDPC coding is used. `tblER_proxy` is only an uncoded sample-error flag and is excluded from the main plots.

A later **NR integration gate** must add exact PUSCH/DMRS mapping, channel coding, 3GPP channel models, impairments, and publication baselines.

## v2.2 integrity and workflow repairs

- The posterior coupling graph changes detection; `edge_mass` is active.
- Uncertainty-off and routing-off ablations are verified to change receiver outputs.
- Sionna is executed, not only imported.
- Posterior PSD, exact tensor shapes, deterministic regeneration, gradient flow, parameter updates, checkpoint reload, pilot separation, and baseline sanity are checked.
- Optuna is blocked unless the smoke report matches the exact deployed package revision.
- Optuna pruning uses a fixed validation bit NLL. It does not compare composite losses whose auxiliary-loss weight is being tuned.
- All hyperparameter sets see the same ordered training stream, and the TPE sampler has an explicit seed.
- Trial caches, best parameters, and downstream training are revision- and contract-checked.
- Evaluation resumes only when its config and checkpoint hash match the saved evaluation contract.
- Reported latency times the receiver forward pass only; data generation and diagnostic metrics are excluded.
- Compact tails of recent Slurm stdout/stderr files are synchronized to GitHub for diagnosis.

## Required order

```text
setup/repair -> smoke -> Optuna -> initial Gate-0 evidence -> scientific review -> larger Gate-0 stress
```

Do not run Optuna unless `outputs/smoke/SMOKE_HEALTH.json` has both:

```json
"overall_pass": true,
"optuna_ready": true
```

The same report will still say `publication_nr_ready: false`. That is intentional.

## Resume behavior

- Optuna stores the study in `outputs/optuna/study.db` and resumes interrupted parameter sets from hash-keyed checkpoints.
- Training saves atomic `last.pt` and `best.pt` checkpoints, including optimizer and RNG states.
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
