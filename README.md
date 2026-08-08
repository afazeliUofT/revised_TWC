# BayesRoute-Rx Gate-0 v2

**Gate-0 v2.1:** CUDA devices are normalized to explicit Sionna device names such as `cuda:0`.
This package tests one receiver principle:

> Learn a positive-semidefinite channel prior from pilot observations, propagate posterior channel uncertainty into detection, and route explicit soft interference cancellation through a posterior coupling graph.

The package runs on Rorqual under `/home/rsadve1/links/scratch/revised_TWC` and is controlled from WSL under `/home/afazeli2006/revised_TWC`.

## Scope

This is **Gate 0: principle validation**. It is not yet a publication-level 3GPP PUSCH simulator.

- QAM mapping is taken from Sionna PHY.
- The smoke test executes real Sionna `Mapper` and `Demapper` blocks.
- The channel and resource grid are compact PyTorch research models.
- Pilot separation uses an explicitly labelled orthogonal DFT codebook. The port, CDM-group, and OCC fields are bookkeeping labels for this Gate-0 abstraction; they are not claimed to reproduce TS 38.211 DMRS mapping.
- No LDPC coding is used. `tblER_proxy` is only an uncoded sample-error flag and is excluded from the main plots.

A later **NR integration gate** must add exact PUSCH/DMRS mapping, channel coding, 3GPP channel models, impairments, and publication baselines.

## What v2 repairs

- The posterior coupling graph now changes detection; `edge_mass` is no longer a no-op.
- Uncertainty-off and routing-off ablations are verified to change receiver outputs.
- Sionna is executed, not only imported.
- Posterior PSD, exact tensor shapes, deterministic regeneration, gradient flow, parameter updates, checkpoint reload, pilot separation, and baseline sanity are checked.
- Optuna is blocked unless the strengthened smoke report has `optuna_ready=true`.
- Optuna results are applied automatically to initial and full Gate-0 runs.
- Optuna, training, and evaluation have restart support.

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

- Optuna stores the study in `outputs/optuna/study.db` and resumes interrupted trial parameter sets from hash-keyed checkpoints.
- Training saves atomic `last.pt` and `best.pt` checkpoints, including optimizer and RNG states.
- Evaluation writes one paired-batch row at a time and skips completed `(baseline, SNR, repetition)` rows.
- Plotting is reproducible from the committed compact CSV outputs.

## Compact evidence synchronized to GitHub

```text
results/from_rorqual/setup/
results/from_rorqual/smoke/
results/from_rorqual/optuna/
results/from_rorqual/reports/
results/from_rorqual/eval/
results/from_rorqual/plots/
```

Heavy checkpoints and the Optuna database remain only on Rorqual.
