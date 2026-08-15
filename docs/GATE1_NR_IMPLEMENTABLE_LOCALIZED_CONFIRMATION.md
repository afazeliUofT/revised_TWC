# Gate-1 NR implementable-localized fixed confirmation

This gate freezes the previously selected 37-parameter LS-anchored localized
receiver. It performs no training, no retuning, and no model selection.

## Experiment

- Two new 12-PRB UMi cases with different cell IDs and DMRS-port ordering.
- 6, 10, and 14 dB.
- 64 paired repetitions per case and SNR.
- Batch size 64 and four transport blocks per batch item.
- 32,768 transport blocks per SNR per receiver; 98,304 in total.
- 2,304 resume-safe CSV rows.

## Primary decision

The primary statistic is the paired TBLER difference, proposed minus LS+LMMSE,
using independently seeded simulation batches as clusters. Both a Student-t
interval and a stratified cluster bootstrap interval are reported. Per-SNR
paired intervals are also reported. A one-sided exact McNemar test on paired
transport-block errors is secondary evidence only.

## Binding outcome

- `GATE1_IMPLEMENTABLE_LOCALIZED_CONFIRMED_BEATS_LS`: continue to broad
  publication-scale generalization, complexity, and latency studies.
- `GATE1_IMPLEMENTABLE_LOCALIZED_FINAL_PARITY_WITH_LS`: stop the claim of
  beating LS+LMMSE. Retain the mechanism only if its complexity or
  interpretability case is strong.
- `GATE1_ABANDON_BAYESROUTE_FOR_LS_BEATING`: stop the architecture search.
- `GATE1_IMPLEMENTABLE_LOCALIZED_CONFIRMATION_INTERFACE_REPAIR_REQUIRED`:
  repair only the failed software or baseline-control path and rerun the same
  frozen experiment.
