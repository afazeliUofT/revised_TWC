# Gate-1 NR grid-scale audit

This gate preserves the completed 4,800-row clean-generalization evidence and the frozen global checkpoint.

It performs two tasks:

1. A supplemental conservative reanalysis of the existing evidence. It corrects the loaded-only subgroup label, reports per-case LS gaps and high-SNR reversals, and records that `delay_spread_s` is not applied to Sionna UMi/UMa by the current context builder.
2. A no-retraining paired audit of the frequency coordinate convention. The existing operator normalizes every allocation to the same coordinate width. The audit compares that convention against fixed physical subcarrier spacing referenced to a 4-PRB, 30-kHz allocation.

The 4-PRB case must be mathematically and numerically equivalent under both coordinate definitions. The 8- and 12-PRB cases determine whether the allocation normalization causes the observed grid-width generalization failure.

The audit uses the frozen `global_r24_cold_lf1_lt0p5` checkpoint and the fixed four-iteration damped extrinsic spatial-LMMSE detector. It does not retrain or tune the receiver.
