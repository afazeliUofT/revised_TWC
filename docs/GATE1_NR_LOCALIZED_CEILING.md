# Gate-1 NR localized delay-Doppler ceiling, precision-corrected revision

This is the binding no-training oracle ceiling for the goal of beating the
standard Sionna NR LS+LMMSE receiver.

## Why this rerun is mandatory

The earlier localized ceiling constructed nearly dependent delay-Doppler atoms
in `complex64` and then promoted the matrix to `complex128` for SVD. Its
relative rank threshold was `1e-10`, far below float32 precision. Float32
roundoff was therefore retained as extra singular directions, and the selected
96-atom basis was reported as rank 96 on all grids.

This revision constructs the atom matrix directly in `complex128`, applies the
same declared `1e-10` relative singular-value threshold, and only then casts the
orthonormal basis to `complex64` for the receiver. For the selected
`ldd_w2_d16_v3_r96_tau3us` basis, the deterministic effective ranks are:

- 4 PRBs: 51;
- 8 PRBs: 69;
- 12 PRBs: 84.

The result synchronized in commit `f5e591d` is retained in Git history but must
not be used for the architecture decision. This precision-corrected 540-row
selection/holdout campaign replaces it.

## Unchanged contracts

- four bounded nominal-rank candidate bases;
- selection on new 4- and 8-PRB cases only;
- fresh 12-PRB holdout not used for selection;
- Sionna LS+LMMSE and perfect-CSI LMMSE references;
- repaired spatial-LMMSE detector;
- exact per-sample coupling graph for batch-dependent posterior covariance;
- no training;
- 252 selection rows and 288 holdout rows.

## Binding decision

- `GATE1_LOCALIZED_CEILING_BEATS_LS` or
  `GATE1_LOCALIZED_CEILING_POSSIBLY_BEATS_LS`: permit exactly one trained,
  LS-anchored localized posterior with a hard final stop.
- `GATE1_ABANDON_BAYESROUTE_FOR_LS_BEATING`: stop this architecture search.
- `GATE1_LOCALIZED_CEILING_INTERFACE_REPAIR_REQUIRED`: repair only the failed
  baseline/control path before making the decision.
