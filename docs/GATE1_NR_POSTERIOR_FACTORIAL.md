# Gate-1 NR Posterior-Factorial Diagnostic

## Purpose

The completed grid audit showed that replacing allocation-normalized frequency
coordinates with fixed physical coordinates **without retraining** makes the
8-PRB and 12-PRB results worse. That result rules out a drop-in coordinate
change, but it does not fairly compare coordinate systems after training. The
current global rank-24 posterior was trained only on 4-PRB cases and contains
only 25 trainable scalar parameters.

This diagnostic performs the missing fair experiment. It keeps the repaired
spatial-LMMSE message detector fixed and retrains six small, explicitly
positive-semidefinite posterior models with equal 4-PRB and 8-PRB exposure.
The 12-PRB grid remains untouched until final evaluation.

## Candidate matrix

| Coordinate system | Single scale | Multi-scale | Context-conditioned multi-scale |
|---|---:|---:|---:|
| Allocation normalized | rank 24 | rank 64 | rank 64 |
| Fixed physical spacing | rank 24 | rank 64 | rank 64 |

The multi-scale model uses four fixed random-Fourier blocks. Nonnegative
spectral weights preserve a valid PSD prior. The context-conditioned model uses
known receiver context only, including allocation width, SCS, DMRS type and
length, stream count, receive-antenna count, mobility, and channel family. It
does not use data bits or the instantaneous channel.

## Fairness controls

- Every candidate receives 800 batches from 4 PRBs and 800 batches from 8 PRBs.
- All candidates use the same random streams and fixed validation streams.
- Candidate selection uses only the trained 4-PRB and 8-PRB grids.
- The 12-PRB seeds are untouched until evaluation.
- The detector, damping, iterations, graph-mass rule, NR LDPC chain, and MCS are
  identical across candidates.
- A factorized control feeds Sionna's LS estimate and error variance into the
  repaired detector. This identifies whether a remaining gap comes from the
  posterior model or from the detector-estimator interface.

## Interpretation of outcomes

- `GATE1_POSTERIOR_FACTORIAL_BEATS_LS`: freeze the architecture and start a
  publication-scale baseline and complexity campaign.
- `GATE1_POSTERIOR_FACTORIAL_NEAR_LS`: add one principled data-aided posterior
  update and retest.
- `GATE1_POSTERIOR_FACTORIAL_TRAINING_EXTENSION_REQUIRED`: extend only the
  selected candidate; do not redesign the architecture yet.
- `GATE1_DETECTOR_ESTIMATOR_INTERFACE_REPAIR_REQUIRED`: align the repaired
  detector with the LS-estimator control before changing the posterior.
- `GATE1_TURBO_REFINEMENT_REQUIRED`: the richer posterior helps, but one
  data-aided update is needed to close the remaining gap.
- `GATE1_POSTERIOR_LOCALIZATION_REQUIRED` or
  `GATE1_POSTERIOR_FAMILY_REDESIGN_REQUIRED`: replace the global RFF prior with
  a localized delay-Doppler basis.

The gate is diagnostic. It does not make the manuscript publication-ready.
