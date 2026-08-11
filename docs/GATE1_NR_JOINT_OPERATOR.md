# Gate-1 joint posterior-operator training

The repaired damped-extrinsic spatial-LMMSE detector is fixed at four
iterations, damping 0.7, edge mass 0.8, and diagonal channel uncertainty.
This gate trains the stochastic channel operator through that detector on a
balanced mixture of UMi Type-1, UMi Type-2, UMa, and CDL-A PUSCH cases.

Four global candidates test rank and time-frequency length scale. A fifth
case-specific candidate is an explicitly labelled diagnostic upper bound. It
is used only to decide whether the remaining error is caused by global
parameter sharing or by the low-rank kernel basis.

The possible scientific outcomes are:

- `GATE1_JOINT_OPERATOR_SUPPORTED`: global joint training is sufficiently
  competitive to proceed to clean new-seed ablations.
- `GATE1_CONFIGURATION_CONDITIONED_OPERATOR_REQUIRED`: independent operators
  work but one global operator does not; build a PSD kernel mixture with
  pilot-conditioned routing.
- `GATE1_OPERATOR_BASIS_EXPANSION_REQUIRED`: case-specific training helps but
  the current kernel basis remains inadequate.
- `GATE1_JOINT_TRAINING_INSUFFICIENT`: revisit the posterior likelihood or
  stochastic operator model.

The historical PGCA/AGMP receiver remains excluded.
