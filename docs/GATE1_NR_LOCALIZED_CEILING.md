# Gate-1 localized delay--Doppler oracle ceiling

This is the final training-free go/no-go gate for the objective of beating
Sionna NR LS+LMMSE.

The gate selects one bounded-rank localized delay--Doppler basis using only new
4- and 8-PRB cases. It then freezes that basis and evaluates a 12-PRB holdout.
The localized receiver is given oracle coefficients for the residual between
the true channel and Sionna's LS estimate. It is therefore a deliberately optimistic engineering ceiling
for an LS-anchored localized Bayesian posterior. A failure is strong practical
evidence to stop this architecture search, although it is not a formal
impossibility theorem for every conceivable nonlinear receiver.

Decision contract:

* `GATE1_LOCALIZED_CEILING_BEATS_LS`: train exactly one LS-anchored localized
  posterior and run one final holdout.
* `GATE1_LOCALIZED_CEILING_POSSIBLY_BEATS_LS`: allow exactly one final trained
  model under a hard stop.
* `GATE1_ABANDON_BAYESROUTE_FOR_LS_BEATING`: stop architecture search. A
  practical learned model is not expected to overcome a failed, true-channel-
  informed ceiling under the same bounded representation.

The maximum nominal basis rank is 128. The 12-PRB holdout is not used for basis
selection. No training or hyperparameter optimization is performed.
