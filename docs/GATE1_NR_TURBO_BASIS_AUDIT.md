# Gate-1 NR Turbo Basis/Update Audit

This no-training gate is required because the prior turbo screen did not improve the pilot-only receiver even when true transmitted data symbols were supplied. Before replacing the global Fourier basis, the audit separates:

1. received-signal / true-symbol alignment;
2. arbitrary tied top-k observation selection;
3. raw versus learned effective observation noise;
4. covariance over-contraction versus mean improvement;
5. the best channel approximation attainable inside the current global basis.

The decisive control is `best_in_global_basis_calibrated`, which projects the true channel onto the frozen feature span and uses the projection residual as an oracle uncertainty. This is diagnostic only and is not a deployable receiver.

Expected evidence: 396 unique paired rows. No training or hyperparameter update is performed.
