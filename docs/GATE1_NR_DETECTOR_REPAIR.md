# Gate-1 detector repair

The completed 2,304-row failure-mode diagnostic isolates the main blocker:
the legacy soft-PIC detector remains far from perfect-CSI LMMSE even when it
is given the true channel. Global LLR-temperature scaling and posterior-
variance rescaling do not remove the TBLER floor.

This repair replaces the scalar-interference soft-PIC front end with a damped
extrinsic LMMSE message detector. For each target stream and data RE, the new
front end:

1. softly cancels only graph-selected strong interferers;
2. retains omitted interferers as Gaussian spatial covariance terms;
3. computes a full receive-antenna LMMSE filter;
4. excludes the target's own previous belief from its message;
5. projects the scalar extrinsic observation onto the QAM alphabet; and
6. damps the updated symbol moments before the next iteration.

The learned posterior operator is unchanged. The first H100 smoke gate checks
single-stream equivalence, permutation equivariance, positive variances,
NR/LDPC integration, true-channel operation, and gradient flow to the learned
posterior parameters. The subsequent 2,688-row resume-safe screen first checks that the one-step true-channel implementation matches a conventional perfect-CSI LMMSE receiver. It then selects iterations and damping on one seed subset and evaluates the selected configuration on disjoint holdout repetitions.

This is a repair gate, not a publication campaign.

The label “extrinsic LMMSE” is used deliberately: this gate does not claim a full expectation-propagation implementation. A full EP precision-subtraction update is reserved as the next repair only if the simpler target-extrinsic LMMSE gate fails.
