# Gate-1 implementable localized receiver

This gate is the single permitted trained follow-up to the precision-corrected
localized oracle ceiling.

The receiver uses Sionna's observable LS estimate and error variance as an
anchor. A fixed, precision-corrected localized delay--Doppler basis represents
the channel-estimation residual. Smooth learned mode curves define a Gaussian
prior over residual coefficients. The coefficient posterior is obtained by
exact conditioning on the observable DMRS residual. The posterior residual mean
is damped and added to LS; its calibrated covariance is passed to the fixed
spatial-LMMSE message detector.

The true channel is used only in the supervised training loss and in reporting
metrics. It is not an input to the receiver at inference.

Training and validation use only 4- and 8-PRB cases. The 12-PRB context is built
only after training and checkpoint selection have completed. The final result
is binding:

- statistically beat LS+LMMSE: proceed to publication-scale evaluation;
- negative point estimate with inconclusive interval: one larger confirmation;
- otherwise: stop the BayesRoute architecture search for beating LS+LMMSE.
## Evidence-schema compatibility

The precision-corrected ceiling report stores the top-level winner as a name
string and the complete selected-basis specification under
`holdout.contract.winner`. The implementable gate resolves the full mapping
from that contract and remains compatible with older reports that stored the
full mapping at top level.

