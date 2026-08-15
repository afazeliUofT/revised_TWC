# Estimator-focused system model

For each resource element, the base station receives the superposition of all
active spatial layers. Layer indices are ordered by user and then by the layer
inside that user. Each layer is assigned one configured DMRS port. The Sionna
PUSCH transmitter produces the exact NR DMRS/CDM/OCC pattern and the receiver
uses the same mapping.

The primary experiment changes only the channel estimator. Every estimator is
followed by the same fixed spatial LMMSE message detector, the same layer
demapper, and the same NR LDPC decoder. Thus, a performance difference cannot
be attributed to a different detector.

The proposed estimator starts from the pilot observation matrix and known NR
configuration. It does not use the true channel, channel-model label, or user
speed at inference. The channel is represented in a fixed localized
frequency-window/delay/Doppler basis. Four positive-semidefinite Gaussian prior
components produce four exact LMMSE posterior estimates. Bayes' rule uses the
pilot marginal likelihood to combine them. The returned channel mean and
posterior covariance are passed to the common detector.
