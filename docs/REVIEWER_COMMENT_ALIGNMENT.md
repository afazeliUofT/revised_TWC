# Reviewer-comment alignment

This package is designed around the main rejection reason: the previous receiver was viewed as a heuristic attention extension without enough receiver-theoretic justification.

Implementation alignment:

1. Pilot reliability is no longer estimated channel power. It is posterior channel uncertainty.
2. Pilot processing is no longer fixed nearest-K attention. It is Gaussian conditioning under a learned positive-semidefinite channel operator.
3. Inter-layer weights are no longer hidden-state attention scores. They are posterior expected interference couplings from the whitened Gram geometry.
4. The smoke test explicitly checks the user/layer/DMRS-port abstraction, pilot orthogonality, tensor shapes, gradient flow, and Sionna availability.
5. Training and evaluation scripts output complexity, BER, TBLER proxy, bit NLL, calibration, channel NMSE, and ablation plots.
