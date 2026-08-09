# Gate-0 v2.4 scientific alignment

The revised implementation is organized around receiver-theoretic quantities rather than generic attention blocks.

| Concern | Gate-0 v2.4 response |
|---|---|
| Heuristic reliability | reliability is the posterior channel covariance; matched-model calibration is checked numerically |
| Heuristic inter-layer weights | graph edges are posterior expected squared whitened-Gram couplings and are verified against Monte Carlo simulation |
| Ignored posterior dependence | full local cross-layer channel covariance enters detection and coupling |
| Fixed nearest-neighbor count | `edge_mass` retains a variable number of interferers according to cumulative physical coupling |
| Arbitrary graph tuning | `edge_mass=0.80` is fixed and ablated; it is not selected by a performance-only Optuna objective |
| Architecture path may be inactive | smoke compares routing-off, default, full-graph, uncertainty-off, and uncertainty-on logits |
| Incorrect or uncalibrated uncertainty | smoke checks `u C u^H`, posterior PSD, normalized matched-model error, and 95% coverage |
| Weak software verification | exact shapes, gradients, parameter update, checkpoint reload, deterministic regeneration, and immutable run contracts are checked |
| Unclear Sionna use | Sionna Mapper/Demapper is executed and the simulator mapping is compared with Sionna symbols |
| Unclear DMRS claim | Gate-0 pilot abstraction is explicitly distinguished from exact 3GPP PUSCH DMRS |
| Missing practical reporting | evaluation records BER, bit NLL, Brier score, ECE, NMSE, coverage, edge density, parameters, receiver-only latency, throughput, and memory |

Gate 0 does not close the publication-level experimental requirements. Exact NR PUSCH/DMRS, LDPC TBLER, 3GPP channels, modern neural/model-driven baselines, data-aided posterior refinement, iteration-dependent effective-noise coupling, spatial covariance, optimized sparse execution, and broad mismatch tests remain a later gate. The Gate-0 edge-density metric is structural; dense reference execution is still used.


## Gate-0 v2.4 workflow integrity
The short screening stage completes only after all twelve named design indices succeed. Failed or interrupted trials are retried by exact parameter identity, and no replacement configuration can satisfy the completion gate.
