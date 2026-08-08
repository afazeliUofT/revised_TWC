# Gate-0 v2 scientific alignment

The revised implementation is organized around receiver-theoretic quantities rather than generic attention blocks.

| Concern | Gate-0 v2 check |
|---|---|
| Heuristic reliability | posterior channel variance is used and its path is ablated |
| Heuristic inter-layer weights | graph edges come from posterior whitened-Gram coupling |
| Fixed neighborhood | `edge_mass` selects a variable number of interferers |
| Architecture path may be inactive | smoke compares routing-off, routing-on, uncertainty-off, and uncertainty-on logits |
| Weak software verification | exact shapes, gradients, parameter update, PSD covariance, checkpoint reload, and deterministic regeneration are checked |
| Unclear Sionna use | real Sionna mapping/demapping is executed |
| Unclear DMRS claim | Gate-0 pilot abstraction is explicitly distinguished from exact 3GPP PUSCH DMRS |
| Missing practical reporting | evaluation records BER, bit NLL, Brier score, ECE, NMSE, edge density, parameters, and latency |

Gate-0 does not close the publication-level experimental requirements. Exact NR PUSCH/DMRS, LDPC TBLER, 3GPP channels, modern neural/model-driven baselines, and broad mismatch tests remain a later gate.
