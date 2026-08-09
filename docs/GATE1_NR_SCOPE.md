# Gate-1 scientific scope

## What is validated

- Sionna-generated PUSCH resource grids and DMRS, not a local DFT pilot model.
- DMRS configuration Type 1 and Type 2, lengths 1 and 2, ports, CDM groups,
  frequency OCC, and time OCC.
- Explicit flattening `(user, within-user layer) -> receiver node -> DMRS port`.
- Sionna LayerMapper and LayerDemapper consistency for one- and two-layer users.
- Sionna TBEncoder/TBDecoder round trip and decoded transport-block error rate.
- 3GPP TR 38.901 UMi, UMa, CDL-A, CDL-C, and CDL-D channels.
- Standard LS channel estimation plus LMMSE detection, perfect-CSI LMMSE, and a
  perfect-CSI K-best reference on a small two-stream case.
- BayesRoute posterior uncertainty, fixed-cardinality coupling controls, PSD
  posterior, gradients, and transfer across DMRS configurations.

## Preliminary evidence only

The optional second job trains only the compact stochastic operator and evaluates
paired decoded TBLER controls. It is designed to decide whether a larger campaign
is justified. It is not sufficient for a TWC submission.

## Excluded for now

- Rejected PGCA+AGMP historical baseline.
- Full publication channel/MCS/bandwidth matrix.
- Sparse execution kernel and wall-clock sparsity claim.
- Final K-best/sphere-decoding complexity sweep.
