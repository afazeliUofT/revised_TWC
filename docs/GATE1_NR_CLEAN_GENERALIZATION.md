# Gate-1 clean new-seed generalization diagnostic

This gate freezes the best global joint posterior operator selected before the
new evaluation seeds. It performs no retraining and no holdout-driven model
selection. The fixed receiver uses the repaired four-iteration spatial-LMMSE
message detector, damping 0.7, edge mass 0.8, and diagonal posterior channel
uncertainty.

The evaluation contains twelve NR PUSCH cases: four new-seed replications and
eight held-out changes covering six streams, Type-2 double-symbol DMRS, higher
mobility, larger delay spread, CDL-C/CDL-D, an eight-PRB grid, and a different
MCS. Every receiver variant sees the same transmitted bits, topology, channel,
and noise realization.

The gate tests the frozen receiver against the old posterior checkpoint,
uncertainty-off with the same graph, a random equal-cardinality graph, a full
graph, graph-off, LS-LMMSE, and perfect-CSI LMMSE. It reports paired confidence
intervals and explicitly separates support for posterior uncertainty from
support for coupling-ranked routing.

This is still a diagnostic. It is not a publication-scale campaign and does
not include the rejected PGCA/AGMP architecture.
