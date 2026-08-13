# Gate-1 posterior extension

This gate performs one predeclared continuation of only the selected
`physical_context_multiscale_r64` posterior. It starts from the audited best
checkpoint, adds 800 batches at 4 PRBs and 800 batches at 8 PRBs, and uses a
cosine learning-rate reduction from 0.001 to 0.0002.

The 12-PRB case is not used for training, validation, model selection, stopping,
or learning-rate selection. A fresh 12-PRB case is evaluated only after the
extension is complete. The gate compares the extended and original checkpoints,
uncertainty off with the same graph, Sionna LS estimate with the same repaired
detector, standard LS-LMMSE, and perfect-CSI LMMSE.

This is the only allowed continuation of the unchanged pilot-only posterior. If
it does not reach LS-LMMSE, the next architectural step is one principled
soft-data-aided posterior update rather than repeated training extensions.
