# Gate-0 mechanism diagnostic

This add-on does not change the v2.4 receiver or its completed initial checkpoint.
It adds one focused diagnostic gate before any larger Gate-0 run.

The diagnostic isolates effects that the first evaluation combined:

- posterior uncertainty in the detector while holding the graph fixed;
- posterior covariance in graph construction while keeping detector uncertainty on;
- full covariance versus diagonal and homoscedastic uncertainty;
- posterior-coupling routing versus a random graph at the same nominal density;
- trained versus untrained stochastic operator;
- learned mismatched versus exactly matched posterior operator;
- perfect-CSI PIC iteration and LLR-overconfidence behavior;
- a perfect-CSI LMMSE marginal reference;
- held-out scalar temperature controls for two overconfident baselines.

The run is resume-safe. It uses three SNR points, ten paired repetitions per point,
and writes all compact evidence under `outputs/eval`, `outputs/reports`,
`outputs/plots`, and `outputs/gates`.

This remains a Gate-0 abstract simulator. It is not a 3GPP PUSCH or coded-TBLER result.
