# Gate-1 NR failure-mode diagnostic

This diagnostic uses the completed preliminary Gate-1 checkpoint. It does not retrain the receiver.
It separates four possible causes of the high-SNR error floors observed in the preliminary campaign:

1. instability of the iterative detector;
2. posterior-variance or LLR calibration mismatch;
3. channel-operator mismatch;
4. weak or misleading graph routing.

Calibration and evaluation use disjoint deterministic seeds. Every evaluation batch is paired across all
receiver variants and is appended atomically. The job is resume-safe.

This is a diagnostic gate, not a publication campaign. The rejected PGCA/AGMP receiver is not included.
