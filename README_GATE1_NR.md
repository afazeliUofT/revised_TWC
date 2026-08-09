# BayesRoute-Rx Gate-1 NR Integration v1

This add-on keeps the successful Gate-0 evidence and introduces a separate
standards-compliant integration gate. It uses Sionna 2.0.1 PUSCH configuration,
PUSCH DMRS generation, layer mapping/demapping, transport-block encoding and
decoding, UMi/UMa/CDL channels, LS-LMMSE, perfect-CSI LMMSE, and K-best.

The rejected PGCA+AGMP receiver is deliberately not included in this gate.

## Gate sequence

1. `RUN_07_SUBMIT_GATE1_NR_SMOKE.py`: strict API, mapping, channel, coding,
   detector, gradient, posterior, and multi-layer health gate.
2. `RUN_08_SUBMIT_GATE1_NR_EVIDENCE.py`: short resume-safe training and decoded
   TBLER campaign. Run only after the smoke report is reviewed.

Do not use the older `RUN_04_SUBMIT_FULL_TRAIN_EVAL.py` for this stage.
Gate-1 preliminary evidence is not a publication-scale campaign.
