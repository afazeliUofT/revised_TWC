# Gate-2: Evidence-Weighted Mixture LMMSE Channel Estimation

## Purpose

This gate replaces the rejected attention modules with one receiver-theoretic
principle: **Bayesian model averaging of a small bank of exact structured LMMSE
channel estimators**. The detector is fixed for every estimator in the primary
comparison. Therefore, measured differences are caused by channel estimation,
not by detector choice.

## Estimator

For pilot observations

\[
\mathbf y_p=\mathbf A_p\mathbf h+\mathbf n,
\qquad \mathbf n\sim\mathcal{CN}(\mathbf 0,\sigma^2\mathbf I),
\]

the channel prior is a finite mixture

\[
p(\mathbf h\mid\mathbf c)=\sum_{k=1}^{K}\pi_k(\mathbf c)
\mathcal{CN}(\mathbf 0,\mathbf C_k).
\]

Each covariance is positive semidefinite by construction in a fixed localized
delay--Doppler basis. For each component, the conditional posterior is the
closed-form Gaussian LMMSE posterior. Component weights are computed from the
pilot marginal likelihood,

\[
\gamma_k(\mathbf y_p)=
\frac{\pi_k\,\mathcal{CN}(\mathbf y_p;\mathbf 0,
\mathbf A_p\mathbf C_k\mathbf A_p^H+\sigma_k^2\mathbf I)}
{\sum_j\pi_j\,\mathcal{CN}(\mathbf y_p;\mathbf 0,
\mathbf A_p\mathbf C_j\mathbf A_p^H+\sigma_j^2\mathbf I)}.
\]

The estimator is

\[
\widehat{\mathbf h}=\sum_k\gamma_k\widehat{\mathbf h}_k.
\]

The weights are posterior model probabilities, not attention scores.

## Why this is interpretable

Every learned quantity has a receiver meaning:

- mixture prior probability;
- smooth delay--Doppler power profile;
- ordered component power;
- component effective-noise scale;
- posterior model-discrepancy floor;
- output uncertainty calibration.

The four-component model has fewer than 128 trainable scalar parameters. The
one-component control reduces to ordinary structured LMMSE channel estimation.

## Primary estimator-only comparison

All estimators below feed the same fixed spatial LMMSE message detector:

1. evidence-weighted mixture estimator;
2. hard component selection;
3. uniform mixture ablation;
4. moment-matched LMMSE estimator using the mixture's second-order covariance;
5. separately trained one-component LMMSE estimator;
6. LS plus linear interpolation;
7. perfect CSI control.

The standard Sionna LS-estimation plus LMMSE-detection chain is retained only as
a secondary end-to-end reference.

## Workflow

1. Deploy the package.
2. Run the H100 smoke gate.
3. Synchronize the smoke evidence to GitHub.
4. Review the GitHub smoke report.
5. Run the resume-safe training/evaluation screen.
6. Synchronize the final evidence to GitHub.

The binary checkpoint is not needed for diagnosis because the complete small
parameter state is exported as JSON under `outputs/reports/` and synchronized
to GitHub. Checkpoint hashes, source hashes, configuration hashes, raw paired
rows, aggregate tables, plots, and Slurm logs are also synchronized.

## Binding classifications

- `GATE2_EVIDENCE_MIXTURE_BEATS_LMMSE_CE`: statistically beats both the
  separately trained one-component LMMSE estimator and the moment-matched
  LMMSE estimator under the same detector.
- `GATE2_EVIDENCE_MIXTURE_ESTIMATION_GAIN_TBLER_INCONCLUSIVE`: significant
  channel-NMSE gain with no material TBLER harm, but TBLER superiority is not
  yet established.
- `GATE2_EVIDENCE_MIXTURE_NO_ADVANTAGE`: stop this estimator family for the
  LMMSE-channel-estimation-beating objective.
- `GATE2_EVIDENCE_MIXTURE_INTERFACE_REPAIR_REQUIRED`: repair only a software or
  fairness-control failure and rerun the identical gate.
