# Reviewer-comment alignment

## Central novelty and theory concern

The rejected paper added standard cross-attention and graph-attention modules
to an existing neural receiver. The new architecture removes PGCA and AGMP.
Its central contribution is a new estimator principle: exact Bayesian model
averaging of structured LMMSE channel estimators.

### Reviewer 1: principled derivation

Addressed by:

- an explicit probabilistic observation model;
- a positive-semidefinite structured mixture prior;
- closed-form component LMMSE posteriors;
- exact marginal-likelihood posterior component weights;
- a proof that the posterior mean is the MMSE estimator under the stated prior;
- a K=1 reduction to classical LMMSE channel estimation;
- a Bayes-risk comparison against every fixed linear estimator under the model;
- explicit failure conditions when the mixture model is misspecified.

### Reviewer 1: interpretation of learned coupling/weights

There are no free attention weights. The routing weights are posterior model
probabilities. Their value is determined by the probability of the observed
DMRS samples under each structured channel prior.

### Reviewer 1: identifiability and generalization

- component total powers are strictly ordered, removing label permutation;
- the basis is fixed and physical;
- parameters are shared across 4, 8, and 12 PRBs, DMRS Types 1 and 2, UMi,
  UMa, and CDL channels;
- evaluation cases have separate cell IDs and pilot-port permutations;
- evaluation cases are not used for training or checkpoint selection.

### Reviewer 2: block-LS and heuristic reliability

LS is now only an explicit baseline/anchor. Estimated channel power is never
called reliability. Reliability is represented by the posterior covariance and
by the marginal probability of the pilot observations.

### Reviewer 2: why pilot processing helps

The analytical answer is direct: each component posterior minimizes conditional
MSE within its Gaussian model, and their evidence-weighted average is the MMSE
estimator for the mixture model. The experiment reports channel NMSE, channel
NLL, posterior coverage, and final TBLER under a common detector.

### Reviewer 2: locality sensitivity

There is no nearest-K attention rule. Locality is encoded by overlapping smooth
frequency windows and physical delay/Doppler atoms. The fixed basis and its
rank are reported for every grid.

### Reviewer 3: users, layers, and DMRS ports

Every case stores:

- user count;
- layers per user;
- total stream count;
- ordered DMRS-port mapping;
- DMRS type, length, and additional positions;
- CDM groups without data.

The smoke gate validates the actual Sionna PUSCH mapping and tensor paths.

### Reviewer 3: complete training information

The configuration, optimizer, learning-rate schedule, batch size, SNR range,
loss terms, validation seeds, early-stopping rule, parameter counts, checkpoint
hashes, and every logged validation score are synchronized to GitHub.

### Complexity and practical tradeoff

The report gives exact trainable parameter counts. The leading estimator cost
is K structured Gaussian solves with shared pilot geometry. The K=1 and
moment-matched controls quantify the price and value of the mixture. A later
publication-scale gate will measure latency and memory only if this gate first
shows an estimation advantage.
