# Gate-1 one-step soft-data-aided posterior refinement

The completed training-only extension converged but remained far behind the
Sionna LS+LMMSE reference. The same repaired detector fed by the Sionna LS
estimate exactly matched standard LS+LMMSE, isolating the remaining problem to
the pilot-only learned channel posterior.

This gate adds one parameter-free outer update. The frozen pilot posterior is
represented by latent moments `(m0,C0)`. The repaired detector provides soft
symbol moments `(xbar,vx)`. For a selected data RE r, the received sample is
approximated by

`y_r = A_r(xbar) z + eta_r`,

where the moment-matched noise variance is

`tau_r = sigma_w^2 + sum_n vx[n,r] E[|h_n[r]|^2 | pilots]`.

Because the symbol moments were inferred from the same received data, the
pseudo-likelihood is applied fractionally with a predeclared information
damping coefficient rho. The Gaussian update is

`C1^{-1} = C0^{-1} + rho A^H R^{-1} A`,

`m1 = C1(C0^{-1}m0 + rho A^H R^{-1}y)`.

Only the most reliable data REs, ranked by aggregate posterior symbol variance,
are used. The physical coupling graph is held fixed so the experiment isolates
the value of channel-posterior refinement. The update adds no trainable
parameter.

The H100 smoke validates the exact fractional Gaussian equations, PSD
covariance, public-posterior equivalence, fixed graph, NR LDPC path, oracle
symbol path, and gradient flow to the frozen posterior operator. The later
screen selects six predeclared `(reliability fraction, information damping)`
settings using only 4- and 8-PRB cases. A new 12-PRB case is evaluated only
after selection.
