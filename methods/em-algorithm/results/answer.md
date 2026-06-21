# The Expectation-Maximization Algorithm

The method solves maximum likelihood problems where the observed datum `y` is an incomplete view of a more convenient complete datum `x`. Let

`g(y | phi) = integral_{X(y)} f(x | phi) dx`

be the observed-data density induced by the complete-data density `f`, and let

`k(x | y, phi) = f(x | phi) / g(y | phi)`

be the conditional density of complete data given the observation. The observed log likelihood is `L(phi) = log g(y | phi)`.

For a current iterate `phi_p`, define

`Q(phi | phi_p) = E_{k(. | y, phi_p)}[log f(X | phi)]`

and

`H(phi | phi_p) = E_{k(. | y, phi_p)}[log k(X | y, phi)]`.

Since `log f(X | phi) = log g(y | phi) + log k(X | y, phi)` on the compatible complete-data set,

`L(phi) = Q(phi | phi_p) - H(phi | phi_p)`.

The update is:

1. E-step: compute the conditional distribution of the complete data under the current parameter and use it to form `Q(phi | phi_p)`.
2. M-step: choose `phi_{p+1}` so that `Q(phi_{p+1} | phi_p) >= Q(phi_p | phi_p)`. Exact maximization gives the usual form; this weaker condition gives a generalized update.

The ascent proof is the essential point. Jensen's inequality gives

`H(phi | phi_p) <= H(phi_p | phi_p)`,

with equality only when the two conditional complete-data distributions agree, up to null sets. Therefore

`L(phi_{p+1}) - L(phi_p)`

is the increase in `Q` plus a nonnegative conditional-distribution term. Raising the expected complete-data log likelihood raises the observed-data likelihood.

For a regular exponential-family complete-data model

`f(x | phi) = b(x) exp(phi t(x)^T) / a(phi)`,

the E-step computes expected complete-data sufficient statistics

`t_p = E[t(X) | y, phi_p]`.

The M-step is the ordinary complete-data maximum-likelihood solve with `t(x)` replaced by `t_p`; equivalently, solve the complete-data likelihood equation using expected sufficient statistics.

For a Gaussian mixture, the complete data add a hidden one-hot component label `z_i`. With current parameters,

`r_ik = pi_k p_k(y_i | theta_k) / sum_j pi_j p_j(y_i | theta_j)`

is the conditional expectation of `z_ik`. The M-step uses fractional sufficient statistics:

`N_k = sum_i r_ik`, `pi_k = N_k / n`,

`mu_k = (sum_i r_ik y_i) / N_k`,

`Sigma_k = (sum_i r_ik (y_i - mu_k)(y_i - mu_k)^T) / N_k`.

The scikit-learn code snapshot in `code/sklearn_base_mixture_current.py` and `code/sklearn_gaussian_mixture_current.py` implements this structure directly: `_estimate_log_prob_resp` computes normalized log responsibilities, `_m_step` exponentiates them and updates weighted Gaussian parameters, and the base loop tracks the lower bound until its change is within tolerance.

The algorithm's guarantee is monotone observed-likelihood ascent under the stated update condition, not global optimization. It can converge to local maxima or stationary points, and its speed depends on the amount of missing information induced by the chosen complete-data representation.
