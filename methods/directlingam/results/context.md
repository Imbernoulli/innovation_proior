# Context: causal discovery from observational linear non-Gaussian data

## Research question

Given only observational data — a sample of a vector `x = (x_1, ..., x_p)` with no
interventions, no time order, no labels saying which variable came first — recover the
*directed* causal graph behind it: both the structure (who points to whom) and the
connection strengths. The data-generating process of interest is a linear acyclic
structural equation model: each variable is a linear function of its direct causes plus an
independent noise term, and the directed graph is a DAG. Writing the noise (the "external
influences") as `e_i`,

```
x_i = sum_{k(j) < k(i)} b_{ij} x_j + e_i,
```

where `k(.)` is some unknown causal order and `b_{ij}` is the strength of `x_j -> x_i`. In
matrix form `x = B x + e`, with `B` permutable to strictly lower triangular (zero diagonal)
under the causal order. The goal is to estimate `B` — equivalently the order `k(.)` and the
weights — from `x` alone.

Why this is hard, and why it matters: in many sciences (sociology, biology, neuroimaging)
controlled experiments are impossible or unethical, so the only handle on causality is
passively observed data. But the natural statistic of such data — its covariance matrix — is
direction-blind. For two correlated variables `x_1` and `x_2`, the models `x_1 -> x_2` and
`x_2 -> x_1` produce the *identical* covariance, so any method that reads only second-order
statistics cannot prefer one over the other. A solution has to break this symmetry with
something covariance does not see, and it has to do so without prior knowledge of the order,
robustly, for tens to a hundred variables.

## Background

**The linear acyclic SEM and its covariance-only ceiling.** Structural equation models and
Bayesian networks (Bollen 1989; Pearl 2000; Spirtes, Glymour & Scheines 1993) model
continuous causal relations. When estimation uses only the covariance structure — as in
Gaussian SEM, the constraint-based PC algorithm (Spirtes & Glymour 1991), and the score-based
GES (Chickering 2002) — the model is in general identifiable only up to its Markov equivalence
class. Within that class many edge directions are interchangeable; for the two-variable case
the direction is completely undetermined. This is a hard ceiling of second-order statistics,
not a defect of any one algorithm.

**Non-Gaussianity lifts the ceiling.** A linear acyclic SEM whose disturbances `e_i` are
mutually independent, non-Gaussian, with nonzero variance, and with no latent confounders is
called LiNGAM (Shimizu, Hoyer, Hyvärinen & Kerminen, JMLR 2006). Solving `x = Bx + e` gives

```
x = A e,   A = (I - B)^{-1},
```

with `A` permutable to lower triangular and unit diagonal. Since the `e_i` are independent and
non-Gaussian, this is precisely the independent component analysis (ICA) model with mixing
matrix `A` and sources `e`. ICA is identifiable: by Comon (1994) the mixing matrix is
recoverable up to permutation, scaling, and sign of the columns, provided at most one source
is Gaussian. So non-Gaussianity converts an unidentifiable covariance problem into an
identifiable ICA problem — the full structure, including all edge directions, becomes
recoverable from observational data. This identifiability is the load-bearing fact of the
whole area; it is what makes the problem solvable at all.

**ICA and its contrast functions.** ICA seeks an unmixing `W` so that the components of `Wx`
are maximally independent. Independence is measured by mutual information, and minimizing the
mutual information of decorrelated components is equivalent to maximizing the sum of their
negentropies (Hyvärinen 1999), where negentropy `J(y) = H(y_gauss) - H(y)` measures
non-Gaussianity and is invariant under invertible linear transforms. Differential entropy is
estimated by simple contrast functions: `J(y) ~ c [E{G(y)} - E{G(nu)}]^2` for a non-quadratic
`G` and standard Gaussian `nu`, with the maximum-entropy family giving optimal contrasts of
the form `G_opt(u) = k1 log f(u) + k2 u^2 + k3` (Hyvärinen 1998). The log-cosh / `tanh`
nonlinearity is the standard robust choice for super-Gaussian sources. A known and important
piece of ICA folklore: the recovered components are insensitive to the exact log-pdf used, as
long as its shape is roughly right.

**The lever that distinguishes cause from effect.** The Darmois–Skitovitch theorem (Darmois
1953; Skitovitch 1953): if two linear combinations of independent variables,
`y_1 = sum_i alpha_i s_i` and `y_2 = sum_i beta_i s_i`, are *independent*, then every `s_j`
with `alpha_j beta_j != 0` must be Gaussian. The contrapositive is the usable form: if some
shared source `s_j` is non-Gaussian and enters both combinations with nonzero weight, then
`y_1` and `y_2` are *dependent*. This theorem ties non-Gaussianity directly to detectable
statistical dependence between linear combinations.

**Exogenous variables.** In a LiNGAM, a variable with no parents (`all b_{ij} = 0` for
`j != i`) satisfies `x_i = e_i`: the noise is observed directly. Acyclicity plus the absence
of latent confounders guarantees at least one such exogenous variable exists. For an exogenous
`x_j`, the least-squares regression coefficient of `x_i` on `x_j` equals the mixing
coefficient `a_{ij}`, because `cov(x_i, x_j) = a_{ij} var(x_j)`.

**Measuring independence in practice.** Least-squares residuals are by construction
uncorrelated with the regressor, so uncorrelatedness cannot be used to test for independence —
a genuine independence measure is required. Mutual information `I(y_1, y_2)` is the canonical
choice. A nonparametric kernel estimator (Bach & Jordan 2002) computes it from the
determinants of Gaussian-kernel Gram matrices, made tractable by incomplete Cholesky
low-rank approximations of rank `M << n`; its regularizer `kappa` and bandwidth `sigma` are
set to `(2e-3, 0.5)` for `n > 1000` and `(2e-2, 1.0)` for `n <= 1000`.

## Baselines

**Gaussian / covariance-based discovery — PC (Spirtes & Glymour 1991) and GES (Chickering
2002).** PC tests conditional independences to build a skeleton and orient what it can; GES
greedily optimizes a decomposable score over DAG space. Both rest on second-order statistics
(or on conditional-independence patterns that, under Gaussianity, are second-order). *Gap:*
they return an equivalence class, leaving many edges undirected; on two variables they cannot
choose a direction at all. They cannot exploit information beyond the covariance matrix.

**ICA-LiNGAM (Shimizu, Hoyer, Hyvärinen & Kerminen, JMLR 2006).** The first estimator for
LiNGAM. (1) Run an ICA algorithm (FastICA with the `tanh` nonlinearity) on the centered data
to estimate `W = A^{-1}`. (2) Permute the rows of `W` to remove zeros from the diagonal —
because the correct correspondence makes the diagonal all nonzero — by solving the linear
assignment problem that minimizes `sum_i 1/|W̃_{ii}|`. (3) Divide each row by its diagonal so
the diagonal is all ones. (4) Set `B̂ = I - W̃'`. (5) Permute `B̂` by a simultaneous row-column
permutation `B̃ = P B̂ P^T` to be as close as possible to strictly lower triangular, measured
by `sum_{i <= j} B̃_{ij}^2`; for many variables, iteratively zero the smallest-magnitude
entries and test for permutability to strict lower triangularity. *Gaps, all observed:*
(i) ICA optimizes a non-convex contrast by iterative search, so it can land in a local optimum
and is not guaranteed to converge to the right unmixing in a finite number of steps; it needs
an initial guess, and gradient variants need a step size, and all variants need a convergence
criterion — algorithmic parameters with no systematic way to set them, and a bad initial guess
can give a wrong answer. (ii) Both permutation steps are *not scale-invariant*: `1/|W̃_{ii}|`
and `sum B̃_{ij}^2` depend on the variances of the variables, yet scale is irrelevant to causal
order, so normalizing variables to unit variance can change — even reverse — the recovered
ordering at finite sample sizes.

**Kernel-based pairwise mutual information (Bach & Jordan 2002).** Not a causal-discovery
method itself but the independence estimator a non-ICA approach could lean on: a consistent
nonparametric `I(y_1, y_2)` from kernel Gram-matrix determinants. *Gap / cost:* it estimates a
two-dimensional dependency, requires a kernel bandwidth and regularizer, and is expensive —
`O(n p^3 M^2 + p^4 M^3)` for the ordering step with low-rank rank `M` — and on small samples kernel
estimates of a 2-D quantity are noisy.

## Evaluation settings

The standard testbed for linear non-Gaussian causal discovery at the time:

- **Synthetic DAGs with controlled sparsity.** Generate a random `p x p` strictly-lower-
  triangular structure by drawing lower-triangle entries as Bernoulli (Kalisch & Bühlmann
  2007), tuning the success probability so the expected in/out degree is a small constant
  (e.g. 2 or 5 adjacent variables). Replace nonzero entries by weights drawn from
  `[-1.5, -0.5] U [0.5, 1.5]` and pick disturbance variances from `[1, 3]` (Silva et al.
  2006). Also dense / full DAGs where every pair is connected.
- **Non-Gaussian disturbances.** Draw `e_i` from a battery of non-Gaussian densities — e.g.
  Student-t (3 and 5 df), double-exponential / Laplace, uniform, exponential, and various
  symmetric and asymmetric mixtures of Gaussians (the 18-distribution suite of Bach & Jordan
  2002) — so results are not tied to one noise shape.
- **Sizes.** Variable counts `p` from ~10 up to 100; sample sizes `n` from 500 to a few
  thousand. After generating `x` per the LiNGAM, the variable order is randomly permuted so the
  method gets no order hint.
- **Metrics.** Compare the estimated connection-strength matrix with the true `B`, for example
  by the Frobenius norm, and separately check whether the recovered order and edge directions
  match the generating DAG.

## Code framework

The method plugs into a standard observational-data harness: a data matrix `X` of shape
`(n_samples, n_variables)` goes in, an adjacency matrix `B` comes out, with the convention that
`B[i, j] != 0` means `x_j -> x_i`. The primitives that already exist are ordinary numerical
ones — covariance/variance, least-squares regression, array bookkeeping. What does *not* yet
exist is the rule that turns covariance-blind observational data into a directed order; that
rule is the single empty slot.

```python
import numpy as np


def regression_coef(xi, xj):
    """LS coefficient of xi on xj (scalar): cov(xi, xj) / var(xj)."""
    return np.cov(xi, xj, bias=True)[0, 1] / np.var(xj)


def residual(xi, xj):
    """Residual of xi after removing its least-squares projection on xj.
    By construction this residual is uncorrelated with xj — but uncorrelated
    is not independent."""
    return xi - regression_coef(xi, xj) * xj


def estimate_parent_coefficients(X, predictors, target):
    """Existing covariance-based regression after an order is known."""
    coefs, *_ = np.linalg.lstsq(X[:, predictors], X[:, target], rcond=None)
    return coefs


def discover_order(X):
    """Return a directed causal order for the variables.

    This is the whole problem: covariance alone cannot direct an edge, so the
    procedure here must read something covariance does not see.
    """
    # TODO: the order-discovery procedure we will design.
    pass


def run_causal_discovery(X: np.ndarray) -> np.ndarray:
    """X: (n_samples, n_variables) -> adjacency B with B[i, j] != 0 meaning j -> i."""
    n = X.shape[1]
    K = discover_order(X)

    # Once the order K is known, edge weights are a routine triangular regression:
    B = np.zeros((n, n))
    for rank, target in enumerate(K):
        predictors = K[:rank]                 # everything earlier in the order
        if predictors:
            B[target, predictors] = estimate_parent_coefficients(X, predictors, target)
    return B
```

The final triangular regression is routine once an order is known. The unresolved part is
`discover_order` — the procedure that extracts a directed order from observational data.
