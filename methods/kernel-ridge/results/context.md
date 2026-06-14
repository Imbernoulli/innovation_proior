## Research question

We are given a training sample `(x_1, y_1), ..., (x_T, y_T)` with `x_t` in `R^n` and real
targets `y_t`, and we must build a regressor that, on a *new* input `x`, predicts a real
value with small average squared error `L = (1/l) sum (y_t - yhat_t)^2` on held-out points.
The honest difficulty is twofold. First, the relationship is generally **nonlinear**: a
straight-line (linear-in-`x`) predictor is too rigid for the kind of curved response surfaces
that arise in practice, so we want a hypothesis class rich enough to bend. Second, the obvious
way to get nonlinearity — map each input into a much richer set of derived features
`phi(x)` (products, powers, basis functions) and fit a linear model *there* — explodes the
number of parameters: a degree-`d` polynomial map of an `n`-dimensional input has on the order
of `n^d` coordinates, so for nontrivial `n` and `d` the feature space has thousands to
millions of dimensions, and any algorithm that explicitly forms, stores, or inverts a matrix
the size of (feature-dimension x feature-dimension) becomes computationally impossible. The
field calls this the "curse of dimensionality" of feature-space regression. On top of that,
with many features and few examples the fit is unstable: the design matrix becomes
ill-conditioned or singular, the coefficients blow up, and the model overfits the noise and
extrapolates wildly. So the goal is a regressor that (1) can represent a flexible nonlinear
function, (2) is computable even when the natural feature space is enormous or infinite, and
(3) is regularized so the fit is stable and generalizes rather than memorizing. Each existing
piece below supplies a subset; none, on its own, supplies all three at once for the
squared-loss regression case.

## Background

By the late 1990s the relevant machinery is largely in place, scattered across statistics,
approximation theory, and the young kernel-machine literature.

**Linear least squares and its conditioning.** Fitting `y = w . x` by minimizing
`sum_t (y_t - w . x_t)^2` has the closed form `w = (X'X)^{-1} X'y` (`X` the `T x n` data
matrix). With `X = U D V'` its singular value decomposition, the OLS estimator's covariance
scales as `sum_j d_j^{-2} v_j v_j'`, so along directions with a small singular value `d_j` the
coefficients have enormous variance, and when `d_j = 0` — exact collinearity, or more features
than examples — `X'X` is singular and the estimator is undefined. This is a concrete,
pre-existing, *observed* failure mode of the plain fit, not a hypothetical.

**Regularization to restore stability.** Two lineages converge on the same fix. In numerical
analysis, Tikhonov regularization (Tikhonov 1963) of an ill-posed linear inverse problem adds
a penalty on the solution norm. In statistics, ridge regression (Hoerl & Kennard 1970) adds a
positive multiple of the identity to `X'X` before inverting, equivalently penalizing the
coefficient norm: `w(lambda) = (X'X + lambda I)^{-1} X'y = argmin ||y - Xw||^2 +
lambda ||w||^2`. Because `X'X` is positive semidefinite and `lambda I` is positive definite,
`X'X + lambda I` is positive definite and hence always invertible; the penalty shrinks the
coefficients along the unstable small-singular-value directions (each is scaled by
`d_j^2/(d_j^2 + lambda)` in `(0,1]`), trading a little bias for a large reduction in variance.
This is the standard stabilizer for the conditioning problem above. It is, however, a model
that is **linear in its inputs**.

**Inner-product-only algorithms and the kernel idea.** A separate, older observation: certain
learning algorithms touch the data *only* through inner products `x_s . x_t`, never through the
raw coordinates individually. Aizerman, Braverman & Rozonoer's potential-function method
(1964) already exploited this — it updates and predicts using scalar products in a
"rectifying" (lifted) space. The consequence, sharpened over the following decades, is that if
an algorithm depends on the data only via inner products, one may *replace* each inner product
`phi(x_s) . phi(x_t)` in a feature space by a function `K(x_s, x_t)` that returns that same
number directly, and never compute `phi` at all. A function `K` is a valid inner product in
*some* feature space exactly when it is symmetric and positive semidefinite (Mercer's
theorem): every Gram matrix `[K(x_i, x_j)]` it produces must be PSD. Examples already in
circulation include the polynomial kernel `K(x, y) = (x . y + 1)^d`, whose feature map is all
monomials up to degree `d` (so the `d=2`, `n=2` map is
`(x1^2, x2^2, sqrt(2) x1, sqrt(2) x2, sqrt(2) x1 x2, 1)` and
`phi(x) . phi(y) = (x . y + 1)^2`), and the Gaussian / radial basis function kernel
`K(x, y) = exp(-gamma ||x - y||^2)`, whose feature map is *infinite*-dimensional and whose
single width parameter `gamma` controls how local the kernel is — large `gamma` makes each
training point influence only its near neighbors (a wiggly, high-capacity fit), small `gamma`
makes the influence broad and the fit smooth. Computing `K` costs the same as one inner
product no matter how large `phi` is, which is precisely the lever against the curse of
dimensionality.

**Finiteness of regularized solutions in function space.** A regularized fit posed directly
over a (possibly infinite-dimensional) reproducing-kernel Hilbert space — minimize an
empirical cost depending on `f` only through its values `f(x_1), ..., f(x_T)`, plus a
nondecreasing penalty on `||f||` — has minimizers that lie in the finite span of the kernel
evaluated at the training points (Kimeldorf & Wahba 1971; Wahba's spline work uses this
same finite-span logic). Decompose `f = f_parallel + f_perp` into the part in
`span{K(., x_i)}` and the part orthogonal to it; the reproducing property
`<f, K(., x_i)> = f(x_i)` makes every training value depend only on `f_parallel`, while
`||f||^2 = ||f_parallel||^2 + ||f_perp||^2`, so any orthogonal component only inflates the
penalty without changing the fit and is driven to zero. The upshot, knowable before any
specific regression algorithm is chosen: an infinite-dimensional regularized fit collapses to
a search over `T` real coefficients.

**The Bayesian / geostatistics mirror.** A related predictive construction appears in spatial
statistics as Kriging and, more generally, as Gaussian-process regression: place a zero-mean
Gaussian prior on the unknown function with covariance given by a kernel, assume Gaussian
observation noise, and compute a posterior mean at a new point from the joint Gaussian
covariances. This provides a probabilistic vocabulary — prior variance, noise variance — for
thinking about smoothing.

## Baselines

These are the prior methods a new regressor would be compared with and would build on.

**Ordinary least squares.** Minimize `sum_t (y_t - w . x_t)^2`; closed form
`w = (X'X)^{-1} X'y`, prediction `w . x`. Simple, unbiased, optimal among linear unbiased
estimators. **Gap:** linear in `x` only; and it breaks down exactly when the inputs are
collinear or outnumber the examples — `X'X` singular, coefficients of unbounded variance —
which is the common regime once one tries to enrich the inputs.

**Ridge regression (Hoerl & Kennard 1970) / Tikhonov regularization.** Minimize
`||y - Xw||^2 + lambda ||w||^2`, closed form `w = (X'X + lambda I)^{-1} X'y`. The added
`lambda I` makes the system positive definite and well conditioned, shrinks the high-variance
directions, and lowers total mean squared error relative to OLS for an appropriate `lambda`.
**Gap:** still a model linear in its inputs. The natural route to nonlinearity is to apply
ridge in a feature space `phi(x)` instead of `x`, but then the matrix to invert is
(feature-dimension x feature-dimension); for a rich `phi` this is enormous, and for an
infinite-dimensional `phi` (the Gaussian-kernel feature map) it does not exist as a finite
object. The method stalls at the point where the feature space is large.

**Explicit nonlinear regression in feature space (e.g. Drucker et al. 1997).** Map inputs
into a high-dimensional feature space and fit a regressor there to obtain nonlinearity in the
original space. **Gap:** computations are carried out *in* the high-dimensional space, so for
nontrivial problems the number of parameters is huge — exactly the curse-of-dimensionality
cost the kernel idea exists to avoid.

**The kernel trick as used in support-vector classifiers (Boser, Guyon & Vapnik 1992;
Aizerman et al. 1964).** Wherever a learning algorithm depends on the data only through inner
products, replace those inner products by a kernel `K`, obtaining a nonlinear method at the
cost of a linear one and never forming `phi`. Support-vector machines use this with a
margin-maximizing (hinge-type) objective for *classification*, solved as a quadratic program.
**Gap:** the trick had been carried furthest for the classification / margin loss and its QP
machinery. For the plain squared-error *regression* fit — the simplest classical estimator —
there is still a practical gap between the stable primal formula and a nonlinear version that
avoids explicit high-dimensional feature coordinates. A dual statement of a closely related
constrained regression problem existed in the support-vector literature, but the regression
lineage did not yet have the same transparent closed-form predictor available as a standard
tool.

**Support-vector regression (Vapnik).** The same kernel model form
`f(x) = sum_t coeff_t K(x_t, x)` but fit with an epsilon-insensitive loss (errors smaller than
`epsilon` are ignored). This yields a *sparse* solution — only the points outside the
epsilon-tube get nonzero coefficients — at the cost of solving a quadratic program rather than
a linear system. **Gap:** no closed form; the QP is slower for medium-sized problems, and the
loss is a design choice that buys sparsity but discards the analytic tractability of squared
error.

**Finite-span regularization results (Kimeldorf & Wahba 1971; Wahba 1990/1997).** Any
minimizer of a kernelized regularized objective lies in `span{K(., x_i)}`, i.e.
`f = sum_i alpha_i K(., x_i)`. **Gap (really an enabling tool):** the theorem guarantees the
finite form exists but, by itself, does not hand over the coefficients for any particular loss
— one still has to substitute the form into the chosen objective and solve.

**Kriging / Gaussian-process regression.** From the Bayesian / geostatistics side, the
posterior-mean predictor is a kernel-weighted linear combination of the training targets.
**Gap:** stated in probabilistic terms (priors, covariances) and historically solved with
"clever matrix manipulations" rather than as a transparent regularized optimization; the
connection to a penalized least-squares fit is not made explicit in that lineage.

## Evaluation settings

The natural pre-existing yardsticks for a nonlinear regressor:

- **The Boston Housing dataset** (UCI repository; studied by Breiman 1996 and others), 506
  cases, 12 continuous and 1 binary attribute per case describing locational, economic, and
  structural features, target the median house price in thousands of dollars (between 5 and
  50). A standard benchmark for nonlinear regression methods, commonly split into a training
  set, a validation set used to select hyperparameters (kernel/penalty), and a held-out test
  set, with the experiment repeated over many random partitions and the per-test errors
  averaged.
- **Metric:** average squared error on the held-out test cases (and the variance of that
  error across repeats). Hyperparameters such as the polynomial degree, the spline order, and
  the penalty strength are selected on the validation split, never on the test split.
- **Comparators run on the identical splits:** the same kernel families (polynomial, spline)
  fed to a support-vector regressor on exactly the same training files, so differences reflect
  the estimator and not the data partition.

## Code framework

The regressor plugs into a standard supervised-regression harness: an object with `fit(X, y)`
that consumes the training inputs and targets and stores whatever it needs, and `predict(X)`
that returns a real value per row. Nothing about how the fit is represented or computed is
fixed yet, so the substrate is only the generic regression interface, a pre-existing
pairwise-similarity primitive, and a stable solver for shifted symmetric systems. The empty
slots are the fitting rule and the prediction rule.

```python
import numpy as np


def pairwise_similarity(A, B, params=None):
    """Return a pre-existing symmetric positive-semidefinite similarity matrix."""
    # TODO: the concrete similarity supplied by the experiment.
    pass


def solve_regularized(M, b, lam):
    """A pre-existing stable solver for a regularized symmetric system:
    returns the solution of (M + lam * I) z = b (positive definite for lam > 0)."""
    n = M.shape[0]
    return np.linalg.solve(M + lam * np.eye(n), b)


class Regressor:
    """Generic supervised regressor. fit() learns from (X, y); predict() scores new X.
    How the learned function is represented and computed is what we will design."""

    def __init__(self, lam=1.0, similarity_params=None):
        self.lam = lam
        self.similarity_params = similarity_params

    def fit(self, X, y):
        # TODO: the fitting rule we will design -- decide what to learn from
        #       (X, y) and how to compute it, then store it on self.
        pass

    def predict(self, X):
        # TODO: the prediction rule we will design -- map new inputs X to
        #       real-valued outputs using what fit() stored.
        pass
```

The harness supplies inputs, targets, a pairwise-similarity primitive, and a shifted-system
solver; the representation of the fit and the form of the prediction are the slots to fill.
