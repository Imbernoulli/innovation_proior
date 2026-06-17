## Research Question

We have training pairs `(x_1, y_1), ..., (x_T, y_T)`, with inputs represented by numeric descriptors and real-valued targets. The immediate goal is a supervised regressor that predicts a new target from a new input under squared error. A useful solution has to satisfy three constraints at once.

First, the response surface may be nonlinear. A straight line in the original coordinates is too rigid when the target changes smoothly but curvedly with the descriptors. Second, the obvious way to make a linear method nonlinear is to build a richer feature vector `phi(x)` and fit a linear model there, but that can make the number of coordinates enormous. Polynomial features grow combinatorially with the input dimension, spline or radial-basis expansions can be much larger still, and an algorithm that forms or inverts a matrix whose size is the feature dimension can become unusable. Third, the fit must be stable. With many correlated descriptors, or more features than examples, ordinary least squares becomes singular or high-variance and can chase noise.

The open design problem is therefore: keep the stability of regularized least squares, let the fitted function be nonlinear in the original input, and make the computation depend on the number of examples rather than on the explicit feature dimension.

## Background

Classical least squares fits a linear predictor by minimizing `sum_t (y_t - w^T x_t)^2`. When the data matrix is `X`, the formal solution is `w = (X^T X)^-1 X^T y`. This is simple and optimal under ideal linear-model assumptions, but it depends on inverting `X^T X`. If the columns are collinear, or if the feature dimension exceeds the number of examples, `X^T X` is singular. Even before exact singularity, small singular values create large coefficient variance.

Ridge regression and Tikhonov regularization address that stability problem by adding a positive diagonal term. The regularized objective is `||y - Xw||^2 + a ||w||^2`, with solution `w = (X^T X + aI)^-1 X^T y` for `a > 0`. The added `aI` lifts the spectrum and shrinks unstable directions. This fixes conditioning, but the model is still linear in the supplied coordinates.

The kernel idea supplies a separate ingredient. If a learning algorithm uses examples only through inner products, then an inner product in a feature space, `phi(u)^T phi(v)`, can be replaced by a function `K(u, v)` that returns that value directly. A valid kernel is symmetric and positive semidefinite on every finite sample, so its Gram matrices behave like inner-product matrices. Polynomial kernels are finite-dimensional examples; spline and radial-basis kernels give access to very large or infinite feature spaces without explicitly enumerating coordinates.

Representer results from the reproducing-kernel Hilbert-space and spline literature give another view of why a finite computation should be possible. For a regularized objective whose empirical loss depends on a function only through its values on the training inputs, any component orthogonal to the span of the training kernel sections changes no fitted value but increases the norm penalty. A minimizer can therefore be represented as a finite sum over training examples.

There is also a probabilistic mirror in Kriging and Gaussian-process regression. A Gaussian prior over functions plus independent Gaussian observation noise gives a posterior mean that is a kernel-weighted combination of the observed targets. That construction supplies a noise-versus-smoothness interpretation for a diagonal regularization constant, but it is historically phrased as probabilistic conditioning rather than as a direct regularized least-squares algorithm.

## Baselines

**Ordinary least squares.** Fit `w` by minimizing squared residuals in the original coordinates. It is computationally simple when `X^T X` is well-conditioned and the relationship is close to linear. Gap: it is rigid and can become singular or high-variance in the correlated, high-feature regime.

**Ridge regression.** Add `a ||w||^2` and solve `(X^T X + aI)w = X^T y`. This stabilizes the linear fit and trades variance for bias. Gap: by itself, it remains linear in the input. Applying it to explicit nonlinear features reintroduces a feature-dimension-sized solve.

**Explicit nonlinear feature regression.** Map each input into `phi(x)` and fit a linear regressor in that space. Gap: the number of parameters and the matrix inverse scale with the feature dimension, which is exactly the quantity that becomes large or infinite for rich nonlinear representations.

**Kernelized support-vector methods.** Maximum-margin classification and support-vector regression already show how kernels avoid explicit feature coordinates. For regression, the epsilon-insensitive version yields a sparse expansion but requires a quadratic program. Gap: sparsity is useful, but the squared-error estimator still lacks the transparent one-shot linear solve in the nonlinear feature setting.

**Kriging / Gaussian-process posterior mean.** This gives a kernel-weighted predictor from a covariance model and a noise model. Gap: the construction lives in a Bayesian/geostatistical vocabulary, and the direct connection to a regularized squared-loss fit is not yet the algorithmic object being filled into a standard supervised-regression scaffold.

## Evaluation Settings

A natural pre-method benchmark for nonlinear regression is the Boston Housing dataset used in the support-vector regression literature: 506 cases, 12 continuous variables and one binary variable, with the target the median house price in thousands of dollars. A standard protocol partitions the data into training, validation, and test subsets, repeats random splits, selects smoothing and kernel parameters on validation data, and reports average squared error on held-out test data.

For the scaling-law scaffold, the analogous setting is a train/test split over model-run descriptors. The training inputs contain numeric quantities that can span orders of magnitude and categorical group labels; the target is a real-valued metric. A fair first-stage regressor may use feature preprocessing learned from training inputs and may tune fixed hyperparameters without looking at the held-out targets. The held-out score should measure real extrapolation or interpolation behavior, not reuse information from the test labels.

The common evaluation principle is the same in both settings: select any regularization or bandwidth parameters without test-target access, fit only on training data, and score real-valued predictions under squared-error-derived metrics.

## Code Framework

The scaffold supplies a supervised-regression interface. The unresolved slots are the learned representation, the fitting rule, and the prediction rule. It is acceptable for the final implementation to call a trusted library primitive for the numerical fit, as long as the surrounding feature preparation is fitted only on training inputs.

```python
import numpy as np


class ScalingLawModel:
    """Generic scaffold shape: fit on descriptors and predict real targets."""

    def __init__(self, benchmark_name, numeric_names=None, categorical_names=None):
        self.benchmark_name = benchmark_name
        self.numeric_names = numeric_names or []
        self.categorical_names = categorical_names or []

    def fit(self, X_num, X_cat, y):
        # TODO: learn a representation and fit a regularized nonlinear regressor.
        raise NotImplementedError

    def predict(self, X_num, X_cat):
        # TODO: transform new descriptors consistently and return real predictions.
        raise NotImplementedError
```
