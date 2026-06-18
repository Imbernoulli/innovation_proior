# Context: feature learning for low-dimensional targets with two-layer networks

## Research question

A regression target depends on a high-dimensional input only through a few hidden directions. The
data are i.i.d. pairs `(x, y)` with

```text
y = g(U x) + xi,
```

where `x in R^d`, `xi` is mean-zero subGaussian noise, and `k << d`. In the source scaling,
there are orthonormal directions `u_1, ..., u_k` and the matrix `U` has rows `u_i^T / sqrt(k)`,
so the link sees the normalized projections `<u_i, x> / sqrt(k)`. The unknowns are both the
subspace spanned by the `u_i` and the link `g`.

Counting parameters suggests that, for fixed `k`, the relevant subspace has only order `d`
degrees of freedom. On isotropic data this makes `n` proportional to `d` the natural statistical
floor. The hard question is whether a standard first-order neural-network training procedure can
reach that floor for broad links and broad covariance structure, rather than paying a price tied to
the algebraic order of the link.

## Statistical structure

The input has zero mean and covariance `Sigma`, with subGaussian control on both `||x||` and
`||U x||`. The scale parameters that matter are

```text
c_x = sqrt(tr(Sigma)),
r_x = || Sigma^{1/2} U^T ||_F,
d_eff = c_x^2 / r_x^2.
```

Because `U` includes the `1/sqrt(k)` normalization, isotropic data gives `r_x^2 = 1` and
`d_eff = d`. If the covariance concentrates variance in directions aligned with the target
subspace, `d_eff` can be much smaller than `d`; if the high-variance directions are unrelated to
the target, it need not improve.

The link is locally Lipschitz on the range where the observed projected data live, with a scale
chosen so that the response has constant-order second moment. This includes polynomially growing
links on the relevant observed ball, not only globally bounded links.

## Known obstacles

For single-index models, online stochastic gradient descent from a random start is governed by the
information exponent: the first nonzero term in the expansion of the population correlation around
the uninformative equator. If that exponent is `s`, the gradient signal near initialization scales
like a small overlap raised to the `s - 1` power. The resulting search phase needs `n ~ d` for
`s = 1`, `n ~ d log d` for `s = 2`, and polynomially larger sample sizes for `s >= 3`.

For multi-index models, the analogous obstruction is leap complexity. Training can proceed by a
saddle-to-saddle sequence in which some coordinates become visible only after earlier coordinates
have been learned. Existing analyses of ordinary two-layer training therefore give costs governed
by the link's hierarchy, not just by the dimension of the hidden subspace.

This creates the central gap: the target is statistically low-dimensional, but the usual gradient
dynamics can spend most of its effort escaping a nearly flat, uninformative initialization.

## Existing baselines

Plain two-layer SGD trains the first layer directly on squared or correlation loss. It is a
homogeneous first-order procedure, but the known rates follow the information or leap exponent and
can be far above the `n ~ d` floor.

Layer-wise or two-stage procedures first try to recover first-layer directions and then fit the
output layer by a convex solve, such as ridge regression on learned features. These methods clarify
the feature-learning step, but they are staged and still rely on the first-layer search signal being
strong enough.

Random-features or fixed-grid methods freeze the first layer and solve only the output weights.
They avoid non-convex first-layer optimization, but without adaptive movement of the features they
do not exploit the hidden low-dimensional structure efficiently.

Convex infinite-width formulations show that, after lifting a wide two-layer network to a
distribution over neurons, the prediction is linear in the distribution and the loss becomes convex
as a functional of that distribution. The missing piece is a general finite-sample, finite-training
guarantee for arbitrary multi-index links under general covariance.

## Code frame

The implementation frame is a two-layer regression harness. It provides samples, a fixed second
layer, a predictor, the squared loss, and the per-batch gradient with respect to first-layer
weights. The design slot is the first-layer update rule: how to scale the gradient, whether to add
regularization drift, whether to add stochasticity, and whether to impose a constraint on the
neuron weights.

```python
import torch


def predict(X, W, a, phi):
    return phi(X @ W.T) @ a


def first_layer_gradient(X, y, W, a, phi, phiprime):
    n = X.shape[0]
    residual = (predict(X, W, a, phi) - y).reshape(1, -1)
    backprop = phiprime(W @ X.T) * a.reshape(-1, 1)
    return (backprop * residual) @ X / n


def update_step(W, X, y, a, phi, phiprime, lr, hparams):
    grad = first_layer_gradient(X, y, W, a, phi, phiprime)
    # Open slot: choose the first-layer training dynamics.
    raise NotImplementedError
```
