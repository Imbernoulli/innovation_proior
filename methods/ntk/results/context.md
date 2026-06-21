# Context

## Research question

Wide artificial neural networks trained by gradient descent generalize
remarkably well, and they do so in a regime that ought to be hostile to it: the
networks are heavily overparametrized, and the loss, viewed as a function of the
parameters, is highly non-convex, riddled with saddle points. Two facts sharpen
the puzzle. First, a sufficiently large network can be trained to fit completely
random labels, yet the same architecture trained on real labels generalizes
(Zhang et al., 2017) — so capacity alone explains neither failure to fit nor
success at generalizing. Second, classical kernel methods exhibit the same
behavior (Belkin et al., 2018), which hints that the right way to understand a
trained network is not through its weights but through the *function* it
computes, in the same language one uses for kernels.

The core question: can the evolution of the network function during training be
characterized in function space — where the cost is convex — rather than in the
non-convex parameter space?

## Background

**Networks as Gaussian processes at initialization.** It has long been known
(Neal, 1996, for one hidden layer; extended to deep networks by Lee et al.,
2018, and Matthews et al., 2018) that as the hidden widths of a fully-connected
network tend to infinity, the network function at initialization converges in
law to a centered Gaussian process. With iid parameters and a per-layer
1/sqrt(width) scaling of the pre-activations, the covariance is given by a
layer-wise recursion. Writing Sigma^(L)(x, x') for the covariance of the
depth-L pre-activations,

    Sigma^(1)(x, x') = (1/n0) x^T x' + beta^2,
    Sigma^(L+1)(x, x') = E_{f ~ N(0, Sigma^(L))}[ sigma(f(x)) sigma(f(x')) ] + beta^2,

where the expectation is over a centered Gaussian f with covariance Sigma^(L)
and sigma is the nonlinearity. The mechanism is a central-limit effect: each
pre-activation is a 1/sqrt(width)-weighted sum of many iid contributions from
the previous layer, so it becomes Gaussian, and the law of large numbers turns
the empirical second moment (1/n) sum_i sigma(a_i(x)) sigma(a_i(x')) into the
Gaussian expectation above. This connects networks at initialization to kernels,
and infinite-width kernels of this form have been used directly in Bayesian
inference and SVMs with results comparable to trained networks (Cho & Saul,
2009; Lee et al., 2018).

**The per-layer scaling and its consequence.** The 1/sqrt(width) factor on each
pre-activation is what makes the large-width limit well-defined. A side effect,
visible by the chain rule, is that the gradient of the network output with
respect to any single weight in a wide layer carries a 1/sqrt(width) factor — so
each individual parameter moves very little under gradient descent when the
layer is wide. The collective effect of many parameters can still be large: a
vanishing per-parameter motion coexists with a finite aggregate effect.

**Random-feature approximation of a kernel.** Rahimi & Recht (2007) approximate
a kernel K by sampling P random functions f^(p) with second moment
E[f^(p)(x) f^(p)(x')] = K(x, x') and forming the *linear* parametrization
f_theta = (1/sqrt(P)) sum_p theta_p f^(p). Because this is linear in theta, its
"feature map" partial_{theta_p} f = (1/sqrt(P)) f^(p) does not depend on theta;
gradient descent on theta is then *exactly* gradient descent in function space
against the empirical kernel (1/P) sum_p f^(p) ⊗ f^(p), which tends to K as
P -> infinity.

**Kernel methods machinery.** Once dynamics live in function space, the relevant
tools are standard. Kernel PCA (Schölkopf et al., 1998) diagonalizes a kernel's
integral operator on the data; kernel ridge regression (Shawe-Taylor &
Cristianini, 2004) gives the minimum-norm interpolant as the ridge tends to
zero, equal to the posterior mean of a Gaussian-process prior. A positive-
definite kernel makes the associated function-space cost strictly decreasing
away from the optimum.

**Positive-definiteness on the sphere.** Two classical results let one certify
positive-definiteness of dot-product kernels. Daniely et al. (2016) give the
"dual" of a function under a Gaussian: if mu = sum_i a_i h_i in Hermite
polynomials, then E_{(X,Y) ~ N(0, [[1, rho],[rho, 1]])}[mu(X) mu(Y)] =
sum_i a_i^2 rho^i. Gneiting (2013), generalizing Schoenberg, states that a
dot-product kernel f(x^T x') on spheres is strictly positive definite across
all dimensions exactly when its power-series coefficients are positive for
infinitely many even and infinitely many odd powers. High-dimensional data
motivates the sphere: data points often share roughly the same norm, so the dot
product carries the signal.

## Baselines

- **Infinite-width Gaussian process at initialization (NNGP; Neal, 1996; Lee
  et al., 2018; Matthews et al., 2018).** Core idea: the prior over functions
  induced by random parameters is a GP with covariance Sigma^(L) above; one can
  do Bayesian inference or ridge regression directly with this kernel.
  Math/algorithm: the layer-wise Sigma recursion, then GP posterior mean
  Sigma(x*, X) Sigma(X, X)^{-1} y.

- **Random-feature kernel descent (Rahimi & Recht, 2007).** Core idea: a linear
  combination of fixed random features makes gradient descent equivalent to
  kernel gradient descent. Math/algorithm: parameters follow
  theta_dot_p = -(1/sqrt(P)) <d, f^(p)>; the function follows kernel descent
  against the (constant) tangent kernel (1/P) sum_p f^(p) ⊗ f^(p).

- **Deep dot-product / arc-cosine kernels (Cho & Saul, 2009).** Core idea:
  compose Gaussian-expectation kernels to build deep kernels with closed forms
  for nonlinearities like ReLU. Math/algorithm: the arc-cosine kernels give
  E[sigma(X) sigma(X')] and E[sigma'(X) sigma'(X')] for jointly Gaussian (X, X')
  in closed form.

- **Mean-field analysis of two-layer networks (Mei et al., 2018).** Core idea:
  track the distribution of neurons in a two-layer network as a measure evolving
  under a PDE in the large-width limit.

## Evaluation settings

- A synthetic low-dimensional set: inputs on the unit circle in R^2 (a stand-in
  for high-dimensional data, whose centered points share roughly equal norm),
  with a smooth scalar target such as a product of coordinates.
- MNIST handwritten digits (28x28, so input dimension 784), used to inspect how
  the function trained by gradient descent behaves across the dataset.
- Architecture knobs that would be varied: depth L, common hidden width n, the
  nonlinearity (ReLU), the bias multiplier beta, the learning rate, and the
  number of random initializations used to estimate distributions.
- Quantities one would measure: how a finite-width network's training dynamics
  behave as width grows; how much that behavior varies across random
  initializations; and how the network function moves through function space
  over the course of training.

## Code framework

The primitives that already exist: numpy/torch tensors, automatic
differentiation, Gaussian-expectation evaluation for a nonlinearity, and a
linear solver. The slots the method will fill:

```python
import numpy as np
import torch


def activation_dual(cov_xx, cov_xpxp, cov_xxp):
    """Gaussian expectations of the nonlinearity and its derivative for a 2x2
    covariance: E[sigma(X) sigma(X')] and E[sigma'(X) sigma'(X')]."""
    # TODO: closed form (or quadrature) for the chosen nonlinearity
    pass


def layer_kernel_recursion(X, Xp, depth, beta):
    """Propagate a pair of inputs through the depth layers, maintaining whatever
    per-layer kernel quantities the dynamics turn out to need."""
    # TODO: the recursion the method will derive
    pass


def limit_object(X, Xp, depth, beta):
    """Whatever large-width-limit object the analysis below turns out to need to
    describe a wide network's training dynamics."""
    # TODO
    pass


class Net(torch.nn.Module):
    """A finite-width fully-connected net. The per-layer scaling is the one knob
    known to matter for a well-defined wide-network limit."""
    def __init__(self, n0, width, depth, beta):
        super().__init__()
        # TODO: layers + the 1/sqrt(width) pre-activation scaling
        pass

    def forward(self, x):
        # TODO
        pass


def finite_net_probe(net, X, Xp):
    """A finite-net counterpart of limit_object, computed from the network's
    automatic-differentiation data, to compare against the large-width limit."""
    # TODO
    pass


def kernel_predict(K_train, K_test, y, ridge=0.0):
    """Generic kernel-regression mean predictor."""
    pass
```
