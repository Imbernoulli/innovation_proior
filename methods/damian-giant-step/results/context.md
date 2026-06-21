# Context: learning low-dimensional polynomial structure with a two-layer net

## Research question

Neural networks trained by gradient descent are often compared against the kernel methods that
arise from linearizing the same networks at initialization. A clean theoretical setting for
studying this comparison is Gaussian regression with a hidden low-dimensional structure:

`x ~ N(0, I_d)`, and `f*(x) = g(<u_1,x>, ..., <u_r,x>)` with `r << d`.

The target is a degree-`p` polynomial, normalized in `L^2(N(0,I_d))`, and the observations are
`y_i = f*(x_i) + epsilon_i`. The ambient distribution itself is isotropic, so the relevant
subspace `S* = span(u_1,...,u_r)` shows up only through correlations between labels and
features. The learning problem is to train a two-layer ReLU network so that its first-layer
features depend on `S*`, then fit the polynomial head. Kernel and random-feature methods for
degree-`p` structure scale like powers of the ambient dimension; the question is how training
the network on this target relates to the hidden dimension `r`.

## Background

The neural tangent kernel and lazy-training analyses explain overparameterized training by
showing that parameters stay close to initialization, so the model behaves like a fixed feature
map, with strong optimization guarantees. Work by Ghorbani, Mei, Misiakiewicz, and Montanari
studies high-dimensional polynomial settings: fixed random features or tangent features stay tied
to random directions, while a fully trained network can align with useful eigendirections.

For Gaussian inputs, Hermite polynomials are the natural coordinates. If `He_k` denotes the
probabilists' Hermite polynomial, then
`E[He_j(z) He_k(z)] = k! delta_{jk}` for `z~N(0,1)`, and contractions against Gaussian
polynomials can be computed through Stein identities. For a polynomial `f*`, the tensors
`C_k = E[grad^k f*(x)]` are its Hermite/Taylor coefficients in Gaussian space. In particular,
`C_2 = H = E[grad^2 f*(x)]` is the average Hessian.

A low-dimensional polynomial satisfies `span(grad^2 f*(x)) subseteq S*` for every `x`, hence
`span(H) subseteq S*`. A standard nondegeneracy condition is that the average Hessian have rank
exactly `r`, so `span(H)=S*`, making `S*` recoverable from second-order information.

## Baselines

**Random features / NTK regression.** Freeze the first-layer or tangent features and fit a
linear readout. This is convex and well understood, with features fixed at their random
initialization. For polynomial targets, the required sample and width scales are governed by the
ambient dimension.

**Generic polynomial regression.** Fit all degree-`p` monomials in `d` variables. The feature
dimension is `Theta(d^p)`, giving a brute-force benchmark that uses all of `d`.

**Specialized low-rank polynomial algorithms.** Chen and Meka give a filtered-PCA plus
geodesic-optimization approach for learning polynomials of few relevant Gaussian dimensions, a
tailored spectral procedure showing that the statistical structure is exploitable.

**Online single-index SGD theory.** Analyses of online SGD for planted-direction problems
classify difficulty using the first nonzero local signal in the population loss, explaining why
some links are harder than others.

## Evaluation settings

The synthetic source task uses isotropic Gaussian inputs and noisy labels from a normalized
degree-`p` polynomial with few relevant directions. Diagnostic examples include single-index
Hermite-polynomial mixtures such as a quadratic term plus a higher-degree term.

The model is a two-layer ReLU network
`f_theta(x) = a^T sigma(Wx+b)`, with width `m`, first-layer rows `w_j`, and scalar readout
weights `a_j`. Reference comparisons are random-feature regression and the linearized NTK
model, evaluated by population or test loss. Transfer is evaluated by keeping the learned
representation from a source polynomial and retraining only the head on a new polynomial that
depends on the same hidden subspace. Reported diagnostics include test loss, whether the learned
features span the relevant subspace, and the target sample complexity when reusing the
representation.

## Code framework

The recipe plugs into a fixed harness: the model, data generator, mini-batch loop, and metrics
are given. The strategy controls only initialization, the optimizer exposed to the driver, the
per-batch update, and an optional full-dataset finalization step.

```python
import torch
import torch.nn as nn


class TwoLayerMLP(nn.Module):
    def __init__(self, d, W):
        super().__init__()
        self.fc1 = nn.Linear(d, W)
        self.fc2 = nn.Linear(W, 1)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


class Strategy:
    def __init__(self, config):
        self.config = config

    def init_two_layer(self, net, config):
        # TODO: initialize the first-layer weights, biases, and readout weights.
        pass

    def make_optimizer(self, net, config):
        # TODO: expose the optimizer used by the fixed driver.
        pass

    def training_step(self, net, opt, x, y, step, config):
        # TODO: define what one mini-batch update does.
        pass

    def finalize(self, net, x_train, y_train, config):
        # TODO: optional full-dataset stage after the driver loop.
        pass


def build_strategy(config):
    return Strategy(config)
```
