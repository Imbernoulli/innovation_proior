# Context: learning low-dimensional polynomial structure with a two-layer net

## Research question

Neural networks trained by gradient descent often beat the kernel methods that arise from
linearizing the same networks at initialization. A clean theoretical setting for isolating this
gap is Gaussian regression with a hidden low-dimensional structure:

`x ~ N(0, I_d)`, and `f*(x) = g(<u_1,x>, ..., <u_r,x>)` with `r << d`.

The target is a degree-`p` polynomial, normalized in `L^2(N(0,I_d))`, and the observations are
`y_i = f*(x_i) + epsilon_i`. The ambient distribution itself is isotropic, so the only way to
find the relevant subspace `S* = span(u_1,...,u_r)` is through correlations between labels and
features. The learning problem is to train a fixed two-layer ReLU network so that its learned
first-layer features depend on `S*`, then fit the polynomial head with far fewer samples than a
generic `d`-dimensional polynomial regression would require.

The desired separation is quantitative. Kernel and random-feature methods for degree-`p`
structure scale like powers of the ambient dimension. A successful gradient-training recipe
should expose where feature movement enters the gradient dynamics and why this movement reduces
the relevant dimension from `d` to `r`.

## Background

The neural tangent kernel and lazy-training analyses explain overparameterized training by
showing that parameters stay close to initialization, so the model behaves like a fixed feature
map. This gives strong optimization guarantees, but it also means the representation is not
adapted to the target. Work by Ghorbani, Mei, Misiakiewicz, and Montanari shows this limitation
even in simple high-dimensional polynomial settings: fixed random features or tangent features
remain tied to random directions, while a fully trained network can align with useful
eigendirections.

For Gaussian inputs, Hermite polynomials are the natural coordinates. If `He_k` denotes the
probabilists' Hermite polynomial, then
`E[He_j(z) He_k(z)] = k! delta_{jk}` for `z~N(0,1)`, and contractions against Gaussian
polynomials can be computed through Stein identities. For a polynomial `f*`, the tensors
`C_k = E[grad^k f*(x)]` are its Hermite/Taylor coefficients in Gaussian space. In particular,
`C_2 = H = E[grad^2 f*(x)]` is the average Hessian.

A low-dimensional polynomial satisfies `span(grad^2 f*(x)) subseteq S*` for every `x`, hence
`span(H) subseteq S*`. To make recovery possible from second-order information, one needs a
nondegeneracy condition: the average Hessian should have rank exactly `r`, so `span(H)=S*`.
Without some such condition, low-dimensionality alone does not make all degree-`p` polynomial
classes easy.

## Baselines

**Random features / NTK regression.** Freeze the first-layer or tangent features and fit a
linear readout. This is convex and well understood, but it cannot rotate its features toward
`S*`. For polynomial targets, the required sample and width scales are governed by the ambient
dimension rather than the hidden dimension.

**Generic polynomial regression.** Fit all degree-`p` monomials in `d` variables. This is the
right brute-force benchmark: the feature dimension is `Theta(d^p)`, so it ignores the promise
that the polynomial depends only on `r` directions.

**Specialized low-rank polynomial algorithms.** Chen and Meka give a filtered-PCA plus
geodesic-optimization approach for learning polynomials of few relevant Gaussian dimensions.
This shows that the statistical structure is exploitable, but the algorithm is a tailored
spectral procedure rather than ordinary gradient descent on the network whose representation is
being studied.

**Online single-index SGD theory.** Analyses of online SGD for planted-direction problems
classify difficulty using the first nonzero local signal in the population loss. They explain why
some links are harder than others, but they are not finite-sample analyses of a fixed
multi-neuron ReLU network learning a reusable representation.

## Evaluation settings

The synthetic source task uses isotropic Gaussian inputs and noisy labels from a normalized
degree-`p` polynomial with few relevant directions. Diagnostic examples include single-index
Hermite-polynomial mixtures such as a quadratic term plus a higher-degree term, where a fixed
kernel can learn the low-degree component yet miss the hidden representation needed for the
whole target.

The model is a two-layer ReLU network
`f_theta(x) = a^T sigma(Wx+b)`, with width `m`, first-layer rows `w_j`, and scalar readout
weights `a_j`. Reference comparisons are random-feature regression and the linearized NTK
model, evaluated by population or test loss. Transfer is evaluated by keeping the learned
representation from a source polynomial and retraining only the head on a new polynomial that
depends on the same hidden subspace.

The key diagnostic question is not just whether the loss goes down. It is whether the learned
features span the relevant subspace and whether a new head can reuse those features with target
sample complexity independent of `d`.

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
        # TODO: initialize the paired first-layer rows, readout signs, and biases.
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
