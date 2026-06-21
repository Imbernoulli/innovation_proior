## Research question

What should the pointwise nonlinearity in a deep network be? The activation is a necessary
architectural choice — without one, a stack of linear layers collapses to a single linear map. The
dominant choice, the ReLU `max(x,0)`, makes a hard, sign-based gating decision: it multiplies the
input by exactly 0 or 1 depending only on whether `x>0`. The question is what pointwise map `f(x)`
to use across vision, language, and speech tasks.

## Background

The lineage of pointwise activations runs from binary threshold units (McCulloch–Pitts, Hopfield),
through sigmoids — which smoothed the hard threshold into a differentiable "firing rate" and made
backpropagation possible — to the ReLU (Nair & Hinton 2010). Saturating sigmoids have a gradient near
zero away from the origin, so error signals attenuate through many layers. ReLU `max(x,0)` has
derivative exactly 1 on the positive half-line and became the default; it is cheap and induces sparsity.
ReLU is non-differentiable at the origin and is identically zero for all negative inputs. The ELU
(Clevert et al. 2015) takes small negative values, `x` for `x≥0` and `α(eˣ−1)` for `x<0`, which can
speed training; it is monotonic and convex.

Two facts about how modern networks are trained are relevant here. First, preactivations into a
unit tend to be approximately normally distributed, an effect strengthened by Batch Normalization,
which explicitly standardizes them. Second, networks are routinely regularized with stochastic
multiplicative masks applied per-unit: dropout multiplies a unit's value by a Bernoulli(0/1) mask
chosen independently of the input; zoneout multiplies by 1 (carries the previous value) stochastically;
adaptive dropout (Standout, Ba & Frey 2013) makes the keep-probability depend on the input through a
learned logistic gate. The activation and the stochastic regularizer are two separate design decisions.

## Baselines

- **ReLU** (Nair & Hinton 2010): `f(x)=max(x,0)=x·1(x>0)`. Hard sign gate; derivative 1 for `x>0`,
  0 for `x<0`, undefined at 0. Cheap, sparse, good positive-half gradient flow.
- **ELU** (Clevert et al. 2015): `x` for `x≥0`, `α(eˣ−1)` for `x<0`. Allows small negative outputs,
  pushing mean activations toward zero and sometimes speeding training. Monotonic and convex, linear in
  the positive domain, with a hyperparameter `α`.
- **Sigmoid / tanh**: smooth and bounded, with a probabilistic "firing-rate" reading; saturate on
  both sides.
- **Stochastic regularizers as a separate axis** — dropout (Bernoulli 0/1 mask, input-independent),
  zoneout (stochastic 1-mask), adaptive dropout / Standout (input-dependent logistic keep-probability).

## Evaluation settings

The natural yardsticks span modalities so the nonlinearity isn't tuned to one domain: MNIST
classification with an 8-layer fully connected net (with and without dropout) and an MNIST deep
autoencoder; Twitter part-of-speech tagging with a small two-layer tagger over pretrained word vectors;
TIMIT frame-level phone recognition with a five-layer wide classifier and dropout; and CIFAR-10 /
CIFAR-100 image classification with convolutional and wide-residual networks using BatchNorm. Optimizers
are Adam (for the smaller tasks) or SGD-with-momentum / Nesterov (for the residual nets); learning rates
are tuned over a small grid `{1e-3,1e-4,1e-5}` with a validation split, and runs are summarized by the
median over several trials. The metric is task accuracy / error (classification error %, log loss for the
autoencoder), with input-noise robustness as a secondary probe.

## Code framework

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class BaselineActivation(nn.Module):
    """Existing pointwise nonlinearities used as controls."""

    def __init__(self, kind="relu", alpha=1.0):
        super().__init__()
        self.kind = kind
        self.alpha = alpha

    def forward(self, x):
        if self.kind == "relu":
            return F.relu(x)
        if self.kind == "elu":
            return F.elu(x, alpha=self.alpha)
        raise ValueError(self.kind)


class PointwiseActivation(nn.Module):
    """Tensor in, same-shape tensor out."""

    def forward(self, x):
        # TODO: design the pointwise map.
        raise NotImplementedError
