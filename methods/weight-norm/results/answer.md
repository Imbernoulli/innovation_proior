# Weight Normalization

## Problem

First-order SGD training is bottlenecked by the curvature of the loss, and curvature is not
invariant to how the model is parameterized — equivalent parameterizations can be far
easier or harder to optimize. Batch normalization improves conditioning by standardizing
each neuron's pre-activation over the minibatch, but in doing so it couples the examples in
a batch, injects high-variance stochastic noise into the gradient, computes different
functions at train and test time, and is awkward in recurrent and noise-sensitive settings
(RNN/LSTM, reinforcement learning, generative models). The goal: a reparameterization of
the basic neuron y = φ(w·x + b) that gives much of batch normalization's conditioning
speed-up while being deterministic, per-example, cheap, and free of any batch dependence.

## Key idea

Reparameterize each weight vector as a scalar magnitude times a unit direction, and run SGD
directly in the new parameters:

    w = (g / ‖v‖) · v,

with v a trainable vector and g a trainable scalar. Then ‖w‖ = g exactly, independent of v:
the length of the weight (g) is decoupled from its direction (v/‖v‖). This is *weight
normalization*. (Optionally g = e^s on a log scale; empirically no benefit, so plain g is
used.) Unlike earlier max-norm/weight-clipping schemes, which optimize in w and project the
norm back after each step, here the optimization is carried out in (g, v, b) themselves, so
the optimizer experiences the better-conditioned geometry directly.

## Gradients

Differentiating L through w = (g/‖v‖)v gives

    ∇_g L = (∇_w L · v) / ‖v‖,
    ∇_v L = (g/‖v‖) ∇_w L − (g ∇_g L / ‖v‖²) v.

Equivalently, with the projector M_w = I − w w'/‖w‖² onto the complement of w,

    ∇_v L = (g/‖v‖) · M_w · ∇_w L,

so the reparameterization (1) scales the ordinary weight gradient by g/‖v‖ and (2) projects
it orthogonal to the current weight. Removing the along-w component — often the dominant
eigenvector of the gradient covariance C, since the v-gradient covariance is
D = (g²/‖v‖²) M_w C M_w — whitens the gradient and improves conditioning. Because ∇_v L ⟂ v,
a plain steepest-descent step grows ‖v‖ by the factor √(1+c²) (c = ‖Δv‖/‖v‖), which shrinks
the effective scale g/‖v‖ and self-stabilizes the effective learning rate, making training
robust to the learning-rate choice.

## Relation to batch normalization

For a single layer with whitened inputs (zero mean, unit variance, independent),
μ[t] = 0 and σ[t] = ‖v‖ for t = v·x, so batch normalization's (t − μ[t])/σ[t] = (v·x)/‖v‖
is exactly weight normalization. In deep nets it is an approximation, but a cheaper,
deterministic, lower-variance one (CNNs have far fewer weights than pre-activations, and
‖v‖ is non-stochastic).

## Data-dependent initialization

Weight normalization does not fix per-layer feature scales the way batch normalization does,
so initialize from one minibatch. Sample v ~ N(0, 0.05²). Feed one minibatch through;
at each neuron compute the normalized pre-activation t = (v·x)/‖v‖ with minibatch mean μ[t]
and std σ[t], and set

    g ← 1/σ[t],   b ← −μ[t]/σ[t],

so every pre-activation starts at zero mean and unit variance (only exactly for that batch).
Recurrent models fall back to standard initialization.

## Mean-only batch normalization

Weight normalization fixes the activation scale but not the mean, so optionally combine it
with a batch normalization that subtracts the minibatch mean only (no variance division):

    t = w·x,   t̃ = t − μ[t] + b,   y = φ(t̃),

with μ[t] running-averaged for test time. The backward pass becomes
∇_t L = ∇_{t̃} L − μ[∇_{t̃} L] — the gradient is centered. The injected noise comes only from
the mean estimate, which is approximately Gaussian (light-tailed), versus the high-kurtosis
noise of full batch normalization's variance estimate. Weight normalization + mean-only
batch normalization combines clean conditioning with gentle regularizing noise.

## Code

```python
import torch
import torch.nn as nn


def _norm_except_dim(v, dim):
    # Euclidean norm of v over every dimension except `dim` (one per output channel).
    if dim == -1:
        return v.norm()
    perm = [dim] + [d for d in range(v.dim()) if d != dim]
    flat = v.permute(*perm).reshape(v.size(dim), -1)
    n = flat.norm(dim=1)
    shape = [1] * v.dim()
    shape[dim] = v.size(dim)
    return n.reshape(shape)


def _compute_weight(module, name, dim):
    g = getattr(module, name + "_g")           # magnitude per output channel
    v = getattr(module, name + "_v")           # direction
    return g * (v / _norm_except_dim(v, dim))  # w = g * v / ||v||


class _WeightNorm:
    def __init__(self, name, dim):
        self.name, self.dim = name, dim

    def __call__(self, module, _inputs):       # runs right before forward()
        setattr(module, self.name, _compute_weight(module, self.name, self.dim))


def weight_norm(module, name="weight", dim=0):
    """Reparameterize module.<name> into magnitude (<name>_g) and direction (<name>_v);
    SGD then runs in those. dim=0 -> one magnitude per output channel."""
    w = getattr(module, name)
    del module._parameters[name]
    module.register_parameter(name + "_g", nn.Parameter(_norm_except_dim(w, dim).data))
    module.register_parameter(name + "_v", nn.Parameter(w.data))
    setattr(module, name, _compute_weight(module, name, dim))
    module.register_forward_pre_hook(_WeightNorm(name, dim))
    return module


@torch.no_grad()
def data_dependent_init(layer, x):
    """One minibatch -> g <- 1/std, b <- -mean/std, per output channel."""
    getattr(layer, "weight_g").fill_(1.0)
    if layer.bias is not None:
        layer.bias.zero_()
    t = layer(x)
    reduce = [d for d in range(t.dim()) if d != 1]
    mean, std = t.mean(dim=reduce), t.std(dim=reduce) + 1e-10
    getattr(layer, "weight_g").copy_((1.0 / std).reshape(getattr(layer, "weight_g").shape))
    if layer.bias is not None:
        layer.bias.copy_(-mean / std)
    return layer


class MeanOnlyBatchNorm(nn.Module):
    def __init__(self, num_features, momentum=0.1):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.momentum = momentum

    def forward(self, t):
        reduce = [d for d in range(t.dim()) if d != 1]
        if self.training:
            mu = t.mean(dim=reduce)
            self.running_mean.mul_(1 - self.momentum).add_(self.momentum * mu.detach())
        else:
            mu = self.running_mean
        shape = [1, -1] + [1] * (t.dim() - 2)
        return t - mu.reshape(shape) + self.bias.reshape(shape)


# usage
conv = weight_norm(nn.Conv2d(3, 96, 3, padding=1), name="weight", dim=0)
# data_dependent_init(conv, first_minibatch)   # one-shot before training
# then train with Adam; the pre-forward hook rebuilds `weight` each forward.
```
