## Research question

The activation function `f(·)` that follows each linear transformation strongly affects how well a deep
network trains, yet the field has effectively standardized on one hand-designed choice — ReLU
`max(x,0)` — because every proposed replacement gives gains that are *inconsistent* across models and
datasets. The question is whether one can *automatically search* the space of scalar activation functions
(one scalar in, one scalar out, so any candidate is a drop-in ReLU replacement requiring no architecture
change) and select candidates by a validation signal.

## Background

ReLU `max(x,0)` (Hahnloser et al. 2000; Jarrett et al. 2009; Nair & Hinton 2010) is the default because
deep networks with ReLUs optimize far more easily than with sigmoid/tanh: for positive inputs the
derivative is exactly 1, so gradients flow undiminished, whereas saturating units kill the gradient in
deep stacks. Its commonly cited virtues are simplicity, sparsity, and this "gradient-preserving"
property on the positive half-line.

Hand-designed alternatives each fix one perceived flaw: Leaky ReLU (Maas et al. 2013) replaces the
negative half with a small slope `αx` (`α=0.01`); PReLU (He et al. 2015) makes that slope a learned
per-channel parameter; ELU (Clevert et al. 2015) uses `α(eˣ−1)` for `x<0` to push mean activations
toward zero; SELU (Klambauer et al. 2017) scales ELU with fixed constants for a self-normalizing fixed
point; softplus `log(1+eˣ)` is a smooth, strictly positive, monotonic approximation to ReLU.

Two background results make an automated search plausible. First, search/meta-learning had recently been
shown to discover traditionally hand-designed components that beat human designs — RNN/CNN architectures
via RL controllers (Zoph & Le 2016; Zoph et al. 2017) and optimizer update rules over a space of
composed primitive functions (Bello et al. 2017). Second, an RNN controller trained with policy-gradient
RL, using validation accuracy of a trained "child" network as the reward, is a known recipe for
searching such combinatorial spaces. On the activation side, the Sigmoid-weighted Linear Unit
`x·σ(x)` (Elfwing et al. 2017, "SiL") had been proposed for neural network function approximation in
reinforcement learning, motivated by the expected-energy of a restricted Boltzmann machine.

## Baselines

- **ReLU** `max(x,0)`: cheap, sparse, derivative 1 on `x>0`.
- **Leaky ReLU** `x` if `x≥0` else `αx`, `α=0.01`; **PReLU**, same with learned per-channel `α`
  (init 0.25): leak a little gradient on the negative side.
- **ELU** `x` if `x≥0` else `α(eˣ−1)`, `α=1`; **SELU**, `λ·ELU` with `α≈1.6733, λ≈1.0507`: negative
  saturation toward `−α`, mean-centering / self-normalizing.
- **Softplus** `log(1+eˣ)`: smooth, monotonic, strictly positive, ReLU-like.
- **GELU** `x·Φ(x)` (Hendrycks & Gimpel 2016): smooth, non-monotonic.
- **SiL** `x·σ(x)` (Elfwing et al. 2017): a non-monotonic unit from RL function approximation.

## Evaluation settings

A search-then-validate protocol. Search: train small "child" networks (ResNet-20 on CIFAR-10, ~10K
steps) for each candidate activation and use validation accuracy as the signal — exhaustive enumeration
for small spaces, an RNN controller trained with PPO (with an EMA-of-reward baseline) for large ones,
parallelized over workers. Validation/generalization: take the top selected candidates and, *with no
change to the published architectures or hyperparameters beyond swapping the activation*, evaluate on
larger image models — preactivation ResNet-164, Wide ResNet 28-10, DenseNet-100-12 on CIFAR-10/100;
Inception-ResNet-v2, Inception-v3/v4, MobileNet, Mobile NASNet-A on ImageNet — and on WMT
English→German machine translation with the Transformer (BLEU). Image models train with SGD-momentum or
RMSProp and He initialization; a few learning rates are tried and the best taken; results are the median
of several runs. Comparisons against the baseline activations are aggregated with a one-sided paired
sign test (wins/ties/losses across models).

## Code framework

```python
import torch
import torch.nn as nn


# --- the search space: scalar activations assembled from unary/binary primitives ---
UNARY = {
    "id": lambda x: x,
    "neg": lambda x: -x,
    "square": lambda x: x ** 2,
    "sigmoid": lambda x: torch.sigmoid(x),
    "tanh": lambda x: torch.tanh(x),
    "exp": lambda x: torch.exp(x),
    "relu": lambda x: torch.relu(x),
    # ... (abs, cube, sqrt, log, sin, cos, erf, max(x,0), min(x,0), ...)
}
BINARY = {
    "add": lambda a, b: a + b,
    "mul": lambda a, b: a * b,
    "sub": lambda a, b: a - b,
    "max": lambda a, b: torch.maximum(a, b),
    # ... (min, div, gaussian, weighted-sum, ...)
}


class CoreUnit:
    """One unit computes b(u1(source1), u2(source2))."""

    def __init__(self, source1, unary1, source2, unary2, binary):
        self.source1 = source1
        self.unary1 = unary1
        self.source2 = source2
        self.unary2 = unary2
        self.binary = binary


def build_candidate(x, core_units):
    """Evaluate a scalar activation candidate from a list of core units."""
    values = [x]
    for unit in core_units:
        u1 = UNARY[unit.unary1](values[unit.source1])
        u2 = UNARY[unit.unary2](values[unit.source2])
        values.append(BINARY[unit.binary](u1, u2))
    return values[-1]


class CandidateActivation(nn.Module):
    """Drop-in scalar activation assembled from the search primitives."""

    def __init__(self, core_units):
        super().__init__()
        self.core_units = core_units

    def forward(self, x):
        return build_candidate(x, self.core_units)


# --- search controller (large spaces): RNN proposes a function string, RL reward = child val-acc ---
class RNNController(nn.Module):
    def sample(self):
        # TODO: autoregressively emit (unary, unary, binary, ...) tokens -> a candidate f.
        pass

    def update(self, candidates, rewards):
        # TODO: policy-gradient (PPO) step toward higher child-network validation accuracy.
        pass
```
