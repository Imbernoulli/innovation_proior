# GELU — the method, distilled

**Problem.** GELU replaces ReLU's hard, sign-based gate `max(x,0)=x·1(x>0)` with a smooth,
magnitude-aware nonlinearity that keeps ReLU's empirical strengths (cheap, deep-net-friendly, recovers
identity for large positive `x`) while softening its defects — the all-negative zero-gradient half-line,
the kink at the origin, and the half-linear positive side — and that carries a probabilistic
interpretation tying it to the stochastic regularizers (dropout, zoneout) networks already use.

**Key idea.** ReLU is a deterministic 0/1 mask on a neuron's value; dropout is a stochastic
input-independent one. Unify them: multiply `x` by a *stochastic, input-dependent* mask
`m ~ Bernoulli(Φ(x))`, where `Φ` is the standard-normal CDF — chosen because (BatchNorm-)standardized
preactivations are approximately `N(0,1)`, so the keep-probability is the probability that a standard
normal reference input falls below this one. For a deterministic activation, take the conditional
expectation:

`GELU(x) = E[m·x | x] = x·E[m | x] = x·Φ(x) = x · ½[1 + erf(x/√2)].`

**Why it works.** As `x→+∞`, `Φ(x)→1` so `GELU→x` (identity, like ReLU); as `x→−∞`, `Φ(x)→0` so
`GELU→0` (suppresses strong negatives, like ReLU); if the gate is `P(X_σ≤x)` for `X_σ~N(0,σ²)` and
`σ→0`, it tends to `1(x>0)` away from the origin and the product at the origin remains zero, so GELU
becomes ReLU exactly. ReLU is its zero-variance limit, and GELU is a Gaussian-smoothing of ReLU.
Unlike ReLU it is smooth and differentiable everywhere (no kink), non-monotonic with a small negative
dip just left of 0, and its negative side is not an entire zero-gradient region. Its derivative is
`Φ(x)+xφ(x)` and its second derivative is `(2−x²)φ(x)`, so it is non-convex and keeps a curved positive
side rather than becoming exactly linear like ReLU/ELU. No new hyperparameters: `μ=0, σ=1`.

**Hyperparameters / variants.** `μ=0, σ=1` fixed. Exact form uses `erf`; two fast approximations to the
same `x·Φ(x)`: the `tanh` form `0.5 x (1 + tanh[√(2/π)(x+0.044715x³)])` and the sigmoid form
`x·σ(1.702x)` (a rescaled logistic gate approximating the Gaussian gate). Using the plain logistic CDF
`σ(x)` gives the related SiLU `x·σ(x)`, but it is a looser match to the Gaussian CDF than the scaled
logistic approximation. Practical tips: train with momentum; use one of the good `Φ`-approximations,
not bare `σ(x)`.

```python
import torch
import torch.nn as nn


class GELU(nn.Module):
    """Gaussian Error Linear Unit: x * Phi(x)."""

    def __init__(self, approximate: str = "none"):
        super().__init__()
        self.approximate = approximate  # "none" | "tanh" | "sigmoid"

    def forward(self, x):
        if self.approximate == "none":
            return x * 0.5 * (1.0 + torch.erf(x / (2.0 ** 0.5)))
        if self.approximate == "tanh":
            c = (2.0 / torch.pi) ** 0.5
            return 0.5 * x * (1.0 + torch.tanh(c * (x + 0.044715 * x ** 3)))
        if self.approximate == "sigmoid":
            return x * torch.sigmoid(1.702 * x)
        raise ValueError(self.approximate)
```
