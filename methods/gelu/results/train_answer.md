The central problem is choosing a pointwise nonlinearity for deep networks. Without one, stacked linear layers collapse to a single linear map, so some nonlinearity is mandatory. ReLU, the dominant choice, applies a hard sign-based gate: it outputs x multiplied by 0 or 1 depending only on whether x is positive. This is cheap and gives good gradient flow on the positive side, but it has clear defects. It is non-differentiable at the origin, it outputs zero for all negative inputs, and any unit whose preactivation drifts negative receives zero gradient forever and stops learning. ELU softens the all-negative-is-zero problem by allowing small negative outputs, but it remains monotonic and convex, with no curvature on the positive side and an extra hyperparameter. Sigmoid and tanh are smooth but saturate, causing gradients to vanish in deep stacks. What is needed is an activation that keeps ReLU's empirical strengths while removing its hard kink and dead-negative half-line, without introducing new hyperparameters or saturating behavior.

A useful observation is that ReLU and dropout are doing structurally similar things. ReLU multiplies a neuron's value by a deterministic 0/1 mask based on sign; dropout multiplies it by a stochastic 0/1 mask independent of the input. Both are multiplicative gates. This suggests unifying them: use a stochastic mask whose keep-probability depends on the input magnitude itself. If preactivations are approximately standard normal, especially after Batch Normalization, then the natural probability that a unit "fires" relative to its peers is the standard-normal cumulative distribution function Φ(x). Multiplying x by a Bernoulli(Φ(x)) mask gives a stochastic activation. Taking the expectation over the mask yields a deterministic nonlinearity: x · Φ(x). This is the Gaussian Error Linear Unit, or GELU.

GELU is defined as GELU(x) = x · Φ(x) = x · ½[1 + erf(x/√2)]. It multiplies the input by a soft Gaussian gate rather than ReLU's hard step. For large positive x, Φ(x) approaches 1, so GELU approaches x and recovers the identity. For large negative x, Φ(x) approaches 0 rapidly, so GELU approaches 0 and suppresses strong negatives. In the limit where the Gaussian gate shrinks to a step function, GELU becomes ReLU exactly; ReLU is the zero-variance limit of the CDF-gated family. Unlike ReLU, GELU is smooth and differentiable everywhere, with no kink at the origin. It is non-monotonic: just below zero it has a small negative dip before returning to zero. Its derivative is Φ(x) + x φ(x), so the negative side is not an entire zero-gradient region. Its second derivative is (2 − x²) φ(x), meaning it is non-convex and keeps curvature on the positive side rather than becoming exactly linear. This makes it more flexible for fitting complicated functions while maintaining stable gradients.

The choice of Φ over other CDFs is principled. SiLU uses the logistic CDF σ(x), giving x · σ(x). That has the same self-gated shape, but it is a convenience rather than a distribution match. Because BatchNorm-standardized preactivations are approximately N(0, 1), the Gaussian gate is the calibrated choice; it weights each input by the probability that a standard-normal reference input falls below it. No new hyperparameters enter because μ = 0 and σ = 1 are fixed by the standardization.

Computationally, erf is more expensive than max. Two fast approximations are commonly used. The tanh approximation is 0.5 x (1 + tanh[√(2/π)(x + 0.044715 x³)]). The sigmoid approximation is x · σ(1.702 x), which is a rescaled logistic gate tracking the Gaussian CDF. Both approximate the same x · Φ(x) target; the tanh form is closer, while the sigmoid form is cheaper. Plain x · σ(x), without the 1.702 rescale, is a looser match and is generally not recommended when approximating GELU. In practice, GELU is a drop-in replacement for ReLU in feedforward networks, convolutional networks, and transformers, and it trains well with standard momentum-based optimizers.

```python
import torch
import torch.nn as nn


class GELU(nn.Module):
    """Gaussian Error Linear Unit: x * Phi(x), Phi the standard-normal CDF.

    The deterministic expectation of multiplying x by a Bernoulli(Phi(x)) mask.
    """

    def __init__(self, approximate: str = "none"):
        super().__init__()
        self.approximate = approximate  # "none" | "tanh" | "sigmoid"

    def forward(self, x):
        if self.approximate == "none":
            # Exact: x * 0.5 * (1 + erf(x / sqrt(2)))
            return x * 0.5 * (1.0 + torch.erf(x / (2.0 ** 0.5)))
        elif self.approximate == "tanh":
            # 0.5 x (1 + tanh[sqrt(2/pi) (x + 0.044715 x^3)])
            c = (2.0 / torch.pi) ** 0.5
            return 0.5 * x * (1.0 + torch.tanh(c * (x + 0.044715 * x ** 3)))
        elif self.approximate == "sigmoid":
            # x * sigmoid(1.702 x): logistic gate rescaled to track Phi
            return x * torch.sigmoid(1.702 * x)
        raise ValueError(self.approximate)
```
