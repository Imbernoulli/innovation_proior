The thing I keep coming back to is that ReLU and dropout, the two pieces I always bolt together, are doing the *same kind* of operation and nobody treats them as one. Look at what each does to a single neuron's value. ReLU takes the preactivation `x` and multiplies it by 0 or 1 — deterministically, by the sign of `x`. Dropout takes a value and multiplies it by 0 or 1 — stochastically, by a Bernoulli draw that ignores the input entirely. Zoneout does almost the same on the RNN side, stochastically multiplying by 1. So I have a deterministic 0/1 mask that depends on the input, and a stochastic 0/1 mask that doesn't. They live in different parts of the architecture and get designed separately, but they're both "multiply this neuron by a 0/1 mask." That coincidence feels like it's pointing somewhere.

Let me push on the discontent with ReLU first, because that's the thing I actually want to replace. ReLU's gate is `1(x>0)`. It's a step function. It throws away every negative input completely — output zero, gradient zero — so a unit that drifts negative is dead and never comes back, and there's a hard kink at the origin with no curvature on either side. The positive half is just the identity; the function has no curvature anywhere, it's piecewise linear. None of that is fatal — ReLU plainly works — but the *form* of the decision bothers me: it gates purely on sign, a binary "is this positive," with no sense of *how* positive, and it makes that decision with infinite sharpness. I'd like something that decides more gently, and that uses the *magnitude* of `x`, not just its sign.

Now braid the two observations together. What if the mask multiplying the neuron isn't deterministic-by-sign (ReLU) and isn't stochastic-input-independent (dropout), but stochastic *and* input-dependent — a 0/1 mask whose probability of being 1 grows with `x`? Concretely, multiply `x` by `m ~ Bernoulli(p(x))`, where `p(x)` is large when `x` is large and small when `x` is very negative. That single object would *be* a nonlinearity (it shapes the neuron's response to its input) and *be* a stochastic regularizer (it's a random mask, like dropout) at the same time. The two design axes I'd been treating as separate would become one.

I need a `p(x)`: a function from the real line into `[0,1]`, increasing, so that the more strongly a unit fires, the more likely its value survives the mask. Any CDF qualifies — it's monotone from 0 to 1 by definition. Which one? Here the training-time fact earns its keep: preactivations into a unit are roughly normally distributed, and Batch Normalization makes that almost exact by standardizing them toward mean 0, variance 1. So the natural scale for "is this input large or small relative to the others" is the standard normal. Set `p(x) = Φ(x)`, the CDF of `N(0,1)`: `P(X ≤ x)` for a standard normal `X`. Then `p(x)` is exactly "the probability that a typical preactivation is below this one" — the mask keeps a unit with probability equal to how much it stands out above the rest. That's a far more honest gate than `1(x>0)`: it's still a 0/1 decision, but the threshold is soft and calibrated to the actual distribution of activations.

So the stochastic object is: multiply `x` by `m ~ Bernoulli(Φ(x))`. This is close in spirit to adaptive dropout, which also makes the keep-probability input-dependent — but adaptive dropout masks the *output* of a nonlinearity with a *logistic* gate, used in tandem with a separate activation; here the mask *is* the only nonlinearity and the gate is the standard normal CDF, chosen because that's the distribution the inputs actually follow. Different intent: I'm not regularizing a nonlinearity, I'm proposing that the masked linear map can *replace* the nonlinearity.

But a network usually wants a deterministic answer at inference, and I'd rather not carry sampling noise through every layer at train time either if I can avoid it. The standard move, the same one dropout uses at test time, is to take the expectation of the stochastic transform. The transform on input `x` is: with probability `Φ(x)` output `x`, with probability `1−Φ(x)` output `0`. Since `x` is fixed when I condition on the preactivation and only the mask is random, its expectation is

`E[m·x | x] = x·E[m | x] = x·Φ(x).`

There it is. The deterministic nonlinearity that is the *expectation* of this input-dependent stochastic mask is simply `x·Φ(x)`. I didn't postulate a curve and then justify it; the curve fell out of "take the mean of the soft, magnitude-aware gate." Call it the Gaussian Error Linear Unit, GELU, because `Φ` is written through the error function:

`GELU(x) = x·Φ(x) = x · ½[1 + erf(x/√2)].`

Read what this does. Where ReLU multiplies `x` by the hard `1(x>0)`, GELU multiplies `x` by the soft `Φ(x)` — it *weights the input by how much greater it is than other inputs*, instead of gating on sign. Loosely, "scale `x` by the probability that a typical input is below it."

Let me check the limits and the shape, because I want to be sure this is a genuine generalization of what works, not something exotic. As `x → +∞`, `Φ(x) → 1`, so `GELU(x) → x`: it recovers the identity on the far positive side, exactly like ReLU. As `x → −∞`, `Φ(x) → 0` fast (Gaussian tail), so `GELU(x) → 0`: it recovers ReLU's suppression of strongly negative inputs. Asymptotically it *is* ReLU. More precisely, if I let the gate use `P(X_σ ≤ x)` with `X_σ ~ N(0, σ²)`, then as `σ → 0` the CDF tends to the hard step `1(x>0)` away from the origin; at the origin the product is still zero. The activation becomes `x·1(x>0)`, exactly ReLU. So ReLU is the zero-variance limit of this CDF-gated family — GELU is a *smoothing* of ReLU by the Gaussian scale, the same way the sigmoid once smoothed the binary threshold. That's reassuring: I'm not abandoning ReLU, I'm putting softness on its gate and reading off the natural setting (`μ=0, σ=1`) from the standardized-preactivation fact, so no new hyperparameters enter.

Now the differences from ReLU/ELU, which are the whole point. Near the origin and slightly negative, `Φ(x)` is small but positive and `x·Φ(x)` dips *below zero* — the function is non-monotonic, with a shallow negative bump for `x` a bit less than 0, then returns to 0. Its derivative is `Φ(x) + xφ(x)`, so the negative side is not an entire zero-gradient half-line the way ReLU's is, even though the local minimum itself has derivative zero. Its second derivative is `(2−x²)φ(x)`, so it is not globally convex and it is not a straight line on the positive half-line; the curvature changes sign at `|x|=√2`. So unlike ReLU it is smooth and differentiable everywhere, no kink; unlike ReLU and ELU it is non-monotonic and non-convex; and unlike ReLU or ELU on `x>0`, it keeps bending instead of becoming exactly linear. That should make it easier to bend toward complicated target functions than a half-linear activation. And it keeps a probabilistic meaning ReLU never had: it's literally the expected value of a calibrated stochastic gate.

The choice of `Φ` over other CDFs deserves a beat, because nothing forces the Gaussian. Take the logistic CDF instead, `σ(x) = 1/(1+e^{−x})`. Then the same expectation construction gives `x·σ(x)` — a Sigmoid Linear Unit, SiLU. It has the same self-gated shape. But `σ` is not the distribution the inputs follow; it's a convenience. The logistic and the standard-normal CDF look similar but are visibly different in the tails and slope, and matching the gate to the *actual* distribution of (BatchNorm-standardized) preactivations is the principled reason to prefer `Φ`. One could also imagine a Cauchy CDF; that route relates to ELU asymptotically. The Gaussian is the one justified by how the inputs are distributed, so that's the default; `μ=0, σ=1` because the inputs are standardized.

The remaining issue is purely computational: `erf` is more expensive than the elementary functions in `max(x,0)`, and I'm calling this nonlinearity once per unit per layer. I want fast, accurate approximations to `x·Φ(x)`. Two work. A `tanh`-based one,

`0.5 x (1 + tanh[√(2/π)(x + 0.044715 x³)]),`

which is a known close approximation to the Gaussian CDF, and a cheaper sigmoid-based one,

`x·σ(1.702 x).`

The second is just SiLU with a scaling constant `1.702` chosen so that `σ(1.702 x)` tracks `Φ(x)` — i.e. the logistic gate, rescaled, approximating the Gaussian gate. Both are fast, easy approximations to the same target; the `tanh` form is the default when I want the closer fit, and the sigmoid form is the cheaper feedforward option. These are *approximations to the same `x·Φ(x)`*, not separate derivations.

Two practical notes I'd hand to anyone using this. Train with momentum, as is standard for deep nets — the smooth curved landscape plays well with it. And actually use a *good* approximation to `Φ`: a bare `σ(x)` (no `1.702` rescale) is a noticeably worse fit to the Gaussian CDF, so if speed forces an approximation, use one of the two above rather than plain `σ(x)`.

The causal chain, start to finish: ReLU and dropout are both 0/1 multiplicative masks on a neuron, one deterministic-by-sign, one stochastic-input-independent → unify them into a *stochastic, input-dependent* mask `m ~ Bernoulli(p(x))` with `p` increasing in `x`, which is simultaneously a nonlinearity and a regularizer → pick `p(x)=Φ(x)` because preactivations are (BatchNorm-)standardized normal, so the gate is the probability that a standard-normal reference input falls below this one → take the conditional expectation for a deterministic activation, `E[m·x | x] = x·Φ(x)` → that smooth, non-monotonic, non-half-linear function recovers ReLU in the zero-variance / large-`x` limits while removing ReLU's kink and all-negative zero-gradient half-line → implement `x·½(1+erf(x/√2))` exactly, or the `tanh`/`x·σ(1.702x)` approximations when speed matters.

```python
import torch
import torch.nn as nn


class GELU(nn.Module):
    """Gaussian Error Linear Unit: x * Phi(x), Phi the standard-normal CDF.

    The deterministic expectation of multiplying x by a Bernoulli(Phi(x)) mask
    -- a soft, magnitude-aware version of ReLU's hard 1(x>0) gate.
    """

    def __init__(self, approximate: str = "none"):
        super().__init__()
        self.approximate = approximate  # "none" (exact erf), "tanh", or "sigmoid"

    def forward(self, x):
        if self.approximate == "none":
            # exact: x * 0.5 * (1 + erf(x / sqrt(2)))
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
