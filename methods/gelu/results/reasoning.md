The thing I keep coming back to is that ReLU and dropout, the two pieces I always bolt together, are doing the *same kind* of operation and nobody treats them as one. Look at what each does to a single neuron's value. ReLU takes the preactivation `x` and multiplies it by 0 or 1 — deterministically, by the sign of `x`. Dropout takes a value and multiplies it by 0 or 1 — stochastically, by a Bernoulli draw that ignores the input entirely. Zoneout does almost the same on the RNN side, stochastically multiplying by 1. So I have a deterministic 0/1 mask that depends on the input, and a stochastic 0/1 mask that doesn't. They live in different parts of the architecture and get designed separately, but they're both "multiply this neuron by a 0/1 mask." That coincidence feels like it's pointing somewhere, and I want to chase it.

Let me push on the discontent with ReLU first, because that's the thing I actually want to replace. ReLU's gate is `1(x>0)`. It's a step function. It throws away every negative input completely — output zero, gradient zero — so a unit that drifts negative is dead and never comes back, and there's a hard kink at the origin with no curvature on either side. The positive half is just the identity; the function has no curvature anywhere, it's piecewise linear. None of that is fatal — ReLU plainly works — but the *form* of the decision bothers me: it gates purely on sign, a binary "is this positive," with no sense of *how* positive, and it makes that decision with infinite sharpness. I'd like something that decides more gently, and that uses the *magnitude* of `x`, not just its sign.

Now braid the two observations together. The two masks I noticed differ along two independent axes: deterministic-vs-stochastic, and input-dependent-vs-input-independent. ReLU is (deterministic, input-dependent); dropout is (stochastic, input-independent). The empty corner of that 2x2 is (stochastic, input-dependent): a 0/1 mask that is random *and* whose probability of being 1 grows with `x`. Concretely, multiply `x` by `m ~ Bernoulli(p(x))`, where `p(x)` is large when `x` is large and small when `x` is very negative. An object like that would shape the neuron's response to its input — so it acts as a nonlinearity — while also being a random mask like dropout — so it acts as a stochastic regularizer. If that holds up, the two design axes I'd been treating as separate collapse into one. Whether it actually "holds up" depends on what falls out when I commit to a `p`, so let me commit to one and see.

I need a `p(x)`: a function from the real line into `[0,1]`, increasing, so that the more strongly a unit fires, the more likely its value survives the mask. Any CDF qualifies — it's monotone from 0 to 1 by definition. Which one? Here the training-time fact earns its keep: preactivations into a unit are roughly normally distributed, and Batch Normalization makes that almost exact by standardizing them toward mean 0, variance 1. So the natural scale for "is this input large or small relative to the others" is the standard normal. Try `p(x) = Φ(x)`, the CDF of `N(0,1)`: `P(X ≤ x)` for a standard normal `X`. Then `p(x)` reads as "the probability that a typical preactivation is below this one" — the mask keeps a unit with probability equal to how much it stands out above the rest. That's a more honest gate than `1(x>0)`: still a 0/1 decision, but with a soft threshold calibrated to the actual distribution of activations. Whether `Φ` specifically is the right choice I'll have to test against the alternatives later; for now it's the natural first guess.

So the stochastic object is: multiply `x` by `m ~ Bernoulli(Φ(x))`. This is close in spirit to adaptive dropout, which also makes the keep-probability input-dependent — but adaptive dropout masks the *output* of a nonlinearity with a *logistic* gate, used in tandem with a separate activation; here the mask *is* the only nonlinearity and the gate is the standard normal CDF, chosen because that's the distribution the inputs actually follow. Different intent: I'm not regularizing a nonlinearity, I'm asking whether the masked linear map can *replace* the nonlinearity.

But a network usually wants a deterministic answer at inference, and I'd rather not carry sampling noise through every layer at train time either if I can avoid it. The standard move, the same one dropout uses at test time, is to take the expectation of the stochastic transform. The transform on input `x` is: with probability `Φ(x)` output `x`, with probability `1−Φ(x)` output `0`. Since `x` is fixed when I condition on the preactivation and only the mask is random, its expectation is

`E[m·x | x] = x·E[m | x] = x·Φ(x).`

So the deterministic nonlinearity that is the *expectation* of this input-dependent stochastic mask is `x·Φ(x)`. I didn't postulate a curve and then justify it; the curve is forced by "take the mean of the soft, magnitude-aware gate." Because `Φ` is written through the error function, `Φ(x) = ½[1 + erf(x/√2)]`, this is

`x·Φ(x) = x · ½[1 + erf(x/√2)].`

Read what this does. Where ReLU multiplies `x` by the hard `1(x>0)`, this multiplies `x` by the soft `Φ(x)` — it *weights the input by how much greater it is than other inputs*, instead of gating on sign. Loosely, "scale `x` by the probability that a typical input is below it."

Before I get attached to it, I need to check whether this is actually a generalization of what works or something exotic, and I'd rather compute than assert. Take the two limits. As `x → +∞`, `Φ(x) → 1`, so the function should approach `x`; as `x → −∞`, `Φ(x) → 0` (Gaussian tail), so it should approach `0`. Let me put numbers on "approach." At `x=5`, `Φ(5)=0.9999997`, so `x·Φ(x) − x = −1.4·10⁻⁶`; at `x=8` the gap is `5·10⁻¹⁵`, i.e. machine-zero. On the negative side, at `x=−5` the value is `−1.4·10⁻⁶` and at `x=−8` it is `−5·10⁻¹⁵`. So well outside `|x|≳5` the function is numerically indistinguishable from `max(x,0)`: identity on the right, zero on the left. The interesting behavior is all in the band roughly `|x|<3`. Good — asymptotically this is ReLU, and the deviation from ReLU is a localized feature near the origin, not a global change.

I can make the ReLU connection sharper by reintroducing the variance I threw away. Let the gate use `P(X_σ ≤ x)` with `X_σ ~ N(0, σ²)`, and watch `σ → 0`. Concretely `x·½[1+erf(x/(σ√2))]`. Let me tabulate it as I shrink `σ`:

```
σ=1.0:   x=-1 → -0.159   x=0.5 → 0.346   x=1 → 0.841
σ=0.3:   x=-1 → -0.000   x=0.5 → 0.476   x=1 → 1.000
σ=0.1:   x=-1 → -0.000   x=0.5 → 0.500   x=1 → 1.000
```

By `σ=0.1` the function already reads `0` for negative `x` and `x` for positive `x` to three decimals — exactly `x·1(x>0)`, ReLU. So ReLU is the zero-variance limit of this CDF-gated family, and what I have is a *smoothing* of ReLU by the Gaussian scale, the same way the sigmoid once smoothed the binary threshold. That's reassuring: I'm not abandoning ReLU, I'm putting softness on its gate and reading off the natural setting (`μ=0, σ=1`) from the standardized-preactivation fact, so no new hyperparameters enter. (I'll name it the Gaussian Error Linear Unit, GELU, since the `erf` is doing the gating.)

Now the differences from ReLU/ELU in the band near the origin, which are the whole point, and I want to see them in actual values rather than trust the shape I'm imagining. Tabulating `x·Φ(x)`:

```
x:        -2      -1     -0.75   -0.5    -0.25    0      0.5     1      2
x·Φ(x): -0.0455 -0.1587 -0.1700 -0.1543 -0.1003  0.0  0.3457 0.8413 1.9545
```

Two things jump out. First, the function goes *negative* for `x<0` and then comes back toward 0 as `x→−∞` (it's `−0.159` at `x=−1` but only `−0.046` at `x=−2`), so it is non-monotonic — there's a dip. Scanning for the bottom of that dip on a fine grid, the minimum sits at `x≈−0.752` with value `≈−0.170`, and the derivative there is `≈−9·10⁻⁵`, i.e. zero to grid precision — confirming it's a genuine interior local minimum, not just me misreading the table. So GELU dips a little below zero just left of the origin and turns back up, something neither ReLU nor ELU does.

Second, let me get the derivatives, both to confirm smoothness and to check the negative-side gradient claim. By the product rule, `d/dx [x Φ(x)] = Φ(x) + x φ(x)` with `φ` the standard-normal density, and differentiating again, `d²/dx² = 2φ(x) + x φ'(x) = 2φ(x) − x²φ(x) = (2 − x²)φ(x)`. I should sanity-check these against finite differences rather than trust my algebra. Central differences of `x·Φ(x)`:

```
x:           -1       0       1
Φ(x)+xφ(x): -0.0833  0.5000  1.0833
numeric  d: -0.0833  0.5000  1.0833
```

and for the second derivative, `(2−x²)φ(x)` versus a second difference: at `x=−2,−1,0,1,2` the closed form gives `−0.108, 0.242, 0.798, 0.242, −0.108` and the numeric values agree to four decimals; at `x=±√2` both give `0`. So the derivative formula is right, and the curvature `(2−x²)φ(x)` changes sign exactly at `|x|=√2`: GELU is convex near the origin and concave outside `|x|>√2`, so it is *not* globally convex and *not* a straight line on the positive half. And the derivative on the negative side is not pinned at zero the way ReLU's is — at `x=−1` it's `−0.083`, at `x=−0.5` it's `+0.13` — even though the single point at the bottom of the dip has zero slope. So a unit sitting at negative preactivation still gets a nonzero gradient and can climb back, which is exactly the "dead ReLU" failure I wanted to remove. Summarizing the contrasts that the numbers establish: unlike ReLU it is smooth and differentiable everywhere, no kink; unlike ReLU and ELU it is non-monotonic and non-convex; and unlike ReLU or ELU on `x>0`, it keeps bending instead of becoming exactly linear. A function that keeps curvature everywhere has more freedom to bend toward complicated targets than a half-linear one — though that's an expectation about expressivity I'd want to confirm empirically, not something these limit checks prove. And it keeps a probabilistic meaning ReLU never had: it's literally the expected value of a calibrated stochastic gate.

The choice of `Φ` over other CDFs deserves a beat, because nothing forces the Gaussian — and this is exactly the kind of claim I shouldn't wave through, since "the inputs are Gaussian so use the Gaussian CDF" is a story, not a measurement. Take the logistic CDF instead, `σ(x) = 1/(1+e^{−x})`. Then the same expectation construction gives `x·σ(x)` — a Sigmoid Linear Unit, SiLU. It has the same self-gated shape. The honest question is how different the gate actually is, so let me compare `σ(x)` against `Φ(x)` numerically over the band that matters, `[−4,4]`. The maximum gap `|σ(x) − Φ(x)|` comes out to `0.117` — and pointwise, at `x=1`, `Φ=0.841` but `σ(1)=0.731`, a `0.11` discrepancy in keep-probability. That's large: the logistic gate keeps a moderately-firing unit with materially lower probability than the calibrated Gaussian gate would. So the two CDFs are *not* interchangeable in the regime where preactivations actually live, and matching the gate to the (BatchNorm-standardized) input distribution is a real choice with a measurable consequence, not a cosmetic one. The Gaussian is the one justified by how the inputs are distributed, so that's the default; `μ=0, σ=1` because the inputs are standardized.

The remaining issue is purely computational: `erf` is more expensive than the elementary functions in `max(x,0)`, and I'm calling this nonlinearity once per unit per layer. I want fast, accurate approximations to `x·Φ(x)`. A `tanh`-based one,

`0.5 x (1 + tanh[√(2/π)(x + 0.044715 x³)]),`

is a known close approximation to the Gaussian CDF. I shouldn't take "close" on faith, so I measure: over `[−5,5]` the maximum deviation of this from the exact `x·Φ(x)` is `4.7·10⁻⁴` — fractions of a percent at the activation's typical scale, well below the noise of training. That's tight enough to use as a drop-in. For something cheaper, a sigmoid-based form,

`x·σ(1.702 x),`

is just SiLU with a scaling constant. The constant `1.702` is supposed to make `σ(1.702 x)` track `Φ(x)`, and I just saw that the *un*scaled `σ(x)` is off by `0.117`, so the rescale had better earn its place. Measuring `|σ(1.702 x) − Φ(x)|` over `[−4,4]`: the maximum is `0.0095` — about an order of magnitude better than plain `σ(x)`'s `0.117`. So the single constant `1.702` really does pull the logistic gate onto the Gaussian gate; at `x=1` it gives `0.846` against `Φ=0.841`, versus plain `σ`'s `0.731`. Both approximations are therefore legitimate: the `tanh` form is the default when I want the closer fit, the sigmoid form the cheaper feedforward option, and they're both approximations to the *same* `x·Φ(x)`, not separate derivations.

That last measurement also settles a practical point I'd otherwise have only asserted. If speed forces an approximation, use one of these two — *not* a bare `σ(x)` with no rescale, because its `0.117` gap to `Φ` makes it a visibly different function, not an approximation to GELU at all. The other practical note: train with momentum, as is standard for deep nets — the smooth curved landscape should play well with it.

The causal chain, start to finish: ReLU and dropout are both 0/1 multiplicative masks on a neuron, one deterministic-by-sign, one stochastic-input-independent → fill the empty corner of that classification with a *stochastic, input-dependent* mask `m ~ Bernoulli(p(x))` with `p` increasing in `x`, which is simultaneously a nonlinearity and a regularizer → pick `p(x)=Φ(x)` because preactivations are (BatchNorm-)standardized normal, so the gate is the probability that a standard-normal reference input falls below this one (and the logistic alternative is numerically a `0.117`-different gate, so the Gaussian choice is load-bearing) → take the conditional expectation for a deterministic activation, `E[m·x | x] = x·Φ(x)` → check the numbers: it equals ReLU to machine precision for `|x|≳5` and in the `σ→0` limit, dips to `−0.170` at `x≈−0.75`, has derivative `Φ(x)+xφ(x)` (nonzero on the negative side) and curvature `(2−x²)φ(x)` flipping sign at `|x|=√2`, all confirmed against finite differences → implement `x·½(1+erf(x/√2))` exactly, or the `tanh` approximation (max error `4.7·10⁻⁴`) / `x·σ(1.702x)` (max error `0.0095`) when speed matters.

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
