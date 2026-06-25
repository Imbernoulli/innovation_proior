I already have a useful pattern in front of me: hard rectification fixed the saturation problem of sigmoid and tanh, but the newer smooth self-gated units keep the raw input and multiply it by a smooth gate. Swish writes this as `x·σ(βx)`, GELU as `x·Φ(x)`. The problem I want to solve is narrower than inventing an activation from nothing: keep the self-gated shape, keep the smooth negative-side behavior that ReLU lacks, and find a pointwise curve whose derivative gives me a mechanical reason to expect stable optimization.

I pin down the properties I am not willing to give up. The function should be smooth and continuously differentiable everywhere, with no ReLU kink. It should preserve a small amount of negative signal, because a hard zero on the whole negative half-plane is exactly how a unit gets stuck with zero output and zero gradient. It should be unbounded above, so large positive activations do not saturate the way sigmoid and tanh do. It should be bounded below, so the negative side cannot run away. And it should stay self-gated: `f(x)=x·h(x)` for a smooth scalar gate `h`.

So the search is over `h`. Swish picks `h=σ`; GELU picks `h=Φ`. What both gates have in common is their asymptotic logic: `h(x)→0` as `x→−∞`, so strongly negative inputs are suppressed gently, and `h(x)→1` as `x→+∞`, so the activation recovers the identity on large positives. If I want a different gate with that same pair of endpoints, I should build it from primitives I can reason about. Softplus is a natural inner piece: `softplus(x)=log(1+eˣ)` is smooth, strictly positive, close to `0` on the far negative side, and close to `x` on the far positive side — a smooth ReLU-like ramp. But softplus itself runs off to `+∞`, so it can't be the gate; I need to squash it back down so the gate tends to `1`. The obvious squash that sends `[0,∞)` to `[0,1)` smoothly is `tanh`. That suggests the gate `h(x)=tanh(softplus(x))`, and the activation

`f(x) = x·tanh(softplus(x)) = x·tanh(ln(1+eˣ)).`

Before I get attached to this, I should check that the two endpoints actually come out right, because the whole point of choosing softplus-then-tanh was to reproduce the `0→1` gate shape, and it would be easy to be off by a scale factor. As `x→+∞`: `softplus(x)→x→∞`, and `tanh(∞)=1`, so `h→1` and `f→x`. Numerically, at `x=20`, `softplus(20)≈20`, `tanh(20)` is `1` to machine precision, so `h(20)=1.000…` — good, the gate hits `1` on its own with no extra constant. As `x→−∞`: `softplus(x)→0⁺` and `tanh(0)=0`, so `h→0` and `f→0`. At `x=−20`, `softplus(−20)≈e^{−20}≈2·10⁻⁹`, and `tanh` of that is the same `2·10⁻⁹`, so `h(−20)≈2·10⁻⁹` — the suppression is in place. Both endpoints land where I wanted, so the gate is the right shape, not just qualitatively but with the asymptotic constants checked.

Now I want to know what the curve does *between* the endpoints, and in particular whether it actually preserves negative signal the way I demanded. For slightly negative `x`, `h(x)` is small and positive but `x` is negative, so the product `x·h(x)` should dip below zero — a negative bump, which is the non-monotonic feature that keeps a unit from dying. I don't want to just assert that; I want to know how deep the bump is, because that number is the range's lower bound. I evaluate `f` on a grid over the negative side:

```
x      f(x)=x·tanh(softplus(x))
-0.5   -0.2207
-1.0   -0.3034
-1.19  -0.3088   <- minimum
-1.5   -0.3034
-2.0   -0.2525
-3.0   -0.1421
```

So the bump bottoms out at `f≈−0.309`, near `x≈−1.19`, and climbs back toward `0` on either side. The minimum is about `−0.31`, which makes the range `[≈−0.31, ∞)`: bounded below, unbounded above, with a shallow non-monotonic dip on the negative side. That is the silhouette I asked for — same qualitative shape as Swish, whose own bump is a touch shallower, with slightly different curvature in the negative region.

That covers the shape. The place to look for a *mechanism* is the first derivative, because backpropagation multiplies the upstream gradient by this scalar at every unit, so its profile is what the optimizer actually feels. Differentiate `f=x·tanh(softplus(x))`. Let `s=softplus(x)`, so `s′=σ(x)` because the derivative of softplus is exactly the sigmoid. With `f=x·tanh(s)`, the product rule gives

`f′(x) = tanh(s) + x·sech²(s)·s′ = tanh(s) + x·sech²(s)·σ(x).`

I want to read structure out of this, so I rewrite the two pieces. The factor `x·σ(x)` in the second term is Swish with `β=1`. And `tanh(s)=tanh(softplus(x))=f(x)/x` whenever `x≠0`. Substituting both,

`f′(x) = sech²(softplus(x))·x·σ(x) + f(x)/x = Δ(x)·swish(x) + f(x)/x,`

with `Δ(x)=sech²(softplus(x))` and `swish(x)=x·σ(x)`.

I should check that this rewrite is actually equal to the derivative and that I haven't dropped a factor, since I divided by `x` to get the `f(x)/x` term. I test it against a direct numerical derivative (central difference) of `f` at a spread of points, comparing the product-rule form, the rewritten form, and the finite difference:

```
x       f'  (finite diff)   f'  (product rule)   f'  (rewrite)
-3.0      -0.09339          -0.09339             -0.09339
-1.0       0.05922           0.05922              0.05922
-0.3       0.40851           0.40851              0.40851
 0.5       0.88642           0.88642              0.88642
 1.0       1.04904           1.04904              1.04904
 2.0       1.06932           1.06932              1.06932
 4.0       1.00443           1.00443              1.00443
```

All three agree to every printed digit, so the identity is literal away from zero and I haven't lost a factor. The only place the rewrite needs care is `x=0`, where I divided by `x`. There the `f(x)/x` term is a removable singularity: `f(x)/x=tanh(softplus(x))→tanh(softplus(0))=tanh(ln 2)`. I can get that limit in closed form — `tanh(ln 2)=(2−½)/(2+½)=(3/2)/(5/2)=3/5` — so by continuous extension `f(x)/x→3/5` at `0`, while `swish(0)=0`, so the rewritten form evaluates to `3/5` there. Does the *direct* derivative agree? A central difference of `f` at `0` gives `f′(0)=0.6000…`, matching `3/5` exactly. So the two forms agree everywhere once that single removable value is filled in.

Now `Δ(x)=sech²(softplus(x))` has a precise role, and I can pin its profile with the same `tanh(ln 2)` computation. Since `softplus(x)≥0` and is increasing, and `sech²` is decreasing on `[0,∞)`, `Δ` runs from near `1` on the far negative side down toward `0` on the far positive side. At zero, `Δ(0)=sech²(softplus(0))=sech²(ln 2)=1−tanh²(ln 2)=1−(3/5)²=16/25=0.64`. Checking the tails numerically: `Δ(−20)=1.0`, `Δ(0)=0.64`, `Δ(20)≈2·10⁻¹⁷`. So `Δ` is a smooth, positive, input-dependent multiplier sitting on the Swish-shaped term of the derivative: it passes that term near full strength when the preactivation is very negative, at `0.64` strength at the origin, and shuts it off for large positives, where instead the gate term `f(x)/x→1` carries the derivative. It is not a literal inverse Hessian, but in backpropagation it acts like an input-dependent rescaling *inside* the activation derivative — the Swish-shaped term is continuously re-scaled rather than passed at one fixed gain. That is the preconditioning analogy I'm willing to defend, and only that far: a smooth local rescaling of gradient flow, not a hard kink and not a constant leak.

One more thing this lets me see, which I didn't put in by hand: because `Δ·swish` and `f(x)/x` add, the derivative can exceed `1`. Tabulating `f′` over `x∈[−5,5]`, its maximum is about `1.088` near `x≈1.49`, and on the negative side it dips slightly negative, with a minimum about `−0.113` near `x≈−2.26` (that negative stretch of `f′` is just the back slope of the negative bump). So gradient magnitude is gently amplified in the active positive region and gently reversed where the bump turns over — a smoothly varying gain, consistent with the rescaling reading rather than a fixed pass-through like ReLU's `1`.

This lines up with the smoothness argument from the optimizer's two sides. A smoother per-unit map produces a smoother output landscape; ReLU builds piecewise-linear facets because every unit has a kink, while this curve is `C∞` (softplus and tanh are both smooth, so the composition is, with no kink anywhere). And the derivative analysis says the same thing from the gradient side: the upstream gradient is multiplied by a scalar that changes continuously with the preactivation, with a shallow negative region rather than a dead half-plane.

Why not just stop at Swish, since it is already smooth and self-gated? Swish gives one fixed smooth gate, `σ(x)`. The construction above keeps the same `0→1` endpoints (I checked them) but changes the *derivative structure*: the Swish-shaped factor reappears, but multiplied by `Δ(x)` and with a separate additive gate term `f(x)/x`. So this isn't "Swish with a new name" — Swish lives inside one term of this derivative, modulated. That difference is exactly the hypothesis I'd want the empirical probes to test: if the input-dependent rescaling story is the operative one, the advantage should show up as stability with depth, tolerance to noise and initializer choice, and a visibly smoother loss landscape, rather than as a new architectural dependency. I'd want to verify that on the depth and landscape probes before claiming it; the derivation only buys me the mechanism, not the win.

Practically it's a pure drop-in: `x * torch.tanh(F.softplus(x))`, no parameters, swap for ReLU/Swish and change nothing else. The one real cost is compute — an extra softplus+tanh versus a single sigmoid for Swish — which is the price for the smoother gradient.

The causal chain, end to end: smooth self-gated units identify the useful shape, so I keep smoothness, a small preserved negative region, unboundedness above, boundedness below, and the form `x·h(x)`; softplus gives a smooth positive inner map with ReLU-like asymptotics; `tanh` squashes it into a gate whose endpoints I checked go from `0` to `1`; the resulting curve `f=x·tanh(softplus(x))` has range `[≈−0.31,∞)` (minimum `≈−0.309` near `x≈−1.19`, found by evaluating it), a shallow non-monotonic negative bump, and identity behavior for large positives; differentiating gives `f′=tanh(s)+x·sech²(s)·σ(x)`, which I verified equals the rewrite `f′=Δ(x)·swish(x)+f(x)/x` away from zero to full numerical precision and at `x=0` via the continuous value `tanh(ln 2)=3/5`; `Δ(x)` (with `Δ(0)=16/25=0.64`) supplies a smooth input-dependent rescaling of gradient flow; the implementation is the parameter-free drop-in `x·tanh(softplus(x))`.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class Mish(nn.Module):
    """Mish: f(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + e^x)).

    Self-gated, smooth (C-inf), non-monotonic; range [~-0.31, inf).
    For x != 0, f' = Delta(x) * swish(x) + f(x)/x, with the
    x = 0 value supplied by continuous extension.
    Parameter-free drop-in for ReLU / Swish.
    """

    def forward(self, x):
        return x * torch.tanh(F.softplus(x))
```
