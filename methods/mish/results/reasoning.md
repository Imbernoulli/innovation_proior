I already have a useful pattern in front of me: hard rectification fixed the saturation problem of sigmoid and tanh, but the newer smooth self-gated units keep the raw input and multiply it by a smooth gate. Swish writes this as `x·σ(βx)`, GELU as `x·Φ(x)`. The problem I want to solve is narrower than inventing an activation from nothing: keep the self-gated shape, keep the smooth negative-side behavior that ReLU lacks, and find a pointwise curve whose derivative gives me a mechanical reason to expect stable optimization.

I pin down the properties I am not willing to give up. The function should be smooth and continuously differentiable everywhere, with no ReLU kink. It should preserve a small amount of negative signal, because a hard zero on the whole negative half-plane is exactly how a unit gets stuck with zero output and zero gradient. It should be unbounded above, so large positive activations do not saturate the way sigmoid and tanh do. It should be bounded below, so the negative side cannot run away. And it should stay self-gated: `f(x)=x·h(x)` for a smooth scalar gate `h`.

So the search is over `h`. Swish picks `h=σ`; GELU picks `h=Φ`. I need the same asymptotic logic: `h(x)→0` as `x→−∞`, so strongly negative inputs are suppressed gently, and `h(x)→1` as `x→+∞`, so the activation recovers the identity on large positives. Softplus is a natural inner primitive here because `softplus(x)=log(1+eˣ)` is smooth, positive, close to `0` on the far negative side, and close to `x` on the far positive side. If I feed that through `tanh`, I get a gate that starts near `0`, rises smoothly, and tends exactly to `1` without an extra scale factor. Other smooth transforms of `softplus(x)` or `eˣ` can be screened as neighboring candidates, but `tanh(softplus(x))` is the clean one: the asymptotes are right, the primitives are standard, and the derivative should stay tractable.

`f(x) = x·tanh(softplus(x)) = x·tanh(ln(1+eˣ)).`

This is the candidate I want to carry forward. The gate is `h(x)=tanh(softplus(x))`: feed `x` through softplus, then squash that nonnegative value through `tanh`. The output is still the input times a smooth gate, but the gate is not the sigmoid gate Swish uses.

Check it has the properties I demanded. Smooth and continuously differentiable everywhere — softplus and tanh are both `C∞`, so the composition is, and there's no kink anywhere (unlike ReLU). Asymptotics: as `x→+∞`, `softplus(x)→x→∞`, `tanh(∞)→1`, so `f→x` — recovers identity, unbounded above, no positive saturation. As `x→−∞`, `softplus(x)→0⁺`, `tanh(0⁺)→0⁺`, so `f→0` — suppresses strong negatives. In between, for slightly negative `x` the gate is small but the product `x·h(x)` dips *below zero*: a shallow negative bump, so it's non-monotonic and preserves a little negative signal, killing the dying-ReLU precondition by design. The minimum of the negative bump sits around `≈ −0.31`, so the range is `[≈−0.31, ∞)` — bounded below, unbounded above. Same qualitative silhouette as Swish, slightly different curvature in the negative region.

The place to look for a mechanism is the first derivative, because backpropagation multiplies by this scalar at every unit. Differentiate `f=x·tanh(softplus(x))`. Let `s=softplus(x)`, so `s′=σ(x)` because the derivative of softplus is sigmoid. With `f=x·tanh(s)`,

`f′(x) = tanh(s) + x·sech²(s)·s′ = tanh(s) + x·sech²(s)·σ(x).`

Rewrite the second piece to expose what is inside. The factor `x·σ(x)` is Swish with `β=1`, and `tanh(s)=f(x)/x` whenever `x≠0`. So, for `x≠0`,

`f′(x) = sech²(softplus(x))·x·σ(x) + f(x)/x = Δ(x)·swish(x) + f(x)/x,`

with `Δ(x)=sech²(softplus(x))` and `swish(x)=x·σ(x)`. At `x=0`, the displayed `f(x)/x` term has to be read by continuous extension: `f(x)/x→tanh(softplus(0))=tanh(log 2)=3/5`, while `swish(0)=0`, so the rewritten form gives `3/5`, exactly matching the direct derivative. The identity is therefore literal away from zero and exact everywhere once that removable singularity is filled in.

Now `Δ(x)` has a precise role. Since `softplus(x)` is nonnegative and increasing, `sech²(softplus(x))` is positive, near `1` on the far negative side, equal to `0.64` at zero, and decays toward `0` on the far positive side. It is not a literal inverse Hessian, but in backpropagation it behaves like an input-dependent scaling inside the activation derivative: the Swish-shaped term is continuously modulated instead of passed at one fixed scale, while the gate term `f(x)/x` carries the activation toward derivative `1` on large positives. That is the preconditioning analogy I can defend: the derivative supplies a smooth local rescaling of gradient flow, not a hard kink and not a constant leak.

This lines up with the smoothness argument. A smoother per-unit map produces a smoother output landscape; ReLU builds piecewise-linear facets because every unit has a kink, while this curve is smooth everywhere. The derivative analysis gives the same story from the optimizer's side: the upstream gradient is multiplied by a smooth scalar that changes continuously with the preactivation, with a shallow negative region rather than a dead half-plane.

Why not stop at Swish, since it is already smooth and self-gated? Because the gate `σ(x)` gives me only one smooth self-gating shape, while `tanh(softplus(x))` keeps the same endpoints and changes the derivative structure in a way I can inspect. The Swish relationship is not "same function with a new name"; it appears inside one term of the derivative, multiplied by `Δ(x)`, with the gate term added separately. That gives me a specific hypothesis to validate with depth, noise, initializer, and landscape probes: if the rescaling story is right, the gain should show up as stability and smoother optimization rather than as a new architectural dependency.

Practically it's a pure drop-in: `x * torch.tanh(F.softplus(x))`, no parameters, swap for ReLU/Swish and change nothing else. The only cost is that it's a touch more expensive than Swish (an extra softplus+tanh vs. a sigmoid), which is the price for the smoother gradient.

The causal chain, end to end: smooth self-gated units identify the useful shape, so I keep smoothness, a small preserved negative region, unboundedness above, boundedness below, and the form `x·h(x)`; softplus gives a smooth positive inner map with ReLU-like asymptotics; `tanh` turns that into a gate tending from `0` to `1`; the resulting curve `f=x·tanh(softplus(x))` has range `[≈−0.31,∞)`, a shallow non-monotonic negative bump, and identity behavior for large positives; differentiating gives the exact identity `f′=tanh(s)+x·sech²(s)·σ(x)`, or `f′=Δ(x)·swish(x)+f(x)/x` away from zero with the removable `x=0` value filled in; the derivative supplies a smooth input-dependent rescaling of gradient flow; the implementation is the parameter-free drop-in `x·tanh(softplus(x))`.

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
