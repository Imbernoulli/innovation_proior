Every few months someone hand-designs a new activation — leaky this, exponential that — argues from a list of properties they think matter (don't die on the negative side, saturate gently, center the mean), and it gives a bump on one model and nothing on the next, so everyone goes back to ReLU. The pattern of failure is the clue: the inconsistency means we don't actually know which properties matter, so designing from a property wishlist is guessing. I'd rather not guess. Let the validation signal choose scalar functions directly, then inspect the shapes it prefers and ask which "essential" ReLU properties were actually necessary.

Why scalar functions specifically — one real input to one real output? Because that's the only family that's a literal drop-in: swap it in for ReLU anywhere, no shape changes, no architecture surgery, the network is identical except for the pointwise map. If the search returned something many-to-one or input-shape-dependent I couldn't just replace ReLU with it. Keeping the contract "scalar in, scalar out" makes any discovery immediately usable.

So I need a search space of scalar functions that's expressive enough to contain something genuinely new but structured enough to actually search. A flat enumeration of arbitrary expressions is hopeless. Borrow the trick that worked for searching optimizer update rules: build functions by composing a small library of primitives. Define one core unit as `b(u₁, u₂)`: choose two sources, pass them through unary functions `u₁` and `u₂`, then combine those two scalar outputs with a binary function `b`. If both sources are the raw preactivation, that is `b(u₁(x), u₂(x));` if I stack units, later units can read earlier scalar outputs too. Unaries are things like `x, −x, x², √x, eˣ, sin, σ, tanh, max(x,0), βx, β` (where `β` is a trainable scalar or per-channel scalar); binaries are `x₁+x₂, x₁·x₂, x₁−x₂, max(x₁,x₂), x₁/x₂`, and so on. Because I want scalar-in/scalar-out, every expression stays pointwise. Stack `n` core units and the space gets large fast — one unit is enumerable, but a few units blows up to around `10¹²` candidates.

Two regimes, then, for the search algorithm. When the space is small (a single core unit, a restricted primitive set), just *exhaustively enumerate* — train a child network with each candidate and rank by validation accuracy. When it's large, I can't enumerate, so use an RNN controller: at each timestep it emits one token of the function (this unary, that binary, this leaf), feeds the choice back autoregressively, and keeps going until a full function string is assembled. The controller is trained with policy-gradient RL — PPO, with an exponential-moving-average of past rewards as the baseline to cut variance — and the *reward* is the validation accuracy of a child network trained with the candidate activation. High-accuracy functions get reinforced; the controller drifts toward good regions of the space.

The expensive part is obvious: every single candidate costs a full child-network training run. So I keep the child small and cheap — ResNet-20 on CIFAR-10, ~10K steps — and parallelize hard: the controller proposes a batch of candidates onto a queue, worker machines pull them, train, and report back validation accuracy, which is aggregated to update the controller. There's a real risk that functions tuned on a tiny child net won't transfer to big models, and I have to test that outside the search loop; but cheap-child-then-validate-large is the only way the search is affordable.

Run it, and look at what comes back rather than what I expected. The useful candidates are not the tangled many-unit expressions; they are mostly one- or two-unit curves. That makes sense once I stare at them: extra expression depth creates wilder derivatives, discontinuities from unsafe divisions, and strange oscillations that a deep network has to optimize through. The stronger pattern is that many good candidates keep the raw preactivation alive until the final combine, something like `b(x, g(x))`. ReLU fits that same template, `max(x, 0)`, with `g(x)=0` and `b=max`; the clean path for `x` seems worth preserving even when I stop hand-designing the curve. Periodic pieces can decorate `x`, and division only looks sane when the denominator is bounded away from zero or vanishes with the numerator, but the promising family is simpler: keep `x`, learn or choose a soft gate for it, and avoid explosive algebra.

So follow that template back to ReLU itself. ReLU can be written as `x·1(x>0)`: the input multiplied by a hard gate. A hard step is exactly the kind of non-smooth hand choice I wanted to stop assuming — but it suggests a move. If I keep the raw `x` and the multiply, but replace the discontinuous gate with a smooth bounded one, what do I get? The smooth bounded substitute for a step is a sigmoid, and if the gate should still respond to the preactivation's sign and scale, it should see `βx` rather than a fixed cutoff. In the core-unit notation this is only one unit: `u₁(x)=x`, `u₂(x)=σ(βx)`, `b(a,b)=a·b`, giving

`f(x) = x · σ(βx),`

with `σ` the logistic sigmoid and `β` a constant or trainable scalar. The input is gated by a sigmoid of a scaled copy of itself — a self-gated unit. I'll call it Swish. With `β=1` this is `x·σ(x)`, the Sigmoid-weighted Linear Unit / SiLU that Elfwing et al. had already used for RL function approximation; here it falls out of the activation search as one core unit, now with an explicit slope knob `β`.

Before I trust this I should check it actually interpolates the things I claimed it would. The construction was supposed to be a *soft* ReLU, so the hard-gate limit has to come back. For positive `β`, as `β → ∞`, `σ(βx)` should pinch to the step `1(x>0)`. Let me put numbers on it at a few points. At `x=−2`: `β=1` gives `−0.238`, `β=5` gives `−0.00009`, `β=20` gives `−2·10⁻⁷` — collapsing to ReLU's `0`. At `x=0.5`: `β=1` gives `0.311`, `β=5` gives `0.462`, `β=20` gives `0.4999`, approaching ReLU's `0.5`. At `x=2`: `β=20` already gives `2.0000`. So `β→∞` does recover `max(x,0)` numerically, not just in the limit-argument. The other end: as `β → 0`, `σ(βx) → 1/2`, so I'd expect `x/2`. Checking `β=0.01`: at `x=−2` I get `−0.990` (vs `x/2=−1.0`), at `x=2` I get `1.010` (vs `1.0`), at `x=±0.5` I get `±0.2506` (vs `±0.25`). It is converging to the line `x/2`. So one formula stretches continuously from a line, through `β=1`/SiLU, to a soft ReLU, and a trainable `β` lets each channel pick where on that family it sits instead of my fixing it.

Now look at the shape between those extremes and what it overturns. Like ReLU, `f` is unbounded above (`σ→1`, so `f≈x` for large positive `x`), so it does not saturate on the positive side. Unlike ReLU it is smooth, and — this I did not put in by hand — it is non-monotonic. For `x` slightly negative, `x` is negative while the gate `σ(βx)` is still positive, so the product is negative; far into the negatives the gate shrinks faster than `x` grows in magnitude, pulling `f` back toward `0`. So there is a dip. How deep, and where? Scanning `f` over `x<0` at `β=1`, the minimum is `f≈−0.278` at `x≈−1.28`, and by `x=−10` it has returned to `−4.5·10⁻⁴`. So the undershoot is real but small and self-limiting, not a runaway negative tail. Whether that controlled dip is a useful degree of freedom or a defect I can't settle from the shape alone — that is exactly what the validation signal, not my intuition, is being asked to decide.

The derivative is where I most expected the ReLU dogma to either hold or crack, so I differentiate. With `s=σ(βx)` and `f=xs`, the product rule gives `f′ = s + x·βs(1−s) = s + βx s(1−s)`, and substituting `xs=f` rewrites it as `βf + s(1−βf)`. I want to be sure I didn't slip a `β`, so I check the two forms against a numeric derivative. At `x=1, β=1`: central difference gives `0.92767`; `s+βxs(1−s)` gives `0.92767`; `βf+s(1−βf)` gives `0.92767`. At `x=−1, β=2`: numeric `−0.09078`, both closed forms `−0.09078`. Across a grid of `x∈{−3,−1,−0.5,0,0.5,1,3}` and `β∈{0.5,1,2}` all three agree to 1e-4. So

`f′(x) = σ(βx) + βx·σ(βx)(1 − σ(βx)) = βf(x) + σ(βx)(1 − βf(x))`

is right, and `β` is the knob that sets how fast `f′` goes to `0` on the far-negative side and to `1` on the far-positive side. The number worth pinning down is whether `f′` stays *below* `1` over the active range at `β=1` — because ReLU's selling point is its exact unit slope on the whole positive half. Scanning `f′` from `0` up: `f′(0.5)=0.740`, `f′(1.0)=0.928`, `f′(1.27)=0.998`, and it first exceeds `1` between `x=1.27` (`0.9981`) and `x=1.28` (`1.0003`). So for all of `0<x<1.28` the slope is under `1`; over much of the active range this unit does *not* preserve gradients at unit scale the way ReLU does on its positive half. (And the crossing at `x≈1.28` is the same magnitude as the dip location `x≈−1.28` on the other side — both are roots of `f′` related by the symmetry of `s(1−s)`, which is a small reassurance the algebra is self-consistent rather than a coincidence I should chase.) That separates two ideas I had been conflating: a non-saturating positive *tail* is still useful, but an exactly unit derivative over *all* positive inputs is not something the search space was forced to reproduce. In residual networks there is already an identity path carrying gradients, so the activation need not shoulder that whole burden alone — which is at least a plausible reason a sub-unit slope here is survivable, though I'd want the large-model validation to confirm it rather than take my word for it.

A couple of implementation cautions fall straight out of the shape. It is a one-line pointwise map: `x * sigmoid(beta * x)`. But if BatchNorm precedes it, the BatchNorm scale parameter should stay learnable; ReLU's positive homogeneity can make that scale look redundant, while this smooth self-gated curve is not scale-equivariant in the same way. And since the derivative is often smaller than ReLU's unit positive slope, the learning rate that was tuned around ReLU is a parameter I should re-tune rather than blindly reuse.

The causal chain, end to end: hand-designed activations give inconsistent gains because we do not know which properties matter, so I search scalar drop-in functions instead of guessing; I build that space from unary/binary core units `b(u₁,u₂)`, enumerate small spaces, and use a PPO-trained RNN controller with child-network validation accuracy as the reward for large spaces; the useful shapes stay simple, preserve a raw `x` path, and avoid unstable division; writing ReLU as `x·1(x>0)` points to a soft self-gate, which one core unit realizes as `u₁=x`, `u₂=σ(βx)`, `b(a,b)=a·b`, giving `x·σ(βx)`; `β→∞` gives ReLU, `β→0` gives `x/2`, and `β=1` gives SiLU; the derivative `f′=βf+σ(βx)(1−βf)` shows why the searched shape keeps the non-saturating positive tail without requiring ReLU's exact unit positive slope; the implementation is the pointwise expression, with BatchNorm scale and learning rate treated as real knobs.

```python
import torch
import torch.nn as nn


class Swish(nn.Module):
    """Swish / SiLU: f(x) = x * sigmoid(beta * x).

    beta=1 (constant) == SiLU; a trainable beta lets each channel adjust
    the sigmoid gate sharpness.
    """

    def __init__(self, num_channels=None, trainable_beta=False, beta_init=1.0):
        super().__init__()
        if trainable_beta:
            shape = (num_channels,) if num_channels is not None else (1,)
            self.beta = nn.Parameter(torch.full(shape, float(beta_init)))
            self.trainable = True
        else:
            self.register_buffer("beta", torch.tensor(float(beta_init)))
            self.trainable = False

    def forward(self, x):
        if self.trainable and self.beta.numel() > 1:
            # per-channel beta over NCHW-style tensors
            shape = [1] * x.dim()
            shape[1] = -1
            b = self.beta.view(*shape)
        else:
            b = self.beta
        return x * torch.sigmoid(b * x)


class SiLU(nn.Module):
    """SiLU == Swish-1: f(x) = x * sigmoid(x)."""

    def forward(self, x):
        return x * torch.sigmoid(x)
```
