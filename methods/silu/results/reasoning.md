Every few months someone hand-designs a new activation — leaky this, exponential that — argues from a list of properties they think matter (don't die on the negative side, saturate gently, center the mean), and it gives a bump on one model and nothing on the next, so everyone goes back to ReLU. The pattern of failure is the clue: the inconsistency means we don't actually know which properties matter, so designing from a property wishlist is guessing. I'd rather not guess. Let the validation signal choose scalar functions directly, then inspect the shapes it prefers and ask which "essential" ReLU properties were actually necessary.

Why scalar functions specifically — one real input to one real output? Because that's the only family that's a literal drop-in: swap it in for ReLU anywhere, no shape changes, no architecture surgery, the network is identical except for the pointwise map. If the search returned something many-to-one or input-shape-dependent I couldn't just replace ReLU with it. Keeping the contract "scalar in, scalar out" makes any discovery immediately usable.

So I need a search space of scalar functions that's expressive enough to contain something genuinely new but structured enough to actually search. A flat enumeration of arbitrary expressions is hopeless. Borrow the trick that worked for searching optimizer update rules: build functions by composing a small library of primitives. Define one core unit as `b(u₁, u₂)`: choose two sources, pass them through unary functions `u₁` and `u₂`, then combine those two scalar outputs with a binary function `b`. If both sources are the raw preactivation, that is `b(u₁(x), u₂(x));` if I stack units, later units can read earlier scalar outputs too. Unaries are things like `x, −x, x², √x, eˣ, sin, σ, tanh, max(x,0), βx, β` (where `β` is a trainable scalar or per-channel scalar); binaries are `x₁+x₂, x₁·x₂, x₁−x₂, max(x₁,x₂), x₁/x₂`, and so on. Because I want scalar-in/scalar-out, every expression stays pointwise. Stack `n` core units and the space gets large fast — one unit is enumerable, but a few units blows up to around `10¹²` candidates.

Two regimes, then, for the search algorithm. When the space is small (a single core unit, a restricted primitive set), just *exhaustively enumerate* — train a child network with each candidate and rank by validation accuracy. When it's large, I can't enumerate, so use an RNN controller: at each timestep it emits one token of the function (this unary, that binary, this leaf), feeds the choice back autoregressively, and keeps going until a full function string is assembled. The controller is trained with policy-gradient RL — PPO, with an exponential-moving-average of past rewards as the baseline to cut variance — and the *reward* is the validation accuracy of a child network trained with the candidate activation. High-accuracy functions get reinforced; the controller drifts toward good regions of the space.

The expensive part is obvious: every single candidate costs a full child-network training run. So I keep the child small and cheap — ResNet-20 on CIFAR-10, ~10K steps — and parallelize hard: the controller proposes a batch of candidates onto a queue, worker machines pull them, train, and report back validation accuracy, which is aggregated to update the controller. There's a real risk that functions tuned on a tiny child net won't transfer to big models, and I have to test that outside the search loop; but cheap-child-then-validate-large is the only way the search is affordable.

Run it, and look at what comes back rather than what I expected. The useful candidates are not the tangled many-unit expressions; they are mostly one- or two-unit curves. That makes sense once I stare at them: extra expression depth creates wilder derivatives, discontinuities from unsafe divisions, and strange oscillations that a deep network has to optimize through. The stronger pattern is that many good candidates keep the raw preactivation alive until the final combine, something like `b(x, g(x))`. ReLU fits that same template, `max(x, 0)`, with `g(x)=0` and `b=max`; the clean path for `x` seems worth preserving even when I stop hand-designing the curve. Periodic pieces can decorate `x`, and division only looks sane when the denominator is bounded away from zero or vanishes with the numerator, but the promising family is simpler: keep `x`, learn or choose a soft gate for it, and avoid explosive algebra.

ReLU can be written as `x·1(x>0)`: the input multiplied by a hard gate. A hard step is exactly the kind of non-smooth hand choice I wanted to stop assuming. The smooth bounded substitute is a sigmoid, and if the gate should still depend on the preactivation's sign and scale, the gate should see `βx`. In the core-unit notation this is only one unit: choose `u₁(x)=x`, choose `u₂(x)=σ(βx)`, and choose `b(a,b)=a·b`. That lands on the strikingly plain candidate:

`f(x) = x · σ(βx),`

where `σ` is the logistic sigmoid and `β` is a constant or trainable scalar, often per channel. The input is gated by a sigmoid of a scaled copy of itself — a self-gated unit. Call it Swish. With `β=1`, it is exactly the Sigmoid-weighted Linear Unit / SiLU, `x·σ(x)`, that Elfwing et al. had already used for RL function approximation; the search has put that self-gating shape in the broader activation-search setting and added the slope knob.

The `β` knob is the whole story. For positive `β`, as `β → ∞`, `σ(βx)` becomes the step `1(x>0)`, so `x·σ(βx) → x·1(x>0) = max(x,0)`: Swish becomes ReLU. As `β → 0`, `σ(βx) → 1/2`, so `x·σ(βx) → x/2`: Swish becomes a scaled linear function. So the same formula connects a line to a soft version of ReLU, and a trainable `β` lets each channel adjust the gate sharpness instead of fixing it by hand.

Look at the shape and what it overturns. Like ReLU, Swish is unbounded above, so it does not saturate on the positive side. Unlike ReLU, it is smooth and non-monotonic: for `x` slightly negative, `x·σ(βx)` dips below zero before returning toward 0 as `x` goes far negative. That small negative dip is not something I would have gotten from the usual monotonic-activation checklist, but the formula makes it unavoidable: negative `x` times a positive gate is negative, and the gate shrinks toward zero only gradually. The question becomes whether this controlled undershoot is a useful degree of freedom rather than a defect.

Differentiate it, because the derivative is where the ReLU dogma cracks. With `f = x·σ(βx)`,

`f′(x) = σ(βx) + βx·σ(βx)(1 − σ(βx)) = βf(x) + σ(βx)(1 − βf(x)).`

`β` controls how fast `f′` approaches 0 on the far-negative side and 1 on the far-positive side. At `β=1`, the derivative's magnitude is below 1 until roughly `x≈1.28`; over much of the active range this unit does not preserve gradients at unit scale the way ReLU does on its positive half. That matters because it separates two ideas I used to conflate: a non-saturating positive tail is still useful, but an exactly unit derivative over all positive inputs is not forced by the search space. In residual networks, there is already an identity path for gradients, so the activation does not have to carry the whole burden by itself.

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
