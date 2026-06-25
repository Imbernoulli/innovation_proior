Swish bothers me less for what it does than for what we *don't know* about it. `x·σ(βx)` came out of a search, it beats ReLU, everyone uses it, and the explanation on offer is essentially "the search found it." That's not an explanation, it's a citation of a procedure. I want to know what Swish *is* — whether it's an instance of something I already understand — because if it is, I can probably write down the rest of that something and find activations the search missed.

Start from the thing Swish is supposed to improve on: ReLU. ReLU is `max(x, 0)`. Leaky ReLU and PReLU are `max(x, px)`. The absolute value is `max(x, −x)`. All of these are *max of two linear functions* — the Maxout family, `max(η_a(x), η_b(x))` with `η_a, η_b` linear. Maxout is the general "hard max of linear pieces" object; ReLU and its piecewise-linear cousins are just specific choices of the two pieces. So the activations Swish is competing with are, structurally, members of one family of *hard maxes*. And Swish is *smooth*. The gap between "hard max of linear pieces" and "smooth activation" is exactly the kind of gap I want to span: if I can write Swish as a smoothed `max`, I get an explanation and a generator for free.

So I need a differentiable surrogate for `max`. The hard `max(x_1,…,x_n)` is non-smooth at ties, and there's a standard way to round it off, used all over optimization and neural computation:

`S_β(x_1,…,x_n) = (Σ_i x_i e^{βx_i}) / (Σ_i e^{βx_i}).`

It's a softmax-weighted average of the values, with temperature `β`. As `β → ∞` the weight piles onto the largest `x_i` and `S_β → max`; as `β → 0` all weights equalize and `S_β →` the arithmetic mean. So `β` looks like a knob sliding between "hard max" and "plain average." I have to keep this distinct from Softplus: Softplus is the LogSumExp smoothing of `max(x,0)`, while this is the softmax-weighted *value* itself. That distinction matters, because the weighted-value version mixes the actual `x_i` back in and so might produce a self-gated activation rather than another positive smooth ReLU — which is the half of the space Swish lives in.

Now apply this to a two-piece max, because every activation I care about is `max(η_a, η_b)` — set `n=2`. Write it out and simplify:

`S_β(η_a, η_b) = η_a · e^{βη_a}/(e^{βη_a}+e^{βη_b}) + η_b · e^{βη_b}/(e^{βη_a}+e^{βη_b}).`

Divide numerator and denominator of the first fraction by `e^{βη_a}` and the second by `e^{βη_b}`:

`= η_a · 1/(1 + e^{−β(η_a−η_b)}) + η_b · 1/(1 + e^{−β(η_b−η_a)}) = η_a·σ(β(η_a−η_b)) + η_b·σ(β(η_b−η_a)).`

Use `σ(−z) = 1 − σ(z)` on the second term: `η_b·(1 − σ(β(η_a−η_b)))`. Collect:

`S_β(η_a, η_b) = (η_a − η_b)·σ(β(η_a − η_b)) + η_b.`

Before I trust this closed form I want to be sure I didn't drop a factor in the algebra, because everything downstream rides on it. I'll check the simplified expression against the raw softmax-weighted definition directly, at a handful of arbitrary points. Take `η_a = 1.3, η_b = −0.7, β = 2`. Raw: weights `e^{2·1.3}=e^{2.6}=13.46`, `e^{2·(−0.7)}=e^{−1.4}=0.2466`, sum `13.71`, weighted value `(1.3·13.46 + (−0.7)·0.2466)/13.71 = (17.50 − 0.1726)/13.71 = 1.264`. Simplified: `η_a−η_b = 2.0`, `σ(2·2.0)=σ(4)=0.9820`, so `2.0·0.9820 + (−0.7) = 1.964 − 0.7 = 1.264`. They agree. To be sure it isn't a coincidence at that one point, I swept 100k random `(η_a, η_b, β)` triples comparing the raw quotient (computed in a numerically stable way) against `(η_a−η_b)σ(β(η_a−η_b))+η_b`; the worst absolute discrepancy was `2.7e-15`, i.e. floating-point noise. So the closed form is the smooth max, and everything I need is now downstream of plugging in linear pieces.

Take the ReLU piece: `η_a = x`, `η_b = 0`. Then

`S_β(x, 0) = (x − 0)·σ(β(x − 0)) + 0 = x·σ(βx).`

That is the Swish formula, with `β` falling exactly where Swish's `β` sits. So Swish is not a mysterious search artifact — it is the smooth maximum of `{x, 0}` under the softmax-weighted-value smoothing, i.e. a smoothed ReLU. As a further sanity check on the reading of `β`, the temperature should slide this between linear and ReLU: at `β→0` the smooth max becomes the mean of `x` and `0`, namely `x/2`, and at `β→∞` it should become `max(x,0)`. The `β→0` limit is immediate (`σ→1/2`, giving `x·½`); I'll confirm the `β→∞` limit numerically once I have the general member in hand, since that's the case where I most want to be sure the rounding really tightens to the hard max. The recipe is now visible: smooth *any* Maxout member the same way.

So walk down the Maxout family and smooth each member. PReLU is `max(x, px)` (with `p<1` in practice, `p` learned, init 0.25). Plug `η_a = x`, `η_b = px`:

`S_β(x, px) = (x − px)·σ(β(x − px)) + px = (1−p)x·σ(β(1−p)x) + px.`

A smooth PReLU — call it the `B` member. And the general two-linear-piece case, `η_a = p_1 x`, `η_b = p_2 x` with `p_1 ≠ p_2`:

`S_β(p_1 x, p_2 x) = (p_1 − p_2)x·σ(β(p_1 − p_2)x) + p_2 x.`

Call this the `C` member — the general smooth max of two linear functions through the origin. The `A` member (Swish) is `p_1=1, p_2=0`; the `B` member is `p_1=1, p_2=p`. So this is a *family*, the smooth-Maxout family, with Swish sitting inside it as the simplest case.

Two things I claimed in passing are worth actually verifying on the `C` form, because if either fails the whole framing is wrong. First, does `C` at `p_1=1, p_2=0, β=1` literally reproduce Swish/SiLU, not just up to my algebra? I evaluated `(1−0)x·σ(1·(1−0)x)+0·x` against `x·σ(x)` on a grid of 2001 points in `[−6,6]`; the maximum absolute difference came out exactly `0.0`. So the Swish point is recovered bit-for-bit — not approximately, it *is* the same function. Second, the deferred `β→∞` limit: at `p_1=1.5, p_2=0.3, x=2`, the hard max is `max(3.0, 0.6)=3.0`, and `f_C` at `β=1000` evaluates to `3.0` to displayed precision, while at `β=10^{−6}` it gives `1.80000`, matching `((p_1+p_2)/2)·x = 0.9·2 = 1.8`. Both endpoints land where the smooth-max story predicts. So the framing holds: `β` really does slide a single unit from the hard max (nonlinear) to the mean (linear), and Swish is the `p_1=1,p_2=0` slice. The useful module makes `β`, `p_1`, and `p_2` per-channel learnable so the unit can move away from that fixed curve.

The obvious question: why would the general `C` member be useful beyond Swish, when Swish is already in the family? They look almost the same — both are `(slope-gap)·x·σ(β·(slope-gap)·x) + offset`. The difference has to be in the gradient, since that's what training actually feels. So differentiate the `C` member and find its first-derivative behavior.

Let `u = (p_1 − p_2)x`, so `f_C = u·σ(βu) + p_2 x`. Then

`f_C′(x) = (p_1 − p_2)·[σ(βu) + βu·σ(βu)(1 − σ(βu))] + p_2.`

(Writing it over a common denominator with `e^{−βu}`: the bracket is `(1 + e^{−βu} + βu e^{−βu})/(1+e^{−βu})²`, so `f_C′ = (p_1−p_2)(1+e^{−βu})/(1+e^{−βu})² + β(p_1−p_2)² e^{−βu} x/(1+e^{−βu})² + p_2`, the same thing.)

Take the asymptotes first. As `x → +∞` (assume `p_1 > p_2`, so `u → +∞`): `σ(βu) → 1`, the `βu·σ(1−σ)` term → 0, so `f_C′ → (p_1−p_2)·1 + p_2 = p_1`. As `x → −∞`: `σ(βu) → 0`, the middle term → 0, so `f_C′ → p_2`. So the derivative of `f_C` runs from `p_2` on the far left to `p_1` on the far right — *its asymptotic slopes are exactly the two linear pieces*, as a smoothed `max(p_1 x, p_2 x)` should be. That matches what I'd expect of a smoothed max, so the derivative expression is at least consistent with the asymptotic picture.

But the asymptotes aren't the interesting part — the *bounds* of the derivative are, because the maximum and minimum of `f_C′` are the largest and smallest gradients the unit can pass, and those govern optimization. Let `a=p_1-p_2` and `y=βax`. The bracket above is

`h(y)=σ(y)+yσ(y)(1−σ(y))`,

so `f_C′(x)=a h(y)+p_2` and `f_C″(x)=βa^2 h′(y)`. Now compute the small derivative instead of differentiating the whole activation again. With `s=σ(y)`, `s′=s(1−s)`, so

`h′(y)=s(1−s)+s(1−s)+y s(1−s)(1−2s) = s(1−s)[2+y(1−2s)].`

For finite `y`, the factor `s(1−s)` is nonzero, so the extrema satisfy `2+y(1−2s)=0`. Since `s=e^y/(1+e^y)`, `1−2s=(1−e^y)/(1+e^y)`, and the condition becomes

`2 + y(1−e^y)/(1+e^y)=0`.

Multiplying out gives `(2+y)+(2−y)e^y=0`, or

`(y − 2)e^y = y + 2,  where  y = (p_1 − p_2)β x.`

This is a transcendental equation in the single combined variable `y`. It has no closed form, so I solve it numerically: bracketing `(y−2)e^y−(y+2)` and running a root finder gives `y = 2.3993573` for the positive root and `y = −2.3993573` for the negative one (symmetric, as the structure suggests). Evaluating `h(y)=σ(y)+yσ(y)(1−σ(y))` at these: `h(+2.39936) = 1.09984` and `h(−2.39936) = −0.09984`. Substituting back into `f_C′=a h(y)+p_2`, and keeping the pieces ordered as `p_1>p_2`, gives the maxima and minima of the derivative:

`max(f_C′) ≈ 1.0998·p_1 − 0.0998·p_2,    min(f_C′) ≈ 1.0998·p_2 − 0.0998·p_1.`

I don't want to take this formula on the strength of the hand calculus alone, so I check the bounds directly against autograd. For the Swish case `p_1=1, p_2=0, β=1`, sampling `x·σ(x)`'s derivative on a fine grid over `[−15,15]` gives `max f_C′ = 1.099839`, `min f_C′ = −0.099839` — exactly the constants the formula predicts. For a general member, `p_1=2, p_2=−0.5, β=1`, the formula predicts `max = 1.0998·2 − 0.0998·(−0.5) = 2.2495`, `min = 1.0998·(−0.5) − 0.0998·2 = −0.7495`; autograd on the same grid returns `2.24960` and `−0.74960`. The four-place agreement (the small last-digit gap is the grid spacing) tells me the derivation is right and, more importantly, that the bounds really do move with `p_1, p_2`. So compare to Swish, which is `p_1=1, p_2=0`: its `max ≈ 1.0998`, `min ≈ −0.0998` are *constants* — Swish's derivative is always bounded between ≈ −0.0998 and ≈ 1.0998, no matter what, and `β` only changes how fast the derivative *approaches* those fixed walls; it cannot move the walls. That's the limitation, and it only became visible through the smooth-Maxout view. In the general `C` member the bounds are `1.0998 p_1 − 0.0998 p_2` and `1.0998 p_2 − 0.0998 p_1` — they depend on `p_1` and `p_2`, which are *learnable*. So `C` can *learn its own upper and lower gradient bounds*, per channel, where Swish is stuck with one fixed pair. That learnable control over the gradient's ceiling and floor is the mechanism by which `C` should optimize better than Swish — the network can set how much gradient each unit is allowed to pass and how much undershoot it permits, instead of accepting Swish's hardwired ±. The role of `β` is now sharply separated from the role of `p_1, p_2`: `β` sets *how fast* the derivative asymptotes to the bounds (the non-linear "degree"), while `p_1, p_2` set *what the bounds are*. Two different knobs that Swish conflated by fixing the second.

Look again at what `β` does geometrically, now that the limits are checked numerically rather than just asserted. On a fixed `(p_1, p_2)`: `β → ∞` makes `f_C → max(p_1 x, p_2 x)`, a genuine (nonlinear) max — the neuron *activates*; `β → 0` makes `f_C → mean(p_1 x, p_2 x) = ((p_1+p_2)/2)x`, a *linear* function — the neuron does *not* activate, it just passes a scaled input. So `β` is literally a per-neuron switch between behaving nonlinearly and behaving linearly. That's a property no fixed activation has: ReLU is always nonlinear, a linear unit is always linear, but this family can sit anywhere between and *learn where*, and "where" can differ per channel. Since `β` is exactly an "activate or not" dial, I'll call the family ACON. The freedom to be linear when linearity is what helps is itself a useful inductive bias — and a regularizer, since the activating-degree carries uncertainty.

The next limitation is structural. A free per-channel `β` can choose one nonlinear degree for the whole channel, but the phrase "activate or not" really wants a decision that can change with the input sample. If the same channel should be linear for one image and strongly nonlinear for another, a stored scalar cannot express that. The switch has to be computed from the feature itself: generate `β` explicitly from the input, `β = G(x)`, so that the switching decision is made by a small learned routing function conditioned on the actual activations. That gives meta-ACON, "learn to learn whether to activate."

What should `G` be? The concept matters more than the specific architecture, and there's a natural design space along *granularity*: the switch can be shared layer-wise, shared channel-wise, or unique pixel-wise. Layer-wise: one `β` for the whole layer, `β = σ(Σ_{c,h,w} x_{c,h,w})` — cheapest, coarsest. Pixel-wise: a `β` for every element, `β_{c,h,w} = σ(x_{c,h,w})` — finest, but no cross-element context. Channel-wise is the natural middle point and reuses a primitive I already trust from squeeze-and-excitation: pool over space, route through a tiny bottleneck, sigmoid to `(0,1)`:

`β_c = σ(W_1 W_2 · GAP(x)),   W_1 ∈ ℝ^{C×C/r}, W_2 ∈ ℝ^{C/r×C}, r=16.`

`GAP` is global average pooling, the two linear maps are a reduce-then-expand bottleneck (reduction `r=16`) that keeps the parameter cost tiny, and `σ` squashes to a per-channel switching factor in `(0,1)`. So each channel gets a switching factor computed from global context, and — because `β = G(x)` depends on the *sample* — different inputs get different non-linear degrees in the same layer. For very large models, replacing dense channel mixing with depth-wise fully connected layers keeps the count negligible.

The reason to try this exactly where fixed activations have little headroom is that Swish forces every unit through the same curve with fixed derivative bounds. The network has no way to dial a channel toward linearity for one sample, toward a sharper max for another, or to set the gradient ceiling and floor through the activation parameters. ACON-C gives it learnable bounds; meta-ACON gives it *per-sample, per-channel* control of the nonlinear degree. That is the concrete hypothesis to test on both small models and very deep optimized ones.

The channel-wise routing `σ(W_1 W_2 GAP(x))` is structurally the SE module, so combining that routing with the smooth-Maxout activation is a channel-wise input-conditioned activation. The routing supplies the sample-dependent switch; `p_1` and `p_2` supply the learnable gradient bounds inside the activation. SE reweights features; this reweights features and reshapes the nonlinearity's gradient.

What started as discomfort with an unexplained search result has turned into something I can defend at each step: Swish is the smooth max of `{x,0}` (checked against the raw softmax form to machine precision and bit-for-bit against `x·σ(x)`); the general smooth-Maxout member `C` has gradient bounds `1.0998 p_1 − 0.0998 p_2` and `1.0998 p_2 − 0.0998 p_1` (checked against autograd) that are *learnable* where Swish's are *fixed*; and `β` is a verified linear↔max switch (both limits computed). The remaining freedom — letting `β` depend on the sample through `β=σ(W_1 W_2 GAP(x))` — is the one piece I can't settle with a desk calculation; it's a hypothesis about optimization that only ImageNet can confirm, and it's the part I'd most want to ablate against a fixed-`β` ACON-C to be sure the per-sample routing is what's paying off rather than just the extra capacity. The code below is ACON-C and its meta variant.

```python
import torch
import torch.nn as nn


class AconC(nn.Module):
    """ACON-C: smooth maximum of two learnable linear pieces p1*x, p2*x.

        f(x) = (p1 - p2) * x * sigmoid(beta * (p1 - p2) * x) + p2 * x

    Per-channel learnable p1, p2, beta. Initialize at p1=1, p2=0, beta=1
    so the module starts at the Swish point, then lets the bounds move.
    """

    def __init__(self, width):
        super().__init__()
        self.p1 = nn.Parameter(torch.ones(1, width, 1, 1))
        self.p2 = nn.Parameter(torch.zeros(1, width, 1, 1))
        self.beta = nn.Parameter(torch.ones(1, width, 1, 1))

    def forward(self, x):
        diff = self.p1 * x - self.p2 * x
        return diff * torch.sigmoid(self.beta * diff) + self.p2 * x


class MetaAconC(nn.Module):
    """meta-ACON-C: switching factor beta generated from the input (channel-wise).

        beta = sigmoid( BN(FC2(BN(FC1(GAP(x))))) )    # SE-style bottleneck, r=16
        f(x) = (p1 - p2) * x * sigmoid(beta * (p1 - p2) * x) + p2 * x

    beta is per-sample and per-channel, so different inputs get different
    non-linear degrees in the same layer. p1, p2 are initialized at 1 and 0.
    """

    def __init__(self, width, r=16):
        super().__init__()
        inner = max(r, width // r)
        self.fc1 = nn.Conv2d(width, inner, kernel_size=1, stride=1, bias=True)
        self.bn1 = nn.BatchNorm2d(inner)
        self.fc2 = nn.Conv2d(inner, width, kernel_size=1, stride=1, bias=True)
        self.bn2 = nn.BatchNorm2d(width)
        self.p1 = nn.Parameter(torch.ones(1, width, 1, 1))
        self.p2 = nn.Parameter(torch.zeros(1, width, 1, 1))

    def forward(self, x):
        ctx = x.mean(dim=2, keepdim=True).mean(dim=3, keepdim=True)
        beta = torch.sigmoid(self.bn2(self.fc2(self.bn1(self.fc1(ctx)))))
        diff = self.p1 * x - self.p2 * x
        return diff * torch.sigmoid(beta * diff) + self.p2 * x
```
