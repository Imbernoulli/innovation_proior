I have a billion-parameter language model and I have to store its weights at two, three, or four bits.
The two regimes I know put me between a rock and a hard place. Quantization-aware training would recover
the accuracy — backprop the task loss through the quantized network and let the weights reorganize to
fit the grid — but for a model this size that means full-model gradients, large data, and thousands of
steps, which I cannot afford. Post-training quantization is the opposite: I round once, gradient-free,
on a handful of calibration sequences, and it is cheap enough — but at low bit-widths it collapses,
because rounding to four or eight levels is irreversible and the round-to-nearest cover does nothing to
fight the coarseness. So the question I actually have to answer is whether there is something with a
*PTQ-grade budget* — no end-to-end backprop, a small calibration set, a short local optimization — that
nonetheless recovers near-QAT accuracy at INT2 and INT3. I want to find the cheapest possible degree of
freedom that buys most of the QAT gain.

Let me look hard at where PTQ actually loses its accuracy, because the fix has to attack that exact
point. The quantizer is `W_q = clamp(round(W/h) + z, 0, 2^N − 1)`, dequantized as `(W_q − z)·h`, and the
single number that controls everything is the scale `h`, which comes from the clipping range:
`h = (xmax − xmin)/(2^N − 1)` set from the per-group extremes. Standard min-max PTQ takes `xmax`, `xmin`
to be the literal maximum and minimum of the group — a no-clip cover. Now picture a group of 128 weights
at two bits: four codes. If one or two weights in the group are outliers — and weight distributions in
these models are heavy-tailed, they always have a few — then `xmax`/`xmin` are stretched out to those
outliers, the four codes are spaced to span that stretched range, and the 126 ordinary weights, which
live in a much narrower band near zero, all get snapped onto a grid far too coarse for them. The error
is dominated by coarsely rounding the bulk, all to faithfully cover a handful of outliers. That is the
disease. And it is irreversible: once I round, the error is frozen.

So the leverage is obvious in hindsight: *do not* faithfully cover the outliers. Clip the range
inward — set `xmax`/`xmin` *smaller* than the literal extremes — so the four codes are spaced to round
the bulk finely, accepting that the outliers get clamped to the rail. The question is how much to clip,
and that depends on the weight distribution *and* on what the clipping costs the model's output, which
is not something a fixed percentile heuristic can know. The amount of clipping that is right for one
group is wrong for another. So I should not *set* the clip; I should *learn* it. Make the clipping range
a trainable quantity, per group, optimized to minimize how much the quantized weights perturb the
output. That is the degree of freedom I will spend my small budget on.

But learning the clip immediately runs into the wall that keeps PTQ gradient-free: the quantizer has a
`round` in it, and `round` is flat almost everywhere — zero derivative — so a gradient cannot flow back
through it to whatever parameterizes the range. I need the straight-through trick: treat `round` as the
identity on the backward pass, so `round_ste(x) = (round(x) − x).detach() + x` is `round(x)` in the
forward and slope-one in the backward. With that in place, the *rest* of the quantizer — the divide by
`h`, the dependence of `h` on the clipping extremes — is an ordinary differentiable function of the
range parameters, and a gradient can reach them. So the round is no longer an obstacle; the only design
choice left is *how to parameterize the clip* so the optimization is well-behaved.

Let me think about what a good parameterization must guarantee. First, the clip should only ever shrink
the range, never expand it past the literal extremes — expanding would waste codes on values that do not
exist, the opposite of what I want. So the learned quantity should be a *ratio* in (0, 1] applied to the
extremes, not a free scale. Second, I want the optimization to start somewhere sane and improve
monotonically: the natural starting point is the min-max cover itself (ratio ≈ 1), which is exactly what
plain PTQ does, so I begin no worse than PTQ and *learn to clip inward* from there. Third, the gradient
of the parameterization should be bounded — at four codes the reconstruction loss in the range is
jagged, because nudging the range reassigns whole blocks of weights across codes, so an unbounded
parameter driven by that gradient would jump around. A bounded, saturating map damps that.

A sigmoid gives me all three at once. Introduce a learnable parameter `γ` per group (per side, since the
positive and negative extremes can want different clips), and set the clipped extreme to
`sigmoid(γ)·extreme`. The sigmoid output is in (0, 1), so the clip can only shrink the range — guarantee
one. If I initialize `γ` to a moderately large positive value, `sigmoid(γ)` starts near 1, so the grid
begins at essentially the min-max cover — guarantee two; `init = 4` gives `sigmoid(4) ≈ 0.982`, near
enough to the cover that I start at PTQ and trim. And the derivative `d sigmoid(γ)/dγ = sigmoid(γ)(1 −
sigmoid(γ))` saturates toward zero as the factor moves to either extreme, damping the jagged
range-gradient — guarantee three. So the parameterization is: two learnable per-group factors
`upbound_factor`, `lowbound_factor`, both initialized to 4, and the calibration computes

```
xmax = sigmoid(upbound_factor) · max(W_group),
xmin = sigmoid(lowbound_factor) · min(W_group),
```

then the scale and zero-point exactly as before from these *clipped* extremes. For symmetric signed
weight quantization that is `h = max(|xmax|, |xmin|)/(2^{N−1} − 1)` (clamped below at a small `CLIPMIN =
1e-5` so a degenerate all-zero group does not produce a zero scale, and above at a large constant); for
asymmetric it is `h = (xmax − xmin)/(2^N − 1)` with a rounded zero-point `z = −round(xmin/h)`. The
fake-quant is then the straight-through `round_ste(W/h) (+z)`, clamped to the code range, times `h`. The
gradient flows: output reconstruction loss → through `round_ste` (identity backward) → through `h`,
which is a differentiable function of `sigmoid(γ)·extreme` → into `γ`. The weights `W` themselves I can
keep fixed or also let move; the *new* learnable thing, the contribution, is the per-group clipping
factor.

Now I have to make sure I optimize this on a PTQ budget, because if I needed the global task loss
through the whole model I would be back to QAT. I do not. The clipping factors are *local*: the scale of
a group only affects that group's layer's output. So I can optimize them block by block. For each
transformer block in turn, run the full-precision block and the quantized block on the small calibration
set, and minimize the *reconstruction* MSE between the two block outputs — a local objective on 128
sequences, a handful of epochs per block, no backprop through the rest of the model. The straight-through
estimator carries the gradient through the round, the sigmoid keeps the factors bounded and stable, and
the per-block locality keeps the cost at PTQ scale. This is the whole trick: the only degree of freedom
I added is a sigmoid-gated per-group clipping ratio, and it is learned by a cheap local reconstruction,
not by global QAT.

Let me sanity-check that this actually fixes the disease I diagnosed. The outlier-stretched range: now
`γ` can learn to pull `xmax` down below the outlier, so the codes pack onto the bulk and the outliers
clamp to the rail — exactly the inward clip I argued for, and chosen per group by the loss rather than a
heuristic. The starting point at `sigmoid(4) ≈ 0.982` means I begin at the cover and the optimization
can only help: if a group has no outliers and the cover is already right, `γ` simply stays near its init
and I lose nothing; if it does, `γ` clips inward as far as the reconstruction loss rewards. And because
the factor is bounded and its gradient saturates, the optimization is stable even at two bits where the
range-loss is jagged. The mechanism is minimal — one extra parameter per side per group, a sigmoid, an
STE round — and it is the smallest thing that turns the uncalibrated min-max cover into a learned,
per-group, loss-aware clip.

There is a natural companion problem this leaves untouched, and I should name it so I do not overclaim.
The clipping factor handles *weight* outliers by shrinking the weight range. But low-bit
weight-*activation* quantization also suffers from *activation* outliers, which clipping the weights
does nothing about; those are better handled by *migrating* the difficulty from activations into weights
via an equivalent channel-wise scaling and shift (a learnable transformation that leaves the layer's
function unchanged but reshapes which tensor carries the outliers). That is a separate, complementary
mechanism, optimized in the same per-block reconstruction loop. For weight-only quantization, though,
the learnable clipping is the whole story, and it is the part that directly attacks the irreversible
rounding error at INT2/INT3.

So the implementation fills the empty slot with the sigmoid-gated learnable clip and nothing else:

```python
import torch, torch.nn as nn

CLIPMIN = 1e-5

def round_ste(x):
    # round in forward, straight-through (gradient = 1) in backward
    return (x.round() - x).detach() + x

class UniformAffineQuantizer(nn.Module):
    def __init__(self, n_bits, group_size, shape, symmetric=False, lwc=False):
        super().__init__()
        self.n_bits = n_bits
        self.qmin, self.qmax = 0, 2 ** n_bits - 1
        self.group_size = group_size
        self.symmetric = symmetric
        self.lwc = lwc
        self.sigmoid = nn.Sigmoid()
        if lwc:
            # one learnable clip factor per group, per side; init so
            # sigmoid(init) ~ 1, i.e. start at the min-max cover.
            init_value = 4.0
            self.upbound_factor = nn.Parameter(torch.ones(shape) * init_value)
            self.lowbound_factor = nn.Parameter(torch.ones(shape) * init_value)

    def calibration(self, x):
        # per-group signed extremes
        xmax = x.amax(dim=-1, keepdim=True)
        xmin = x.amin(dim=-1, keepdim=True)
        if self.lwc:
            # learnable inward clip in (0,1): shrink the range, never expand it
            xmax = self.sigmoid(self.upbound_factor) * xmax
            xmin = self.sigmoid(self.lowbound_factor) * xmin
        if self.symmetric:
            abs_max = torch.max(xmax.abs(), xmin.abs())
            scale = abs_max / (2 ** (self.n_bits - 1) - 1)
            scale = scale.clamp(min=CLIPMIN, max=1e4)
            zero_point = None
        else:
            scale = (xmax - xmin).clamp(min=CLIPMIN) / (2 ** self.n_bits - 1)
            zero_point = -(xmin / scale).round()
        return scale, zero_point

    def fake_quant(self, x, scale, zero_point):
        x_int = round_ste(x / scale)               # STE round -> gradient reaches scale -> factors
        if zero_point is not None:
            x_int = x_int + zero_point
        x_int = x_int.clamp(self.qmin, self.qmax)  # clip to the code range
        x_dq = x_int
        if zero_point is not None:
            x_dq = x_dq - zero_point
        return x_dq * scale                        # dequantize

    def forward(self, x):
        # x reshaped to (..., group_size) by the caller
        scale, zero_point = self.calibration(x)
        return self.fake_quant(x, scale, zero_point)
```

The causal chain: PTQ collapses at low bit-widths because the min-max cover spends its few codes
faithfully spanning outliers and coarsely rounds the bulk, irreversibly → the fix is to clip the range
inward, by an amount that must be chosen per group against the loss, so *learn* it → learning needs a
gradient through the round, supplied by the straight-through estimator → parameterize the clip as a
sigmoid-gated factor so it can only shrink the range, starts at the min-max cover (`sigmoid(4) ≈ 0.982`)
and improves monotonically, with a saturating gradient that stays stable at two bits → optimize the
factors with a PTQ-grade *per-block local reconstruction* MSE on a small calibration set, no end-to-end
backprop → recovering near-QAT low-bit accuracy at a PTQ budget, with activation outliers left to a
complementary equivalent-transformation mechanism in weight-activation settings.
