I have a billion-parameter language model and I have to store its weights at two, three, or four bits.
The two regimes I know put me between a rock and a hard place. Quantization-aware training would recover
the accuracy — backprop the task loss through the quantized network and let the weights reorganize to
fit the grid — but for a model this size that means full-model gradients, large data, and thousands of
steps, which I cannot afford. Post-training quantization is the opposite: I round once, gradient-free,
on a handful of calibration sequences, and it is cheap enough — but at low bit-widths it collapses,
because rounding to four or eight levels is irreversible and the round-to-nearest cover does nothing to
fight the coarseness. So the question I actually have to answer is whether there is something with a
*PTQ-grade budget* — no end-to-end backprop, a small calibration set, a short local optimization — that
nonetheless recovers near-QAT accuracy at INT2 and INT3. I want the cheapest possible degree of freedom
that buys most of the QAT gain.

Let me look hard at where PTQ actually loses its accuracy, because the fix has to attack that exact
point. The quantizer is `W_q = clamp(round(W/h) + z, 0, 2^N − 1)`, dequantized as `(W_q − z)·h`, and the
single number that controls everything is the scale `h`, which comes from the clipping range:
`h = (xmax − xmin)/(2^N − 1)` set from the per-group extremes. Standard min-max PTQ takes `xmax`, `xmin`
to be the literal maximum and minimum of the group — a no-clip cover. Now picture a group of weights at
two bits: four codes. If one or two weights in the group are outliers — and weight distributions in
these models are heavy-tailed, they always have a few — then `xmax`/`xmin` are stretched out to those
outliers, the four codes are spaced to span that stretched range, and the ordinary weights, which live
in a much narrower band near zero, all get snapped onto a grid far too coarse for them. The error is
dominated by coarsely rounding the bulk, all to faithfully cover a handful of outliers. That is the
disease. And it is irreversible: once I round, the error is frozen.

I want to see the size of this effect before I commit to a fix, so let me work a concrete group by
hand. Take fifteen "bulk" weights drawn uniformly in ±0.1 and one outlier at 1.0 — exactly the
heavy-tailed shape I am worried about — and quantize it asymmetrically at two bits the way the code will:
`scale = (xmax − xmin)/(2^N − 1)`, `z = −round(xmin/scale)`, code each weight as `clamp(round(W/scale)+z,
0, 3)`, dequantize. With the literal min-max cover the range runs from about −0.05 to 1.0, so the four
codes are spaced ≈0.32 apart and the entire bulk in ±0.1 collapses onto a single code — every one of
the fifteen small weights dequantizes to roughly zero. The mean-squared error over the group comes out
to about 1.8e-3. So the disease is real and I can measure it.

Now the tempting move is the aggressive one: clip the range hard, throw the outlier away, and spend all
four codes on the bulk. Let me actually try it rather than assume it works. Shrinking the upper extreme
to `0.5·xmax` gives the group MSE ≈ 1.9e-2; to `0.2·xmax`, ≈ 6e-2; to `0.1·xmax`, ≈ 7e-2. Clipping
*hard* makes things an order of magnitude **worse**, not better. The reason is plain once I see the
numbers: the outlier still has to be represented, and when I crush the range down to the bulk, that one
weight at 1.0 gets clamped to a rail sitting far below it, and its squared dequant error alone — (1.0
minus wherever the top code lands)² — swamps whatever I saved on the bulk. So "ignore the outlier" is
the wrong instinct. Let me sweep the clip ratio finely instead of guessing. The group MSE is minimized
not near zero clip but around `0.94–0.95·xmax`: a *gentle* inward trim that buys ≈1.1x lower error at
two bits, and ≈1.14x at three. The optimum is a moderate clip, and the curve is genuinely non-monotone —
it dips a little, then climbs steeply if I over-clip.

That sweep is the thing that actually settles the design. There is no fixed percentile I could write down
that lands on 0.94 for this group and on something else for the next, because where the optimum sits
depends on the shape of the bulk and on how far the outlier reaches — both of which vary group to group.
A heuristic clip cannot know that. And the right metric is not even this per-group weight MSE: the model
cares about the *layer output*, where every bulk weight is exercised by every token while the outlier
weight feeds a single input dimension, so the bulk error accumulates far more than the weight-MSE on one
toy group lets on. The honest reading of my little experiment is that it confirms the *direction* —
inward clipping helps — and, more usefully, it rules out the aggressive version and shows the right
amount is a delicate, group-specific tradeoff. That points away from *setting* the clip and toward
*learning* it: make the clipping range a trainable quantity, per group, optimized directly against how
much the quantized weights perturb the output. That is the degree of freedom I will spend my budget on.

Learning the clip immediately runs into the wall that keeps PTQ gradient-free: the quantizer has a
`round` in it, and `round` is flat almost everywhere — zero derivative — so a gradient cannot flow back
through it to whatever parameterizes the range. I need the straight-through trick: treat `round` as the
identity on the backward pass, so `round_ste(x) = (round(x) − x).detach() + x` is `round(x)` in the
forward and slope-one in the backward. With that in place, the *rest* of the quantizer — the divide by
`h`, the dependence of `h` on the clipping extremes — is an ordinary differentiable function of the range
parameters, and a gradient can reach them. So the round is no longer an obstacle; the only design choice
left is *how to parameterize the clip* so the optimization is well-behaved.

Three things I want from the parameterization, and I should check that a candidate actually delivers them
rather than just hoping. First, the clip should only ever shrink the range, never expand it past the
literal extremes — expanding wastes codes on values that do not exist, the opposite of what the sweep
told me to do. So the learned quantity should be a *ratio* in (0, 1] applied to the extremes, not a free
scale. Second, I want to start somewhere sane: the natural starting point is the min-max cover itself
(ratio ≈ 1), which is exactly what plain PTQ does, so I begin no worse than PTQ and learn to clip inward
from there. Third, the gradient of the parameterization should be bounded and ideally small near the
start — the sweep above was visibly jagged near the optimum, because nudging the range reassigns whole
blocks of weights across codes, and an unbounded parameter driven by that jagged gradient would jump
around.

A sigmoid is the obvious candidate for the first two: a learnable `γ` per group (per side, since the
positive and negative extremes can want different clips), with the clipped extreme set to
`sigmoid(γ)·extreme`. The output lives in (0, 1), so the clip can only shrink the range — the first
requirement holds by construction. For the second I need the init to put `sigmoid(γ)` near 1; let me
just compute it. `sigmoid(4) = 0.982014`, so initializing `γ = 4` starts the grid at 98.2% of the
min-max cover — essentially the cover, close enough that I begin at plain PTQ and trim from there. The
third requirement I should not take on faith, since the whole point was to damp a jagged gradient, so let
me tabulate the derivative `d sigmoid(γ)/dγ = sigmoid(γ)(1 − sigmoid(γ))`:

```
 γ  | sigmoid(γ) | sigmoid(γ)(1−sigmoid(γ))
----+-----------+------------------------
 -4 |   0.0180   |   0.0177
  0 |   0.5000   |   0.2500
  2 |   0.8808   |   0.1050
  4 |   0.9820   |   0.0177
  6 |   0.9975   |   0.0025
```

At the init `γ = 4` the local gradient is 0.0177 — about a fourteenth of the 0.25 it would be at the
midpoint, and it keeps shrinking as `γ` grows. So the sigmoid does damp the jagged range-gradient near
where I start and at both extremes, exactly the saturating behavior I wanted; the parameter cannot lurch.
All three requirements check out. So the parameterization is: two learnable per-group factors
`upbound_factor`, `lowbound_factor`, both initialized to 4, and the calibration computes

```
xmax = sigmoid(upbound_factor) · max(W_group),
xmin = sigmoid(lowbound_factor) · min(W_group),
```

then the scale and zero-point exactly as before from these *clipped* extremes. For symmetric signed
weight quantization that is `h = max(|xmax|, |xmin|)/(2^{N−1} − 1)` (clamped below at a small `CLIPMIN =
1e-5` so a degenerate all-zero group does not produce a zero scale — and indeed, an all-zero group gives
`xmax = xmin = 0`, raw scale 0, which without the clamp divides by zero; with it the scale floors at
1e-5 — and above at a large constant); for asymmetric it is `h = (xmax − xmin)/(2^N − 1)` with a rounded
zero-point `z = −round(xmin/h)`. The fake-quant is then the straight-through `round_ste(W/h) (+z)`,
clamped to the code range, times `h`. The gradient flows: output reconstruction loss → through
`round_ste` (identity backward) → through `h`, which is a differentiable function of `sigmoid(γ)·extreme`
→ into `γ`. The weights `W` themselves I can keep fixed or also let move; the *new* learnable thing, the
contribution, is the per-group clipping factor.

Now I have to make sure I optimize this on a PTQ budget, because if I needed the global task loss through
the whole model I would be back to QAT. I do not. The clipping factors are *local*: the scale of a group
only affects that group's layer's output. So I can optimize them block by block. For each transformer
block in turn, run the full-precision block and the quantized block on the small calibration set, and
minimize the *reconstruction* MSE between the two block outputs — a local objective on 128 sequences, a
handful of epochs per block, no backprop through the rest of the model. The straight-through estimator
carries the gradient through the round, the sigmoid keeps the factors bounded and stable, and the
per-block locality keeps the cost at PTQ scale. The only degree of freedom I added is a sigmoid-gated
per-group clipping ratio, learned by a cheap local reconstruction rather than by global QAT.

Let me close the loop on the disease I measured. The min-max cover spent its four codes faithfully
spanning the outlier and collapsed the bulk onto one code; the learnable factor lets `γ` pull `xmax`
down toward the optimum the sweep found — a moderate inward trim, not the aggressive crush that backfired
— chosen per group by the loss rather than by any fixed rule. The starting point at `sigmoid(4) = 0.982`
means I begin at the cover and can only improve from there: if a group has no outlier worth clipping, `γ`
simply stays near its init and I lose nothing; if it does, `γ` trims inward exactly as far as the
reconstruction loss rewards, and no farther, since I saw that over-clipping is punished. And because the
factor is bounded with a gradient that is ≈0.018 at the init, the optimization stays stable even at two
bits where the range-loss is jagged. The mechanism is small — one extra parameter per side per group, a
sigmoid, an STE round — which is what turns the uncalibrated min-max cover into a learned, per-group,
loss-aware clip.

There is a natural companion problem this leaves untouched, and I should name it so I do not overclaim.
The clipping factor handles *weight* outliers by shrinking the weight range. But low-bit
weight-*activation* quantization also suffers from *activation* outliers, which clipping the weights does
nothing about; those are better handled by *migrating* the difficulty from activations into weights via
an equivalent channel-wise scaling and shift (a learnable transformation that leaves the layer's function
unchanged but reshapes which tensor carries the outliers). That is a separate, complementary mechanism,
optimized in the same per-block reconstruction loop. For weight-only quantization, though, the learnable
clipping is the whole story, and it is the part that directly attacks the irreversible rounding error at
INT2/INT3.

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
