The deepening-factor sweep answered the tension cleanly, and in the direction I was less confident
about going in: `l=2` won at 5.43%, `l=1` was clearly too weak (6.69%), and `l=3`/`l=4` got
progressively worse again (5.65%, 5.93%) rather than continuing to improve. That's not a monotonic
capacity-per-unit story — if richer per-unit expressiveness were the dominant effect, `l=4` should have
been the best point in the sweep, and instead it's the worst apart from `l=1`. The shape is a peak at
`l=2`, which is exactly what the shortcut-density story predicts: going from `l=2` to `l=3` or `4`
trades away more shortcut connections than the extra per-unit depth is worth, at this fixed budget,
while `l=1` fails for the opposite reason — one convolution per unit just isn't enough transformation
to be useful regardless of how many units there are. Both block-structure axes are now closed: the
kernel pattern doesn't matter much (rung 1) and the two-convolution block is a real optimum, not an
arbitrary default (rung 2). Every further parameter I spend from here goes into either more blocks
(depth `d`) or wider convolutions (`k`) — the axis this whole design process has been building toward.

Depth's exchange rate is already known going in, and it's poor. Thin residual networks — the
strongest thin reference available, 110/164/1001 layers at 1.7M/1.7M/10.2M parameters — buy each
further fraction of a percent of accuracy by very roughly doubling layer count, and going from 164 to
1001 layers (a 6x increase in depth) buys a comparatively modest further gain while pushing parameters
from 1.7M to 10.2M. Width's exchange rate, by contrast, needs to be worked out rather than assumed,
because it's genuinely non-obvious in which direction it's expensive. The number of residual blocks
`d` enters the parameter count linearly — twice as many blocks is twice the parameters, holding
channel counts fixed. But the widening factor `k` multiplies *every* convolution's channel count, and
a convolution's parameter count is (roughly) `in_channels x out_channels x kernel_area`: doubling `k`
doubles both `in_channels` and `out_channels` for every convolution inside every block, so it costs
roughly `4x` per convolution, not `2x`. That's the "quadratic in k, linear in d" asymmetry stated
directly. I want a concrete number attached to that before I trust it, not just the algebra. Take a
16-layer skeleton (the shallowest depth in the family, `(16-4)/6=2` blocks per group) at the smallest
setting `k=1`: stage widths are `16, 16, 32, 64`, essentially the thin baseline, and a network that
shallow and that narrow comes out to well under a quarter-million parameters — call it order `0.2M`.
Push the same 16-layer skeleton to `k=8`: stage widths become `16, 128, 256, 512`, and the parameter
count lands around `11M` — a rough sixty-fold increase for an eight-fold widening. Eight times the
width for roughly sixty times the parameters is the `k^2` scaling showing up exactly where the algebra
says it should, because `k` enters both the input and output dimension of nearly every convolution in
the network. Width is genuinely, quadratically expensive per unit of widening — which means the case
for spending budget on it can't rest on parameter-efficiency; it has to rest on something width does
per parameter that depth doesn't.

That something is wall-clock, and rung 1 already handed me real evidence for it, not just an
assertion. At matched parameter count, the block-type sweep's timings weren't uniform across
network shapes: `B(3,3)` at depth 28 (two `3x3` convolutions per block, fewer, larger blocks) trained
at 67.5s/epoch, the same as `B(3,1)` at depth 40 despite `B(3,1)` having more sequential blocks to
step through — and `B(3,1,3)` at depth 22, the shallowest of the two-plus-`1x1` variants, was fastest
at 59.9s/epoch. The pattern across that table, even though it wasn't the question rung 1 was asking,
is consistent with the general GPU argument: many small sequential operations underuse a GPU's
parallel throughput more than fewer, larger ones do at the same total compute. A thousand small
convolutions chained end to end is a poor fit for hardware built for parallel work on large tensors;
fewer, wider convolutions give the hardware more to chew on per launch. If that holds up at the much
larger depth and width extremes I'm about to test, the real comparison I should be making isn't
accuracy-per-parameter, where depth already has the advantage of linear scaling — it's accuracy-per-
training-time, where width's parallelism-friendliness might pay back its quadratic parameter cost.

I don't yet know the shape of the tradeoff at the scale that matters, so I want a grid, not a single
comparison. Two separate questions are tangled together in "should I go wide": at a fixed, modest
depth, does turning up `k` keep helping, or does it saturate or overfit past some point given only
horizontal-flip/crop augmentation? And at a fixed, large `k`, does depth still help the way it does in
the thin regime, or does the degradation problem depth alone exhibits (very deep thin nets degrading
past a point, independent of overfitting) reappear once the network is already wide? Varying only one
of `k` or depth at a time would leave the other question unanswered, and a single point in the
`(depth, k)` plane can't tell me whether I'm looking at a k-effect, a depth-effect, or an interaction
between them. So: a grid. Depths from 16 (the shallowest depth this convention supports beyond the
absolute minimum) up to 40 (already established as a working budget from the last two rungs), and `k`
from 2 up to a value large enough that I actually see whether returns diminish — I'll go to 12, since
stopping at, say, 4 or 6 wouldn't tell me whether the accuracy curve is still rising or has already
flattened. At each depth I'll test the range of `k` values that keeps parameter count in a broadly
similar band across depths, so I'm reading off a genuine depth-vs-width tradeoff surface rather than
just "bigger numbers win," which would be true of almost anything and uninformative about the ratio
that matters. Every cell reports both CIFAR-10 and CIFAR-100 test error — CIFAR-100 is the sharper
overfitting stress test at this parameter scale, ten times fewer images per class than CIFAR-10, so if
width without regularization is going to show diminishing or negative returns anywhere in this budget
range, CIFAR-100 is where I'd expect to see it first.

One comparison I'm specifically watching for, because it's the cleanest test of the "depth-to-width
ratio is arbitrary, not a law" hypothesis this whole design process is circling: whether a wide-but-
much-shallower network can match a thin extremely-deep one at comparable parameter count. The
strongest thin reference sits at 1001 layers, 10.2M parameters. If some cell in this grid — thirty to
forty layers deep, not a thousand — lands in the same parameter neighborhood and gets a competitive or
better test error while training in a small fraction of the time (which the timing argument above
predicts it should), that's direct evidence against the idea that extreme depth itself provides some
regularization or representational benefit that width can't substitute for. I don't get to assume that
outcome — it's exactly what the grid is for — but it's the specific comparison I'll be looking to
extract once the numbers are in, alongside the more basic question of where in `(depth, k)` space
CIFAR-10 and CIFAR-100 error are actually minimized within this budget range.
