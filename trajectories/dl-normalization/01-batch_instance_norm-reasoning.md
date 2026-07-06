The whole task reduces to one choice — which feature positions share a mean and variance — so before I
reach for anything elaborate I want to start from the move that is *most defensible* given what the prior
art already tells me, and let the measured numbers point at the headroom. The three layers in the lineage
each pick a different population: the batch layer pools per channel across the whole batch and spatial
grid, instance pools each channel's own spatial map alone, layer pools everything in one image. The batch
layer is the de-facto standard and it plainly trains these nets; the others were built for different
problems (recurrent models, style transfer). But the lineage also names a specific tension I can exploit
right at the first rung, and it is sharper than "small batch is noisy," because here the batch is a
comfortable 128 and small-batch noise is not the pressure. The tension is between **what the batch
statistic keeps and what the instance statistic throws away**, and it is the cleanest first thing to
attack because it needs no new design space — only a way to *blend* two members of the family I already
have.

Let me be precise about that tension, because it decides the shape of the rung. The batch statistic for a
channel `c` is `mean`/`var` over `(N, H, W)` — every position of that channel across all 128 images and
the whole spatial grid. So the standardization a feature receives is keyed to *the dataset's distribution
for that channel*: it pulls every image's channel-`c` activations onto a common scale that the running
average has learned to represent. That is exactly what you want for recognition — it preserves the
*relative* differences between images that carry the class signal, because the normalizing constant is
shared across the batch and does not depend on the particular image. The cost is that it also preserves
whatever per-image global style or contrast lives in that channel: if two images of the same object differ
only in illumination or texture statistics, the batch layer leaves that difference largely intact, because
it never normalizes *within* an image. The instance statistic is the mirror image. It standardizes each
channel of each image by that image's *own* spatial mean and variance — `(H, W)` only — so it explicitly
removes per-image, per-channel contrast and style; this is precisely why it dominates style transfer,
where wiping out the content image's style is the goal. But for recognition that same removal is a
liability: by re-centering and re-scaling every channel of every image to the same per-image moments, it
discards the global magnitude information that distinguishes one image's channel response from another's —
the very between-image signal the batch layer worked to keep.

There is a second, quieter cost to the instance statistic that I should quantify now, because it decides
how much I can trust the instance branch on the deep parts of these nets. The instance moment for a
channel is estimated from `H·W` numbers and nothing else. On the He-2016 CIFAR ResNets the spatial grid
shrinks stage by stage — 32×32 at the 16-channel stage, 16×16 at 32 channels, 8×8 at 64 channels — so the
instance branch estimates a mean and a variance from 1024, then 256, then only **64** values on the
deepest stage. The batch branch, pooling the same channel over all 128 images, works from 131072, 32768,
and 8192 values respectively — always a factor of `N = 128` more. If I treat each channel's activations as
roughly independent, the relative fluctuation of a sample variance is `√(2/(n−1))`, so the instance
variance on the deepest stage carries about `√(2/63) ≈ 0.178`, roughly **18%** noise on the per-channel
scale *every forward pass*, against the batch branch's `√(2/8191) ≈ 0.016`, under 2%. That is an order of
magnitude gap, and it is worst exactly where the feature maps are coarsest and the channels most abstract.
So the instance branch is not just throwing away between-image scale; on the deep stages it is also
standardizing by a noisy estimate. Both effects push the same way — against instance on the parts of the
net I care most about — and I will want to remember this when I read the numbers.

I could reach further on this first rung — bring the layer statistic in too, or make the whole family a
learned mixture from the start — but that would be the wrong first move, and not for timidity. The point of
a first rung is to isolate *one* axis cleanly so the numbers say something unambiguous about it. Batch and
instance are the two members whose disagreement the lineage names most sharply here, and they differ along
a single interpretable axis — how much of each image's own style to remove — so a two-way blend along that
axis is a controlled experiment: if it helps, I have learned that per-image style removal is a live lever;
if it does not, I have learned the opposite, and either way the next rung starts from a fact rather than a
guess. Opening the full family at once would confound several axes and leave me unable to read which one
moved the needle. And there is a concrete reason to start with *this* axis specifically rather than, say,
reaching for the layer statistic first: batch is the known-good incumbent and instance is its most direct
opposite, so the two-way blend has batch normalization sitting inside it at `g ≡ 1` as a strict fallback —
the experiment cannot cost me the baseline behavior except through the optimizer's own choices, which is
the safest possible way to probe an unfamiliar direction. So the two-way blend is not a hedge; it is the
minimal probe that puts one clean question to the data.

So I have two members of the same family that fail in *opposite* directions: batch keeps the
discriminative between-image scale but also keeps nuisance style; instance kills nuisance style but also
kills the discriminative scale (and estimates it noisily on coarse maps). The classic move when two
estimators err oppositely is to interpolate between them — and crucially, the right interpolation weight
is almost certainly not the same for every channel. Some channels of a conv layer carry mostly style-like,
contrast-like information (low-level texture filters, color blobs) where I want the instance behavior;
others carry shape and structure that is class-relevant across images, where I want the batch behavior. A
*single* global blend cannot serve both. Before I commit to per-channel, let me actually price the cheaper
option, because "per-channel" is a decision I should have to earn: a single scalar gate for the whole
layer is one parameter instead of `C`, and it is genuinely simpler. But a scalar gate forces every channel
of the layer onto the same point on the batch-versus-instance dial — it must wash out the same fraction of
per-image style from the edge detectors and the shape channels alike, when the whole premise is that those
two kinds of channel want *opposite* treatment. A single number cannot express "instance here, batch
there," which is the entire content of the idea. So the gate has to be per channel; the scalar version
throws away the mechanism to save one parameter, and I will check below that `C` parameters cost almost
nothing. What I want is a **per-channel gate** that learns, channel by channel, how much of each image's
own style to wash out versus how much of the batch-level scale to keep — and to learn it from the task
loss, so the network decides per channel which of the two behaviors helps classification. That is the
whole idea of the first rung: at each `CustomNorm`, compute both the batch-normalized and the
instance-normalized version of the feature, and combine them per channel with a learnable weight.

Before I write it down I want to see the mechanism on a case simple enough to compute by hand, because the
"between-image scale" argument is easy to say and easy to get wrong. Take a channel `c` that is spatially
flat within each image — every position carries the same value `a_n`, which varies from image to image;
this is the extreme of a channel that encodes a per-image global offset and nothing spatial, and the `a_n`
across images are exactly the between-image signal I claimed the batch layer keeps. Run instance on it:
`mean_IN[n,c] = a_n`, `var_IN[n,c] = 0`, so `x̂_IN[n,c] = (a_n − a_n)/√(0 + ε) = 0` — *every image maps to
zero*. Instance normalization annihilates this channel completely; the entire spread of `a_n`, the whole
between-image signal, is gone. Now batch on the same channel: `mean_BN[c] = ā` (the batch mean of the
`a_n`, since spatial averaging is trivial on a flat channel), `var_BN[c] = s²_a` (their batch variance),
so `x̂_BN[n,c] = (a_n − ā)/√(s²_a + ε)` — the ordering and spread of the `a_n` survive intact. The blend
`g·x̂_BN + (1−g)·x̂_IN` collapses on this channel to `g·(a_n − ā)/√(s²_a + ε)`: the between-image signal
survives *scaled by `g`*. This is the whole idea made arithmetic — the gate is literally the fraction of
between-image DC signal this channel gets to keep, and it says something sharp about the danger too: if the
optimizer ever drives `g → 0` on a channel that carries class-discriminative offset, that channel's signal
is deleted, not merely attenuated. The gate is a knob over exactly the quantity the two branches disagree
about.

Now make it concrete against *this task's edit surface*, because the contract is narrow — `CustomNorm`
must be a drop-in for `nn.BatchNorm2d`, constructed as `CustomNorm(C)`, mapping `[B, C, H, W]` to the same
shape, stable in train and eval — and the details of how I compute the two branches matter for both
correctness and what I can expect from the numbers. Let `g_c ∈ [0, 1]` be the per-channel gate; I store it
as a raw real parameter `ρ_c` and pass it through a sigmoid so it stays in the unit interval without a
constraint in the optimizer, `g_c = σ(ρ_c)`. The mixed output, before the affine, is
`x̂ = g · x̂_BN + (1 − g) · x̂_IN`, broadcasting `g` over `(B, H, W)` so each channel uses its own gate.
When `g_c = 1` the channel is pure batch normalization; when `g_c = 0` it is pure instance normalization;
in between it is a true blend of the two standardizations.

There is a real fork here that I want to resolve deliberately rather than by reflex: I am blending the two
*standardized features*, but I could instead blend the raw moments — form `mean = g·mean_BN + (1−g)·mean_IN`
and `var = g·var_BN + (1−g)·var_IN`, then standardize once by those mixed moments. The two are not the
same map; they agree only at the endpoints `g ∈ {0,1}`. Blending features keeps each branch a genuine
unit-variance standardization and makes the output a clean convex interpolation of two well-scaled signals,
so the endpoints are *exactly* batch and instance and the interior is an honest slider between them.
Blending the moments instead standardizes by a `var` that is a convex mix of two different populations'
variances, which does not render either population to unit scale, and mixes a per-channel center with a
per-image-per-channel center under one square root — harder to reason about, and with no clear benefit for
a two-way BN/IN slider. So for this rung I blend the standardized features. After the mixture, a single
learnable per-channel affine `γ_c x̂ + β_c` restores representational power — and here is a deliberate
choice worth naming: I apply *one* affine to the *combined* normalized feature, not a separate
scale-and-shift folded into each branch. Folding a `(γ, β)` into each branch would double the affine
parameters and, worse, confound the thing I am trying to measure — the effect of the *blend* — with the
effect of extra affine capacity. The two branches differ only in their standardization statistics; once
they are blended, one per-channel `(γ, β)` is enough to set each channel's output distribution, and a
single affine keeps the parameter count and the optimization surface as close to the plain batch layer as
possible, so any difference in accuracy is attributable to the blend and not to capacity I quietly added.

The next decision is how to compute the batch branch, and this is where I have to be careful to match the
substrate rather than import a heavier machinery the layer does not need. The textbook batch layer keeps a
running mean and variance, accumulated by an exponential average during training and *frozen* for
inference, precisely so that test time — where there may be no batch — still has a population statistic.
That running-average mechanism is the source of the train/test discrepancy the lineage warned about, and
it is genuinely needed when the deployment batch is one image. But here the batch is a fixed 128 in both
train and eval (the loop evaluates over the same 128-image batches), so I do not need a frozen running
buffer to have a stable statistic — I can compute the batch mean and variance *directly from the current
batch* in `forward`, identically in train and eval. That is the honest thing to do for this contract: it
keeps the layer's behavior a single, consistent function of the input, with no hidden state to accumulate
or freeze, and no momentum hyperparameter to tune. So the batch branch is `mean_BN = x.mean over (0,2,3)`,
`var_BN = x.var over (0,2,3)` with the biased (divide-by-`m`) estimator, and `x̂_BN = (x − mean_BN) /
sqrt(var_BN + ε)`. The instance branch is the same arithmetic with the batch axis removed:
`mean_IN = x.mean over (2,3)`, `var_IN = x.var over (2,3)`, `x̂_IN = (x − mean_IN) / sqrt(var_IN + ε)`,
which gives a separate `(mean, var)` per `(image, channel)`. The biased variance is the right one for a
normalization statistic — it is the moment I am standardizing by, not an unbiased estimator I am reporting,
and consistency between the subtraction and the division is what matters. I take `ε = 1e-5` for the usual
numerical floor.

Computing batch statistics on the fly is only safe because the eval batch is a full 128, and it is worth
checking the degenerate end to see *how* the safety is bought. If this layer were ever evaluated with a
single image, `mean_BN` over `(0,2,3)` reduces to a mean over `(2,3)` — the batch axis has length one — so
it becomes exactly `mean_IN`, and `var_BN` collapses onto `var_IN` too: at batch 1 the two branches
coincide and the gate loses all meaning, and the "batch" statistic is no longer a population statistic at
all. That is the failure mode the running-average buffer exists to prevent, and my choice to drop the
buffer is a bet that eval always sees the full 128 so the batch moment is a real population estimate. It
holds under this loop, but I should file it as the one assumption the on-the-fly batch branch is leaning
on: the batch branch is only a batch branch as long as the batch is large, and its behavior is tied to
whatever 128 images happen to be together in the eval batch.

Let me sanity-check the gate initialization, because it sets the whole network's *prior* over the blend.
If I want the rung to begin life behaving like the standard batch layer — the thing I know trains these
nets — the gate should start *batch-leaning*. Initializing the raw parameter `ρ_c = 1.0` gives
`g_c = σ(1) ≈ 0.731`, so every channel starts about three-quarters batch, one-quarter instance, and the
task loss then pushes each channel's gate toward whichever end helps. Two competing initializations are
worth pricing against this. Setting `ρ_c = 0` starts every gate at exactly `0.5` — maximally uncertain,
which sounds neutral, but on the deep ResNets it means washing out a full half of the between-image scale
from step zero, before the optimizer has any evidence about which channels can tolerate it; by the DC-channel
computation above, that halves every channel's between-image signal at initialization on a task where that
signal is the class signal. Setting `ρ_c` large — say 4, giving `g ≈ 0.982`, almost pure batch — looks
safer still, but the sigmoid derivative there is `σ'(4) = σ(4)(1−σ(4)) ≈ 0.018`, so the gate is nearly
frozen and gradient descent can barely move it; the blend would be present in name only. At `ρ = 1` the
derivative is `σ'(1) = σ(1)(1−σ(1)) ≈ 0.197`, an order of magnitude larger — the gate sits in the
well-conditioned middle of the sigmoid where a unit of gradient moves it about 0.2, so it starts near the
known-good batch behavior *and* remains free to slide. That is the deliberate, safe initialization: begin
near the batch layer, keep the gate mobile. I do *not* register a running buffer, a momentum, or a
per-branch affine; the only new degrees of freedom over the plain batch layer are the `C` gate parameters,
one per channel.

And those are nearly free, which is what lets me treat any accuracy change as attributable to the blend
rather than to added capacity. One gate parameter per channel means, summed over a ResNet-56 whose norm
layers carry 16, 32, and 64 channels across its three stages, on the order of a couple of thousand new
scalars against a backbone of roughly 0.85M parameters — well under half a percent, far too few to matter
as capacity or to need regularization. The compute cost is two mean/variance reductions instead of one, so
roughly twice a plain batch layer's forward arithmetic, which is negligible against the convolutions; the
memory cost is holding `x̂_BN` and `x̂_IN` momentarily, a transient doubling of the normalization output.
None of this is a real budget pressure. So the rung is cheap, and cleanly a test of one idea.

Now reason about what this rung must do, because that is the point of running it as the first rung rather
than a later one. The blend is strictly more flexible than either endpoint: at `g = 1` it reproduces the
batch layer exactly, so in principle it can do no worse than batch normalization *if* the gates simply stay
near 1. The bet is that letting some channels slide toward instance — washing out per-image style on the
channels where style is nuisance — buys accuracy on top of that. But there are two ways this bet can come
back thin or negative, and naming them now is what makes the next rung's diagnosis honest. First, the
gradient signal on `ρ_c` is indirect — the gate only helps if the optimizer reliably discovers, through
the classification loss alone, which channels want instance behavior, and under a fixed schedule with no
re-tuning it may simply leave most gates near their batch-leaning init, in which case the rung lands
*near* plain batch and the blend buys little. Second, and more concerning, computing the batch statistic
directly from the current batch with no running average means the **eval** pass standardizes by the
*eval* batch's own moments rather than a stable population statistic — fine when the eval batch is a full
128, but it makes the batch branch's behavior depend on batch composition in a way the running-average
layer was specifically built to avoid, and on the deep CIFAR-100 ResNets, where many channels carry
class-discriminative global scale, washing any of it out toward instance can *cost* accuracy rather than
add it. Instance normalization's known weakness for recognition — destroying the between-image scale, and
estimating it from as few as 64 values on the deep stages — does not disappear just because it is gated; it
only becomes *optional* per channel, and if the optimizer mis-allocates the gates, the instance branch
actively removes signal the batch branch was keeping. These two faults are entangled by construction: both
live in the same forward pass, both hit the harder tasks hardest, and the numbers alone will not cleanly
separate "instance removed scale" from "on-the-fly batch coupled to composition." I should keep that in
mind, because if this rung underperforms I will not yet know which fault dominated.

So here is the falsifiable expectation I will read against the numbers. On the two deep CIFAR-100 ResNets —
the harder, primary targets, where between-image scale is most class-relevant and the instance estimate is
noisiest — I expect this blend to land *at or modestly below* a clean batch layer: the instance branch has
the most to lose there, and the gated mixture, computed batch-statistics-on-the-fly, gives up the
running-average stability for a flexibility the hard 100-class task may not reward. On MobileNetV2 /
FashionMNIST — a near-ceiling, low-class-count run where per-image contrast is more nuisance than signal —
the instance behavior on some channels could plausibly *help*, so I would expect the blend to be at least
even and perhaps slightly ahead there. If that split holds, it tells me something I can build on: that
*blending toward instance* is not the lever on the recognition tasks that matter, and that the open
question for the next rung is not "batch vs instance" but "is there a per-image statistic that escapes the
noise of any single channel's spatial map while staying batch-independent" — i.e. whether pooling over
something larger than one channel's spatial map, and never crossing the batch axis at all, recovers what
this two-way blend leaves on the table. The distilled module that lands this rung is in the answer.
