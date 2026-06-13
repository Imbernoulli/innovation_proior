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

So I have two members of the same family that fail in *opposite* directions: batch keeps the
discriminative between-image scale but also keeps nuisance style; instance kills nuisance style but also
kills the discriminative scale. The classic move when two estimators err oppositely is to interpolate
between them — and crucially, the right interpolation weight is almost certainly not the same for every
channel. Some channels of a conv layer carry mostly style-like, contrast-like information (low-level
texture filters, color blobs) where I want the instance behavior; others carry shape and structure that is
class-relevant across images, where I want the batch behavior. A *single* global blend cannot serve both.
What I want is a **per-channel gate** that learns, channel by channel, how much of each image's own style
to wash out versus how much of the batch-level scale to keep — and to learn it from the task loss, so the
network decides per channel which of the two behaviors helps classification. That is the whole idea of the
first rung: at each `CustomNorm`, compute both the batch-normalized and the instance-normalized version of
the feature, and combine them per channel with a learnable weight.

Now make it concrete against *this task's edit surface*, because the contract is narrow — `CustomNorm`
must be a drop-in for `nn.BatchNorm2d`, constructed as `CustomNorm(C)`, mapping `[B, C, H, W]` to the same
shape, stable in train and eval — and the details of how I compute the two branches matter for both
correctness and what I can expect from the numbers. Let `g_c ∈ [0, 1]` be the per-channel gate; I store it
as a raw real parameter `ρ_c` and pass it through a sigmoid so it stays in the unit interval without a
constraint in the optimizer, `g_c = σ(ρ_c)`. The mixed output, before the affine, is
`x̂ = g · x̂_BN + (1 − g) · x̂_IN`, broadcasting `g` over `(B, H, W)` so each channel uses its own gate.
When `g_c = 1` the channel is pure batch normalization; when `g_c = 0` it is pure instance normalization;
in between it is a true blend of the two standardizations. After the mixture, a single learnable
per-channel affine `γ_c x̂ + β_c` restores representational power — and here is a deliberate choice worth
naming: I apply *one* affine to the *combined* normalized feature, not a separate scale-and-shift folded
into each branch. The two branches differ only in their standardization statistics; once they are blended,
one per-channel `(γ, β)` is enough to set each channel's output distribution, and a single affine keeps the
parameter count and the optimization surface as close to the plain batch layer as possible, so any
difference in accuracy is attributable to the *blend*, not to extra affine capacity.

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

Let me sanity-check the gate initialization, because it sets the whole network's *prior* over the blend.
If I want the rung to begin life behaving like the standard batch layer — the thing I know trains these
nets — the gate should start *batch-leaning*. Initializing the raw parameter `ρ_c = 1.0` gives
`g_c = σ(1) ≈ 0.73`, so every channel starts about three-quarters batch, one-quarter instance, and the
task loss then pushes each channel's gate toward whichever end helps. This is the deliberate, safe
initialization: I begin near the known-good batch behavior and let gradient descent decide, per channel,
how far to slide toward instance. I do *not* register a running buffer, a momentum, or a per-branch affine;
the only new degrees of freedom over the plain batch layer are the `C` gate parameters, one per channel.

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
add it. Instance normalization's known weakness for recognition — destroying the between-image scale — does
not disappear just because it is gated; it only becomes *optional* per channel, and if the optimizer
mis-allocates the gates, the instance branch actively removes signal the batch branch was keeping.

So here is the falsifiable expectation I will read against the numbers. On the two deep CIFAR-100 ResNets —
the harder, primary targets, where between-image scale is most class-relevant — I expect this blend to land
*at or modestly below* a clean batch layer: the instance branch has the most to lose there, and the gated
mixture, computed batch-statistics-on-the-fly, gives up the running-average stability for a flexibility the
hard 100-class task may not reward. On MobileNetV2 / FashionMNIST — a near-ceiling, low-class-count run
where per-image contrast is more nuisance than signal — the instance behavior on some channels could
plausibly *help*, so I would expect the blend to be at least even and perhaps slightly ahead there. If
that split holds, it tells me something I can build on: that *blending toward instance* is not the lever on
the recognition tasks that matter, and that the open question for the next rung is not "batch vs instance"
but "is there a per-image *all-channel* statistic that escapes the noise of any single channel's spatial
map while staying batch-independent" — i.e. whether bringing the layer statistic and other normalizers
into the mix, rather than just batch and instance, recovers what this two-way blend leaves on the table.
The distilled module that lands this rung is in the answer.
