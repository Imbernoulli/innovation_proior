The batch-instance blend told me exactly which lever does *not* work here, and it told me in the numbers I
most cared about. On the two deep CIFAR-100 ResNets the gated mixture landed *below* a clean batch layer —
66.06 on ResNet-56 and 68.65 on ResNet-110 — and only on the near-ceiling MobileNetV2 / FashionMNIST run
did it come in respectable at 93.64. Before I move, let me read the internal structure of those three
numbers, because the shape of the failure is as informative as the fact of it. ResNet-110 sits 2.59 points
above ResNet-56 (68.65 vs 66.06) — a gap I read as capacity, the deeper stack simply having more of it, not
as anything the blend did well; both are the primary hard targets and both underperformed the incumbent.
The FashionMNIST run at 93.64 is where the context warned there is little headroom, so its being respectable
is close to uninformative on its own. That is the split I half-expected — the blend hurts exactly where
between-image scale is most class-relevant and is harmless where per-image contrast is more nuisance — and
it is diagnostic. The whole premise of the first rung was that letting some channels slide toward instance
normalization would wash out nuisance per-image style and buy accuracy; instead, on exactly the tasks where
between-image scale is most class-relevant, the instance branch *cost* accuracy.

But I have to be honest about *what* the numbers can and cannot pin down, because I named two faults at rung
one and these three numbers do not, by themselves, tell them apart. One fault: the instance branch, gated in
per channel, removes the global per-image magnitude that the deep 100-class ResNets rely on to tell images
apart — so any gate the optimizer pushed below 1 on a shape-carrying channel actively deleted signal, and on
the deepest 64-channel stage it estimated that removal from only 64 spatial values. The other fault: I
computed the batch branch's mean and variance *directly from the current batch* in `forward`, with no
running average — fine for stability at batch 128, but it ties the standardization to the particular batch's
composition, and on a deep stack that adds a second source of variance on top of the instance branch's
removal of scale. Both faults hurt the harder tasks more; both are silent on the near-ceiling one. The split
I see is consistent with either fault, or both, and I cannot regress them apart from a single seed on three
configs. What I *can* do is design the next rung so that it removes both at once, and — better — so that its
result will discriminate between them. That is the standard I hold this rung to. So I stop asking "how much
of each image's own style to remove" and ask the question the lineage actually poses: which feature
positions should share one statistic, if I want a population that is large, stable, and never crosses the
batch axis the wrong way.

Let me re-read the family in one notation, because that is where the right move becomes visible. Index a
feature position by `i = (i_N, i_C, i_H, i_W)` — which image, which channel, which row, which column. Every
member of the family does the identical arithmetic: pick a set `S_i` of positions that share one
`(mean, var)`, standardize, then a per-channel affine. The only thing distinguishing the methods is the
*membership rule* for `S_i`. The batch layer: `S_i = {k : k_C = i_C}` — same channel, spanning `(N, H, W)`.
Instance: `S_i = {k : k_N = i_N, k_C = i_C}` — same image *and* same channel, just `(H, W)`. Layer:
`S_i = {k : k_N = i_N}` — same image, all of `(C, H, W)`. Written this way, the first rung's failure has a
crisp reading: the instance set fixes `k_C = i_C`, so each channel is standardized using *only its own*
spatial map — `H·W` values, nothing else — which is both a small, noisy sample on the coarse feature maps
of a deep CIFAR ResNet (32×32, 16×16, 8×8 spatial, so as few as 64 values at the deepest stage) and, worse,
blind to every correlated neighbor channel.
That is the under-sharing extreme. The layer set goes to the opposite extreme: it drops the `k_C` condition
entirely, pooling *all* channels of an image into one statistic — but conv channels are different filters
(edges, colors, textures) whose response distributions genuinely differ, so one shared mean/variance over
all of them is too coarse.

Let me make that over-sharing concrete, because it is easy to wave at and worth an actual number. Suppose a
layer has an edge-detector channel whose activations sit around `N(0, 1)` and a colour-blob channel around
`N(0, 100)` — a hundred-fold variance gap is unremarkable between a sharp high-frequency filter and a broad
low-frequency one. The layer statistic pools both into one variance, roughly the average, `≈ 50.5`, so it
divides every channel by `√50.5 ≈ 7.1`. The edge channel, whose real scale was 1, is squashed to about
`0.14` standard deviations; the colour channel, whose real scale was 10, comes out at about `1.4`. The
shared standardization has mis-scaled the two channels by a factor of ten relative to each other, and the
per-channel affine downstream now has to spend its `γ` undoing damage the normalization created. That is the
over-sharing extreme made arithmetic: one statistic over unlike filters serves neither.

So the two batch-free options are opposite failures of *how many channels share a statistic*: instance
shares with exactly one channel (its own), layer shares across all of them. Both fix `k_N = i_N` — same
image — which is what makes them batch-independent, and that is the property I now want, because the first
rung's second problem was precisely the batch-coupled branch. Before I commit to the move between them, let
me price the alternatives that stay on the old axis, because I should have to reject them explicitly. I
could keep the blend and *fix only its batch branch* — give it a proper running mean and variance so eval
no longer depends on batch composition. But that reintroduces exactly the momentum hyperparameter and
train/eval discrepancy the lineage warned about, and it leaves the instance branch's scale-removal and
64-value noise untouched; it patches one fault on an axis the numbers already told me is losing. I could
retreat to pure batch — but that is just reverting to the scaffold, abandoning any headroom, and it keeps
the on-the-fly composition coupling unless I also add the running buffer. I could go to pure instance — but
rung one already showed instance-leaning hurts the ResNets, and pure instance is the extreme of it. Every
one of these either stays on the batch-versus-instance axis or re-imports the running-buffer machinery. The
move that leaves the axis entirely and stays batch-free sits *between* the two batch-free extremes: share a
statistic across *some* channels — more than one, fewer than all. Group the channels into contiguous blocks
and pool within a block. This is not splitting the difference for its own sake; it is the granularity the
data has been pointing at all along. Classical vision descriptors normalize *within groups* — a HOG block
is a group of orientation bins normalized together, SIFT does the same, VLAD and Fisher Vectors are stacks
of sub-vectors each normalized within itself — because coefficients inside a group belong together while
different groups deserve their own scale. Conv channels behave the same way: a filter and its horizontal
flip have near-identical response distributions and are natural groupmates; orientation, frequency, and
texture each induce families of interdependent channels. The interior of the dial is where the method
lives, and the two extremes I just rejected are its boundary.

Make the membership rule explicit. Split the `C` channels into `G` contiguous groups of `C/G` channels
each; two channels are in the same group when `⌊k_C/(C/G)⌋ = ⌊i_C/(C/G)⌋`. Then
`S_i = {k : k_N = i_N, ⌊k_C/(C/G)⌋ = ⌊i_C/(C/G)⌋}` — for a given image and group, the set is all `C/G`
channels of the group across all `H·W` positions. Standardize by that group's `(mean, var)`, then a
per-channel affine `γ x̂ + β` (kept per-channel even though the statistic is per-group, so each channel can
re-scale after the shared standardization and no representational power is lost). Check the extremes fall
out, because the interpolation claim rests on it: at `G = 1` there is one group, `⌊k_C/C⌋ = 0` for every
channel, so the channel condition is vacuous and `S_i = {k : k_N = i_N}` — the layer set exactly. At
`G = C` each group is one channel, `⌊k_C/1⌋ = k_C`, so the condition becomes `k_C = i_C` and
`S_i = {k : k_N = i_N, k_C = i_C}` — instance exactly. So grouping *interpolates* between the two batch-free
extremes, and any `1 < G < C` is a new intermediate that neither of the endpoints was — and the endpoints
are precisely the two failures I want to avoid.

Now I have to be careful, because pinning this to *this task's edit surface* is where the clean story meets
the awkward channel counts, and the awkwardness changes what group normalization actually does here. The
backbones run normalization on channel counts of 16, 32, 64 (the CIFAR ResNet stages) and a spread inside
MobileNetV2's inverted residuals (16, 24, 32, 64, 96, 160, 320, 1280, plus expansion hiddens). A fixed `G`
of, say, 32 would be illegal on a 16-channel layer (more groups than channels) and on any count not
divisible by 32. So the rule has to *adapt* `G` to `C`: start from a sensible round default of 32 groups,
cap it at the channel count, `num_groups = min(32, num_features)`, and then walk it *down* until it divides
the channel count — `while num_features % num_groups != 0: num_groups -= 1`. Let me trace what that
actually produces, because the trace is the design. On the 16-channel ResNet stage: `min(32,16) = 16`,
`16 % 16 = 0`, so 16 groups, one channel each — which *is* instance normalization on that layer. On the
32-channel stage: `min(32,32) = 32`, one channel per group again — instance. On the 64-channel stage:
`min(32,64) = 32`, two channels per group — the only genuinely grouped stage in the ResNets. So on the CIFAR
ResNets group normalization is instance-equivalent on two of three stages and a two-channel group on the
third. On MobileNetV2 the trace is similar: 16 → 16 groups (1/group), 24 → 24 (1/group), 32 → 32 (1/group),
64 → 32 (2/group), 96 → 32 (3/group), 160 → 32 (5/group), 320 → 32 (10/group), 1280 → 32 (40/group); and an
expansion hidden like 144, which is `2⁴·3²` with no divisor between 25 and 32, decrements all the way from
32 down to 24 (6/group) — a reminder the loop can walk several steps. The termination is worth one check:
`num_groups` decreases from `min(32, C)` and `C % 1 = 0` for every `C`, so in the worst case it halts at one
group; it can never underflow past 1 or divide by zero. The decrement is a graceful fallback, not a separate
design — where the channel count is small or awkwardly factored, the layer simply slides toward the
instance-equivalent end of the same dial it already lives on.

The default of 32 groups is itself a choice I should be able to defend rather than inherit. The number
trades two pressures against each other: more groups means finer channel granularity (closer to instance,
each group a purer set of like filters) but a smaller within-group sample; fewer groups means a larger,
steadier sample but coarser pooling that starts to conflate unlike filters (toward the layer extreme I just
rejected with the ten-fold mis-scaling). On these narrow backbones the pressure is asymmetric. The widest
common ResNet stage is 64 channels, so any default of 32 or more is already capped down to at most 32
groups there and to the channel count on everything narrower; a default of 16 would coarsen the 64-channel
stage to 4-channel groups with no benefit I can see, while a default well above 32 would simply be clamped
back by `min(32, C)` on every layer here and never take effect. So 32 is the smallest round anchor that
lets the *wider* MobileNetV2 layers (96, 160, 320, 1280) actually group several channels together while
leaving the ResNet stages where the cap and the divisibility walk put them. It is a heuristic, and its
brittleness on awkward counts like 144 is exactly the rigidity I flag at the end — but as a fixed anchor for
this rung it is the defensible round number, not an arbitrary one.

This trace forces me to be honest about the mechanism, and the honesty sharpens the prediction. If group
normalization is instance-equivalent on two of the three ResNet stages, then the story cannot be "grouping
uses cross-channel structure the instance branch lacked" on those stages — there *is* no cross-channel
pooling when the group is one channel. So why should this recover the ResNets at all, when it is, if
anything, *more* instance-like than the 73%-batch blend was? Two things differ, and separating them is the
point. First, the spatial sample. Group normalization keeps `(H, W)` in `S_i` always, so even a one-channel
group pools a full spatial map — 1024 values at the 16-channel stage, 256 at the 32-channel stage — and on
the two-channel 64-channel stage it pools `2 × 8 × 8 = 128` values against the deep blend's instance branch,
which had only `1 × 8 × 8 = 64`. Doubling the sample there cuts the variance-of-variance roughly in half; on
the idealized `√(2/(n−1))`, the relative noise on the per-group scale falls from about 0.178 to about 0.125,
near a 30% reduction, exactly where the maps are coarsest. Second, and I think dominant: group normalization
has *no batch axis at all*. Its statistic is computed from one image, identically in train and eval, with no
running buffer, no momentum, and no dependence on which 128 images share the eval batch. That removes the
first rung's second fault outright — not by fixing the batch branch but by deleting it. So my expectation
is that the recovery, if it comes, is driven mainly by killing the batch-composition coupling and stabilizing
the estimate, not by the between-image-scale story — and this is the discriminating prediction I promised
myself: because group normalization is *more* instance-like than the blend on the narrow ResNet stages, if
between-image-scale removal had been the dominant fault at rung one, this rung would make the ResNets *worse*,
not better. A clear recovery would therefore implicate the on-the-fly batch coupling as the fault that
mattered; a flat or negative result on the ResNets would say scale removal was the real cost after all. The
numbers will adjudicate a question rung one left genuinely open.

It helps to see the reduction as a shape derivation, because it makes both the cost and the per-channel
affine choice concrete. Take the deepest ResNet stage, a feature map `[128, 64, 8, 8]` at `G = 32`.
`nn.GroupNorm` reshapes the channel axis into `[128, 32, 2, 8, 8]` — batch, group, within-group channel,
height, width — computes mean and variance over the last three axes `(2, 8, 8)`, giving one `(mean, var)`
pair per `(image, group)`, a `[128, 32]` grid of moments; it standardizes with those moments broadcast back
over the collapsed axes and then reshapes to `[128, 64, 8, 8]`. Each pair is estimated from `2·8·8 = 128`
values, and the whole operation is a single pass of two reductions — the same order of arithmetic as one
batch layer, no extra buffers. The affine that follows is per *channel*, all 64 of them, not per group, and
that is deliberate: two channels sharing a group were standardized by one common `(mean, var)`, but they are
still different filters and may want different output scales, so giving each its own `γ, β` lets the layer
re-separate what the shared standardization pooled — no representational power is lost by sharing the
statistic, because the affine is left free to un-share the scale. That is the same reasoning that kept the
affine per-channel at rung one; here it also quietly answers why an interior `G` is not a compromise on
expressiveness: the group only shares the *estimate*, never the final scale.

I should also check that the instance-equivalent stages are not silently reintroducing rung one's noise, and
the spatial samples say they are not. Group normalization at the 16-channel stage pools 1024 values per
one-channel group and at the 32-channel stage 256 — on the idealized `√(2/(n−1))`, relative noise of about
4% and 6% on the per-group scale, against the deep blend's 18% at 64 values. So even where group
normalization collapses to instance-equivalent on the ResNets, it is a *stable* instance because it always
keeps the whole spatial map in `S_i`, whereas the blend's instance branch was noisiest precisely on the
coarse deep stage. This is why I keep insisting the driver is batch-freeness plus sample stability rather
than grouping per se: on two of three ResNet stages there is no grouping at all, only a batch-free,
spatially-large per-image statistic — and that alone should be a better-behaved layer than a batch/instance
blend whose batch half is composition-coupled at eval.

That same trace makes me *less* sure about MobileNetV2, and for the mirror reason. Many of its inverted-residual
layers are narrow — 16, 24, 32 channels — and on every one of those the decrement rule collapses to one
channel per group, so group normalization there is exactly instance normalization, inheriting its
under-sharing with no cross-channel help. Worse for this task, group normalization is *strictly* batch-free
by construction, so on a network where the batch of 128 is comfortable and per-image contrast on FashionMNIST
may be as much signal as nuisance, it can never use the batch statistic even where the batch statistic would
help. So I would not bet on a gain on the near-ceiling run; roughly even is the honest expectation, and a
small slip would not surprise me. I delegate the actual moment computation to the framework's
group-normalization primitive `nn.GroupNorm(num_groups, num_features)`, which reshapes the channels into the
group axis, reduces over the within-group channels and all spatial positions with the biased variance,
standardizes, and applies the per-channel `γ, β` — exactly the `S_i` I derived, with `ε = 1e-5`.

So the delta from rung one is concrete and addresses its measured failure directly. Where the gated blend
slid along batch-versus-instance and lost accuracy on the deep ResNets by deleting between-image scale and
coupling to batch composition, this rung abandons that axis entirely: it pools within channel groups inside
one image, a population that is batch-free (fixing the composition coupling), spatially large and — on the
wider layers — cross-channel (stabilizing the noisy isolated-channel estimate), and adaptively sized to
these small layer widths. Here is the falsifiable expectation against rung one's numbers. On the two
CIFAR-100 ResNets — 66.06 and 68.65 under the blend — I expect group normalization to *recover and exceed*
that, driven by the batch-free stability I just argued for, with the clearest gain on the deeper ResNet-110
where stable per-layer statistics have more blocks over which to compound. On MobileNetV2 / FashionMNIST —
93.64 under the blend — I am less sure of a gain and would accept roughly even, because its many small,
awkwardly factored channel counts collapse the group statistic toward instance-equivalent and the strictly
batch-free rule denies it the comfortable batch-128 the blend could still reach. If that read holds — clear
recovery on the deep CIFAR ResNets, roughly even or a touch down on the saturated FashionMNIST run — it
confirms that the productive axis is *grouping inside one image and never touching the batch axis*, not
*batch-vs-instance weighting*, and it sets up the open question for the next rung: group normalization
commits to one fixed grouping and one fixed pooling rule per layer, chosen by a hand-written heuristic
rather than by the task, and on the narrow layers that heuristic strands it at the instance-equivalent
boundary with the batch statistic locked out. The remaining headroom is whether *letting the layer choose*,
rather than fixing the rule by hand, can reach the statistics this rung cannot. The distilled module that
lands this rung is in the answer.
