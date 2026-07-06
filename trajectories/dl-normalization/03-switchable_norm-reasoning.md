Group normalization did most of what I bet it would, and the way to see what it left behind is to put the
two rungs' numbers side by side and read the differences rather than the levels. Against the blend's 66.06 /
68.65 / 93.64, group normalization posted 67.90 / 70.43 / 93.16 — deltas of +1.84 on ResNet-56, +1.78 on
ResNet-110, and −0.48 on MobileNetV2 / FashionMNIST. Two things jump out of those three numbers. First, the
recovery on the ResNets is real and almost exactly equal on the two depths — +1.84 versus +1.78 — which is
mildly against the specific thing I predicted at the last rung, that the gain would be *clearest* on the
deeper ResNet-110 because stable statistics compound over more blocks. It did not scale with depth; the two
gains are within six hundredths of each other. In hindsight that fits the mechanism I actually argued for:
the dominant fix was deleting the batch axis, and batch-freeness is a per-layer property that applies
uniformly, block by block, rather than something that accumulates with depth. So the ResNet result confirms
the batch-composition coupling was the fault that mattered at rung one — a more-instance-like layer got
*better*, which it could not have if between-image-scale removal had been the dominant cost — and it tells
me the depth-compounding story I told was the wrong reason for a right prediction.

Second, and this is the number I have to build on: group normalization *lost* on MobileNetV2 / FashionMNIST,
slipping from the blend's 93.64 to 93.16. That −0.48 is not noise I can wave away, because it inverts the
ranking on that config — the blend, the weaker method on the ResNets, is the *better* method on FashionMNIST,
and it is better there by exactly the thing group normalization gave up. The reason is built into the last
rung's own adaptive grouping rule. To stay legal on the small, awkwardly factored channel counts in
MobileNetV2's inverted residuals, the rule caps groups at the channel count and decrements to divisibility,
so on a 16-, 24-, or 32-channel layer it collapses to one channel per group — instance normalization on that
layer. MobileNetV2 has many such narrow layers, so on exactly the network where the layers are narrowest,
group normalization quietly reverts to the instance-equivalent boundary, and — being *strictly* batch-free
by construction — it can never reach for the batch statistic even though the batch here is a comfortable 128
in both train and eval. The blend still had a batch branch; group normalization deleted it wholesale. So the
0.48 points the blend holds on FashionMNIST are, almost by elimination, the value of the batch corner on the
narrow layers, the one corner group normalization structurally cannot touch.

Put the three configs together and the shape of the remaining problem is stark: no single method so far is
best on all three. The blend wins FashionMNIST (93.64) and loses both ResNets; group normalization wins both
ResNets (67.90, 70.43) and loses FashionMNIST. Each method is best exactly where it can use the statistic the
other cannot — group normalization's batch-free per-image pooling on the deep ResNets, the blend's batch
branch on the narrow FashionMNIST layers. What I want is the *union*: a layer that behaves like batch-free
group/instance pooling where that wins and can dial in the batch statistic where *that* wins, and — crucially
— decides which, per layer, without my hand-writing the rule that stranded group normalization on the narrow
layers. The last rung's failure was not its choice of statistic but its *rigidity*: one fixed grouping and one
fixed pooling rule at every layer, chosen by a heuristic rather than by the task. So the question for this
rung is whether I can let each layer *learn* which statistic it uses.

Let me price the ways to do that, because "learn it" admits several forms and I should reject the ones that
miss the diagnosis. I could keep group normalization and try to *learn the group count* `G` — but `G` is a
discrete structural choice, not differentiable, and relaxing it (Gumbel, a continuous group assignment) is
heavy machinery; worse, it stays inside the batch-free family, so it can slide between instance and layer but
never recovers the batch corner, which is exactly the corner the FashionMNIST miss says I need. Learning `G`
optimizes the wrong axis. I could go per-channel and give every channel its own selection over statistics —
but rung one already tried per-channel selection on this substrate, and the gates mostly failed to earn their
keep under the fixed schedule; a per-channel weighting over three statistics is `3C` parameters chasing a
signal the optimizer under-used at `C`, and it would confound "which corner this layer leans toward" with
"per-channel reshaping." I could add a *proper* running-buffer batch corner — but that re-imports the
momentum and the train/eval discrepancy the lineage warned about, and the edit surface here computes batch
statistics on the fly with a plain parameter and no buffer, stable at 128, which is the honest choice I made
at rung one. What is left, and what actually targets the diagnosis, is to compute *all three* corner
statistics — instance, layer, batch — and let each layer learn a weighting over them, end to end from the
classification loss. That reaches the batch corner group normalization was denied, keeps the selection
per-layer and cheap, and turns the hand-set rigidity into a continuous knob the optimizer sets wherever it
helps.

One more thing to settle before I build it: which corners go in the mix. Batch and instance are obviously in
— they are the two the FashionMNIST and ResNet numbers say I need to trade between. But should I include the
layer corner at all, given that the last rung *rejected* pure layer normalization as over-sharing, the
ten-fold mis-scaling of unlike filters? I think yes, and the reason is exactly the caveat I will hit below.
As a standalone rule the layer statistic is too coarse, but as one corner of a convex mix it is nearly free
— one weight slot the softmax can drive to zero if it is useless — and it is the mix's only *batch-free,
cross-channel* option. Instance is batch-free but single-channel; batch is cross-channel but crosses the
batch axis. The layer corner is the one that pools across channels within an image without touching the
batch, which is precisely the flavour of statistic group normalization's grouping was reaching for. Since my
three corners cannot reproduce a genuine `G`-group average, the layer corner is the closest stand-in I have
for "batch-free pooling over more than one channel," and letting the optimizer blend a little of it with
instance is how this rung approximates what group normalization did on its grouped stages. So I keep all
three: instance and batch because the numbers demand them, layer because it is the cheap batch-free
cross-channel corner and the mix can discard it wherever it does not earn its slot.

Let me build it carefully against the family notation, because the three statistics are *not* independent
quantities I have to recompute from scratch — they are nested reductions of the same feature map and I want
to be precise about which axes each one spans. Take `x` of shape `[B, C, H, W]`. The **instance** statistic
is per `(image, channel)`: `mean_IN = x.mean over (2,3)`, `var_IN = x.var over (2,3)`, the spatial map of
each channel alone. The **layer** statistic is per image: `mean_LN = x.mean over (1,2,3)`,
`var_LN = x.var over (1,2,3)`, all channels and all spatial of one image. The **batch** statistic is per
channel: `mean_BN = x.mean over (0,2,3)`, `var_BN = x.var over (0,2,3)`, that channel across the whole batch
and grid. All three with the biased (divide-by-`m`) variance, consistent with the standardization. These are
the three corners; the layer's job is to mix them.

There is a correctness point in the shapes that I want to check before I mix, because the three corners do
not live on the same tensor shape. With `keepdim`, `mean_IN` is `[B, C, 1, 1]` — a moment for every image
and channel; `mean_LN` is `[B, 1, 1, 1]` — one per image, no channel axis; `mean_BN` is `[1, C, 1, 1]` — one
per channel, no batch axis. A weighted sum of tensors with shapes `[B, C, 1, 1]`, `[B, 1, 1, 1]`, and
`[1, C, 1, 1]` broadcasts to `[B, C, 1, 1]`, so the combined `mean` is per `(image, channel)` — the finest of
the three grains — with the layer term contributing the same value to every channel of an image and the batch
term the same value to every image of a channel. That is exactly what I want: the mix is defined per
`(image, channel)`, and each corner injects its own coarser structure into that grid by broadcasting. The
same holds for the variance. So the combined moment `x − mean` subtracts a `[B, C, 1, 1]` tensor from
`[B, C, H, W]`, broadcasts cleanly, and the output keeps shape — the drop-in contract holds. The compute is
three mean/variance reductions instead of one, roughly three times a batch layer's reduction arithmetic and
still negligible against the convolutions; the three corners are cheap because they are just different axis
choices on the same tensor.

Now the mix, and here I make a different choice than rung one made, deliberately. At rung one, blending two
statistics, I combined the two *standardized features* — `g·x̂_BN + (1−g)·x̂_IN` — to keep each branch a
clean unit-variance signal. Blending three corners with *separate control over centering and scaling*, that
form no longer serves; the natural object to mix is the moments themselves. So I keep two small sets of
learnable weights — one for the mean, one for the variance — each a 3-vector over `{IN, LN, BN}`, passed
through a softmax so they form a convex combination that sums to one: `mean_w = softmax(mean_weight)`,
`var_w = softmax(var_weight)`, and the combined statistics are
`mean = mean_w[0]·mean_IN + mean_w[1]·mean_LN + mean_w[2]·mean_BN` and likewise for `var`. Standardize by
the *combined* mean and variance, `x̂ = (x − mean) / sqrt(var + ε)`, then one per-channel affine `γ x̂ + β`.

Two design points here are load-bearing, and the first is why the weights are *separate* for mean and
variance rather than one shared vector. The right *centering* and the right *scaling* need not come from the
same corner. Take a channel that carries a per-image DC offset — a between-image signal I want to keep, so I
want batch-level centering that preserves where each image sits — but that also carries per-image contrast in
its spatial variance, nuisance I want to wash out, which is instance-level scaling. With one shared weight
the layer cannot say "center like batch, scale like instance"; it must pick one corner for both. Two vectors
let `mean_w` lean toward BN while `var_w` leans toward IN, expressing exactly that decomposition — and by the
DC-channel arithmetic from rung one, that is a real degree of freedom, not a cosmetic one: it is the
difference between keeping a channel's between-image offset while flattening its within-image contrast, and
losing both together. The cost is three extra scalars per layer, which I will show is nothing.

Let me run that channel through the arithmetic to be sure the separate weights actually buy the decomposition
and are not just a story. Take two images on one channel. Image one has spatial mean 5 and spatial variance 4
(contrast, std 2); image two has spatial mean 1 and variance 36 (std 6) — so the between-image signal is the
offset gap 5 versus 1, and the nuisance is the mismatched contrast 2 versus 6. The batch statistic over both
is `mean_BN = 3` and, since the batch variance carries within-image spread plus the between-image mean spread,
`var_BN = ½[(4 + (5−3)²) + (36 + (1−3)²)] = ½(8 + 40) = 24`, std ≈ 4.9. Now compare the three centering/scaling
choices. Pure instance (`mean_w`, `var_w` both on IN) sends image one to `(x−5)/2` and image two to `(x−1)/6`,
both centered at 0 — the offset gap is erased, the between-image signal gone. Pure batch (both on BN) sends
both to `(x−3)/4.9`: image one lands at center `2/4.9 ≈ 0.41` with spread `2/4.9 ≈ 0.41`, image two at
`−0.41` with spread `6/4.9 ≈ 1.22` — the offset gap survives but image two still carries three times image
one's contrast, the nuisance retained. Separate weights with `mean_w` on BN and `var_w` on IN send image one
to `(x−3)/2` — center `1`, spread `1` — and image two to `(x−3)/6` — center `−0.33`, spread `1`: the offset
gap is preserved (1 versus −0.33) *and* both contrasts are flattened to unit scale. Only the mixed
assignment achieves both, and no single shared weight can produce it, because centering-like-batch and
scaling-like-instance are literally different rows of the two weight vectors. So the sixth scalar is earning
a decomposition the fifth cannot.

The second point pins this to the task's edit surface: the weights are **per-layer, shared across all
channels** — a single 3-vector each, applied identically to every channel — not a per-channel weighting. The
contract gives the layer `nn.Parameter(torch.ones(3))` for the mean and the same for the variance: three
numbers per layer, softmaxed. The budget makes this obviously right. Six new scalars per layer against a
per-channel affine of `2C ≥ 32` scalars is a rounding error; summed over the roughly fifty-odd normalization
layers of a ResNet-56 it is on the order of a few hundred parameters against a 0.85M backbone — far too few
to act as capacity, so any accuracy change is attributable to *which corner each layer selects*, decided
globally for the layer, exactly the clean-attribution discipline I held at both earlier rungs. And it is the
right granularity for the diagnosis: the thing that failed at the last rung was a per-*layer* rigidity (this
layer is stuck at instance-equivalent), so a per-layer knob is precisely what unsticks it; per-channel would
be solving a problem I do not have evidence for.

Now I have to be honest about what this rung does and does not contain, because it is tempting to call it a
strict generalization of everything so far and that is not quite true. It genuinely contains the three
*corners*: put all of `mean_w` and `var_w` on the batch slot and the layer is exactly the on-the-fly batch
statistic; all on instance, exactly instance; all on layer, exactly layer — verify by substitution, each
corner is recovered when its weight is one and the others zero. So it strictly generalizes the three named
normalizations. But it does *not* contain group normalization's interior. A `G`-group statistic is an average
over a specific subset of `C/G` channels; my mix offers `mean_IN` (this channel's own map), `mean_LN` (the
average over *all* channels), and `mean_BN` (this channel across the batch), and no convex combination of
"my own" and "everyone's" reproduces "my group of eight's" — the partial average over a chosen block is
simply not on offer from these three corners. So on the ResNet stages where group normalization genuinely
grouped (the 64-channel, two-per-group stage), this rung can only *approximate* it as a blend of instance and
layer, not reproduce it. Where group normalization had *collapsed* to instance-equivalent, though — the
narrow 16- and 32-channel ResNet stages and the many narrow MobileNetV2 layers — this rung can reproduce it
exactly by putting `mean_w` and `var_w` on the instance slot. So the honest statement is a trade: I give up
exact grouping on the handful of layers that were genuinely grouped, and in exchange I gain the batch corner
on every layer. The FashionMNIST number tells me that trade is worth making — the batch corner is where the
last rung's headroom sat — but I should hold it as a bet with a known cost, not pretend the new layer
dominates the old one everywhere by construction.

The softmax is doing more than making the weights interpretable, and it is worth checking it is load-bearing
for stability, not decoration. Because the combined variance is `Σ var_w[k]·var_k` with `var_w` non-negative
and summing to one, it is a convex combination of three non-negative variances, so it is itself non-negative
and `sqrt(var + ε)` is always real — the standardization can never take the root of a negative number. If I
had used unconstrained linear weights instead, a variance combination could go negative on some batch and the
layer would emit NaNs; the softmax is what forecloses that. It also forces the combination to be a genuine
interpolation among the three normalizations rather than an arbitrary linear combination that could amplify
or cancel corners in ways that destabilize training. Two ones per slot at initialization make
`softmax([1,1,1]) = (1/3, 1/3, 1/3)`, so every layer starts as an equal blend of the three corners, and the
softmax Jacobian at the uniform point is well-conditioned — no corner is saturated, all three receive
comparable gradient — so the optimizer is free to slide each layer toward whatever corner or interior point
the loss prefers. Worth noting against rung one: that uniform start puts only `1/3` weight on the on-the-fly
batch corner, against the blend's `σ(1) ≈ 0.73`, and it puts `1/3` on the layer corner the blend never had —
so this rung begins *less* batch-coupled and with a wider reach than the first.

That on-the-fly batch corner is the one residual risk on this rung, and I want to name it precisely. As at
rung one, I compute the batch moments directly from the current batch in `forward`, no running buffer, in
both train and eval; at 128 images that is a large, stable sample — `mean_BN` over `128·H·W` values — so the
eval-composition coupling is small *unless* the optimizer leans the mix heavily onto the batch corner on a
deep stack, where the coupling could add variance the way it did at rung one. The difference now is that the
mix can *learn its way out*: if leaning on batch hurts a given layer, the optimizer can shift `mean_w` and
`var_w` onto the layer or instance corners, which are strictly batch-free. That adaptivity is the whole point
— but it is a hope about what the optimizer finds, not a guarantee, so I will watch for it as the thing that
could go wrong. No running buffers, `ε = 1e-5`, affine `weight`/`bias` at 1 and 0, and the two weight
3-vectors at all-ones so every layer starts uniform.

So the falsifiable expectations against the prior numbers are concrete. On MobileNetV2 / FashionMNIST —
group normalization's weak point at 93.16, where the blend's 93.64 already showed the batch corner is worth
about half a point on the narrow layers — I expect this rung to *recover and exceed* group normalization and
to reach at least the blend's level, because it can dial in exactly the batch statistic the narrow layers
were denied while keeping the batch-free corners everywhere else; this is where I expect the clearest gain
over the last rung. On the two CIFAR-100 ResNets — group normalization at 67.90 and 70.43 — I expect
*match or modest gain*: the mix can reproduce group normalization exactly on the narrow instance-equivalent
stages, approximate its two-channel grouping on the 64-channel stage as an instance/layer blend, and turn the
batch corner *off* where it hurt, and the separate mean/variance freedom can add a little on top — but I would
not expect a large jump, since group normalization already suited the deep stacks well. The risk that would
falsify the bet is the on-the-fly batch corner: if the optimizer over-weights batch on the deepest stack and
the eval-composition coupling bites, ResNet-110 could come in *at or below* 70.43 rather than above. If this
rung clears group normalization across the board — and especially if it reclaims the FashionMNIST config the
blend and group normalization each half-won — it confirms that the productive move at the top of this ladder
is to stop fixing the statistic per layer and *learn* the convex combination of the family's corners, letting
each layer reach the batch statistic or the batch-free ones as its own loss dictates. The distilled module
that lands this rung is in the answer.
