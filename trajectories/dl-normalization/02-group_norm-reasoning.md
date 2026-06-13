The batch-instance blend told me exactly which lever does *not* work here, and it told me in the numbers I
most cared about. On the two deep CIFAR-100 ResNets the gated mixture landed *below* a clean batch layer —
66.06 on ResNet-56 and 68.65 on ResNet-110 — and only on the near-ceiling MobileNetV2 / FashionMNIST run
did it come in respectable at 93.64. That is the split I half-expected, and it is diagnostic. The whole
premise of the first rung was that letting some channels slide toward instance normalization would wash out
nuisance per-image style and buy accuracy; instead, on exactly the tasks where between-image scale is most
class-relevant, the instance branch *cost* accuracy. Two things were going on at once and the rung
entangled them. One: the instance branch, gated in per channel, removes the global per-image magnitude that
the deep 100-class ResNets rely on to tell images apart — so any gate the optimizer pushed below 1 on a
shape-carrying channel actively deleted signal. Two: I computed the batch branch's mean and variance
*directly from the current batch* in `forward`, with no running average — fine for stability at batch 128,
but it ties the standardization to the particular batch's composition, and on a deep stack that adds a
second source of variance on top of the instance branch's removal of scale. The lesson is sharp: the axis
I was sliding along — *batch versus instance, weighted per channel* — is not the axis that helps these
recognition tasks. Blending toward instance is the wrong direction. So I should stop asking "how much of
each image's own style to remove" and ask the question the lineage actually poses: **which feature
positions should share one statistic, if I want a population that is large, stable, and never crosses the
batch axis the wrong way.**

Let me re-read the family in one notation, because that is where the right move becomes visible. Index a
feature position by `i = (i_N, i_C, i_H, i_W)` — which image, which channel, which row, which column. Every
member of the family does the identical arithmetic: pick a set `S_i` of positions that share one
`(mean, var)`, standardize, then a per-channel affine. The only thing distinguishing the methods is the
*membership rule* for `S_i`. The batch layer: `S_i = {k : k_C = i_C}` — same channel, spanning `(N, H, W)`.
Instance: `S_i = {k : k_N = i_N, k_C = i_C}` — same image *and* same channel, just `(H, W)`. Layer:
`S_i = {k : k_N = i_N}` — same image, all of `(C, H, W)`. Written this way, the first rung's failure has a
crisp reading: the instance set fixes `k_C = i_C`, so each channel is standardized using *only its own*
spatial map — `H·W` values, nothing else — which is both a small, noisy sample on the coarse feature maps
of a deep CIFAR ResNet (16×16, 8×8, 4×4 spatial) and, worse, blind to every correlated neighbor channel.
That is the under-sharing extreme. The layer set goes to the opposite extreme: it drops the `k_C` condition
entirely, pooling *all* channels of an image into one statistic — but conv channels are different filters
(edges, colors, textures) whose response distributions genuinely differ, so one shared mean/variance over
all of them is too coarse. The over-sharing extreme.

So the two batch-free options are opposite failures of *how many channels share a statistic*: instance
shares with exactly one channel (its own), layer shares across all of them. Both fix `k_N = i_N` — same
image — which is what makes them batch-independent, and that is the property I now want, because the first
rung's second problem was precisely the batch-coupled branch. The resolution sits right between the two
extremes: share a statistic across *some* channels — more than one, fewer than all. Group the channels into
contiguous blocks and pool within a block. This is not splitting the difference for its own sake; it is the
granularity the data has been pointing at all along. Classical vision descriptors normalize *within
groups* — a HOG block is a group of orientation bins normalized together, SIFT does the same, VLAD and
Fisher Vectors are stacks of sub-vectors each normalized within itself — because coefficients inside a
group belong together while different groups deserve their own scale. Conv channels behave the same way: a
filter and its horizontal flip have near-identical response distributions and are natural groupmates;
orientation, frequency, and texture each induce families of interdependent channels. The interior of the
dial is where the method lives, and the two extremes I just rejected are its boundary.

Make the membership rule explicit. Split the `C` channels into `G` contiguous groups of `C/G` channels
each; two channels are in the same group when `⌊k_C/(C/G)⌋ = ⌊i_C/(C/G)⌋`. Then
`S_i = {k : k_N = i_N, ⌊k_C/(C/G)⌋ = ⌊i_C/(C/G)⌋}` — for a given image and group, the set is all `C/G`
channels of the group across all `H·W` positions. Standardize by that group's `(mean, var)`, then a
per-channel affine `γ x̂ + β` (kept per-channel even though the statistic is per-group, so each channel can
re-scale after the shared standardization and no representational power is lost). Check the extremes fall
out: `G = 1` collapses the channel condition (every channel maps to group 0), giving the layer set
exactly; `G = C` makes each group a single channel, giving instance exactly. So grouping *interpolates*
between the two batch-free extremes, and any `1 < G < C` is a new intermediate that neither of the
endpoints was — and the endpoints are precisely the two failures I want to avoid.

Why should an interior `G` beat both ends, and why is it the right answer to the first rung's failure
specifically? At an interior `G` the statistic for a channel is pooled over its whole *group* — `C/G`
channels over the full spatial grid — so correlated channels stabilize each other's estimate and the
cross-channel structure that instance threw away is used. That directly fixes the under-sharing that made
the gated-instance branch noisy and signal-destroying on the deep ResNets. And because the set never
crosses the batch axis (`k_N = i_N` is in the rule), the statistic is computed identically in train and
eval from one image alone — there is no running buffer to accumulate and freeze, and no dependence on batch
composition. That fixes the *second* problem from rung one, the batch-coupled branch I computed on the fly:
group normalization simply has no batch axis in its definition, so the eval pass standardizes by the
image's own group statistic, the same function it used in training. One root cause for the first rung's two
symptoms — instance's removal of scale and the batch branch's composition dependence — and a single move
that removes both: pool within channel groups inside one image.

I should check the sample size, because the first rung's instance branch was hurt partly by estimating
over too few values. On a ResNet `BasicBlock` with, say, 64 channels and a 16×16 feature map, a group at
`G = 32` holds `C/G = 2` channels, so the pool is `2 × 16 × 16 = 512` values per `(image, group)` — and at
the 32-channel stage with an 8×8 map a group of 1–2 channels pools `64`–`128` values. These are far larger
and, crucially, batch-independent samples than a single channel's spatial map gave instance, and they are
guaranteed regardless of batch because I always keep the spatial axes in `S_i`. Keeping `(H, W)` in the set
is doing real work: it is what makes even one channel-group on one image a comfortable sample. So the
estimate is stable and never touches `N`.

Now the part that pins this rung to *this task's edit surface*, because the contract is narrow and the
channel counts here are small and uneven, which is exactly where the grouping rule needs care. The
backbones run normalization on channel counts of 16, 32, 64 (the CIFAR ResNet stages) and a spread of
counts inside MobileNetV2's inverted residuals (16, 24, 32, 64, 96, 160, 320, 1280, plus expansion
hiddens). A fixed `G` of, say, 32 would be illegal on a 16-channel layer (more groups than channels) and on
any count not divisible by 32. So the rule has to *adapt* `G` to `C`: start from a sensible round default
of 32 groups, cap it at the channel count, `num_groups = min(32, num_features)`, and then walk it
*down* until it divides the channel count — `while num_features % num_groups != 0: num_groups -= 1`. This
is the literal edit I make. On a 64-channel layer it gives 32 groups (2 per group); on 32 channels, 32
groups (1 per group — which, note, is exactly instance normalization on that layer, the `G = C` boundary);
on 16 channels, 16 groups (again 1 per group, instance-equivalent); on a 96-channel MobileNetV2 layer it
backs off from 32 to 32 (96 is divisible by 32, 3 per group); on 160 it lands on 32 (5 per group); on 24 it
caps at 24 then settles at 24 (1 per group). The decrement is a graceful fallback, not a separate design:
where the channel count is small or awkwardly factored, the layer simply slides toward the
instance-equivalent end of the same dial it already lives on. I delegate the actual moment computation to
the framework's group-normalization primitive `nn.GroupNorm(num_groups, num_features)`, which reshapes the
channels into the group axis, reduces over the within-group channels and all spatial positions, standardizes,
and applies the per-channel affine — exactly the `S_i` I derived, with the biased variance and the
per-channel `γ, β` built in.

So the delta from rung one is concrete and addresses its measured failure directly. Where the gated blend
slid along batch-versus-instance and lost accuracy on the deep ResNets by deleting between-image scale and
coupling to batch composition, this rung abandons that axis entirely: it pools within channel groups inside
one image, a population that is batch-free (fixing the composition coupling), large and cross-channel
(fixing the noisy isolated-channel estimate), and adaptively sized to these small layer widths. Here is the
falsifiable expectation against rung one's numbers. On the two CIFAR-100 ResNets — 66.06 and 68.65 under
the blend — I expect group normalization to *recover and exceed* that, because the very thing that hurt the
blend there (instance's scale removal, computed per isolated channel) is replaced by a stable group
statistic that uses cross-channel structure without ever forcing the over-removal of per-image scale; the
gain should be clearest on the deeper ResNet-110 where stable per-layer statistics matter most across the
many blocks. On MobileNetV2 / FashionMNIST — 93.64 under the blend — I am less sure of a gain and would
accept roughly even: many of its layers have small or awkward channel counts where the decrement rule
collapses the group statistic toward instance-equivalent (1 channel per group), so on those layers group
normalization *is* essentially instance normalization and inherits its under-sharing, and the near-ceiling
task leaves little headroom either way. If that read holds — clear recovery on the deep CIFAR ResNets,
roughly even on the saturated FashionMNIST run — it confirms that the productive axis is *grouping inside
one image*, not *batch-vs-instance weighting*, and it sets up the open question for the next rung: group
normalization commits to one fixed grouping and one fixed pooling rule per layer, so the remaining headroom
is whether *learning* the combination of batch/layer/instance statistics per layer — rather than fixing the
group structure by hand — squeezes out more, especially on the layers where the decrement rule left group
normalization stuck at the instance-equivalent boundary. The distilled module that lands this rung is in
the answer.
