I want to start from the piece of prior art that reacts most directly against a constraint here, because
that constraint reshapes the whole idea. The thing that bugs me is dropout: it is the best injected noise
we have, and it has quietly disappeared from the
convolutional layers of every strong modern architecture, kept at most on a final dense head. That is
strange, because the convolutional layers are where the spatial parameters live. The reason it fails
there is correlation. A convolutional feature map is one kernel slid over a spatially smooth input;
neighboring output positions look at heavily overlapping receptive fields, so two adjacent activations
in the same channel carry almost the same number. Zero one of them and the next layer's convolution,
sliding over the hole, simply reads the same edge or texture off the surviving neighbors. I removed a
unit and removed nothing — the information leaked around the hole, and the "thinned sub-network" I
thought I sampled was never actually thinned. The cure, in the activation-masking world, is to remove
information at the *scale of correlation*: delete a contiguous `block_size × block_size` square so there
are no surviving correlated neighbors inside the hole to carry the signal onward. That is the
DropBlock idea, and it is where my intuition starts.

But this task does not let me build that. The activation-masking version is a layer inserted into the
forward graph — it samples a Bernoulli seed field, expands each seed into a square with a stride-1
max-pool, multiplies the activation tensor by the keep-mask, rescales by the realized survival
fraction, and is a no-op at inference. Every one of those operations mutates the model's forward pass.
The edit surface here is a single function, `compute_regularization`, that returns one differentiable
scalar *added to the loss* — it cannot insert a layer, cannot mask an activation that the next
convolution will consume, and cannot change inference behavior because it only contributes to the
training loss. So I must take the *insight* of DropBlock — discourage the network from relying on
contiguous spatial regions — and re-express it as a loss-side penalty on something I am allowed to
touch.

Before I settle on what to touch, let me enumerate the re-expressions available, because the honest one
is not the first that comes to mind. The most faithful transplant keeps acting on *activations*: register
forward hooks on the conv layers, capture the intermediate feature maps, and penalize a spatial
co-activation statistic on them directly — the thing DropBlock literally attacks. I reject it on cost and
fragility. The forward pass that produced `outputs` has already happened by the time my function is
called; to see intermediate maps I must re-run the network or leave hooks resident, either way carrying
an extra activation-sized backward graph through every conv on every step, across three full 200-epoch
runs where the contract asks for something cheap. And a hook-based penalty is stateful machinery bolted
onto a function meant to return a pure scalar — the right penalty in the wrong container. A second
candidate acts on the weights through a different notion of structure: penalize the spatial *total
variation* of the kernel, `Σ |w[i+1] − w[i]|`, to make kernels smooth. I reject this on mechanism: a
smooth kernel is not a de-concentrated one — a smooth Gaussian blob is about as *contiguous* a detector
as exists, all its energy in one soft central lump — so a smoothness penalty would reward the very
contiguous-region reliance I want to discourage. That leaves acting on the weights through their *energy*
distribution, cheap and differentiable every step. So the question narrows: what penalty on the
convolutional *weights* discourages contiguous spatial co-activation, in the spirit of dropping
contiguous blocks of the feature map?

Let me think about where contiguous spatial structure lives in a convolution. The output at a position
is the inner product of the kernel with the input patch under it. A kernel whose weight energy is
concentrated in a tight, contiguous spatial cluster of taps responds maximally to a contiguous patch
of input — it is, in effect, a detector that fires on a small contiguous region. That is exactly the
kind of reliance DropBlock fights at the activation level: the network leaning on a contiguous chunk of
the map. So if I want a weight-side analogue, I should penalize convolutional kernels whose magnitude is
spatially clustered within local blocks of the kernel itself. A kernel whose energy is spread out, or
small inside any local block, is less of a contiguous-region detector; a kernel with a hot contiguous
sub-block is more of one. Penalizing the local block energy of the kernel pushes filters away from being
sharp contiguous-region detectors and toward smoother, more distributed responses — the weight-space
echo of "don't let any contiguous region carry everything."

How do I measure "local block energy" of a kernel differentiably? Take a Conv2d weight `w` of shape
`[out_c, in_c, kH, kW]`. I care about spatial structure, not which input channel it came from, so I
collapse the input-channel axis. There is a choice here worth a sentence: I could collapse by *max* over
input channels — take the single hottest channel's spatial pattern — or by *mean*. Max would let one
loud input channel dictate the whole filter's penalty and is non-smooth in a way that jitters the
gradient; the natural energy aggregate is the mean, which reads the filter's *typical* spatial profile
across the channels it mixes. So `w_sq = w.pow(2).mean(dim=1, keepdim=True)`, shape `[out_c, 1, kH, kW]`.
This is a small spatial energy map for each filter. Now I want the energy *within local contiguous
blocks* of that map. A block of side `block_size` centered at each tap, averaged, is exactly an average
pool with kernel `block_size`, stride 1, and `padding = block_size // 2` to keep the spatial size:
`local = F.avg_pool2d(w_sq, block_size, stride=1, padding=pad)`. Each entry of `local` is the mean
squared weight magnitude in the `block_size × block_size` neighborhood around a tap. Take the mean of
`local` over all taps and filters and I have a scalar that is large when filters pack their energy into
tight contiguous blocks and small when energy is spread thin — the gentle distributional nudge I want,
rather than a max over blocks that a single sharp detector could dominate.

One subtlety in `avg_pool2d` at the borders: with `padding = 1` the default `count_include_pad=True`
holds the divisor fixed at `block_size² = 9` even where a window pokes into the zero pad, so on a `3 × 3`
map `mean(local)` is not the flat mean of the nine taps but a *center-weighted* average of the squared
energies, scaled overall by `≈ 0.605`. The center-weighting is benign and mildly on-target — a center-hot
kernel is the prototypical contiguous-region detector — and the `0.605` scale simply folds into the
conservative strength I pick next.

Two design parameters fall out. `block_size` is the spatial scale of the "contiguous region" I am
discouraging — the analogue of the activation-masking block side. The CIFAR architectures here use
`3 × 3` convolutions almost everywhere (every `BasicBlock` conv, every VGG conv, the MobileNetV2
depthwise convs), so the kernels themselves are `3 × 3`. A block of side `3` is therefore the whole
kernel — the smallest contiguous spatial scale that still spans a genuine local neighborhood rather than
a single tap. So `block_size = 3`, and I only apply the penalty to layers whose kernel is at least
`block_size` in both spatial dimensions (`m.kernel_size[0] >= block_size` and `w.size(-1), w.size(-2) >=
block_size`); the `1 × 1` pointwise convolutions in VGG's classifier path and MobileNetV2's expansion /
projection layers have no spatial structure to regularize and are correctly skipped. How many layers
survive that filter matters, because the count feeds the next decision. ResNet-56 is a `3×3`-conv stem
plus `BasicBlock [9,9,9]` — so on the order of `1 + 54 = 55` genuine `3 × 3` conv banks (its downsampling
shortcuts are `1 × 1` and drop out). VGG-16-BN contributes its thirteen `3 × 3` conv layers. MobileNetV2
keeps only its `3 × 3` stem and the seventeen `3 × 3` depthwise convs — about eighteen — while its dozens
of `1 × 1` expand/project convs are skipped. So the same fixed penalty sees `~55`, `13`, and `~18`
contributing layers: a better-than-`4×` spread. Summed unnormalized, ResNet would carry roughly four
times VGG's total from depth alone, so the single `lambda_max` I share across all three would mean
something different on each. Dividing the accumulated penalty by the layer count (`reg = reg / count`)
makes the term depth-independent — a per-layer average — and that is what lets one strength travel. The
second parameter is the strength `lambda`, the most delicate choice, and I want to pin it with arithmetic.

Kaiming init gives each `3 × 3` conv weight
variance `2/fan_in` with `fan_in = 9·in_c`, so for an `in_c = 64` layer `E[w²] ≈ 2/576 ≈ 3.5e-3`. Then
`w_sq ≈ 3.5e-3`, `mean(local) ≈ 0.605 · 3.5e-3 ≈ 2.1e-3`, and at full strength `lambda_max · mean(local)
≈ 1e-4 · 2.1e-3 ≈ 2.1e-7`. Against a cross-entropy that starts near `ln(100) ≈ 4.6` on the CIFAR-100
pairs, the penalty's *value* is on the order of `1e-7` of the loss — utterly negligible as a number, and
it stays tiny even if the weights grow through training (at `w ≈ 0.1` the term is `~6e-7`). So this term
never competes with the data loss on magnitude; whatever it does, it does entirely through the *gradient*
it adds to the weights, which is `~lambda · w` per weight — of order `1e-4 · 0.06 ≈ 6e-6`, a gentle shove
against the much larger data gradient. That is exactly the intent: a light, complementary nudge to the
*spatial shape* of filters, not a second weight decay. It rides on top of L2 (`5e-4`) and BatchNorm, both
already doing heavy magnitude control, and the quantity I penalize is on the same scale as the squared
weights L2 already shrinks — so if I pushed `lambda_max` hard I would just fight L2 for a marginal shape
preference and lose accuracy. `lambda_max = 1e-4`, an order of magnitude below the L2 coefficient, is the
natural conservative setting: it perturbs filter shape without dominating the magnitude control L2 already
provides.

Now the wall, and it is the same one the activation-masking version hit, transplanted. If I switch on
even this small penalty from step zero, I am shaping filters that have not learned anything yet. Early in
training the convolutional weights are near their Kaiming-random initialization, the representation is
meaningless, and a shaping force that discourages contiguous kernel energy is fighting the network's
attempt to find *any* useful filter structure — the sign of that gradient opposes the sharpening of the
first real edge and texture detectors. And these three architectures are BatchNorm-heavy — every conv is
followed by a BN — and BN couples the scale of the conv weights to the running statistics in a way that
makes the early dynamics fragile; a weight-shaping force applied before BN has settled can perturb the
very statistics the rest of training depends on.

But the arithmetic I just did cuts against that worry. The penalty's gradient scales with `w`, and at
init `w` is small, so the penalty already *self-attenuates* early without my doing anything: it is
quadratic in weights that start tiny. So do I even need an explicit schedule? On pure magnitude, no — the
early penalty is `~1e-7` and cannot shock anything, and the self-attenuation is real. But the concern that
survives is not magnitude, it is *direction and guarantee*: a small but persistent shaping gradient still
points against feature discovery during the chaotic early phase, and I want a hard promise that BN settles
undisturbed, not a soft one. The clean way to get that is to return *exactly zero* for the opening stretch,
not a small number I have to trust. So on top of the soft `w²` attenuation I keep an explicit, blunt
delayed start — the weight-side analogue of ramping the drop probability from zero: penalty *off* entirely
for the first stretch, then a linear ramp to target.

Concretely, I compute training progress `progress = epoch / max(total_epochs - 1, 1)` from the `config`
the loop hands me. For the first 20% of training (`progress < 0.2`) I return exactly zero — let the
network build features and let BatchNorm settle, with no interference. After that I ramp linearly:
`adjusted = (progress - 0.2) / 0.8` runs from 0 to 1 over the remaining 80%, and `lam = lambda_max *
adjusted`. So the penalty starts at zero at the 20% mark and reaches its full strength `lambda_max` only
at the end of training, when the filters are mature and over-fitting is the live problem. This delayed,
warmed-up schedule is exactly the "start gentle, get harsh" principle, and it makes the method robust to a
slightly-too-aggressive target: even if `lambda_max` were a touch high, the network had a long runway to
build filters before the penalty bit, and the BN statistics were never shocked.

The pieces fit the contract: the penalty reads only `model` and `config['epoch']`,
`config['total_epochs']`, ignores the batch (a pure weight-shape penalty does not need it), returns a
scalar on `outputs.device`, and is differentiable in the weights through the square, pool, and means. It
adds essentially nothing to step cost — an average pool over tiny `3 × 3` maps and a couple of reductions,
dwarfed by the convolutions — and changes nothing else in the pipeline. The full scaffold body is in the
answer.

What should I expect this to do? By construction it is a *weak* regularizer, for three reasons the
arithmetic has made concrete. It is far removed from the failure it targets — the activation-masking
DropBlock removes information from the feature map directly, while my version only reshapes filter weights
through a `~1e-6`-scale gradient and never touches an activation. It lands on a pipeline that already runs
L2 on the same weights and BN on the activations, so the marginal room for a third weight-shape penalty is
thin, on a quantity that lives at the same scale L2 already shrinks. And the delayed warm-up means it
reaches `lambda_max` precisely when the cosine schedule has driven the learning rate toward zero and the
weights are barely moving — least leverage exactly at full strength. Together these point to a result at
or slightly below a plain well-regularized baseline.

Per architecture the leverage is weakest where the gap is smallest or the target is missed: VGG-16-BN's
real capacity sink is its 512-wide dense head, which a conv-only penalty never touches; ResNet-56's
residual paths let the network route around any single filter's shape preference; and MobileNetV2 /
FashionMNIST is nearly saturated in the mid-90s with almost no gap left, its depthwise banks the least
spatial structure to reshape. So I expect this rung near the floor of the field, plausibly the weakest,
with the largest shortfall on the harder CIFAR-100 pairs and a near-tie on FashionMNIST. If it instead
*led*, that would tell me the contiguous-energy preference does more real work than I credit — but I do
not expect it. What it establishes is the floor, and the argument for moving the point of action off
filter *shape* toward the two blind spots the prior art flagged: the output distribution, and the
*spectrum* of the weights rather than their local spatial energy.
