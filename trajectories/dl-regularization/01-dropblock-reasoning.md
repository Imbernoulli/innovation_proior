I am starting the climb at the rung that reacts most directly to the prior art I just laid out, and I
want to be honest from the first line about a constraint that reshapes the whole idea. The thing that
bugs me is dropout: it is the best injected noise we have, and it has quietly disappeared from the
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

Before I settle on what to touch, let me actually enumerate the re-expressions available to me, because
the honest one is not the first one that comes to mind. The most faithful transplant would keep acting
on *activations*: I have `inputs` and I have `model`, so I could register forward hooks on the conv
layers, run the batch through, capture the intermediate feature maps, and penalize a spatial
co-activation statistic on them directly — the thing DropBlock literally attacks. This is the tempting
option precisely because it is faithful. I kill it on cost and fragility. The forward pass to produce
`outputs` has already happened by the time my function is called; to see intermediate maps I must either
re-run the network or leave hooks resident, and either way I am now carrying an extra activation-sized
backward graph through every conv on every step, on three full 200-epoch runs where the contract asks
for something cheap-every-step. Worse, a hook-based penalty is stateful machinery bolted onto a function
that is supposed to be a pure scalar — exactly the kind of forward-graph coupling this edit surface was
built to forbid. It is the *right* penalty in the wrong container. A second candidate acts on the
weights but through a different notion of "structure": penalize the spatial *total variation* of the
kernel, `Σ |w[i+1] − w[i]|`, to make kernels smooth. I reject this one on mechanism, not cost. A smooth
kernel is not the same as a de-concentrated kernel — a smooth Gaussian blob is about as *contiguous* a
detector as exists, all its energy piled into one soft central lump — so a smoothness penalty would
actively reward the very contiguous-region reliance I am trying to discourage. TV answers a different
question than the one DropBlock poses. That leaves acting on the weights through their *energy*
distribution, which is the thing I can touch cheaply, every step, differentiably, and which maps onto
the DropBlock question honestly. So the question narrows: what penalty on the convolutional *weights*
discourages contiguous spatial co-activation, in the spirit of dropping contiguous blocks of the feature
map?

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
`local` over all taps and filters, and I have a scalar that is large when filters pack their energy into
tight contiguous blocks and small when energy is spread thin. Penalizing it nudges filters toward the
latter — a square, a mean, an average pool, a mean, with autograd handling the gradient. There is also
the option of penalizing the *max* local block instead of the mean — the single hottest block per filter,
the worst-off region. That is more surgical but non-smooth, and it would let a single legitimately sharp
detector dominate the whole term; the mean is the gentle distributional nudge I actually want here, so
mean it is.

I should not wave my hands at `avg_pool2d` at the borders, because with `padding = 1` the answer depends
on a flag. The default is `count_include_pad=True`, which means the pooling divisor stays fixed at
`block_size² = 9` even where the window pokes into the zero pad — so a corner output averages four real
taps against five implicit zeros, all divided by nine. Let me trace what `mean(local)` therefore computes
on a `3 × 3` map, because it is not quite the flat mean of the nine taps I might have assumed. Averaging
the nine `3 × 3`-windowed outputs counts each kernel tap once for every output window that covers it. The
center tap sits under all nine windows; each edge-center tap under six; each corner tap under four —
`9 + 4·6 + 4·4 = 49` coverings, so `mean(local) = (1/81)(9·a_center + 6·Σa_edge + 4·Σa_corner)`, a
*center-weighted* average of the squared energies, scaled overall by `49/81 ≈ 0.605`. The center tap
carries weight `9/81 ≈ 0.111`, a corner `4/81 ≈ 0.049` — roughly `2.25×` more pressure on center-hot
kernels than corner-hot ones. I did not design that emphasis; it falls out of `count_include_pad`. But it
is benign and, if anything, mildly on-target: a center-hot kernel is the prototypical contiguous-region
detector, so weighting it hardest is the right sign. I could flatten it with `count_include_pad=False`,
but that buys nothing real and complicates the fill, so I accept the center-weighting and keep the
default. The `0.605` overall scale is simply absorbed into the conservative strength I pick next.

Two design parameters fall out. `block_size` is the spatial scale of the "contiguous region" I am
discouraging — the analogue of the activation-masking block side. The CIFAR architectures here use
`3 × 3` convolutions almost everywhere (every `BasicBlock` conv, every VGG conv, the MobileNetV2
depthwise convs), so the kernels themselves are `3 × 3`. A block of side `3` is therefore the whole
kernel — the smallest contiguous spatial scale that still spans a genuine local neighborhood rather than
a single tap. So `block_size = 3`, and I only apply the penalty to layers whose kernel is at least
`block_size` in both spatial dimensions (`m.kernel_size[0] >= block_size` and `w.size(-1), w.size(-2) >=
block_size`); the `1 × 1` pointwise convolutions in VGG's classifier path and MobileNetV2's expansion /
projection layers have no spatial structure to regularize and are correctly skipped. It is worth counting
how many layers actually survive that filter, because the count feeds the next decision. ResNet-56 is a
`3×3`-conv stem plus `BasicBlock [9,9,9]` — three stages of nine two-conv blocks — so on the order of
`1 + 54 = 55` genuine `3 × 3` conv banks (its downsampling shortcuts are `1 × 1` and drop out). VGG-16-BN
contributes its thirteen `3 × 3` conv layers. MobileNetV2 keeps only its `3 × 3` stem and the seventeen
`3 × 3` depthwise convs — about eighteen — while its dozens of `1 × 1` expand/project convs are skipped.
So the same fixed penalty sees `~55`, `13`, and `~18` contributing layers across the three nets: a
better-than-`4×` spread. If I summed the per-layer term without normalizing, ResNet would carry roughly
four times VGG's total from depth alone, and the single `lambda_max` I have to share across all three
would mean something different on each. So I divide the accumulated penalty by the number of contributing
layers (`reg = reg / count`), which makes the term depth-independent — a per-layer average — and that is
what lets one strength travel. The second parameter is the strength `lambda`. This is the most delicate
choice, and I want to reason about it with arithmetic rather than guess.

Let me get the scale of this thing at initialization on paper. Kaiming init gives each `3 × 3` conv weight
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

Here I have to be honest with myself, because the arithmetic I just did cuts against the drama. I showed
the penalty's gradient scales with `w`, and at init `w` is small — so the penalty already *self-attenuates*
early, softly, without my doing anything: it is quadratic in weights that start tiny. So do I even need an
explicit schedule, or is the `w²`-scaling enough? Let me not overstate the danger. On pure magnitude, the
early penalty is `~1e-7` and cannot "shock" anything; the self-attenuation is real. But the concern that
actually survives is not magnitude, it is *direction and guarantee*: a small but persistent shaping
gradient still points against feature discovery during the chaotic early phase, and I want a hard promise
that BN settles undisturbed, not merely a soft one. The clean way to get a hard promise is to return
*exactly zero* for the opening stretch, not a small number I have to trust. So I keep an explicit delayed
start as belt-and-suspenders on top of the soft `w²` attenuation, and I keep it deliberately blunt.
The activation-masking version solved the analogous problem by ramping the drop probability linearly from
zero; I do the weight-side analogue: penalty *off* entirely for the first stretch, then a linear ramp to
target.

Concretely, I compute training progress `progress = epoch / max(total_epochs - 1, 1)` from the `config`
the loop hands me. For the first 20% of training (`progress < 0.2`) I return exactly zero — let the
network build features and let BatchNorm settle, with no interference. After that I ramp linearly:
`adjusted = (progress - 0.2) / 0.8` runs from 0 to 1 over the remaining 80%, and `lam = lambda_max *
adjusted`. So the penalty starts at zero at the 20% mark and reaches its full strength `lambda_max` only
at the end of training, when the filters are mature and over-fitting is the live problem. This delayed,
warmed-up schedule is exactly the "start gentle, get harsh" principle, and it makes the method robust to a
slightly-too-aggressive target: even if `lambda_max` were a touch high, the network had a long runway to
build filters before the penalty bit, and the BN statistics were never shocked.

Let me make sure the pieces fit the contract and the substrate. The penalty reads only `model` (for the
conv weights) and `config['epoch']`, `config['total_epochs']` (for the schedule); it ignores `inputs`,
`outputs`, `targets`, which is fine — a pure weight-shape penalty does not need the batch. It returns a
scalar on `outputs.device` (using `outputs.device` is the clean way to land on the right GPU without
touching the batch values). It is differentiable in the weights through the square, pool, and means. It
adds essentially nothing to step cost: an average pool over tiny `3 × 3` kernel maps and a couple of
reductions, dwarfed by the convolutions themselves. And it changes nothing else — same architecture,
same Kaiming init, same SGD, same cosine schedule, same evaluation. It is a clean fill of the editable
region. The full scaffold body is in the answer.

Now let me reason carefully about what this rung should actually do, because I am deliberately starting
the ladder here and I want a falsifiable expectation. The honest assessment is that this is a *weak*
regularizer by construction, for three compounding reasons that the arithmetic above has now made
concrete. First, it is far removed from the failure it is supposed to fix: the activation-masking
DropBlock removes information from the feature map directly and forces the network to look elsewhere; my
version only gently reshapes filter *weights* through a `~1e-6`-scale gradient and never touches an
activation, so the mechanism by which it improves generalization is indirect and diluted. Second, it
lands on a pipeline that already has two strong regularizers — L2 on the same weights and BN on the
activations — so the marginal room for a third weight-shape penalty is thin, and I even showed the
quantity it penalizes lives on the same scale L2 already shrinks. Third, the delayed warm-up means the
penalty is at meaningful strength only over the last fraction of training, and it reaches `lambda_max`
precisely when the cosine schedule has driven the learning rate toward zero and the weights are barely
moving — so the penalty has the least leverage exactly when it is finally at full strength. Each of these
alone would blunt the effect; together they make me expect this rung to land *at or slightly below* a
plain well-regularized baseline rather than above it.

I can be sharper per architecture. On VGG-16-BN, the most over-parameterized of the three — a plain
deep stack with a 512-wide dense head on CIFAR-100 — there is the most over-fitting to fight, but the
penalty targets only its thirteen `Conv2d` filter banks and never touches the dense classifier head that
is the real capacity sink, so a tiny conservative weight-shape term applied late is unlikely to move it.
On ResNet-56 the residual structure already controls gradient flow, and although the penalty applies
broadly across its `~55` `3 × 3` banks, the residual paths give the network easy ways to route around any
single filter's shape preference, so again I expect little movement. On MobileNetV2 / FashionMNIST the
task is the easiest (10 classes, a near-saturated accuracy regime in the mid-90s), so there is almost no
generalization gap left to close — and the penalty only reaches its `~18` depthwise banks, whose
single-input-channel filters have the least spatial structure to reshape — so it has nearly nothing to
grab.

So my falsifiable expectation, before any number comes back, is: this rung should sit near the floor of
the three regularizers, plausibly *the weakest*, with the largest shortfall on the harder CIFAR-100
pairs (ResNet-56 and VGG-16-BN), where the indirect weight-shape mechanism is most clearly out-muscled
by the over-fitting it is supposed to fight, and a near-tie with the others on the saturated
MobileNetV2 / FashionMNIST pair where everything works. If instead it *led* the field, that would
falsify my reading that a late, conservative weight-shape penalty is too indirect to beat the
output-side and weight-spectrum penalties the prior art's own design menu points at — and would tell me
the contiguous-energy preference is doing more real work than I credit it. I do not expect that. What I
expect is a weak baseline that establishes the floor of the ladder and motivates moving the point of
action away from filter *shape* and toward the two places the prior art flagged as the real blind spots:
the output distribution, and the *spectrum* of the weights rather than their local spatial energy. Those
are the natural next places to look.
