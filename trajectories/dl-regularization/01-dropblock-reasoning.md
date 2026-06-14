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
touch. The thing I can touch every step, differentiably, is the model's parameters. So the question
becomes: what penalty on the convolutional *weights* discourages contiguous spatial co-activation,
in the spirit of dropping contiguous blocks of the feature map?

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
collapse the input-channel axis by taking the per-output-channel, per-spatial-tap mean squared
magnitude: `w_sq = w.pow(2).mean(dim=1, keepdim=True)`, shape `[out_c, 1, kH, kW]`. This is a small
spatial energy map for each filter. Now I want the energy *within local contiguous blocks* of that map.
A block of side `block_size` centered at each tap, averaged, is exactly an average pool with kernel
`block_size`, stride 1, and `padding = block_size // 2` to keep the spatial size:
`local = F.avg_pool2d(w_sq, block_size, stride=1, padding=pad)`. Each entry of `local` is the mean
squared weight magnitude in the `block_size × block_size` neighborhood around a tap. Take the mean of
`local` over all taps and filters, and I have a scalar that is large when filters pack their energy into
tight contiguous blocks and small when energy is spread thin. Penalizing it nudges filters toward the
latter. This is the weight-space reading of "discourage reliance on contiguous regions," and it is fully
differentiable — a square, a mean, an average pool, a mean — with autograd handling the gradient.

Two design parameters fall out. `block_size` is the spatial scale of the "contiguous region" I am
discouraging — the analogue of the activation-masking block side. The CIFAR architectures here use
`3 × 3` convolutions almost everywhere (every `BasicBlock` conv, every VGG conv, the MobileNetV2
depthwise convs), so the kernels themselves are `3 × 3`. A block of side `3` is therefore the whole
kernel — the smallest contiguous spatial scale that still spans a genuine local neighborhood rather than
a single tap. So `block_size = 3`, and I only apply the penalty to layers whose kernel is at least
`block_size` in both spatial dimensions (`m.kernel_size[0] >= block_size` and `w.size(-1), w.size(-2) >=
block_size`); the `1 × 1` pointwise convolutions in VGG's classifier path and MobileNetV2's expansion /
projection layers have no spatial structure to regularize and are correctly skipped. The second
parameter is the strength `lambda`. This is the most delicate choice, and I want to reason about it
rather than guess.

Here is the wall, and it is the same one the activation-masking version hit, transplanted. If I switch
on a strong penalty from step zero, I am shaping filters that have not learned anything yet. Early in
training the convolutional weights are near their Kaiming-random initialization, the representation is
meaningless, and a penalty that discourages contiguous kernel energy is just fighting the network's
attempt to find *any* useful filter structure. Worse, these three architectures are BatchNorm-heavy —
every conv is followed by a BN — and BN couples the scale of the conv weights to the running statistics
in a way that makes the early dynamics fragile; a weight-shaping force applied before BN has settled can
destabilize the very statistics the rest of training depends on. So a constant penalty from the start is
a blunt instrument: early it harms, late it helps. The activation-masking version solved the analogous
problem by ramping the drop probability linearly from zero. I will do the weight-side analogue: keep the
penalty *off* entirely for the first stretch of training, then ramp it linearly to its target.

Concretely, I compute training progress `progress = epoch / max(total_epochs - 1, 1)` from the `config`
the loop hands me. For the first 20% of training (`progress < 0.2`) I return exactly zero — let the
network build features and let BatchNorm settle, with no interference. After that I ramp linearly:
`adjusted = (progress - 0.2) / 0.8` runs from 0 to 1 over the remaining 80%, and `lam = lambda_max *
adjusted`. So the penalty starts at zero at the 20% mark and reaches its full strength `lambda_max` only
at the end of training, when the filters are mature and over-fitting is the live problem. This delayed,
warmed-up schedule is exactly the "start gentle, get harsh" principle, and it is what makes the method
robust to a slightly-too-aggressive target: even if `lambda_max` is a touch high, the network had a long
runway to build filters before the penalty bit, and the BN statistics were never shocked.

How large is `lambda_max`? This penalty rides on top of L2 weight decay (`5e-4`) and BatchNorm, both
already doing heavy regularization, and the quantity I am penalizing — mean squared weight magnitude
inside local blocks — is on the same scale as the squared weights L2 already shrinks. So this is meant
to be a light, complementary nudge to the *spatial shape* of filters, not a second weight decay. If I
push it hard I will fight L2 and the data loss for a marginal shape preference and lose accuracy. I want
something conservative — a fraction of a percent of the loss scale. `lambda_max = 1e-4` is the natural
conservative setting: an order of magnitude below the L2 coefficient, so it perturbs filter shape gently
without dominating the magnitude control L2 already provides. I also average `local` over the number of
contributing layers (`reg = reg / count`) so the penalty's scale does not grow with network depth — a
56-layer ResNet and a 16-layer VGG see a comparably-scaled term, which matters because `lambda_max` has
to be one value across all three architectures.

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
regularizer by construction, for three compounding reasons. First, it is far removed from the failure it
is supposed to fix: the activation-masking DropBlock removes information from the feature map directly
and forces the network to look elsewhere; my version only gently reshapes filter *weights* and never
touches an activation, so the mechanism by which it improves generalization is indirect and diluted.
Second, it lands on a pipeline that already has two strong regularizers — L2 on the same weights and BN
on the activations — so the marginal room for a third weight-shape penalty is thin. Third, the delayed
warm-up means the penalty is at meaningful strength only over the last fraction of training, and it
reaches `lambda_max` precisely when the cosine schedule has driven the learning rate toward zero and the
weights are barely moving — so the penalty has the least leverage exactly when it is finally at full
strength. Each of these alone would blunt the effect; together they make me expect this rung to land
*at or slightly below* a plain well-regularized baseline rather than above it.

I can be sharper per architecture. On VGG-16-BN, the most over-parameterized of the three — a plain
deep stack with a 512-wide dense head on CIFAR-100 — there is the most over-fitting to fight, but also
the most BN to perturb, and a tiny conservative weight-shape penalty applied late is unlikely to move a
network whose dense classifier head is the real capacity sink (and which I do not regularize, since the
penalty targets only `Conv2d` layers with kernels `>= 3`). On ResNet-56 the residual structure already
controls gradient flow and the filters are `3 × 3` throughout, so the penalty does apply broadly, but
the residual paths give the network easy ways to route around any single filter's shape preference, so
again I expect little movement. On MobileNetV2 / FashionMNIST the task is the easiest (10 classes, a
near-saturated accuracy regime in the mid-90s), so there is almost no generalization gap left to close —
the penalty has nothing to grab.

So my falsifiable expectation, before any number comes back, is: this rung should sit near the floor of
the three regularizers, plausibly *the weakest*, with the largest shortfall on the harder CIFAR-100
pairs (ResNet-56 and VGG-16-BN), where the indirect weight-shape mechanism is most clearly out-muscled
by the over-fitting it is supposed to fight, and a near-tie with the others on the saturated
MobileNetV2 / FashionMNIST pair where everything works. If instead it *led* the field, that would
falsify my reading that a late, conservative weight-shape penalty is too indirect to beat the
output-side and weight-spectrum penalties the next rungs will try — and would tell me the contiguous-
energy preference is doing more real work than I credit it. I do not expect that. What I expect is a
weak baseline that establishes the floor of the ladder and motivates moving the point of action away
from filter *shape* and toward the two places the prior art said were the real blind spots: the output
distribution, and the *spectrum* of the weights rather than their local spatial energy. Those are the
next two rungs.
