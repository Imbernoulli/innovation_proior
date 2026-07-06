The block is the whole game here, but every candidate is a variation on one fill of the same scaffold, so
the place to start is the plainest non-trivial change I can make to the default and the one I most trust
not to hurt: put a single learnable scalar on the residual branch. Before I reach for anything structural —
rewiring the block's internals or changing what the branch itself computes — I want to know what the *minimal*
deviation from the default block does, because whatever it does calibrates every rung after it. The default
is the post-activation basic block: two 3×3 conv–BN layers with a ReLU between them, the shortcut added,
and a final ReLU after the addition, `H = ReLU(F(x) + shortcut(x))`. It already works. The question is
whether the network is getting the *amount* of residual right, block by block, and whether I can simply let
it choose.

Let me be precise about what "the amount of residual" means and why the fixed block might be getting it
wrong. A residual block commits to a fixed mixing rule: the branch output `F(x)` is added to the shortcut at
full weight, `1·F(x)`. That weight is a hard architectural constant, identical in the first block and the
last, identical at depth 20 and depth 110. But the right contribution of a block is surely not constant.
Early blocks, building the low-level features every later block depends on, may want to write strongly; deep
blocks, refining an already-rich representation, may want only a small correction — exactly the picture
where the optimal `F` is "near zero, a gentle nudge." With the weight pinned at 1, the only lever the
network has to dial a block down is to drive all of `F`'s conv weights toward zero. And that is a bad lever:
it fights the same conditioning the residual reformulation was meant to relieve, it commits the full
convolutional machinery — two 3×3 kernels, on the order of `9·C²` weights each — to expressing what is
really a single scalar decision ("be quieter"), and it couples that decision to whatever *shape* the block
is simultaneously trying to learn. Put a number on how lopsided that is: a stage-3 block has `C = 64`, so its
two convs carry about `2·9·64² ≈ 7.4×10⁴` weights, and to dial that block down the optimizer must walk all
seventy-odd-thousand of them toward zero *in concert* — against weight decay, and against every gradient that
also wants those same weights to shape features — to accomplish what one scalar could do by moving a single
coordinate. Seventy thousand coupled knobs standing in for one is the concrete form of "the fixed magnitude is
the wrong lever," and it is exactly the redundancy a single `alpha` removes. So the minimal hypothesis is to give each block one extra degree of
freedom that sits exactly on this axis: a scalar `alpha` multiplying its residual branch,
`H = ReLU(shortcut(x) + alpha·F(x))`, and let SGD learn how loud each block should be.

Count the cost before I commit, because "minimal" has to be literal. One scalar per block means 9 extra
parameters on ResNet-20 (`[3,3,3]` blocks), 27 on ResNet-56 (`[9,9,9]`), 54 on ResNet-110 (`[18,18,18]`).
Against a ResNet-110 that carries on the order of 1.7M weights, 54 scalars is about `3×10⁻⁵` of the
parameter count — not a capacity change I could even measure as capacity. Whatever this rung does, it does
through *reweighting*, not through added expressiveness, and that is exactly what I want from a calibration
probe: it isolates the single question "was the fixed magnitude wrong" and answers it without confounding
the answer with new capacity that could soak up gains for unrelated reasons.

Now the design decision that actually defines this rung, and it is *not* the one the obvious reference would
push me toward. The natural place to look for "scale the residual by a learnable scalar" is the lineage that
initializes that scalar to **zero** — start every block as a pure identity, `H = ReLU(x)`, and let the gates
open up during training so a freshly-initialized deep net begins life perfectly signal-preserving. That
story is about *trainability at extreme depth*: at zero-init the forward map is the identity and the
input–output Jacobian is exactly the identity, so gradients propagate without warm-up or normalization, and
it has been used to train networks thousands of layers deep. It is a real and attractive idea. But I have to
ask whether it fits *this* substrate, and when I trace it through, it does not — and the place where it
breaks is the load-bearing decision of this whole rung.

Here is the mismatch, and I want to trace it concretely rather than assert it. The zero-init story assumes
the residual branch is the *only* thing standing between input and output — that with `alpha = 0` the block
collapses to a clean identity. In this scaffold the block is **post-activation**: there is a ReLU *after*
the addition and BN *inside* the branch, so at `alpha = 0` the block is not the identity, it is
`H = ReLU(shortcut(x))`. Trace the two shortcut cases separately, because they differ. On a within-stage
block the shortcut is a bare identity, so `alpha = 0` gives `ReLU(x)` — and that equals `x` only because `x`
here is the output of the previous block's final ReLU and is therefore already non-negative elementwise; it
is the identity by an accident of the substrate, not by construction. On a transition block — the first
block of stages 2 and 3, stride 2, channels doubling — the shortcut is a 1×1 conv followed by BN, whose
output is roughly zero-mean, so `ReLU(BN(conv(x)))` rectifies away about half of it and is nowhere near a
pass-through. There are only two such blocks per net, but the point stands: `alpha = 0` does not even
deliver the clean identity the zero-init story is selling, so the one property that would justify paying for
it is not present here.

And decisively, the substrate already *has* trainability solved. BN after every conv keeps the forward and
backward signals healthy at depth 110, and the schedule is a fixed cosine over 200 epochs with no warm-up to
remove. The disease zero-init cures — an un-normalized deep net that cannot start — is simply not present. So
if I zero-init `alpha` I am not buying a stability I need; I am throwing away the front of training. Every
block would contribute nothing until its gate crawls off zero, and — as I will work out below — weight decay
is actively pulling gates *down*, so the lift-off is not even free: the data gradient must first overcome
that decay just to reach an `alpha` of order 1. For however many epochs that takes, the network is
effectively shallow, and against a fixed 200-epoch budget those are epochs I do not get back. Zero-init
trades early capacity for an early stability the BN substrate already provides.

So I init the gate at **one**. At `alpha = 1` the block is bit-for-bit the default scaffold block:
`shortcut(x) + 1·F(x)` is *literally* `shortcut(x) + F(x)`, the same tensor, the same forward, the same
gradients — the edit is a no-op at step zero. That algebraic identity is the whole point. The network begins
training as the proven post-activation ResNet, full depth from the first step, and the gate is a pure
*correction* the optimizer may apply: pull a block below 1 if it over-writes, push it above 1 if it wants to
write louder than the default allows. Initializing at 1 makes the floor of this rung the baseline itself —
the worst case is "the gates sit near 1 and I recover the baseline," and the upside is "some blocks discover
they should be quieter or louder." Initializing at 0 makes the floor strictly worse and bets the whole rung
on the gates lifting fast enough against a decay that opposes them. Given an already-trainable substrate and
a fixed budget, 1 is the only defensible choice, and it is the single decision that separates this rung from
the zero-init lineage it superficially resembles.

Where exactly does the scalar multiply? On the residual branch and nothing else. The shortcut must stay
un-gated, because the entire value of the identity path is that it is *always* open, carrying signal and
gradient untouched; gating the shortcut would reintroduce the Highway failure mode where a drifting gate
closes the highway exactly where a deep net needs it most. So `alpha` multiplies `F(x)` after its final BN
and before the addition, leaving `shortcut(x)` at full strength, and it sits *inside* the post-activation
structure with the block's own ordering otherwise untouched: `out = ReLU(BN(conv1(x)))`, then
`out = BN(conv2(out))`, then `out = shortcut(x) + alpha·out`, then a final `ReLU` on the sum. I am
deliberately not touching the activation order on this rung — reordering BN and ReLU is a different lever,
and holding it fixed is what keeps `alpha` the *only* changed variable, so whatever the numbers show is a
clean signal about magnitude and nothing else.

There is a redundancy I owe myself an honest look at, because if I skip it I do not really understand what
`alpha` buys. The branch already ends in `bn2`, and BatchNorm carries a learnable per-channel scale `gamma`.
So the network can *already* rescale the branch — channel by channel — by moving `bn2`'s `gamma`, and in
particular it can already quiet a block by driving all of that block's `gamma` small. If BN's `gamma`
subsumes a uniform down-scale, what does a separate scalar `alpha` add? Three concrete things, and the fact
that they are marginal is itself a prediction. First, `alpha` is a *single* parameter carrying a single
decision, so its gradient is the sum over all channels and spatial positions of `g·F` — a much stronger,
lower-variance signal than each `gamma_c`'s gradient, so SGD can move block-level loudness faster and more
decisively through one knob than by nudging `C` separate `gamma`s in concert. Second, `alpha` is *decoupled*
from BN's normalization job: `gamma` also sets the per-channel scale the final ReLU and everything
downstream calibrate to, so overloading it with "this whole block is quiet" entangles the volume decision
with the feature-shaping decision. Third, weight decay on one scalar gives a clean, interpretable standing
pressure to quiet unused blocks, rather than spreading that pressure across `C` channels each with its own BN
dynamics. But notice the flip side: because `gamma` was already most of the way there, `alpha`'s *marginal*
effect should be small — the same reasoning that justifies the knob predicts it will not move the needle
much. I take that as a feature; it is exactly the modest, clean signal a calibration rung should return.

Now make the training dynamics of `alpha` concrete, because the phrase "weight decay touches it, benignly"
is too glib and I should work out whether it matters. The gradient of the loss with respect to `alpha` is
`<g, F(x)>`, where `g` is the gradient entering the pre-ReLU sum — the correlation between the incoming
gradient and this block's branch output — so SGD moves `alpha` up when the block's contribution aligns with
reducing the loss. Weight decay adds a term `wd·alpha` to that gradient, so each step shrinks `alpha`
multiplicatively by `(1 − lr·wd)` before the data gradient acts. Over the schedule this compounds: 50k CIFAR
images at batch 128 is about 391 steps per epoch, roughly 78k steps over 200 epochs, and cosine annealing
from `lr = 0.1` gives a mean learning rate near 0.05, so the summed learning rate is on the order of
`3.9×10³`. An `alpha` with *no* opposing data gradient would therefore decay by
`exp(−wd·Σlr) = exp(−5×10⁻⁴·3.9×10³) ≈ exp(−1.95) ≈ 0.14` by the end of training — and that is before
momentum amplifies the effective step. So the pull is not the cosmetic nudge I might have waved away. What it
means is that `alpha`'s resting value is set by a genuine tug-of-war: at equilibrium the data gradient must
balance the decay, `<g, F(x)> + wd·alpha ≈ 0`, so `alpha* ≈ −<g, F(x)>/wd`. A block the loss actively wants
loud defends its `alpha` near or above 1; a block the network is not using has nothing to oppose the decay,
and its `alpha` is pulled down toward that ~0.14 floor. Momentum only sharpens this: at `momentum = 0.9` the
effective step size in steady state is amplified by roughly `1/(1 − 0.9) = 10×`, so an idle gate decays even
faster than the no-momentum estimate and ~0.14 is really a conservative ceiling on where an unused gate
rests. That is not a nuisance — it turns `alpha` into a *usage readout*, a per-block scalar that quiets
blocks the network does not need and keeps the ones it does, the same standing incentive already acting on
the conv weights but now expressed on one interpretable knob. The mechanistic signature I would expect, if
this reading is right, is that the blocks doing real reshaping — the two stride-2 transition blocks and the
first block of each stage — hold `alpha` up, while the mid-to-late within-stage refining blocks let it drift
below 1; the aggregate accuracy cannot show me each gate, but it should be *consistent* with a network that
found a few blocks worth turning down and lost nothing by it.

Should the scale even be learnable, or would a fixed constant do? Some deep architectures simply multiply
every branch by a fixed factor — `1/√2`, say — to keep the running variance from compounding across blocks.
But a fixed scale is a *guess* at the right loudness, applied identically to every block, and that uniform
rigidity is precisely the thing I am trying to remove: my whole hypothesis is that the correct loudness
*varies* — strong in the early feature-building blocks, small in the late refining ones. A single constant
cannot express that variation, and picking it well would require knowing the per-block answer in advance. So
the scale has to be learned, and learned *per block*, so each block finds its own resting loudness under the
tug-of-war worked out above. That is the difference between calibrating the axis and merely re-guessing a
point on it.

One more choice worth walking rather than assuming: should the gate be a single scalar per block, or a
per-channel vector? A per-channel gate would be a length-`C` parameter — 16, 32, or 64 numbers per block
depending on the stage — which for ResNet-110 sums to `18·(16+32+64) = 2016` parameters, still tiny. So cost
does not decide it. What decides it is the hypothesis I am testing. The coarse question is "how loud is this
*block*," and a single scalar answers exactly that. A per-channel gate answers a *different* question —
"which channels of this block matter" — a channel-selection question orthogonal to residual magnitude and
better left out of the present test entirely. Folding it in here would
confound the clean magnitude signal with a channel-selection signal, and I would not be able to read either
afterward. So a single per-block scalar, broadcast across channels and space: `alpha` of shape `(1,)`
multiplying an `(B, C, H, W)` branch, broadcasting cleanly over every channel and pixel. One scalar, one
branch, one axis.

It is worth a moment on what that scalar physically controls at init, to be sure `alpha = 1` is truly the
no-op I claimed and to understand what a learned departure would do. At initialization `F(x) = BN(conv2(…))`
has roughly unit per-channel variance (BN normalizes, and the BN weight starts at 1), and the shortcut
carries the running representation with some variance `v`. Treating them as roughly independent, the pre-ReLU
sum has variance about `v + alpha²·1`. At `alpha = 1` that is `v + 1` — the baseline, exactly. If SGD pushes
a block's `alpha` to 1.5 its variance contribution grows to `2.25×`; to 0.5 it shrinks to `0.25×`. So `alpha`
is a per-block volume knob on the *variance* the branch injects into the stream, and because the next
block's BN renormalizes, what survives is the *ratio* of this block's injected variance to the variance
already flowing — i.e. how much this block perturbs the running representation relative to what is already
there. That is the right physical picture of the single degree of freedom I am adding, and it confirms the
init is a genuine identity rather than a near-identity.

I should run the same variance check at the two transition blocks per net, because those are the ones where I
earlier worried `alpha = 0` is not even an identity, and I want to be sure the *init* is clean there even
though the zero limit is not. At a transition the shortcut is `BN(conv1x1(x))`, whose output BN normalizes to
roughly unit per-channel variance, and the branch `F(x) = BN(conv2(…))` is likewise ~unit variance; treating
them as roughly independent, the pre-ReLU sum has variance about `1 + alpha²`. At `alpha = 1` that is ~2 — and
that is *exactly* what the baseline transition block sums as well, since the default also adds these two
~unit-variance tensors with no scale between them. So `alpha = 1` is the bit-for-bit baseline at the transition
blocks too, not just the identity-shortcut majority: the no-op holds at *every* block in the net. And the same
picture tells me what a learned departure would mean there — pulling a transition's `alpha` below 1 lets the
cheap reprojected shortcut dominate the sum, pushing it above 1 lets the freshly-extracted 3×3 branch features
dominate. It is the same volume knob, now trading a 1×1 reprojection against the block's real convolutional
work, which is exactly why I expect the two transitions to be among the blocks that *hold* `alpha` up.

The backward direction deserves the same check, because gradient flow is the axis the whole design lives on.
Ignore the post-add ReLU's mask for a moment and differentiate `out = shortcut(x) + alpha·F(x)` with respect
to the block input `x`: the gradient reaching `x` is `g_shortcut + alpha·(∂F/∂x)ᵀ g`, the sum of an
*un-scaled* path through the shortcut and a *scaled* path through the branch. So the scalar rescales only the
branch's contribution to the upstream gradient; the shortcut's contribution is untouched no matter what
`alpha` does. This is the backward-side statement of the same decision that put the gate on the branch and
not the shortcut: even if SGD drives some block's `alpha` far from 1, the identity gradient path stays open
at full strength, so the gate can never throttle the highway that carries gradient to the bottom of the
stack. It confirms the intervention is safe in exactly the sense a deep net cares about — the worst a
misbehaving gate can do is mute its own branch's gradient, never the shared through-line.

The implementation follows directly: one `nn.Parameter` of shape `(1,)` initialized to ones, registered so
SGD trains it alongside the convs and so weight decay reaches it (the tug-of-war above). The edit relative to
the default is exactly this — declare `self.alpha = nn.Parameter(torch.ones(1))` in the constructor, and
change the addition line from `out += self.shortcut(x)` to `out = self.shortcut(x) + self.alpha * out`.
Everything else — the two convs, the two BNs, the inter-conv ReLU, the dimension-matching 1×1 shortcut, the
final ReLU — is the scaffold default, untouched. The full module is in the answer. It is the smallest
meaningful change to the default, and that is the point: it sits on the residual-magnitude axis and on
nothing else.

Now reason about what this floor should do, because running it first is only worth it if I commit to a
prediction. At initialization the network *is* the baseline, so anything the gates do is a departure the
optimizer chose, and the honest expectation is modest — reinforced by the `gamma`-redundancy above, which
says most of the block-scaling ability was already present. On the shallow easy setting — ResNet-20 on
CIFAR-10 — the baseline block is already near the ceiling of what 20 layers can extract from a 10-class
problem, and a per-block scalar has little structural slack to exploit; I expect the gates to settle close to
1 and the accuracy to land right around a well-tuned vanilla ResNet-20, with any gain coming from a handful
of blocks discovering they should write a little more or a little less. On the deeper CIFAR-100 settings the
scalar has *more* to do — more blocks (27 and 54), more reason for their contributions to differ — but the
same property that makes this rung safe makes it weak: a single scalar per block is a very low-bandwidth
correction. It can rescale a block; it cannot change *what* the block computes, and it cannot help where the
bottleneck is the activation ordering or the feature reuse rather than the residual magnitude. So I expect
roughly baseline-level accuracy everywhere, strongest on ResNet-20/CIFAR-10 where "match the baseline" is
itself a high number, and the deeper nets left wanting — because what limits a 56- or 110-layer net is more
likely how cleanly gradients and features pass *through* the identity path, which a residual-branch scalar
does not touch.

The falsifiable shape I am committing to, against the numbers this rung returns: the gated block sits close
to a strong vanilla ResNet on all three settings, best on ResNet-20/CIFAR-10, with the deep CIFAR-100 nets
showing the most unexploited headroom. If instead the gate produced a large jump on the deep CIFAR-100 nets,
I would be wrong about the bandwidth and this magnitude axis would be the main lever after all — I do not
expect that, and the three-setting `test_acc` table will tell me which. And the diagnosis is already the
pointer to the next rung: if the gated block lands near the baseline and the deep CIFAR-100 nets are where
the room is, then the lever to reach for is not *how much* residual each block adds but *how* the block is
wired around the identity path — the order of BN and ReLU relative to the convolutions and the addition,
because that is what governs whether the shortcut is a truly clean highway at depth. That is the variable I
held fixed here on purpose, and it is the natural thing to attack next.
