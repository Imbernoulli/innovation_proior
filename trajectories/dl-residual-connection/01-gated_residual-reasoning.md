The block is the whole game here, and every candidate is a variation on one fill of the same default, so I
start with the plainest non-trivial change I trust not to hurt: put a single learnable scalar on the
residual branch. Before anything structural — rewiring the block or changing what the branch computes — I
want to know what the *minimal* deviation from the default does, because it calibrates everything I try after it.
The default is the post-activation basic block, `H = ReLU(F(x) + shortcut(x))`, and it already works. The
question is whether the network is getting the *amount* of residual right, block by block, and whether I can
simply let it choose.

A residual block commits to a fixed mixing rule: `F(x)` is added at full weight, `1·F(x)`, identical in the
first block and the last, at depth 20 and depth 110. But the right contribution of a block is surely not
constant — early blocks, building the low-level features every later block depends on, may want to write
strongly; deep blocks, refining an already-rich representation, may want only a small correction. With the
weight pinned at 1, the only lever the network has to dial a block down is to drive all of `F`'s conv weights
toward zero — a bad lever: it fights the same conditioning the residual reformulation relieves, and it
commits the full convolutional machinery to expressing a single scalar decision ("be quieter"). Concretely,
a stage-3 block (`C = 64`) carries `2·9·64² ≈ 7.4×10⁴` conv weights, and to quiet it the optimizer must walk
all seventy-odd-thousand toward zero in concert, against weight decay — to do what one scalar could do by
moving a single coordinate. So the minimal hypothesis is one extra degree of freedom on exactly this axis: a
scalar `alpha` multiplying the residual branch, `H = ReLU(shortcut(x) + alpha·F(x))`.

The cost is literal: one scalar per block is 9 extra parameters on ResNet-20, 27 on ResNet-56, 54 on
ResNet-110 — about `3×10⁻⁵` of a 1.7M-weight net. Whatever this change does, it does through *reweighting*, not
added expressiveness, which is what I want from a calibration probe: it isolates "was the fixed magnitude
wrong" without confounding the answer with new capacity.

Now the decision that actually defines this step, and it is *not* the one the obvious move pushes
toward. The natural lineage for "scale the residual by a learnable scalar" initializes that scalar to
**zero** — start every block as a pure identity and let the gates open during training, so a fresh deep net
begins perfectly signal-preserving. That story is about trainability at extreme depth: at zero-init the
input–output Jacobian is exactly the identity, gradients propagate without warm-up or normalization. It is a
real idea, but it does not fit *this* substrate. This block is **post-activation**: a ReLU *after* the
addition, BN *inside* the branch, so at `alpha = 0` the block is not the identity, it is `H = ReLU(shortcut(x))`.
On a within-stage block that gives `ReLU(x)`, which equals `x` only because `x` is the previous block's
post-ReLU output and already non-negative — identity by accident, not construction. On a transition block
(stride 2, channels doubling) the shortcut is a 1×1 conv then BN, roughly zero-mean, so `ReLU(BN(conv(x)))`
rectifies away about half of it. So `alpha = 0` does not even deliver the clean identity the zero-init story
sells.

And decisively, the substrate already has trainability solved: BN after every conv keeps signals healthy at
depth 110, and the schedule is a fixed cosine with no warm-up to remove. The disease zero-init cures is
simply not present. If I zero-init `alpha` I am not buying stability I need; I am throwing away the front of
training — every block contributes nothing until its gate crawls off zero, and weight decay is pulling gates
*down* (worked out below), so lift-off is not even free. Against a fixed 200-epoch budget those are epochs I
do not get back.

So I init the gate at **one**. At `alpha = 1` the block is bit-for-bit the default: `shortcut(x) + 1·F(x)`
is literally the baseline tensor, the same forward, the same gradients — a no-op at step zero. The network
begins as the proven post-activation ResNet, full depth from the first step, and the gate is a pure
correction the optimizer may apply: below 1 if a block over-writes, above 1 if it wants to write louder. The
floor is the baseline itself; init at 0 makes it strictly worse and bets everything on
gates lifting fast against a decay that opposes them.

The scalar multiplies the residual branch and nothing else. The shortcut stays un-gated, because the value
of the identity path is that it is *always* open; gating it would reintroduce the Highway failure where a
drifting gate closes the highway exactly where a deep net needs it most. So `alpha` multiplies `F(x)` after
its final BN, before the addition, with the block's ordering otherwise untouched — I am deliberately not
reordering BN and ReLU here, so `alpha` is the *only* changed variable and whatever the numbers show is a
clean magnitude signal.

There is a redundancy I owe myself an honest look at. The branch ends in `bn2`, and BatchNorm carries a
learnable per-channel `gamma`, so the network can *already* rescale the branch channel by channel, and quiet
a block by driving its `gamma` small. What does a separate `alpha` add? Three things. It is a *single*
parameter whose gradient is the sum over all channels and positions of `g·F` — a stronger, lower-variance
signal, so block-level loudness moves faster through one knob than through `C` separate `gamma`s in concert.
It is *decoupled* from BN's normalization job, so the volume decision does not entangle with `gamma`'s
feature-shaping and downstream-calibration role. And weight decay on one scalar gives a clean standing
pressure to quiet unused blocks. But the flip side is a prediction: because `gamma` was already most of the
way there, `alpha`'s *marginal* effect should be small — the same reasoning that justifies the knob says it
will not move the needle much. That modest, clean signal is exactly what this calibration probe should return.

The training dynamics of `alpha` are a genuine tug-of-war, not the benign touch "weight decay just shrinks
it" makes it sound. Its loss gradient is `<g, F(x)>`, the correlation between the incoming gradient and the
block's branch output; weight decay adds a standing `wd·alpha` pull toward zero. So the resting value is
`alpha* ≈ −<g, F(x)>/wd`: a block the loss wants loud defends its `alpha` near or above 1, while an unused
block has nothing to oppose the decay and drifts down. That turns `alpha` into a per-block usage readout —
the same standing incentive already on the conv weights, now on one interpretable knob. Aggregate accuracy
cannot show me each gate, but it should be consistent with a net that quieted a few blocks and lost nothing
by it.

The scale has to be *learned*, and learned per block. A fixed constant like `1/√2` is a single guess at the
right loudness applied identically everywhere, which is precisely the uniform rigidity I am removing — the
whole hypothesis is that correct loudness *varies*. And it has to be a single scalar per block, not a
per-channel vector: cost does not decide it (per-channel would still be only ~2000 params on ResNet-110),
the hypothesis does. "How loud is this block" is the coarse question a scalar answers exactly; a per-channel
gate answers a *different*, channel-selection question that would confound the clean magnitude signal. So
`alpha` of shape `(1,)`, broadcast across channels and space.

Read as variance, `alpha` is a per-block volume knob: at init the branch injects ~`alpha²` of variance onto
the shortcut's, and since the next block's BN renormalizes, what survives is the *ratio* of injected to
already-flowing. Backward, `out = shortcut(x) + alpha·F(x)` gives an un-scaled shortcut gradient
path plus a scaled branch path, so however far SGD drives `alpha` the identity gradient path stays open at
full strength — the gate mutes its own branch, never the shared through-line.

The implementation is exactly one edit: declare `self.alpha = nn.Parameter(torch.ones(1))` and change
`out += self.shortcut(x)` to `out = self.shortcut(x) + self.alpha * out`, everything else the default. The
full module is in the answer.

What should this floor do? At init the network *is* the baseline, so anything the gates do is a departure
SGD chose, and — reinforced by the `gamma`-redundancy above — the honest expectation is modest. On
ResNet-20/CIFAR-10 the baseline is already near the ceiling of what 20 layers extract from 10 classes, so I
expect gates near 1 and accuracy around a well-tuned vanilla ResNet-20. On the deeper CIFAR-100 nets the
scalar has more to do, but it stays a very low-bandwidth correction: it rescales a block, it cannot change
what the block computes, and it cannot help where the bottleneck is activation ordering or feature reuse
rather than magnitude. So I expect roughly baseline-level accuracy everywhere, strongest on ResNet-20/CIFAR-10,
with the deep CIFAR-100 nets left holding the most unexploited headroom — because what limits a 56- or
110-layer net is more likely how cleanly gradients and features pass *through* the identity path, which a
residual-branch scalar does not touch. That through-line, and the order in which the block wires BN and ReLU
around it, is the variable I held fixed here on purpose, and where I look next.
