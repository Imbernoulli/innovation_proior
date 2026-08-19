The kernel sweep confirmed the shape I predicted: accuracy climbed from
79.92% at kernel 3 (the reordered starting point) through 80.35% at 5,
80.57% at 7, held flat at 80.57% at 9, and turned down to 80.47% at 11 —
returns saturate right at 7x7, essentially matching Swin's minimum local
window size, while FLOPs barely moved across the whole sweep (4.07G to
4.29G). That's a genuinely interesting convergence: a convolution growing
its kernel from the small-kernel side and windowed attention restricting
itself from the global side land on the same locality budget. I'll adopt
7x7 and carry it forward — this is the last change to what any single
convolution can physically see. Every macro-level and receptive-field
question I set out with is now closed: stage layout, stem, spatial-vs-
channel mixing separation, block shape, kernel size. What's left is a
different kind of question entirely, one I've been setting aside since
rung 1 while I dealt with bigger structural levers first: the layer-level
details inside the block that I copied over from vanilla ResNet without
examining, out of inertia rather than any argument that they're right for
this new block.

Three ResNet conventions have survived every rewrite so far, untouched: an
activation function after *every* convolution including the 1x1s, a
normalization layer before every activation, and BatchNorm specifically as
that normalization. All three were reasonable defaults for the original
ResNet bottleneck. None of them were re-derived for the block this ladder
has actually built — narrow-depthwise(7x7)-wide-1x1-wide-1x1-narrow, an
arrangement that looks much more like a Transformer sublayer pair than like
a ResNet bottleneck at this point. So the natural move is to go back to the
Transformer block as the reference and check each convention against it,
one at a time, in the order that seems least risky first — because unlike
the earlier rungs, several of these are genuinely coupled (removing a norm
changes what the next norm sees; changing normalization type changes what
removing a norm costs), and I'd rather isolate them sequentially than
change three things at once and not know which one moved the needle.

**Activation function: ReLU vs. GELU.** This is the safest place to start
because it changes nothing structural — same number of activations, same
positions, only the function itself. ReLU has been the ConvNet default
since well before this exploration began, valued mainly for simplicity and
efficient implementation. The most advanced Transformer-family models
(BERT, GPT-2, and the ViT lineage this whole exploration has been
referencing) use GELU instead, a smoother version of ReLU with a non-zero
gradient for small negative inputs rather than a hard cutoff at zero. I
have no strong structural argument for why a ConvNet specifically should
prefer one over the other — this isn't like the depthwise/kernel-size
changes, where I could point to a receptive-field or mixing-structure
reason — so I'm testing it mainly because it's a one-line swap with
essentially no cost, and consistency with the Transformer reference this
whole ladder has followed is reason enough to check. I don't expect much
movement either way.

**Number of activations per block.** This one I do expect to matter, and
here's the structural argument. Count the activations in a Transformer
sublayer pair: attention has none inside it (the softmax is not commonly
counted as a activation function that's varied), and the feedforward
sublayer — narrow, expand 4x, project narrow — has exactly *one*
nonlinearity, sitting between its two linear layers. My current block, by
contrast, still follows the ResNet convention of an activation after
*every* convolution: after the depthwise conv, after the first 1x1, after
the second 1x1 — three activations total. If I map my block onto the
Transformer sublayer pair the way I've been doing throughout this ladder —
depthwise conv as the spatial-mixing sublayer, the two 1x1s as the
feedforward sublayer — the feedforward analogue should have exactly one
activation, between its two linear layers, not one after each. So: drop the
activation after the depthwise conv and after the second (projection) 1x1,
keeping only the one between the two 1x1s. Unlike the GELU swap, this
removes real nonlinear capacity from the network — two fewer nonlinearities
per block, network-wide — so I take this to be a bet with actual
directional stakes, not a free consistency check. My prior leans toward
this still helping despite removing nonlinearities, because a nonlinearity
placed right after a channel-mixing-only 1x1 projection back down to the
block's narrow output, immediately before a residual addition, forces the
residual branch's output through a hard floor at exactly the point where
it's about to be summed with the identity path — a fully linear projection
back into the residual stream, by contrast, lets the block contribute
smoothly-scaled positive and negative corrections. But I hold this loosely;
removing capacity can just as easily cost accuracy as removing redundancy,
and I want the measurement to settle it rather than my intuition.

**Number of normalization layers.** Same argument, applied to
normalization instead of activation: Transformers normalize sparingly too —
typically one norm before attention, one before the feedforward sublayer,
not a norm after every internal linear layer. My block currently
normalizes after every convolution, again a straight ResNet holdover. I'll
cut down to a single normalization layer, placed before the two 1x1 convs
(after the depthwise conv, before the channel-mixing pair) — the position
that best matches "normalize before the sublayer that does the heavy
lifting," mirroring pre-norm Transformer convention. This is a second real
capacity/stability change, not a free swap, and it interacts with the
activation change above: fewer norms means the remaining norm has to do
more of the work of keeping the block well-conditioned across many stacked
layers, three or nine per stage. I don't have a strong directional
prior here beyond noting that if this over-reduces stability, I'd expect to
see it directly as a measurement, not as a subtle effect — training
instability from too little normalization tends to show up sharply rather
than as a small accuracy dip, so a clean positive-or-flat result would be
reassuring evidence this wasn't over-cut.

**BatchNorm vs. LayerNorm.** This is the change I'm most wary of, because
there's a specific documented failure mode working against it: substituting
LayerNorm for BatchNorm in an otherwise-unmodified ResNet is known to
produce suboptimal performance. That fact predates this whole exploration
and is exactly why BatchNorm has stayed the default for ConvNets even as
Transformers standardized on LayerNorm. If I took that fact in isolation, I
should probably leave BatchNorm alone here. But I don't think that
precedent transfers cleanly to *this* network, because the network the
precedent was measured on and the network I'm about to test it on differ in
several ways that specifically bear on why BatchNorm tends to help: BN's
main value comes from computing per-channel statistics across the batch
dimension, which stabilizes training partly by smoothing over batch
composition — valuable when a network is deep, uses large batches with SGD,
and has no other mechanism keeping activations well-scaled. This network
now trains with AdamW rather than SGD, already carries LayerScale gating
every residual branch, uses far heavier regularization (Stochastic Depth,
label smoothing, aggressive augmentation) than the ResNet the historical
failure was documented on, and — if the two changes above hold up — has
already been stripped down to one normalization layer per block rather
than three. That's a different enough training regime that I don't think
the old result settles the question here; it's evidence to take seriously,
not a foregone conclusion. LayerNorm's own case for fitting this block
specifically: it normalizes across channels at each spatial position
independently of batch composition, which is exactly how the Transformer
reference this ladder has been following normalizes its own residual
stream, and it's also less sensitive to batch size, a practical
consideration given this network already trains at batch size 4096. I'll
test the substitution directly, at the network's current normalization
count (one per block), and let the historical warning and the
Transformer-reference argument settle against each other with a number
rather than a guess.

**Separate downsampling layers.** The last remaining structural mismatch
with the Transformer-family reference, and the riskiest change in this
rung. Every downsampling in this network so far has been folded into the
first block of each new stage — a strided convolution somewhere inside that
block's residual path handles both the spatial mixing and the resolution
change at once, the ResNet convention I've never revisited even while
changing everything about what's inside the block. Swin, by contrast,
inserts a dedicated, separate layer *between* stages purely for
downsampling — a patch-merging step that isn't otherwise part of any
block's forward pass. This is a bigger structural change than anything
else in this rung: it's not reweighting an existing computation, it's
adding a new kind of layer with a job no existing layer currently has. I'll
implement it as a 2x2, stride-2 convolution sitting between stages,
separate from the blocks on either side. I expect this to be the riskiest
sub-step precisely because it's furthest from anything already validated
in this network — resolution changes and the surrounding normalization
have been co-adapted throughout the entire ladder, and inserting a new,
previously-untested transition point is the kind of change that can
destabilize training outright rather than just shift accuracy by a few
tenths, especially at 4x downsampling points where activation statistics
already shift sharply. If it does destabilize, my fallback hypothesis is
that the fix is more normalization at exactly the points where resolution
changes and where the network's statistics are least settled: a
normalization layer immediately before each new separate downsampling
layer, plus one right after the stem (which is itself an aggressive 4x
downsampling step that currently has no norm of its own beyond whatever
follows it), plus one after the final global pooling before the
classification head — bracketing every point in the network where spatial
information gets abruptly compressed with a normalization layer on at
least one side of it.

So the test for this rung, run as five sequential, cumulative sub-steps on
top of the rung-5 baseline (kernel 7x7, all else unchanged): GELU in place
of ReLU; then reduce to a single activation per block; then reduce to a
single normalization layer per block; then substitute that remaining norm
from BatchNorm to LayerNorm; then add separate downsampling layers with
the bracketing normalization scheme just described. Each is measured on
its own before the next is added, the same cumulative-ablation discipline
this whole ladder has used. I expect the first step to be close to flat,
the middle three to be the ones carrying real weight (with genuine
uncertainty on sign for the norm-count and BN-vs-LN sub-steps given the
documented precedent working against the latter), and the last to be the
one most likely to either meaningfully help or meaningfully hurt training
stability, depending on whether the bracketing normalization I've reasoned
my way into is actually sufficient. Whatever this sequence lands on becomes
the final architecture this exploration commits to.
