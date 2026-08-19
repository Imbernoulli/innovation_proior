The kernel sweep landed exactly where I'd hoped: accuracy climbed from
79.92% at kernel 3 through 80.57% at kernel 7, held flat at kernel 9, and
turned down at kernel 11, with FLOPs barely moving across the whole sweep.
That closes every macro-level and receptive-field question I set out with —
stage layout, stem, spatial-vs-channel mixing separation, block shape,
kernel size. What's left is a different kind of question: the layer-level
details inside the block that I copied over from vanilla ResNet without
ever examining, out of inertia rather than any argument that they're right
for the block this ladder has actually built.

Three ResNet conventions have survived every rewrite so far, untouched: an
activation function after *every* convolution including the 1x1s, a
normalization layer before every activation, and BatchNorm specifically as
that normalization. All three were reasonable defaults for the original
bottleneck. None of them were re-derived for the arrangement I now have —
narrow -> depthwise(7x7) -> 1x1 expand 4x -> 1x1 project -> narrow, which
looks much more like a Transformer sublayer pair than a ResNet bottleneck.
So I go back to the Transformer block as the reference and check each
convention against it, one at a time, in order of least risky first —
several of these are genuinely coupled (removing a norm changes what the
next norm sees; changing normalization type changes what removing a norm
costs), so I isolate them sequentially rather than changing three things
at once and losing track of which one moved the needle.

**Activation function: ReLU vs. GELU.** The safest place to start, because
it changes nothing structural — same number of activations, same
positions, only the function itself. ReLU has been the ConvNet default
since before this exploration began; the most advanced Transformer-family
models use GELU instead, a smoother version with a non-zero gradient for
small negative inputs rather than a hard cutoff at zero. I have no strong
structural argument for why a ConvNet specifically should prefer one over
the other — unlike the depthwise/kernel-size changes, I can't point to a
receptive-field or mixing-structure reason — so I test it mainly because
it's a one-line swap with essentially no cost, and consistency with the
Transformer reference this whole ladder has followed is reason enough to
check. I don't expect much movement either way.

**Number of activations per block.** This one I do expect to matter, and
here's the structural argument. Count the activations in a Transformer
sublayer pair: attention has none inside it, and the feedforward sublayer —
narrow, expand 4x, project narrow — has exactly *one* nonlinearity,
between its two linear layers. My block, by contrast, still carries an
activation after every convolution — after the depthwise conv, after the
first 1x1, after the second 1x1 — three total. Mapping my block onto the
Transformer sublayer pair the way I've mapped it throughout this ladder
(depthwise conv as the spatial-mixing sublayer, the two 1x1s as the
feedforward sublayer), the feedforward analogue should have exactly one
activation, not one after each. So: drop the activation after the
depthwise conv and after the projection 1x1, keeping only the one between
the two 1x1s. Unlike the GELU swap, this removes real nonlinear capacity —
two fewer nonlinearities per block, network-wide — so this is a bet with
actual directional stakes. My prior leans toward this still helping,
because a nonlinearity placed right after the projection 1x1, immediately
before the residual addition, forces the residual branch's output through
a hard floor at exactly the point it's about to be summed with the
identity path; a fully linear projection back into the residual stream
lets the block contribute smoothly-scaled positive and negative
corrections instead. But I hold that loosely and want the measurement to
settle it.

**Number of normalization layers.** Same argument, applied to
normalization: Transformers normalize sparingly too, typically one norm
before attention and one before the feedforward sublayer, not a norm after
every internal linear layer. My block currently normalizes after every
convolution, again a straight ResNet holdover. I cut down to a single
normalization layer, placed before the two 1x1 convs — after the depthwise
conv, before the channel-mixing pair — the position that best matches
"normalize before the sublayer that does the heavy lifting," mirroring
pre-norm Transformer convention. This is a second real capacity/stability
change, and it interacts with the activation change above: fewer norms
means the remaining norm carries more of the burden of keeping the block
well-conditioned across many stacked layers, three or nine per stage. I
don't have a strong directional prior beyond noting that if this
over-reduces stability, I'd expect it to show up sharply, as training
instability, rather than as a small accuracy dip — so a clean
positive-or-flat result would be reassuring evidence this wasn't over-cut.

**BatchNorm vs. LayerNorm.** The change I'm most wary of, because there's
a specific documented failure mode working against it: substituting
LayerNorm for BatchNorm in an otherwise-unmodified ResNet is known to
produce suboptimal performance, which is exactly why BatchNorm has stayed
the ConvNet default even as Transformers standardized on LayerNorm. Taken
in isolation, that precedent argues for leaving BatchNorm alone. But I
don't think it transfers cleanly to *this* network. BN's main value comes
from computing per-channel statistics across the batch dimension, which
stabilizes training partly by smoothing over batch composition — valuable
when a network is deep, trained with plain SGD, and has no other mechanism
keeping activations well-scaled. This network now trains with AdamW rather
than SGD, already carries LayerScale gating every residual branch, uses
far heavier regularization than the ResNet the historical failure was
documented on, and — if the two sub-steps above hold up — has already been
stripped to one normalization layer per block rather than three. That is a
different enough regime that I take the old result as evidence to weigh,
not a foregone conclusion. LayerNorm's own case: it normalizes across
channels at each spatial position independently of batch composition,
exactly how the Transformer reference normalizes its own residual stream,
and it is less sensitive to batch size — a practical consideration given
this network already trains at batch size 4096. I test the substitution
directly, at the block's current normalization count, and let the
historical warning and the Transformer-reference argument settle against
each other with a number.

**Separate downsampling layers.** The last remaining structural mismatch
with the Transformer-family reference, and the riskiest change in this
rung. Every downsampling so far has been folded into the first block of
each new stage — a strided convolution inside that block's residual path
handles spatial mixing and the resolution change at once, the ResNet
convention I've never revisited even while rewriting everything inside the
block. The hierarchical Transformer reference instead inserts a dedicated,
separate layer between stages purely for downsampling, no different job
shared with any block's forward pass. I implement it as a 2x2 stride-2
convolution sitting between stages, separate from the blocks on either
side. This is the riskiest sub-step precisely because it is furthest from
anything already validated in this network: resolution changes and the
surrounding normalization have co-adapted through the entire ladder, and
inserting a previously-untested transition point can destabilize training
outright rather than just shift accuracy by a few tenths, especially at 4x
downsampling points where activation statistics already shift sharply. My
fallback hypothesis, in case it does destabilize, is that the fix is more
normalization at exactly the points where resolution changes and
statistics are least settled: a normalization layer immediately before
each new separate downsampling layer, plus one right after the stem
(itself an aggressive 4x downsampling step with no norm of its own beyond
whatever follows it), plus one after the final global pooling before the
classification head — bracketing every point where spatial information
gets abruptly compressed with a normalization layer on at least one side.
I build this bracketing scheme in from the start rather than waiting to
see the bare version fail, since an unnormalized resolution-change
boundary is a plausible enough instability point that I'd rather not spend
a training run finding out the hard way.

So the proposal for this rung is five sequential, cumulative sub-steps on
top of the rung-5 baseline (kernel 7x7, all else unchanged): GELU in place
of ReLU; then reduce to a single activation per block; then reduce to a
single normalization layer per block; then substitute that remaining norm
from BatchNorm to LayerNorm; then add separate downsampling layers with
the bracketing normalization scheme just described. Each is measured on
its own before the next is added, the same cumulative-ablation discipline
this whole ladder has used. I expect the first step to be close to flat,
the middle three to carry the real weight (genuine uncertainty on sign for
the norm-count and BN-vs-LN sub-steps given the documented precedent
against the latter), and the last to be the one most likely to either
meaningfully help or meaningfully hurt, depending on whether the
bracketing normalization I've reasoned my way into is actually sufficient.
Whatever this sequence lands on becomes the final architecture this
exploration commits to.

```python
# rung 6: micro design — activation count, norm count, norm kind, separate downsampling.
import torch
import torch.nn as nn
import torch.nn.functional as F

class MicroDesignBlock(nn.Module):
    """narrow -> depthwise(7) -> norm -> 1x1 expand 4x -> act -> 1x1 project -> narrow.
    Sub-step 1-2 use GELU and a single activation; sub-step 3-4 collapse to a
    single norm and swap it to LayerNorm; downsampling (sub-step 5) is pulled
    out of this block entirely -- see SeparateDownsample below.
    """
    expand_ratio = 4

    def __init__(self, dim, norm_layer, act_layer=nn.GELU):
        super().__init__()
        self.dw = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim, bias=False)
        self.norm = norm_layer(dim)                      # single norm, after dw, before the 1x1 pair
        hidden = dim * self.expand_ratio
        self.pw_expand = nn.Conv2d(dim, hidden, 1)
        self.act = act_layer()                            # single activation, between the two 1x1s
        self.pw_project = nn.Conv2d(hidden, dim, 1)

    def forward(self, x):
        identity = x
        out = self.dw(x)                                  # no activation, no norm here (sub-steps 2-3)
        out = self.norm(out)
        out = self.pw_expand(out)
        out = self.act(out)                                # the one nonlinearity in the block
        out = self.pw_project(out)                          # no activation here either (sub-step 2)
        return out + identity                                # no stride/downsample handled in-block

class SeparateDownsample(nn.Module):
    """Standalone 2x2 stride-2 conv between stages, bracketed by a norm on the
    reduced-resolution side -- the fix for training the un-normalized boundary
    that in-block downsampling used to cover implicitly."""
    def __init__(self, dim_in, dim_out, norm_layer):
        super().__init__()
        self.norm = norm_layer(dim_in)
        self.reduce = nn.Conv2d(dim_in, dim_out, kernel_size=2, stride=2)

    def forward(self, x):
        return self.reduce(self.norm(x))

# LayerNorm placed after the stem and after the final global pool too,
# bracketing every abrupt resolution/statistics change in the network.
```
