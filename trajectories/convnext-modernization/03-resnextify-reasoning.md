The macro-design rung landed almost exactly where I expected: 78.82% to
79.51%, +0.69 points, a real but modest gain, confirming my prior that
redistributing existing blocks and simplifying the stem is a small lever
compared to whatever lives inside each block. Both sub-steps — stage ratio
alone (79.36%) and stage ratio plus patchify stem (79.51%) — moved in the
same small-positive direction with no surprises, which tells me the outer
skeleton I inherited from Swin-T (stage depths (3, 3, 9, 3), 4x4 patchify
stem) is at minimum not costing anything, and I can now build on it with
confidence rather than revisit it. FLOPs sit at 4.42G, close to Swin-T's
4.5G, so I have essentially no budget headroom left before I'd be
comparing an unfairly larger network — any FLOPs increase from here needs a
matching decrease somewhere else, or a clear justification.

Now the block itself. ResNet-50's bottleneck computes its spatial mixing
with a full (non-grouped) 3x3 convolution: every output channel is a
learned combination of every input channel in its receptive field, mixing
space and channel information simultaneously in one operation. There's a
well-established alternative for the same operation, grouped convolution:
split both input and output channels into groups and only convolve within
each group, cutting FLOPs roughly in proportion to the number of groups. The
established strategy that comes with grouped convolution, rather than just
using it to save compute and stopping there, is to reinvest the saved FLOPs
into width — more groups means each group is cheap, so widen the network
back toward (or past) its original FLOPs budget while keeping the improved
group structure. Taken to its extreme, grouped convolution with the number
of groups equal to the number of channels is depthwise convolution: each
output channel depends on exactly one input channel's spatial neighborhood,
nothing else. That's the extreme worth trying first, for a reason specific
to the broader goal of this exploration rather than to compute savings
alone: it produces a clean separation of what mixes space and what mixes
channels. A depthwise conv only ever looks across the spatial window of a
single channel — it cannot mix information *between* channels. The 1x1
convolutions on either side of it in the bottleneck, meanwhile, only ever
look at a single spatial position — they cannot mix information *across*
space. So a depthwise-conv-plus-1x1-convs block factors "mix space" and
"mix channels" into two completely separate operations, rather than one
operation doing both at once the way a standard 3x3 conv does. That
factorization is worth calling out explicitly because it's exactly the
structure of a self-attention block: the attention operation mixes
information spatially (a weighted sum over other spatial positions, done
per-channel) and contributes nothing to channel mixing, while the
feedforward sublayer mixes channels and contributes nothing spatially. If
part of the value Transformer-style networks are extracting comes from
keeping those two kinds of mixing cleanly separated rather than conflated,
depthwise convolution is the direct convolutional analogue of the spatial-mixing
half, and it's worth testing as a hypothesis in its own right, not
only as a FLOPs-saving trick.

The obvious risk, and I should be explicit that I expect to see it before I
see the fix: a depthwise conv alone has far less representational capacity
per parameter than a full 3x3 conv at the same channel count, since it
literally cannot combine information across channels at all — that job gets
pushed entirely onto the 1x1 layers, which in the *current* block are still
sized for a full-mixing 3x3 conv, not compensated for a channel-mixing
bottleneck that just lost a big chunk of its job. So I predict that
swapping to depthwise convolution *by itself*, at the network's current
channel widths, will *cost* accuracy relative to the 79.51% macro-design
point, not gain it — the FLOPs will drop substantially (a depthwise conv is
much cheaper than a full conv at the same channel count) but so, I expect,
will accuracy, since I'm removing cross-channel mixing capacity from the
spatial layer without yet replacing it anywhere. This isn't a reason not to
try it; it's the predicted first half of a two-part move, and I want it
measured as its own point specifically so I can see the size of the
capacity hole before deciding how much width to add back.

The second half is the standard fix: widen. If depthwise convolution frees
up FLOPs (because grouped-to-the-limit convolution is FLOPs-cheap per
channel), the established response is to spend those freed FLOPs on more
channels rather than banking the savings, since a wider network gives every
layer — including the 1x1s that now carry the entire channel-mixing burden —
more capacity to work with. I need a concrete target width, not an
open-ended "wider," and there's a natural one already sitting in this
exploration: Swin-T's channel count. Swin-T's stages run at 96 channels at
the first stage (I've already matched its stage ratio and patch size, so
matching its width too keeps every structural comparison to it apples-to-
apples, and gives me one less free parameter to justify by taste). ResNet-50
currently runs at 64 base channels; I'll widen to 96. This should roughly
restore, and by the stated strategy possibly exceed, the FLOPs that were cut
by going depthwise, landing the network back near the 4.4-4.5G budget I've
been tracking rather than leaving depthwise's cheapness on the table
unused.

So the concrete test: depthwise convolution replaces the 3x3 conv in the
bottleneck, groups equal to channel count, first at the current width (64),
then again after widening the base channel count from 64 to 96 (matching
Swin-T). I expect the first measurement to be a real regression below
79.51% — my best guess is somewhere in the mid-to-high 70s given how much
cross-channel capacity a full-group depthwise conv removes at unchanged
width, though I don't have a way to pin the exact number without training
it — and the second to at least recover the macro-design point, ideally
exceed it, since width is the standard, well-supported answer to exactly
this kind of capacity loss. If widening doesn't recover the loss, that
would be evidence the depthwise-plus-1x1 factorization is a net capacity
loss even after accounting for the freed compute, and depthwise convolution
would not be worth carrying forward into the rest of the ladder — but I
expect it will recover, because the freed FLOPs from going to the grouping
extreme are large enough that a fairly modest width increase (64 to 96, not
some enormous multiple) should already restore, and the 1x1 layers doing
the channel mixing get wider along with everything else, which is exactly
where the lost capacity needs to land.
