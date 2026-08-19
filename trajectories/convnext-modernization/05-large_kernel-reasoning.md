Inverting the block paid off cleanly: FLOPs dropped from 5.27G to 4.64G —
the narrowed downsampling shortcuts more than offset the depthwise conv now
running at the expanded, 4x width — while accuracy nudged up slightly,
80.50% to 80.64%. That confirms the rearrangement bought back budget
discipline essentially for free, and puts the network back closer to the
4.5G band I've been tracking against Swin-T. Nothing about that rung
changed how much of the image each layer actually sees, though — same 3x3
depthwise kernel throughout this entire ladder, same as vanilla ResNet's
3x3 spatial convs going all the way back. That's worth stopping on. The
single most distinguishing property of self-attention, more than any of the
macro or block-shape edits I've made so far, is that it isn't local: a
standard attention layer's receptive field is the entire spatial extent it
attends over. Even Swin's windowed variant, which deliberately restricts
attention to local regions for tractability, still uses windows at least
7x7 — more than four times the linear extent, and about five times the
area, of the 3x3 kernel every convolution in this network has used since
rung 0. I've spent four rungs matching Swin-T's macro layout, block shape,
and channel-mixing structure without touching the one property that most
obviously distinguishes what each layer can even see. Large kernel-sized
convolutions have existed since early ConvNets, but 3x3 became, and stayed,
the standard because stacking small kernels has efficient, well-optimized
GPU implementations that a single large kernel historically didn't match —
an engineering-cost argument, not evidence that a bigger receptive field per
layer is undesirable. It's worth revisiting now specifically because so
much else about the block has changed since that convention was set.

But I can't just widen the kernel on the current block and expect a clean
read, because of where the depthwise conv currently sits. After rung 4, the
order inside the block is: narrow input, 1x1 expand to 4x width, depthwise
conv *at that expanded width*, 1x1 project back to narrow. Depthwise conv
FLOPs scale linearly with channel count (each channel processes its own
kernel independently, so cost is channels x kernel_area x spatial_positions).
Right now the depthwise conv sits at the most expensive place to grow it: 4x
the boundary channel count. If I sweep the kernel from 3x3 up toward 7x7,
9x9, 11x11 while the layer sits there, kernel area grows roughly with the
square of kernel size, multiplied by an already-4x-inflated channel count —
the FLOPs cost of exploring large kernels compounds badly with the current
ordering, and I'd be paying for channel width and spatial extent
simultaneously in the one layer I'm trying to grow.

There's a reordering that avoids this, and it isn't arbitrary — it mirrors
something already established in the Transformer block I've been using as a
reference throughout this whole ladder. A Transformer block's sublayers run
in a fixed order: the spatial-mixing operation (multi-head self-attention)
comes *first*, operating on the model's narrow residual-stream width, and
only afterward does the feedforward sublayer expand to 4x width and project
back down. The expensive, structurally complex operation runs at low
channel count; the cheap, dense, generic 1x1-style mixing runs at high
channel count and does the bulk of the parameter-heavy work. My current
block has this backwards relative to that convention — the depthwise conv,
which is about to become the expensive, structurally complex operation
(large-kernel spatial mixing) is sitting at the wide, expensive position,
while the 1x1s are the ones I'd expect to comfortably absorb high channel
count. So: move the depthwise conv from between the two 1x1s to *before*
the first one. The block becomes narrow input -> depthwise conv (now at the
narrow boundary width) -> 1x1 expand 4x -> 1x1 project back to narrow. This
is a direct, mechanical parallel to attention-before-MLP, and it's also the
efficiency argument I need: with the depthwise conv now scaled by the
narrow boundary width instead of the 4x-expanded width, growing its kernel
size costs roughly a quarter of what it would have cost in the rung-4
ordering, for the same kernel area. That's the prerequisite this reordering
buys — large kernels only become affordable to sweep once the expensive
spatial op is sitting at the cheap end of the channel-width spectrum.

I want to test this reordering as its own point, at the *unchanged* 3x3
kernel size, before touching kernel size at all — separating "did moving
the layer change anything by itself" from "did growing the kernel help,"
the same discipline I used splitting depthwise-conv-alone from
depthwise-conv-plus-width in rung 3. And I expect this reordering, on its
own, to cost accuracy rather than gain it, at least temporarily. The reason:
this changes what the very first operation inside the block sees. Under the
rung-4 ordering, the block's input goes through a 1x1 conv first — a full,
dense per-position linear mixing across every input channel — before any
spatial operation touches it. Under the new ordering, the raw block input
goes straight into a depthwise conv, which (at kernel size 3, unchanged for
this sub-test) mixes each channel only with its own small spatial
neighborhood, with no cross-channel mixing having happened yet at all. It's
the same total set of operations, just resequenced, so I'd expect any
capacity loss to be transient rather than structural — but re-sequencing
where information first gets combined inside a stack this deep plausibly
disturbs whatever the optimizer had settled into, and I don't have a strong
reason to expect that disturbance to be free. My prediction: the
reorder-only point comes in below 80.64%, a real but I'd guess modest dip,
since nothing about the block's total capacity changed, only the order
operations run in — this should read as a *temporary* cost paid for
unlocking cheaper large kernels, not a verdict on the reordering itself.

With the reorder done and cheap large kernels within reach, the actual
question this rung is trying to answer: does a bigger depthwise receptive
field help, and if so, how far do the gains extend? I want to sweep rather
than jump straight to one size, because I have no principled way to pick a
single "right" kernel size in advance — my only real anchor point is Swin's
choice of window size, which sits at a minimum of 7x7, and that's a
reasonable candidate but not something I want to commit to without
comparison points on both sides of it. Sweep 3 (matching the pre-existing
default, now re-measured after the reorder), 5, 7, 9, and 11, keeping every
other block dimension fixed. My prediction has a specific shape, not just
"bigger is better": I expect accuracy to climb as the kernel grows from 3
toward somewhere around 7, then flatten — plateau rather than continue
climbing, and possibly turn over and decline at the largest sizes. The
reasoning: a depthwise conv's per-channel spatial context is the thing
being grown here, and there should be a point past which enlarging that
context stops adding new information the network can use, either because
the effective receptive field from stacking many layers already covers
enough context by that point, or because a wider kernel spends more
parameters on redundant, highly-correlated neighboring positions rather
than genuinely new signal, or (less benignly) because optimizing an even
larger set of per-channel spatial weights gets harder without a
proportional payoff. Swin's own choice of minimum window size gives me a
concrete number to check the plateau against: if the point where gains
level off lands anywhere near 7, that would suggest ConvNets and windowed
attention are converging on a similar answer to "how much local spatial
context is enough" from two different directions — attention choosing it
as a locality restriction on an otherwise-global operation, convolution
approaching it from the small-kernel side and growing outward. If instead
the plateau sits well below or well above 7, that would say the two
architectures' locality sweet spots don't actually coincide, and matching
Swin's window size elsewhere in this exploration was more coincidence than
principle. Either way, once the sweep locates where returns flatten, I'll
adopt that kernel size and carry it forward as the network's spatial extent
for the rest of the ladder — this is the last change to what each
convolution can physically see; everything after this is per-layer
micro-design (activations, normalization) rather than receptive field.
