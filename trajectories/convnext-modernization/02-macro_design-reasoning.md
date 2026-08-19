The recipe swap alone moved ResNet-50 from 76.13% to 78.82% — a 2.69-point
gain at identical FLOPs, with the architecture untouched. That's a solid
chunk of the road to Swin-T's 81.30%, and it confirms training procedure was
carrying real weight in the original comparison, but it still leaves close
to 2.5 points on the table with zero architectural credit claimed yet. Every
change from here on can be attributed to the network design alone, since the
recipe is now frozen. Time to open the block.

I don't want to start by touching the convolution operator itself — grouped
convolution, depthwise convolution, kernel size, inverted bottlenecks — all
of that changes what each block computes and how much it costs, and I'd
rather not conflate a "how compute is spent inside a block" question with a
"how compute is distributed across the network" question. The second
question is cheaper to answer first and orthogonal to the first: it only
asks how many blocks go in each stage and how the input is initially
downsampled, without touching what a block does internally. That's a
macro-level, structural question, and I can settle it by directly copying
two structural facts about Swin-T that have nothing to do with attention.

**Stage compute ratio.** ResNet-50's stage depths are (3, 4, 6, 3) — the
third stage ("res4," the one that operates on the 14x14 feature map) is
disproportionately heavy relative to the others, a choice that as far as I
know was driven mainly by compatibility with downstream detection heads that
read off that resolution, not by an ImageNet-classification-optimal search.
Swin-T uses a materially different ratio, 1:1:3:1, spreading compute more
evenly across the first three stages relative to ResNet's roughly
1:1.33:2:1. This is a purely combinatorial redistribution of the same kind
of block (still a residual block, still the same operations inside it) —
literally just how many of them per stage — so it's about as low-risk an
architectural edit as exists: same operator, same total parameter budget in
the same ballpark, different allocation. I'll change ResNet-50's depths from
(3, 4, 6, 3) to (3, 3, 9, 3), which is the closest analogue to Swin-T's ratio
expressible as integer block counts and which happens to land ResNet-50's
FLOPs close to Swin-T's 4.5G budget as a side effect (more depth moves into
the cheaper early-and-mid stages relative to the disproportionately
expensive res4 stage). I have no strong basis for claiming this exact ratio
is optimal for a ConvNet specifically — it's entirely possible a systematic
search over compute distributions would find something better, block-search
studies on this exact question already exist — but "copy the ratio a strong
peer architecture already uses at this FLOPs class" is a well-motivated
starting point that costs nothing extra to try, and refining the allocation
further isn't the point of this rung.

**Stem design.** Separately, and for a different reason, I want to revisit
how the very first layer processes the raw image. A ResNet stem is a 7x7
stride-2 convolution followed by a max pool, giving 4x spatial downsampling
using two operations chained together. Every Transformer-family
architecture handles this differently: ViT uses a single large-kernel,
non-overlapping "patchify" convolution (kernel size matching the patch
size, so each output pixel comes from a disjoint input region), and Swin-T
specifically uses a 4x4-kernel, stride-4 patchify layer to match its 4-stage
hierarchy's expected input resolution at the first stage. The rationale for
patchify isn't really about matching Transformers for its own sake — it's
that natural images are highly spatially redundant at the pixel level, and
aggressively, non-overlappingly downsampling right at the entry point costs
one convolution instead of two, with no overlapping receptive fields to
reconcile. If a network is going to aggressively downsample 4x in its first
layer regardless, doing it with one clean non-overlapping conv rather than a
strided conv plus a max pool is at minimum a simplification, and plausibly
loses nothing since both operations are doing fundamentally the same
job — spatial redundancy reduction — just packaged differently. I'll replace
the 7x7-stride-2-conv-plus-maxpool stem with a single 4x4, stride-4
convolution, matching Swin-T's patch size exactly (which is also required
for the two networks' stage resolutions to line up, useful for keeping the
FLOPs comparison apples-to-apples going forward). The channel count and
normalization/activation immediately after the stem follow the network's
current convention (BatchNorm + ReLU, unchanged from ResNet, since I haven't
touched normalization yet).

I'm proposing both of these together, as one macro-design rung, because
neither is really a hypothesis about *capacity* or about a new computational
primitive — they're both "does the network's outer skeleton, independent of
what's inside each block, matter" questions, and bundling them means one
training run settles both at once rather than spending two full runs on
what I expect to be a smaller, structural effect. I will still be able to
read the sub-step contribution of each individually if the two are reported
separately.

What do I expect? Modest movement, and I want to be explicit about why my
prior here is *modest* rather than large. Neither of these edits changes the
fundamental operator a block computes (still a 3x3-conv-based bottleneck,
still BatchNorm, still ReLU), the total parameter count is barely disturbed,
and the stem in particular touches only the very first layer of a 50-layer
network — whatever information is lost or preserved by choosing patchify
over strided-conv-plus-maxpool is a rounding error against everything that
happens in the 49 layers after it. The stage-ratio change has slightly more
room to matter, since it changes where roughly half the network's depth
sits, but redistributing existing blocks among stages, without changing
what any individual block computes, is a coarser lever than changing the
block itself. My expectation is that this rung is a "control for the outer
skeleton before committing to it" step more than a "close a big chunk of the
remaining gap" step — I'd be mildly surprised by a swing larger than roughly
a point in either direction, and I'd read a strong negative surprise as
evidence that ResNet's original stem or compute distribution was doing more
work than I credited it for, which would need explaining before I trust any
skeleton I inherit from Swin-T for the rest of the ladder. Either way, this
locks in the macro shape — stage depths (3, 3, 9, 3), patchify stem — that
every subsequent rung's block-level edits will sit inside.
