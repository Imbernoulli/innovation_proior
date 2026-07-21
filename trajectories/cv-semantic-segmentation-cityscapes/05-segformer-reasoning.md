79.61, 81.01 with multi-scale-and-flip, and the gains are getting thin. Lay the sequence out: from-scratch
encoder–decoder 69.10, pretrained dilated backbone 72.25 (+3.15), pooling pyramid 77.85 (+5.60), separable
atrous pyramid with a decoder 79.61 (+1.76). The context jump was the big one; the step right after it,
refining *how* that context is gathered and adding a boundary decoder, returned less than a third of it. The
ms+flip gaps say the same from the side — +1.95, +1.11, +1.33, +1.40 — a boundary-flavored last step
nudging the augmentation gap up as I expected, reassuring about the mechanism but a sign I am now polishing
rather than transforming. Two rounds of diminishing returns on the same move — bolt another context module
onto the convolutional backbone — is the signal to stop tuning the move and question it.

Step back and notice the shape of everything I have built. A convolutional backbone whose units are
*locally* receptive — a stack of small kernels — and on top of it I keep bolting *context modules* whose
entire job is to undo that locality: pyramid pooling, then atrous pyramids, each a hand-designed way to
reach context the convolutions cannot reach on their own. I am fighting the backbone's inductive bias. Every
context module patches the same underlying fact: a CNN grows its receptive field stage by stage, its
effective receptive field stays smaller than its theoretical one, and so the global view segmentation needs
is never native — always reconstructed afterward by pooling or dilation, with hand-tuned scales and rates I
had to choose. The atrous rates `(1, 12, 24, 36)`, the pool scales `(1, 2, 3, 6)` are my guesses at what
context scales matter, baked into the architecture.

What if the backbone itself were globally receptive, so context is native rather than a bolt-on? The
operation that does this is self-attention: each location attends to *every* other and aggregates by content
similarity in a single layer — no growing of receptive field, no chosen dilation rates, every token seeing
the whole image from layer one, and *which* far context to use learned per-pixel from data rather than
fixed by a pool scale. And it is not just "more context" but *content-adaptive* context, a qualitative
change: the ASPP samples three fixed dilation offsets regardless of what is there, the PPM averages fixed
rectangles regardless of what falls in them, both gathering context at locations the architecture chose in
advance. Attention lets each query pull from *whichever* other locations are relevant to it — a pixel on a
road/sidewalk seam attending to the curb line and building base that resolve it, a pixel on a pole attending
to the sign it belongs to, wherever those sit, near or far. The thing I kept hand-tuning across three steps —
*which* context scales and offsets matter — becomes a learned, per-pixel, per-image quantity, which is why I
expect this to reach errors the fixed-offset modules structurally could not: the hard remaining cases are
exactly the ones where the disambiguating context is somewhere the fixed offsets did not look.

Three problems stand between me and using attention as a segmentation backbone. First, cost: self-attention
is quadratic in tokens. A transformer wants fairly high resolution for fine boundary tokens; at a 1024×1024
crop a 1/4-resolution first stage tokenizes to a 256×256 grid — 65,536 tokens — and a dense token-by-token
score matrix is 65,536² ≈ 4.3 billion entries per head per layer, and that is just the first stage. Orders
of magnitude worse than the already-heavy ASPP, so full dense attention at segmentation resolution is out.
The fix: when forming the keys and values, first spatially reduce the sequence — reshape the tokens back to
a 2-D map, apply a strided convolution shrinking it by a factor R per side, flatten again. The queries stay
at full resolution (one per output location, so the result is still a dense per-pixel map) but attend over a
key/value set R² times smaller. That drops the attention cost by R² while every query stays globally
receptive — it still attends across the whole reduced image. And a strided conv rather than an average pool
for the reduction, because it is *learned*: the network decides what to keep when it summarizes the keys
rather than blurring R×R blocks into their mean the way the PPM did. The reduction touches only keys and
values, never the queries, so it costs a strided conv per attention layer and never coarsens the output
grid.

Work the reduction through the stages, since R must be large where the map is large and can relax where it
is small. Take R = 8, 4, 2, 1 across the four stages. Stage 1: 65,536 queries, keys reduced 8× per side to
32×32 = 1,024, so the score matrix is ≈6.7×10⁷ instead of 4.3×10⁹ — a 64× cut, exactly R² = 8². Stage 2:
16,384 queries, R = 4 gives 1,024 keys, a 16× cut. Stage 3: 4,096 queries, R = 2, a 4× cut. Stage 4: 1,024
queries, R = 1 — no reduction, because at ~1,000 tokens the full matrix is only ~10⁶ entries. Each stage's
reduced key grid lands around 32×32, so attention cost is roughly balanced across stages rather than
exploding at the fine end, and the queries stay dense throughout so the hierarchy still emits dense feature
maps for the decoder. The efficient-attention block is in the answer.

Second, the single-resolution problem. A plain vision transformer tokenizes the image once into a coarse
grid and keeps that resolution throughout — fine for classifying a whole image, useless for dense
prediction, which needs *both* the fine map for boundaries and the coarse map for high-level semantics, the
multi-scale features a CNN backbone naturally produces across its stages. So I build the transformer
*hierarchically*: four stages, each beginning by re-tokenizing its input into a coarser grid, emitting
features at 1/4, 1/8, 1/16, 1/32 resolution — a feature pyramid made of attention. And the tokenizing
matters: cutting the image into *non-overlapping* patches (the plain-ViT way) severs local continuity at
every patch boundary, poison for a task whose whole point is boundaries. So I use *overlapping* patch
embedding — a strided convolution with a kernel larger than its stride. Stage 1 tokenizes with a 7×7 conv at
stride 4, padding 3: stride 4 sets the 1/4 grid (1024 → 256), and the 7-wide kernel stepping 4 overlaps its
neighbor by 7 − 4 = 3 pixels, so no pixel sits on a hard seam; stages 2–4 each use a 3×3 at stride 2,
padding 1, halving resolution (1/4 → 1/8 → 1/16 → 1/32) with an overlap of 1. Contrast the plain-ViT 16×16
conv at stride 16, kernel *equal* to stride, zero overlap — every 16-pixel boundary a wall the tokens cannot
see across, a gridded blind spot exactly where a segmentation model must place boundaries. The overlap costs
almost nothing and buys continuous coverage.

Third, position. Attention is permutation-invariant, so transformers add positional encodings — but a
*fixed-length* positional table is tied to the resolution it was trained at, and segmentation runs at test
resolutions and crops different from training, where interpolating the table hurts exactly the fine
localization I care about. I would rather carry position implicitly. A convolution already encodes position:
its zero-padding leaks absolute location and its locality encodes relative arrangement. So I drop explicit
positional encodings and slip a depthwise 3×3 convolution into the feed-forward network between its two
linear layers — the Mix-FFN — injecting the positional/local information attention lacks at every layer,
with no resolution-tied table to interpolate. This is what lets the same trained weights run at the training
crop, at the sliding-window tiles, and at whole 1024×2048 frames with no lossy interpolation step — a
property a fixed table structurally cannot have, and worth one cheap depthwise conv per block. The Mix-FFN
block is in the answer.

Now the decoder, where the globally-receptive backbone pays off twice. With a CNN backbone the deep features
had a *small* effective receptive field, so I needed a heavy context module on top to manufacture global
context — the decoder did real work. But my transformer stages are already globally receptive at every
level, so the features arriving at the decoder *already carry global context*. The decoder no longer
aggregates context; it only *fuses the four scales* and predicts. So it can be radically simple: push each
stage's feature through one linear layer to a common channel width, bilinearly upsample all four to the
finest 1/4 resolution, concatenate, fuse with a single linear layer, classify — an all-MLP head, because the
encoder did the hard part. The head is in the answer; trace its shapes to see how little it does. With the
smallest encoder the four stages emit widths about 32, 64, 160, 256 at 1/4…1/32; each stage's 1×1 maps its
width to a common 256, giving four 256-channel maps, `resize` lifts all to 1/4 (256×256 on the crop),
concatenate to 4 × 256 = 1,024 channels, the fusion 1×1 folds to 256, the classifier maps to 19. Four linear
projections, four upsamples, one concat, one linear fuse, one classifier — no atrous or pooling pyramid, no
stack of 3×3 convs growing a receptive field, because every map the head receives was produced by
globally-receptive attention and already carries whole-scene context at its scale.

The design is internally consistent: spatial-reduction attention makes global receptivity *affordable*, the
hierarchy makes it *multi-scale*, overlapping patches and the Mix-FFN conv supply the *locality and position*
pure attention lacks, and because the encoder is globally receptive the decoder collapses to an all-MLP
fuse. Every context module I hand-designed — the pool scales, the atrous rates, the positional tables —
dissolves into the backbone or becomes unnecessary. The bet is against the 79.61 of the strongest
convolutional model: a backbone globally receptive from layer one, learning *which* context each pixel needs
rather than reading it off fixed scales, should produce cleaner predictions where long-range scene reasoning
matters, and the all-MLP decoder should hold up because the encoder already carries the context the
convolutional decoders had to manufacture. One honesty note: this changes the setting. The convolutional
models shared the R-50-D8 backbone and the 512×1024 crop; this uses its own Mix Transformer backbone at a
1024×1024 crop with sliding-window inference, so its mIoU is reported under that configuration rather than
the held-fixed one — an architecture-family comparison, not a drop-in swap.

I check the smallest configuration first, because the sharpest test of the thesis is whether even a *tiny*
globally-receptive encoder is competitive — if the idea only works at enormous scale it is a weaker claim.
The smallest encoder lands at 76.54 mIoU: in the same league as the pooling pyramid's 77.85 and clearly
above the pretrained-FCN's 72.25, but *below* the best convolutional 79.61. So a tiny attention backbone
with a trivial decoder already matches a hand-designed context module without yet beating the strongest —
exactly the competitive-at-tiny-scale evidence that the structure is right and the remaining gap is
capacity, not concept. Scaling the same architecture up — wider stages, more blocks, nothing structurally
new — the large encoder reaches 82.25 mIoU, clearly past 79.61. A globally-receptive backbone converts scale
directly into segmentation quality, which is what a design whose bottleneck is capacity rather than a fixed
context recipe should do. On the metric pair I expect the usual ~1–1.5 point lift from multi-scale-and-flip
in line with the last steps, if anything a little less, since scale reasoning is native to attention rather
than reconstructed. Making context native to the backbone rather than bolting it on was the right structural
move: every pool scale, every atrous rate, every positional table I hand-designed has dissolved into the
encoder or been made unnecessary.
