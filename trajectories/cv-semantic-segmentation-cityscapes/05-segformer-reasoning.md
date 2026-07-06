79.61, 81.01 with multi-scale-and-flip, and the gains are getting thin. Lay the ladder out and the trend
is unmistakable: from-scratch encoder–decoder 69.10, pretrained dilated backbone 72.25 (+3.15), pooling
pyramid 77.85 (+5.60), separable atrous pyramid with a decoder 79.61 (+1.76). The context jump was the
big one; the rung right after it, refining *how* that context is gathered and adding a boundary decoder,
returned less than a third of it. And the ms+flip gaps tell the same story from the side — +1.95, +1.11,
+1.33, +1.40 — a boundary-flavored last rung nudging the augmentation gap up exactly as I expected, which
is reassuring about the mechanism but also a sign I am now polishing rather than transforming. Two rungs
of diminishing returns on the same move — bolt another context module onto the convolutional backbone —
is the signal to stop tuning the move and question it. Step back and notice the shape of everything I have
built. I have a convolutional backbone whose units are *locally* receptive — a stack of small kernels —
and then on top of it I keep bolting *context modules* whose entire job is to undo that locality: pyramid
pooling,
then atrous pyramids, each one a hand-designed way to reach context the convolutions can't reach on
their own. I'm fighting the backbone's inductive bias. Every context module is a patch over the same
underlying fact: a CNN has to *grow* its receptive field stage by stage, its effective receptive field
stays smaller than its theoretical one, and so the global view that segmentation needs is never native —
it's always reconstructed afterward by pooling or dilation, with hand-tuned scales and rates that I had
to choose. The atrous rates `(1, 12, 24, 36)`, the pool scales `(1, 2, 3, 6)` — those are my guesses
at what context scales matter, baked into the architecture.

What if the backbone itself were globally receptive, so context isn't a bolt-on but native to every
layer? The operation that does this is self-attention: each location attends to *every* other location
and aggregates information weighted by content similarity, in a single layer. No growing of receptive
field, no chosen dilation rates — every token sees the whole image from layer one, and *which* far-away
context to use is learned per-pixel from the data rather than fixed by a pool scale. That directly
attacks the thing all my context modules were working around.

The difference is not just "more context" — it is *content-adaptive* context, and that is a qualitative
change from everything below. The ASPP samples three fixed dilation offsets regardless of what is at those
offsets; the PPM averages fixed rectangular regions regardless of what falls in them. Both gather context
at locations the architecture chose in advance. Attention instead lets each query pull from *whichever*
other locations are relevant to it, weighted by learned content similarity: a pixel on a road/sidewalk
seam can attend specifically to the curb line and the building base that resolve it, and a pixel on a pole
can attend to the sign it belongs to, wherever those happen to sit in the frame — near or far, above or
below — because the weighting is computed from the tokens, not from a fixed grid. So the very thing I kept
hand-tuning across three rungs — *which* context scales and offsets matter — becomes a learned, per-pixel,
per-image quantity. That is why I expect this to reach errors the fixed-offset context modules structurally
could not: the hard remaining cases are exactly the ones where the disambiguating context is somewhere the
fixed offsets did not look.

Three problems stand between me and using attention as a segmentation backbone, and I have to solve all
three or it won't work.

First, cost. Self-attention is quadratic in the number of tokens, and I should put a number on "quadratic"
before waving it away. The transformer wants a fairly high input resolution to have fine tokens for
boundaries; at a 1024×1024 crop, a 1/4-resolution first stage tokenizes to a 256×256 grid — that is
65,536 tokens. Dense attention forms a token-by-token score matrix, so 65,536² ≈ 4.3 billion entries per
head per layer, and that is just the first stage. The attention matrix is hopeless — orders of magnitude
worse than the already-heavy ASPP, which at least ran fixed-size kernels. So full dense attention at
segmentation resolution is out. I need attention that is *globally receptive* — every query still sees the
whole image — but that does not pay the full N² for it. The fix: when I form the keys and values for an
attention layer,
first spatially reduce the sequence — reshape the tokens back to a 2-D map, apply a strided convolution
that shrinks it by a factor R in each dimension, and flatten again. The queries stay at full resolution
(one per output location, so I still get a dense per-pixel result) but they attend over a key/value set
that's R² times smaller. That drops the attention cost by R² while keeping every query globally
receptive — it still attends across the whole (reduced) image. This is the lever that makes attention
affordable at the high resolutions segmentation needs.

Work the reduction through the stages, because R has to be large where the map is large and can relax
where the map is small. Take the reduction schedule R = 8, 4, 2, 1 across the four stages, matched to
their resolutions. Stage 1: 256×256 = 65,536 queries, keys/values reduced 8× per side to 32×32 = 1,024,
so the score matrix is 65,536 × 1,024 ≈ 6.7×10⁷ instead of 4.3×10⁹ — a 64× cut, exactly R² = 8². Stage 2:
128×128 = 16,384 queries, R = 4 gives 32×32 = 1,024 keys, 1.7×10⁷ instead of 2.7×10⁸, a 16× cut. Stage 3:
64×64 = 4,096 queries, R = 2 gives 32×32 keys, a 4× cut. Stage 4: 32×32 = 1,024 queries, R = 1 — no
reduction, because at ~1,000 tokens the full matrix is already only ~10⁶ entries, cheap. So the schedule
spends the reduction where the tokens are dense and stops paying for it once the grid is coarse enough
that dense attention is free — and notice each stage's reduced key grid lands around 32×32, so the
attention cost is roughly balanced across stages rather than exploding at the fine end. The queries stay
dense throughout, which is what I need: a per-pixel output at every stage, so the hierarchy still produces
dense feature maps for the decoder to fuse.

One small choice inside the reduction: I shrink the key/value map with a *strided convolution*, not an
average pool. A pool would just blur R×R blocks into their mean before attention reads them, throwing
away structure the same way the PPM did; a strided conv is *learned*, so the network can decide what to
keep when it summarizes the key/value tokens — it can preserve the distinctions that matter for matching
rather than averaging them out. It is the same lesson from two rungs back (learned beats fixed pooling for
summarization), applied now to the attention keys instead of the context branch. The reduction only
touches keys and values, never the queries, so it costs a strided conv per attention layer and buys the
R² saving without ever coarsening the output grid.

```python
class EfficientMultiheadAttention(MultiheadAttention):
    def __init__(self, embed_dims, num_heads, sr_ratio=1, norm_cfg=dict(type='LN'), **kw):
        super().__init__(embed_dims, num_heads, **kw)
        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = Conv2d(embed_dims, embed_dims, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = build_norm_layer(norm_cfg, embed_dims)[1]

    def forward(self, x, hw_shape, identity=None):
        x_q = x
        if self.sr_ratio > 1:                 # shrink the key/value sequence
            x_kv = nlc_to_nchw(x, hw_shape)
            x_kv = self.sr(x_kv)              # strided conv: HxW -> (H/R)x(W/R)
            x_kv = nchw_to_nlc(x_kv)
            x_kv = self.norm(x_kv)
        else:
            x_kv = x
        if identity is None:
            identity = x_q
        out = self.attn(query=x_q, key=x_kv, value=x_kv)[0]
        return identity + self.dropout_layer(self.proj_drop(out))
```

Second, the single-resolution problem. A plain vision transformer tokenizes the image once into a
coarse grid and keeps that resolution throughout — fine for classifying a whole image, useless for
dense prediction, which needs *both* the fine map for boundaries and the coarse map for high-level
semantics, exactly the multi-scale features a CNN backbone naturally produces across its stages. So I
build the transformer *hierarchically*: four stages, each beginning by re-tokenizing its input into a
coarser grid, so the stages emit features at 1/4, 1/8, 1/16, 1/32 resolution — a feature pyramid, like
a CNN backbone, but made of attention. And the tokenizing matters: if I cut the image into
*non-overlapping* patches (the plain-ViT way), I sever the local continuity right at every patch
boundary, which is poison for a task whose whole point is getting boundaries right. So I use
*overlapping* patch embedding — a strided convolution with a kernel larger than its stride — so adjacent
patches share pixels and local continuity is preserved across the whole tokenization. Each stage's
patch-embed conv both downsamples (sets the stage's resolution) and overlaps (keeps continuity).

Make the tokenizer concrete so "overlapping" is a spec and not a slogan. The first stage tokenizes the
image with a 7×7 convolution at stride 4 with padding 3: stride 4 sets the 1/4-resolution grid (1024 →
256), and because the kernel is 7 wide but steps only 4, each patch's window overlaps its neighbor's by
7 − 4 = 3 pixels, so no pixel sits on a hard seam between two tokens. The remaining three stages each use
a 3×3 convolution at stride 2 with padding 1, halving the resolution (1/4 → 1/8 → 1/16 → 1/32) with a
kernel-minus-stride overlap of 1 each. Contrast the plain-ViT tokenizer: a 16×16 convolution at stride
16, kernel *equal* to stride, zero overlap — every 16-pixel patch boundary is a wall the tokens cannot
see across, and a segmentation model asked to place a boundary exactly at such a wall has been handed a
gridded blind spot. The overlap costs almost nothing (a slightly larger kernel on the downsampling conv)
and buys continuous coverage, which for a boundary task is not optional. Tracing the resolutions confirms
the pyramid I wanted: 256×256, 128×128, 64×64, 32×32 — the 1/4…1/32 feature stack a CNN backbone gives,
here produced by strided overlapping convolutions between attention stages rather than by pooling.

Third, position. Attention is permutation-invariant — it has no idea where tokens are — so transformers
add positional encodings. But a *fixed-length* positional encoding is tied to the resolution it was
trained at, and segmentation runs at test resolutions and crops different from training; interpolating
the position table hurts. I'd rather the network carry position *implicitly*. A convolution already
encodes position — its padding leaks absolute location, and its locality encodes relative arrangement.
So I drop explicit positional encodings entirely and instead slip a depthwise 3×3 convolution into the
feed-forward network between its two linear layers. That conv injects the positional/local information
attention lacks, at every layer, with no resolution-tied table to interpolate. The feed-forward becomes
a "mix" of an MLP and a conv:

```python
class MixFFN(BaseModule):
    def __init__(self, embed_dims, feedforward_channels, act_cfg=dict(type='GELU'), **kw):
        super().__init__()
        fc1 = Conv2d(embed_dims, feedforward_channels, 1)               # 1x1 = linear
        pe_conv = Conv2d(feedforward_channels, feedforward_channels, 3, # depthwise 3x3:
                         padding=1, groups=feedforward_channels)        #   carries position
        fc2 = Conv2d(feedforward_channels, embed_dims, 1)
        self.layers = Sequential(fc1, pe_conv, build_activation_layer(act_cfg), fc2)

    def forward(self, x, hw_shape, identity=None):
        out = nlc_to_nchw(x, hw_shape)
        out = self.layers(out)
        out = nchw_to_nlc(out)
        if identity is None:
            identity = x
        return identity + self.dropout_layer(out)
```

The resolution-invariance is the whole reason this beats a positional table for *this* task, and it is
worth being concrete about why. A learned positional encoding is a table with one entry per token
position, so it is fixed to the token grid it was trained on. Train at a 1024×1024 crop and stage 1's
table has 256×256 entries; but segmentation does not evaluate on a single fixed grid — the images are
1024×2048, larger than the crop, so inference runs a sliding window, and any global-context model wants to
see whole scenes at test time. The moment the test grid differs from 256×256, the table has to be
bilinearly interpolated to the new size, and interpolating a *learned* position code is lossy in a way
that degrades exactly the fine localization I care about. A depthwise 3×3 convolution has none of this
brittleness: it is a fixed 9-weight-per-channel operator that runs identically at any spatial size, and it
encodes position implicitly — its zero-padding at the borders leaks absolute location (a token near an
edge sees padding on one side), and its locality encodes the relative arrangement of neighbors. So the
Mix-FFN carries position without ever committing to a resolution, which means the same trained weights run
at the training crop, at the sliding-window tiles, and at whole 1024×2048 frames with no interpolation step
to lose detail. That is a property a fixed table structurally cannot have, and it is why I would rather pay
one cheap depthwise conv per block than maintain a positional table.

Now the decoder. This is where the globally-receptive backbone pays off twice. With a CNN backbone the
deep features had a *small* effective receptive field, so I needed a heavy context module (ASPP, PPM) on
top to manufacture global context — the decoder had to do real work. But my transformer stages are
already globally receptive at every level, so the features arriving at the decoder *already carry global
context*. The decoder no longer has to aggregate context; it only has to *fuse the four scales* and
predict. So it can be radically simple: take each stage's feature, push it through one linear layer to a
common channel width, bilinearly upsample all four to the finest (1/4) resolution, concatenate, fuse
with a single linear layer, and classify. No atrous pyramid, no pooling pyramid, no decoder convolutions
to grow a receptive field — an all-MLP head, because the encoder already did the hard part.

```python
class SegformerHead(BaseDecodeHead):
    def __init__(self, interpolate_mode='bilinear', **kwargs):
        super().__init__(input_transform='multiple_select', **kwargs)
        self.convs = nn.ModuleList([                  # 1x1 = per-stage linear unify
            ConvModule(self.in_channels[i], self.channels, 1, norm_cfg=self.norm_cfg,
                       act_cfg=self.act_cfg) for i in range(len(self.in_channels))])
        self.fusion_conv = ConvModule(self.channels * len(self.in_channels),
                                      self.channels, 1, norm_cfg=self.norm_cfg)

    def forward(self, inputs):
        inputs = self._transform_inputs(inputs)        # 4 stages: 1/4,1/8,1/16,1/32
        outs = []
        for idx in range(len(inputs)):
            x = inputs[idx]
            outs.append(resize(self.convs[idx](x), size=inputs[0].shape[2:],
                               mode=self.interpolate_mode,
                               align_corners=self.align_corners))   # all -> 1/4 res
        out = self.fusion_conv(torch.cat(outs, dim=1))  # fuse the 4 scales (one linear)
        return self.cls_seg(out)
```

Trace the head's shapes to see how little it does, because that is the point. With the smallest encoder
the four stages emit channel widths of about 32, 64, 160, 256 at resolutions 1/4, 1/8, 1/16, 1/32. Each
stage's 1×1 `conv` maps its width to a common decoder width — 256 — so I now have four 256-channel maps at
the four resolutions. `resize` lifts all four to the finest, 1/4 (256×256 on the crop). Concatenate:
4 × 256 = 1,024 channels at 1/4. The single `fusion_conv` 1×1 folds 1,024 → 256, and `cls_seg` maps
256 → 19. That is the entire decoder: four linear projections, four bilinear upsamples, one concat, one
linear fuse, one classifier. There is no atrous pyramid, no pooling pyramid, no stack of 3×3 convs growing
a receptive field — the head never has to *gather* context because every one of the four maps it receives
was produced by globally-receptive attention and already carries whole-scene context at its scale. The
head's only job is to align the four scales onto one grid and let a linear layer weigh them, which is why
a pure-MLP head suffices where the convolutional rungs needed an ASPP or a PPM doing real work.

The whole design is internally consistent: spatial-reduction attention makes global receptivity
*affordable*, the hierarchy makes it *multi-scale*, overlapping patch embedding and the Mix-FFN's
depthwise conv supply the *locality and position* that pure attention lacks, and because the encoder is
globally receptive the decoder collapses to an all-MLP fuse. Every context module I hand-designed across
the previous rungs — the pool scales, the atrous rates, the explicit positional tables — has been
dissolved into the backbone or made unnecessary.

The bet, against the 79.61 of the strongest convolutional rung. A backbone that is globally receptive
from layer one, learning *which* context each pixel needs rather than reading it off fixed pool/atrous
scales, should produce cleaner predictions especially where long-range scene reasoning matters — and the
lightweight all-MLP decoder should hold up precisely because the encoder already carries the context the
convolutional decoders had to manufacture. One honesty note before the numbers: this rung changes the
setting, unlike the four before it. The convolutional rungs shared the R-50-D8 backbone and the 512×1024
crop; this uses its own Mix Transformer backbone at a 1024×1024 crop with sliding-window inference, so its
mIoU is reported under that configuration rather than the held-fixed one. The comparison is therefore
architecture-family to architecture-family, not a drop-in swap, and I will say so.

I check the smallest configuration first, because the sharpest test of the thesis is whether even a
*tiny* globally-receptive encoder is competitive — if the idea only works at enormous scale it is a weaker
claim. The smallest encoder lands at 76.54 mIoU. Read that honestly: it is in the same league as the
pooling pyramid's 77.85 and clearly above the pretrained-FCN's 72.25, but it is *below* the best
convolutional rung's 79.61 — so a tiny attention backbone with a trivial decoder already matches a
hand-designed context module, but does not by itself beat the strongest one. That is exactly the
competitive-at-tiny-scale result I wanted as evidence the structure is right, and it tells me the
remaining gap is capacity, not concept. So I scale the same architecture up — wider stages, more blocks,
nothing structurally new — and the large encoder reaches 82.25 mIoU, clearly past 79.61. Scaling a
globally-receptive backbone converts directly into segmentation quality, which is what a design whose
bottleneck is capacity rather than a fixed context recipe should do. On the metric pair I expect the
usual ~1–1.5 point lift from multi-scale-and-flip on top of each single-scale number, in line with the
+1.3–1.4 the last two convolutional rungs showed; I only have the single-scale values as the landing
here, but a globally-receptive model should if anything be *less* dependent on test-time scale
augmentation, since scale reasoning is native to attention rather than reconstructed.

If a globally-receptive hierarchical attention backbone with a trivial decoder matches the best
convolutional context module at tiny scale and clearly beats it when scaled up, then the right structural
move was to make context native to the backbone rather than bolt it on — every pool scale, every atrous
rate, every positional table I hand-designed across the previous rungs dissolved into the encoder or made
unnecessary — and that is where the ladder ends.
