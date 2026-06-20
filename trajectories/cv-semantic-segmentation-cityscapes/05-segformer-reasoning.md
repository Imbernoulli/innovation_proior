79.61, and the gains are getting thin. Step back and notice the shape of everything I've built. I have
a convolutional backbone whose units are *locally* receptive — a stack of small kernels — and then on
top of it I keep bolting *context modules* whose entire job is to undo that locality: pyramid pooling,
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

Three problems stand between me and using attention as a segmentation backbone, and I have to solve all
three or it won't work.

First, cost. Self-attention is quadratic in the number of tokens. If a token is a pixel of even an OS-4
feature map, the token count is enormous and the attention matrix is hopeless — far worse than the
already-heavy ASPP. So full dense attention at segmentation resolution is out. I need attention that's
*globally receptive but sub-quadratic*. The fix: when I form the keys and values for an attention layer,
first spatially reduce the sequence — reshape the tokens back to a 2-D map, apply a strided convolution
that shrinks it by a factor R in each dimension, and flatten again. The queries stay at full resolution
(one per output location, so I still get a dense per-pixel result) but they attend over a key/value set
that's R² times smaller. That drops the attention cost by R² while keeping every query globally
receptive — it still attends across the whole (reduced) image. This is the lever that makes attention
affordable at the high resolutions segmentation needs.

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
convolutional decoders had to manufacture. The smallest configuration of this transformer (the tiny MIT
encoder) is what I check first, at the transformer's native crop with sliding-window inference; the
prediction is that even the tiny model lands in the same league as the convolutional heads, and that
scaling the same architecture up to a large encoder pushes the mIoU clearly past 79.61. If a globally-
receptive hierarchical attention backbone with a trivial decoder beats the best convolutional context
module at this fixed task, then the right structural move was to make context native to the backbone
rather than bolt it on — and that is where the ladder ends.
