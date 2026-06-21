DeepLabV3+ reaches **79.61 mIoU**, the strongest convolutional rung, and the gains are getting thin. Stepping back to see the shape of everything I have built: I have a convolutional backbone whose units are *locally* receptive — a stack of small kernels — and on top of it I keep bolting *context modules* whose entire job is to undo that locality. Pyramid pooling, then atrous pyramids: each is a hand-designed way to reach context the convolutions cannot reach on their own. I am fighting the backbone's inductive bias. Every context module is a patch over the same fact: a CNN has to *grow* its receptive field stage by stage, its effective receptive field stays smaller than its theoretical one, and so the global view that segmentation needs is never native — it is always reconstructed afterward by pooling or dilation, with hand-tuned scales and rates I had to choose. The atrous rates `(1, 12, 24, 36)`, the pool scales `(1, 2, 3, 6)` — those are my guesses at which context scales matter, baked into the architecture.

I propose **SegFormer**: make context *native* by building the backbone out of self-attention, so context is not a bolt-on but a property of every layer. Self-attention has each location attend to *every* other location and aggregate information weighted by content similarity, in a single layer — no growing of receptive field, no chosen dilation rates; every token sees the whole image from layer one, and *which* far-away context to use is learned per-pixel from the data rather than fixed by a pool scale. Three obstacles stand between me and using attention as a segmentation backbone, and all three must be solved. First, cost. Self-attention is quadratic in the number of tokens, and at segmentation resolution the token count is enormous and the attention matrix hopeless. The fix is *efficient (spatial-reduction) attention*: when I form the keys and values for an attention layer, I first reshape the tokens back to a 2-D map, apply a strided convolution that shrinks it by a factor $R$ in each dimension, and flatten again. The queries stay at full resolution — one per output location, so the result is still dense per-pixel — but they attend over a key/value set that is $R^2$ times smaller, dropping the attention cost by $R^2$ while every query remains globally receptive, attending across the whole (reduced) image. This is the lever that makes attention affordable at the high resolutions segmentation needs.

Second, the single-resolution problem. A plain vision transformer tokenizes the image once into a coarse grid and keeps that resolution throughout — fine for classifying a whole image, useless for dense prediction, which needs *both* a fine map for boundaries and a coarse map for high-level semantics, exactly the multi-scale features a CNN backbone naturally produces across its stages. So I build the transformer *hierarchically*: four stages, each beginning by re-tokenizing its input into a coarser grid, so the stages emit features at $1/4$, $1/8$, $1/16$, $1/32$ resolution — a feature pyramid, like a CNN backbone, but made of attention. And the tokenizing matters: cutting the image into *non-overlapping* patches (the plain-ViT way) severs local continuity right at every patch boundary, which is poison for a task whose whole point is getting boundaries right. So I use *overlapping* patch embedding — a strided convolution with a kernel larger than its stride — so adjacent patches share pixels and continuity is preserved across the whole tokenization; each stage's patch-embed conv both downsamples (sets the stage's resolution) and overlaps (keeps continuity). Third, position. Attention is permutation-invariant — it has no idea where tokens are — so transformers add positional encodings, but a *fixed-length* positional table is tied to the resolution it was trained at, and segmentation runs at crops different from training, where interpolating the table hurts. I would rather carry position *implicitly*. A convolution already encodes position — its padding leaks absolute location, its locality encodes relative arrangement — so I drop explicit positional encodings entirely and instead slip a depthwise $3\times3$ convolution into the feed-forward network between its two linear layers. That conv injects the positional and local information attention lacks, at every layer, with no resolution-tied table to interpolate; the feed-forward becomes a *Mix-FFN*, a mix of an MLP and a conv.

The decoder is where the globally-receptive backbone pays off twice. With a CNN backbone the deep features had a *small* effective receptive field, so I needed a heavy context module — ASPP, PPM — on top to manufacture global context; the decoder had to do real work. But my transformer stages are already globally receptive at every level, so the features arriving at the decoder *already carry global context*. The decoder no longer has to aggregate context; it only has to *fuse the four scales* and predict. So it collapses to an *all-MLP head*: take each stage's feature, push it through one linear layer ($1\times1$ conv) to a common channel width, bilinearly upsample all four to the finest $1/4$ resolution, concatenate, fuse with a single linear layer, and classify. No atrous pyramid, no pooling pyramid, no decoder convolutions to grow a receptive field — the head can be trivial precisely because the encoder already did the hard part. The whole design is internally consistent: spatial-reduction attention makes global receptivity *affordable*, the hierarchy makes it *multi-scale*, overlapping patch embedding and the Mix-FFN's depthwise conv supply the *locality and position* pure attention lacks, and because the encoder is globally receptive the decoder is an all-MLP fuse. Every context module I hand-designed across the previous rungs — the pool scales, the atrous rates, the explicit positional tables — has been dissolved into the backbone or made unnecessary. The bet against $79.61$ is that a backbone globally receptive from layer one, learning *which* context each pixel needs rather than reading it off fixed scales, produces cleaner predictions especially where long-range scene reasoning matters, and that the lightweight all-MLP decoder holds up precisely because the encoder already carries the context the convolutional decoders had to manufacture. The smallest configuration (the tiny MIT-B0 encoder) is what I check first, at the transformer's native crop with sliding-window inference; it lands in the same league as the convolutional heads at $76.54$ mIoU, and scaling the same architecture up to the large MIT-B5 encoder pushes the mIoU clearly past the best convolutional rung to $82.25$. The right structural move was to make context native to the backbone rather than bolt it on — and that is where the ladder ends.

```python
class EfficientMultiheadAttention(MultiheadAttention):
    """Global attention with a spatially-reduced key/value sequence."""
    def __init__(self, embed_dims, num_heads, sr_ratio=1,
                 norm_cfg=dict(type='LN'), **kw):
        super().__init__(embed_dims, num_heads, **kw)
        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = Conv2d(embed_dims, embed_dims,
                             kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = build_norm_layer(norm_cfg, embed_dims)[1]

    def forward(self, x, hw_shape, identity=None):
        x_q = x
        if self.sr_ratio > 1:
            x_kv = nlc_to_nchw(x, hw_shape)
            x_kv = self.sr(x_kv)                 # HxW -> (H/R)x(W/R)
            x_kv = nchw_to_nlc(x_kv)
            x_kv = self.norm(x_kv)
        else:
            x_kv = x
        if identity is None:
            identity = x_q
        out = self.attn(query=x_q, key=x_kv, value=x_kv)[0]
        return identity + self.dropout_layer(self.proj_drop(out))

class MixFFN(BaseModule):
    """FFN with a depthwise 3x3 conv carrying positional information."""
    def __init__(self, embed_dims, feedforward_channels,
                 act_cfg=dict(type='GELU'), **kw):
        super().__init__()
        fc1 = Conv2d(embed_dims, feedforward_channels, 1)
        pe_conv = Conv2d(feedforward_channels, feedforward_channels, 3,
                         padding=1, groups=feedforward_channels)
        fc2 = Conv2d(feedforward_channels, embed_dims, 1)
        self.layers = Sequential(fc1, pe_conv,
                                 build_activation_layer(act_cfg), fc2)

    def forward(self, x, hw_shape, identity=None):
        out = nlc_to_nchw(x, hw_shape)
        out = self.layers(out)
        out = nchw_to_nlc(out)
        if identity is None:
            identity = x
        return identity + self.dropout_layer(out)

class SegformerHead(BaseDecodeHead):
    """All-MLP head: unify, upsample, concat, fuse, classify."""
    def __init__(self, interpolate_mode='bilinear', **kwargs):
        super().__init__(input_transform='multiple_select', **kwargs)
        self.interpolate_mode = interpolate_mode
        self.convs = nn.ModuleList([
            ConvModule(self.in_channels[i], self.channels, 1,
                       norm_cfg=self.norm_cfg, act_cfg=self.act_cfg)
            for i in range(len(self.in_channels))])
        self.fusion_conv = ConvModule(self.channels * len(self.in_channels),
                                      self.channels, 1, norm_cfg=self.norm_cfg)

    def forward(self, inputs):
        inputs = self._transform_inputs(inputs)        # 1/4, 1/8, 1/16, 1/32
        outs = []
        for idx in range(len(inputs)):
            x = inputs[idx]
            outs.append(resize(self.convs[idx](x), size=inputs[0].shape[2:],
                               mode=self.interpolate_mode,
                               align_corners=self.align_corners))
        out = self.fusion_conv(torch.cat(outs, dim=1))
        return self.cls_seg(out)
```

The encoder is the Mix Transformer (MIT): four hierarchical stages, each an overlapping patch-embed conv followed by efficient-attention + Mix-FFN blocks, emitting the $1/4$–$1/32$ pyramid the all-MLP head fuses. SegFormer uses its own MIT backbone (not the R-50-D8 of the convolutional rungs) and runs at a $1024\times1024$ crop with sliding-window inference, so its mIoU is reported under that configuration rather than the $512\times1024$ / R-50-D8 setting.
