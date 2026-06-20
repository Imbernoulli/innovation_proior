**Problem (from step 4).** DeepLabV3+ reaches 79.61, but the whole convolutional line fights the
backbone's inductive bias: a CNN is locally receptive, so global context is never native — it is
reconstructed afterward by hand-designed context modules (pool scales `(1,2,3,6)`, atrous rates
`(1,12,24,36)`, explicit positional choices). The scales are guesses baked into the architecture.

**Key idea — SegFormer (hierarchical transformer encoder + all-MLP decoder).** Make context *native* by
using self-attention, which is globally receptive in one layer and learns per-pixel which context to
use. Three enabling pieces solve the three obstacles: (1) **Efficient (spatial-reduction) attention** —
shrink the key/value sequence with a strided conv (queries stay dense), making global attention
sub-quadratic and affordable at segmentation resolution; (2) a **hierarchical encoder** of four stages
emitting 1/4, 1/8, 1/16, 1/32 features (a feature pyramid made of attention) with **overlapping patch
embedding** (strided conv, kernel > stride) to preserve local continuity across tokenization; (3)
**Mix-FFN** — a depthwise 3×3 conv inside the feed-forward network that carries position implicitly, so
*no* resolution-tied positional encoding is needed. Because every encoder stage is already globally
receptive, the decoder collapses to a **lightweight all-MLP head**: a per-stage linear unify, bilinear
upsample all four to 1/4 resolution, concatenate, fuse with one linear layer, classify.

**Why it works.** Global receptivity from layer one dissolves the context modules the convolutional
rungs had to bolt on; the network learns which long-range context each pixel needs instead of reading it
off fixed scales. Spatial-reduction attention buys the global view affordably; overlapping patches and
the Mix-FFN conv restore the locality and position pure attention lacks. The decoder can be trivial
precisely because the encoder already carries context — the work the ASPP/PPM decoders did is done
upstream.

**Change / code.** Efficient attention, Mix-FFN, and the all-MLP head.

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

The encoder is the Mix Transformer (MIT): four hierarchical stages, each an overlapping patch-embed
conv followed by efficient-attention + Mix-FFN blocks, emitting the 1/4–1/32 pyramid the all-MLP head
fuses. **Setting note:** SegFormer uses its own MIT backbone (not the R-50-D8 of the convolutional
rungs) and runs at a 1024×1024 crop with sliding-window inference, so its mIoU is reported under that
configuration rather than the 512×1024 / R-50-D8 setting.

**Target.** Beat the strongest convolutional rung (DeepLabV3+, 79.61 mIoU at R-50-D8) on Cityscapes
val by making context native to a globally-receptive hierarchical attention backbone with a trivial
all-MLP decoder — even the smallest MIT encoder landing in the same league, and a large MIT encoder
pushing the mIoU clearly past 79.61.
