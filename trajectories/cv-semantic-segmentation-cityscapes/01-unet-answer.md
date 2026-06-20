**Problem (from baseline).** A classification backbone with a 1×1 classifier and a single bilinear
upsample predicts at a heavily downsampled grid, so boundaries are blurry and thin/small classes are
lost: the deep feature map has the semantics but threw away the resolution, and the shallow maps that
still hold the resolution are ignored.

**Key idea — U-Net.** A symmetric **encoder–decoder with skip connections at every resolution level**.
The encoder (contracting path) is blocks of two 3×3 convs with 2× downsampling between them, saving
each stage's feature map. The decoder (expanding path) mirrors it: at each level, upsample the
lower-resolution decoder feature, **concatenate** the matching-resolution encoder map (the *skip*),
and fuse with two 3×3 convs. The upsampled decoder feature supplies coarse semantics; the concatenated
skip supplies sharp same-resolution spatial detail. A final 1×1 conv maps the full-resolution decoder
feature to class scores.

**Why it works.** Localization information never went into the deep map — it lives in the shallow
encoder maps. Reinjecting the matching-resolution encoder feature at *each* upsampling step means every
decoder level fuses semantics with detail already aligned to the true spatial structure at that scale,
so reconstruction error does not accumulate down the expanding path. Concatenation (not addition) lets
the fusing convs learn how to combine the coarse-semantic and fine-spatial signals channel by channel.
The result sharpens boundaries and recovers thin/small structures the single-upsample baseline lost.

**Change / code.** The decoder upsample-and-fuse block and the U-shaped forward pass.

```python
class UpConvBlock(nn.Module):
    """Upsample the deep feature, concat the matching-resolution skip, fuse."""
    def __init__(self, conv_block, in_channels, skip_channels, out_channels,
                 num_convs=2, upsample_cfg=dict(type='InterpConv')):
        super().__init__()
        self.conv_block = conv_block(in_channels=2 * skip_channels,
                                     out_channels=out_channels,
                                     num_convs=num_convs)
        self.upsample = build_upsample_layer(cfg=upsample_cfg,
                                             in_channels=in_channels,
                                             out_channels=skip_channels)

    def forward(self, skip, x):
        x = self.upsample(x)               # deep feature -> skip resolution
        out = torch.cat([skip, x], dim=1)  # [skip ; upsampled]
        out = self.conv_block(out)
        return out

class UNet(nn.Module):
    def forward(self, x):
        enc_outs = []
        for enc in self.encoder:                       # contracting path
            x = enc(x)
            enc_outs.append(x)                         # save skip per level
        dec_outs = [x]
        for i in reversed(range(len(self.decoder))):   # expanding path
            x = self.decoder[i](enc_outs[i], x)        # upsample + fuse skip
            dec_outs.append(x)
        return dec_outs
```

Each encoder/decoder block is a `BasicConvBlock` of two `3×3` conv+BN+ReLU; the upsample is a learned
`InterpConv` (interpolate-then-1×1-conv). Output stride is set by how far the U descends and climbs
back; the final decoder feature is at full resolution and goes through the `1×1` `conv_seg`.
